"""Bridge validated WebSocket camera observations into ROS 2 topics."""

from __future__ import annotations

import asyncio
import json
import queue
import signal
import threading
import time
from typing import Optional

from geometry_msgs.msg import PoseStamped, Vector3Stamped
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Bool, Float32, String
import websockets

from .rotation_utils import arm_frame, matrix_to_quaternion
from .vision_protocol import EnableRearmLatch, ProtocolError, TransportLagGuard, VisionPacket, parse_packet


class VisionBridgeNode(Node):
    """Run WebSocket I/O off the ROS executor and publish only validated data."""

    def __init__(self) -> None:
        super().__init__("vision_bridge_node")
        self.declare_parameter("websocket_host", "0.0.0.0")
        self.declare_parameter("websocket_port", 8765)
        self.declare_parameter("receive_timeout_sec", 0.5)
        self.declare_parameter("max_transport_lag_sec", 0.25)
        self.declare_parameter("pose_score_min", 0.6)
        self.host = str(self.get_parameter("websocket_host").value)
        self.port = int(self.get_parameter("websocket_port").value)
        self.timeout = float(self.get_parameter("receive_timeout_sec").value)
        self.max_transport_lag = float(self.get_parameter("max_transport_lag_sec").value)
        self.pose_score_min = float(self.get_parameter("pose_score_min").value)
        if self.timeout <= 0.0:
            raise ValueError("receive_timeout_sec must be positive")
        if self.max_transport_lag <= 0.0:
            raise ValueError("max_transport_lag_sec must be positive")
        if not 0.0 <= self.pose_score_min <= 1.0:
            raise ValueError("pose_score_min must be in [0, 1]")

        self.pose_pub = self.create_publisher(PoseStamped, "/teleop/human_pose", 1)
        self.pose_score_pub = self.create_publisher(Float32, "/teleop/pose_score", 1)
        self.gesture_pub = self.create_publisher(String, "/teleop/gesture", 1)
        self.gesture_score_pub = self.create_publisher(Float32, "/teleop/gesture_score", 1)
        self.enabled_pub = self.create_publisher(Bool, "/teleop/enabled", 1)
        self.recalibrate_pub = self.create_publisher(Bool, "/teleop/recalibrate", 1)
        self.depth_features_pub = self.create_publisher(Vector3Stamped, "/teleop/depth_features", 1)
        self.pinch_ratio_pub = self.create_publisher(Float32, "/teleop/pinch_ratio", 1)
        self.depth_clutch_pub = self.create_publisher(Bool, "/teleop/depth_clutch", 1)
        self.orientation_held_pub = self.create_publisher(Bool, "/teleop/orientation_held", 1)
        self.transport_lag_pub = self.create_publisher(Float32, "/teleop/transport_extra_lag_ms", 1)
        self.experiment_marker_pub = self.create_publisher(String, "/teleop/experiment_marker", 1)
        self.latest_hud = {
            "state": "DISABLED",
            "in_flight": False,
            "human_delta": [0.0, 0.0, 0.0],
            "target_delta": [0.0, 0.0, 0.0],
            "clipped": False,
            "motion_limited": False,
            "enabled_axes": [False, True, True],
            "orientation_held": False,
            "depth_clutch": False,
        }
        self.create_subscription(String, "/teleop/hud", self._hud_callback, 1)
        self.create_subscription(String, "/teleop/state", self._state_callback, 1)

        self.incoming: queue.Queue[tuple[VisionPacket, float]] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.last_receive_monotonic: Optional[float] = None
        self.last_orientation: Optional[tuple[float, float, float, float]] = None
        self.watchdog_disabled = True
        self.enable_latch = EnableRearmLatch()
        self.bad_packet_count = 0
        self.timer = self.create_timer(0.02, self._drain_latest)
        self.network_thread = threading.Thread(
            target=self._network_main, name="vision-websocket", daemon=True
        )
        self.network_thread.start()
        self.get_logger().info(f"WebSocket listening on ws://{self.host}:{self.port}")

    async def _connection(self, websocket, _path=None) -> None:
        """Validate a connection's increasing sequence and retain newest packet."""
        previous_seq: Optional[int] = None
        previous_source_time_ms: Optional[int] = None
        lag_guard = TransportLagGuard(self.max_transport_lag * 1000.0)
        peer = getattr(websocket, "remote_address", "unknown")
        self.get_logger().info(f"Vision sender connected: {peer}")
        try:
            async for raw in websocket:
                try:
                    packet = parse_packet(raw, previous_seq, previous_source_time_ms)
                except ProtocolError as exc:
                    self.bad_packet_count += 1
                    if self.bad_packet_count <= 3 or self.bad_packet_count % 100 == 0:
                        self.get_logger().warning(f"Rejected vision packet: {exc}")
                    continue
                previous_seq = packet.seq
                previous_source_time_ms = packet.source_time_ms
                if not lag_guard.accept(packet.source_time_ms, time.monotonic() * 1000.0):
                    self.bad_packet_count += 1
                    if self.bad_packet_count <= 3 or self.bad_packet_count % 100 == 0:
                        self.get_logger().warning("Rejected delayed vision packet backlog")
                    continue
                try:
                    self.incoming.get_nowait()
                except queue.Empty:
                    pass
                self.incoming.put_nowait((packet, lag_guard.latest_extra_delay_ms))
                await websocket.send(json.dumps({
                    "v": 1,
                    "type": "telemetry",
                    **self.latest_hud,
                    "recording_ready": self.experiment_marker_pub.get_subscription_count() > 0,
                }, separators=(",", ":")))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if not self.stop_event.is_set() and rclpy.ok():
                self.get_logger().warning(f"Vision sender disconnected: {peer}")

    async def _serve(self) -> None:
        async with websockets.serve(
            self._connection, self.host, self.port, ping_interval=10, ping_timeout=10, max_size=65536
        ):
            while not self.stop_event.is_set():
                await asyncio.sleep(0.05)

    def _network_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as exc:  # reported into ROS log; executor remains responsive
            if not self.stop_event.is_set():
                self.get_logger().error(f"WebSocket server stopped: {type(exc).__name__}: {exc}")

    def _publish_disabled(self) -> None:
        self.enable_latch.force_lock()
        self.enabled_pub.publish(Bool(data=False))
        self.recalibrate_pub.publish(Bool(data=False))
        self.depth_clutch_pub.publish(Bool(data=False))
        self.orientation_held_pub.publish(Bool(data=True))
        self.watchdog_disabled = True

    def _hud_callback(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(value, dict):
            self.latest_hud = value

    def _state_callback(self, message: String) -> None:
        self.latest_hud["state"] = message.data

    def _drain_latest(self) -> None:
        packet: Optional[VisionPacket] = None
        transport_lag_ms = 0.0
        while True:
            try:
                packet, transport_lag_ms = self.incoming.get_nowait()
            except queue.Empty:
                break
        now = time.monotonic()
        if packet is not None:
            self.last_receive_monotonic = now
            self.transport_lag_pub.publish(Float32(data=float(transport_lag_ms)))
            self._publish_packet(packet)
        if self.last_receive_monotonic is None or now - self.last_receive_monotonic > self.timeout:
            if not self.watchdog_disabled:
                self.get_logger().warning("Vision timeout: teleoperation disabled")
                self._publish_disabled()

    def _publish_packet(self, packet: VisionPacket) -> None:
        pose_usable = packet.pose_valid and packet.pose_score >= self.pose_score_min
        orientation = self.last_orientation
        orientation_held = False
        if pose_usable:
            if packet.orientation_valid:
                try:
                    orientation_array = matrix_to_quaternion(
                        arm_frame(packet.shoulder, packet.elbow, packet.wrist)
                    )
                    orientation = tuple(float(v) for v in orientation_array)
                    self.last_orientation = orientation
                except ValueError:
                    orientation_held = orientation is not None
                    pose_usable = orientation is not None
            else:
                orientation_held = orientation is not None
                pose_usable = orientation is not None

        # Score is published before pose so the mapper can gate this same frame.
        self.pose_score_pub.publish(Float32(data=float(packet.pose_score)))
        if pose_usable and orientation is not None:
            message = PoseStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "camera"
            message.pose.position.x = packet.wrist[0] - packet.shoulder[0]
            message.pose.position.y = packet.wrist[1] - packet.shoulder[1]
            message.pose.position.z = packet.wrist[2] - packet.shoulder[2]
            message.pose.orientation.x, message.pose.orientation.y, message.pose.orientation.z, message.pose.orientation.w = orientation
            self.pose_pub.publish(message)

        gesture = packet.gesture if packet.hand_valid else "Unknown"
        gesture_score = packet.gesture_score if packet.hand_valid else 0.0
        self.gesture_pub.publish(String(data=gesture))
        self.gesture_score_pub.publish(Float32(data=float(gesture_score)))
        features = Vector3Stamped()
        features.header.stamp = self.get_clock().now().to_msg()
        features.header.frame_id = "camera"
        features.vector.x = packet.depth_world_m
        features.vector.y = packet.image_reach_ratio
        features.vector.z = packet.image_arm_scale
        self.depth_features_pub.publish(features)
        self.pinch_ratio_pub.publish(Float32(data=float(packet.pinch_ratio)))
        self.experiment_marker_pub.publish(String(data=packet.experiment_marker))
        self.orientation_held_pub.publish(Bool(data=orientation_held))
        effective_enabled = self.enable_latch.update(packet.enabled, pose_usable)
        self.enabled_pub.publish(Bool(data=effective_enabled))
        self.recalibrate_pub.publish(Bool(data=bool(packet.recalibrate and effective_enabled)))
        self.depth_clutch_pub.publish(Bool(data=bool(packet.depth_clutch and effective_enabled)))
        self.latest_hud["orientation_held"] = orientation_held
        self.latest_hud["depth_clutch"] = bool(packet.depth_clutch and effective_enabled)
        self.watchdog_disabled = not effective_enabled

    def destroy_node(self) -> bool:
        self.stop_event.set()
        if self.network_thread.is_alive():
            self.network_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.stop_event.set()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
