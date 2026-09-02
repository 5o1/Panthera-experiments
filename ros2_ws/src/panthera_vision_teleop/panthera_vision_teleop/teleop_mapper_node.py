"""Safely retarget human arm observations to Panthera end-effector targets."""

from __future__ import annotations

import math
import json
import signal
import time
from typing import Optional

from example_interfaces.msg import Bool as ExampleBool
from geometry_msgs.msg import PoseStamped, Vector3Stamped
import numpy as np
from panthera_interfaces.msg import ArmStatus, EndPoseEuler, PosCmd
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Bool, Float32, String

from .teleop_logic import (
    DepthClutch,
    GestureDebouncer,
    MappingConfig,
    PerAxisCommandGate,
    PinchHysteresis,
    TargetPose,
    TeleopMapping,
    TeleopState,
    TeleopStateMachine,
    limit_vector_step,
)


class TeleopMapperNode(Node):
    """Own calibration, dry-run output and at-most-one in-flight robot command."""

    def __init__(self) -> None:
        super().__init__("teleop_mapper_node")
        defaults = {
            "command_rate_hz": 5.0,
            "pose_score_min": 0.6,
            "gesture_score_min": 0.7,
            "gesture_stable_frames": 5,
            "pinch_close_ratio": 0.35,
            "pinch_open_ratio": 0.65,
            "pinch_stable_sec": 0.6,
            "gripper_cooldown_sec": 0.75,
            "position_alpha": 0.25,
            "position_alpha_axes": [0.25, 0.25, 0.25],
            "rotation_alpha": 0.25,
            "position_gain": [0.0, 0.25, 0.12],
            "max_position_offset": [0.0, 0.05, 0.04],
            "position_deadband_m": [0.0, 0.002, 0.002],
            "max_position_velocity_mps": [0.02, 0.04, 0.03],
            "max_position_acceleration_mps2": [0.08, 0.15, 0.12],
            "position_axis_matrix": [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            "max_rotation_deg": 20.0,
            "orientation_axes": [1.0, 0.0, 0.0],
            "max_angular_velocity_degps": 10.0,
            "max_angular_acceleration_degps2": 30.0,
            "stale_timeout_sec": 0.5,
            "enable_orientation": False,
            "open_labels": ["Open_Palm", "snake_open"],
            "close_labels": ["Closed_Fist", "snake_closed"],
            "publish_robot_commands": False,
            # The stock gesture model may classify a relaxed hand as
            # Open_Palm.  Real gripper output therefore has its own gate.
            "publish_gripper_commands": False,
            "command_position_tolerance_m": 0.006,
            "command_rotation_tolerance_rad": 0.06,
            "command_timeout_sec": 16.0,
            "command_enter_delta_m": [0.006, 0.006, 0.006],
            "command_exit_delta_m": [0.003, 0.003, 0.003],
            # The official driver blocks while executing /pos_cmd, so keep
            # each individually uninterruptible move short.
            "max_command_step_m": 0.012,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        axis = np.asarray(self.get_parameter("position_axis_matrix").value, dtype=float).reshape(3, 3)
        config = MappingConfig(
            position_gain=np.asarray(self.get_parameter("position_gain").value, dtype=float),
            max_position_offset=np.asarray(self.get_parameter("max_position_offset").value, dtype=float),
            position_deadband_m=np.asarray(self.get_parameter("position_deadband_m").value, dtype=float),
            max_position_velocity_mps=np.asarray(self.get_parameter("max_position_velocity_mps").value, dtype=float),
            max_position_acceleration_mps2=np.asarray(self.get_parameter("max_position_acceleration_mps2").value, dtype=float),
            position_axis_matrix=axis,
            position_alpha=float(self.get_parameter("position_alpha").value),
            position_alpha_axes=np.asarray(self.get_parameter("position_alpha_axes").value, dtype=float),
            rotation_alpha=float(self.get_parameter("rotation_alpha").value),
            max_rotation_rad=math.radians(float(self.get_parameter("max_rotation_deg").value)),
            orientation_axes=np.asarray(self.get_parameter("orientation_axes").value, dtype=float),
            max_angular_velocity_radps=math.radians(float(self.get_parameter("max_angular_velocity_degps").value)),
            max_angular_acceleration_radps2=math.radians(float(self.get_parameter("max_angular_acceleration_degps2").value)),
            stale_timeout_sec=float(self.get_parameter("stale_timeout_sec").value),
            enable_orientation=bool(self.get_parameter("enable_orientation").value),
        )
        self.mapping = TeleopMapping(config)
        self.pose_score_min = float(self.get_parameter("pose_score_min").value)
        self.publish_robot_commands = bool(self.get_parameter("publish_robot_commands").value)
        self.publish_gripper_commands = bool(self.get_parameter("publish_gripper_commands").value)
        self.command_position_tolerance = float(self.get_parameter("command_position_tolerance_m").value)
        self.command_rotation_tolerance = float(self.get_parameter("command_rotation_tolerance_rad").value)
        self.command_timeout = float(self.get_parameter("command_timeout_sec").value)
        self.max_command_step = float(self.get_parameter("max_command_step_m").value)
        if self.max_command_step <= 0.0:
            raise ValueError("max_command_step_m must be positive")
        self.command_gate = PerAxisCommandGate(
            self.get_parameter("command_enter_delta_m").value,
            self.get_parameter("command_exit_delta_m").value,
        )
        self.gestures = GestureDebouncer(
            self.get_parameter("open_labels").value,
            self.get_parameter("close_labels").value,
            float(self.get_parameter("gesture_score_min").value),
            int(self.get_parameter("gesture_stable_frames").value),
        )
        self.pinch = PinchHysteresis(
            float(self.get_parameter("pinch_close_ratio").value),
            float(self.get_parameter("pinch_open_ratio").value),
            float(self.get_parameter("pinch_stable_sec").value),
            float(self.get_parameter("gripper_cooldown_sec").value),
        )

        self.debug_target_pub = self.create_publisher(PoseStamped, "/teleop/debug_target", 1)
        self.debug_gripper_pub = self.create_publisher(Bool, "/teleop/debug_gripper", 1)
        self.classifier_gripper_pub = self.create_publisher(Bool, "/teleop/diagnostics/classifier_gripper_candidate", 1)
        self.state_pub = self.create_publisher(String, "/teleop/state", 1)
        self.command_in_flight_pub = self.create_publisher(Bool, "/teleop/command_in_flight", 1)
        self.hud_pub = self.create_publisher(String, "/teleop/hud", 1)
        self.human_delta_pub = self.create_publisher(Vector3Stamped, "/teleop/diagnostics/human_delta_camera", 1)
        self.raw_offset_pub = self.create_publisher(Vector3Stamped, "/teleop/diagnostics/raw_robot_offset", 1)
        self.deadbanded_offset_pub = self.create_publisher(Vector3Stamped, "/teleop/diagnostics/deadbanded_robot_offset", 1)
        self.limited_offset_pub = self.create_publisher(Vector3Stamped, "/teleop/diagnostics/limited_robot_offset", 1)
        self.filtered_offset_pub = self.create_publisher(Vector3Stamped, "/teleop/diagnostics/filtered_robot_offset", 1)
        self.motion_limited_offset_pub = self.create_publisher(Vector3Stamped, "/teleop/diagnostics/motion_limited_robot_offset", 1)
        self.robot_pose_pub = self.create_publisher(PosCmd, "/pos_cmd", 1) if self.publish_robot_commands else None
        self.robot_gripper_pub = (
            self.create_publisher(ExampleBool, "/gripper_cmd", 1)
            if self.publish_robot_commands and self.publish_gripper_commands
            else None
        )

        self.create_subscription(PoseStamped, "/teleop/human_pose", self._human_pose, 1)
        self.create_subscription(Float32, "/teleop/pose_score", self._pose_score_callback, 1)
        self.create_subscription(String, "/teleop/gesture", self._gesture_callback, 1)
        self.create_subscription(Float32, "/teleop/gesture_score", self._gesture_score_callback, 1)
        self.create_subscription(Float32, "/teleop/pinch_ratio", self._pinch_ratio_callback, 1)
        self.create_subscription(Bool, "/teleop/depth_clutch", self._depth_clutch_callback, 1)
        self.create_subscription(Bool, "/teleop/enabled", self._enabled_callback, 1)
        self.create_subscription(Bool, "/teleop/recalibrate", self._recalibrate_callback, 1)
        self.create_subscription(EndPoseEuler, "/end_pose_euler", self._robot_pose, 1)
        self.create_subscription(ArmStatus, "/arm_status", self._arm_status, 1)

        self.pose_score = 0.0
        self.depth_clutch_active = False
        self.depth_clutch = DepthClutch()
        self.gesture = "Unknown"
        self.source_enabled = False
        self.arm_ok = False
        self.latest_measured_position: Optional[np.ndarray] = None
        self.latest_measured_rpy: Optional[np.ndarray] = None
        self.in_flight_target: Optional[TargetPose] = None
        self.in_flight_since: Optional[float] = None
        self.robot_command_lockout = False
        self.state_machine = TeleopStateMachine()
        self.last_published_state: Optional[TeleopState] = None
        rate = float(self.get_parameter("command_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("command_rate_hz must be positive")
        self.timer = self.create_timer(1.0 / rate, self._tick)
        mode = "TRUE ARM COMMANDS" if self.publish_robot_commands else "DRY-RUN"
        gripper_mode = "enabled" if self.robot_gripper_pub is not None else "disabled"
        self.get_logger().warning(f"Teleop mapper started in {mode} mode; real gripper commands {gripper_mode}")
        self._publish_state()

    def _pose_score_callback(self, msg: Float32) -> None:
        self.pose_score = float(msg.data)

    def _human_pose(self, msg: PoseStamped) -> None:
        if self.pose_score < self.pose_score_min:
            return
        p, q = msg.pose.position, msg.pose.orientation
        try:
            gated_depth = self.depth_clutch.update(
                float(p.z), self.depth_clutch_active or not self.source_enabled
            )
            self.mapping.update_human([p.x, p.y, gated_depth], [q.x, q.y, q.z, q.w], time.monotonic())
            if self.mapping.calibrated:
                self.state_machine.inputs_ready()
                self._publish_state()
        except ValueError as exc:
            self.get_logger().warning(f"Rejected human pose: {exc}")

    def _enabled_callback(self, msg: Bool) -> None:
        requested = bool(msg.data)
        effective = requested and self.pose_score >= self.pose_score_min
        if not effective:
            changed = self.source_enabled
            self.source_enabled = False
            self.mapping.set_enabled(False)
            self.depth_clutch_active = False
            self.depth_clutch.reset()
            self.state_machine.source_enable(False, False)
            self._publish_state()
            if changed:
                self.in_flight_target = None
                self.command_gate.reset()
                self.get_logger().warning("Teleoperation disabled; no new robot targets will be sent")
            return
        next_state = self.state_machine.source_enable(True, self.mapping.calibrated)
        if next_state in (TeleopState.STALE_LOCK, TeleopState.FAULT_LOCK):
            self.source_enabled = False
            self.mapping.set_enabled(False)
            self._publish_state()
            return
        changed = not self.source_enabled
        self.source_enabled = True
        calibrated = self.mapping.set_enabled(True)
        if calibrated:
            self.state_machine.inputs_ready()
        self._publish_state()
        if changed:
            self.robot_command_lockout = False
            self.get_logger().info("Teleoperation enabled; calibration captured" if calibrated else "Teleoperation enabled; waiting for human and robot poses")

    def _recalibrate_callback(self, msg: Bool) -> None:
        if msg.data and self.source_enabled:
            if self.mapping.calibrate():
                self.state_machine.inputs_ready()
                self._publish_state()
                self.robot_command_lockout = False
                self.in_flight_target = None
                self.command_gate.reset()
                self.get_logger().info("Teleoperation zero poses recalibrated")
            else:
                self.get_logger().warning("Recalibration requested before both poses were available")

    def _robot_pose(self, msg: EndPoseEuler) -> None:
        xyz = np.array([msg.x, msg.y, msg.z], dtype=float)
        rpy = np.array([msg.roll, msg.pitch, msg.yaw], dtype=float)
        try:
            self.mapping.update_robot(xyz, rpy)
        except ValueError as exc:
            self.get_logger().error(f"Rejected robot feedback: {exc}")
            return
        self.latest_measured_position, self.latest_measured_rpy = xyz, rpy
        if self.mapping.calibrated:
            self.state_machine.inputs_ready()
            self._publish_state()
        if self.in_flight_target is not None:
            position_error = float(np.linalg.norm(xyz - self.in_flight_target.position))
            rotation_error = float(np.max(np.abs(_wrapped_angle_delta(rpy, self.in_flight_target.rpy))))
            if position_error <= self.command_position_tolerance and rotation_error <= self.command_rotation_tolerance:
                self.in_flight_target = None
                self.in_flight_since = None

    def _arm_status(self, msg: ArmStatus) -> None:
        fault_free = bool(
            msg.arm_enabled
            and not msg.error_message
            and not any(msg.motor_faults)
            and not any(msg.joint_at_limit)
        )
        # Only an idle controller may receive the next point target.  A normal
        # moving state pauses output but does not itself create a lockout.
        self.arm_ok = fault_free and msg.motion_status == 0
        if not fault_free and self.source_enabled and self.publish_robot_commands:
            if not self.robot_command_lockout:
                self.get_logger().error("Unsafe arm status received; robot command output locked")
            self.robot_command_lockout = True
            self.in_flight_target = None
            self.in_flight_since = None
            self.command_gate.reset()
            self.state_machine.fault()
            self._publish_state()

    def _gesture_callback(self, msg: String) -> None:
        self.gesture = msg.data

    def _depth_clutch_callback(self, msg: Bool) -> None:
        self.depth_clutch_active = bool(msg.data and self.source_enabled)

    def _gesture_score_callback(self, msg: Float32) -> None:
        transition = self.gestures.update(self.gesture, float(msg.data), self.source_enabled)
        if transition is None:
            return
        self.classifier_gripper_pub.publish(Bool(data=transition))

    def _pinch_ratio_callback(self, msg: Float32) -> None:
        transition = self.pinch.update(float(msg.data), time.monotonic(), self.source_enabled)
        if transition is None:
            return
        self.debug_gripper_pub.publish(Bool(data=transition))
        self.get_logger().info("Pinch command: OPEN" if transition else "Pinch command: CLOSE")
        if self.robot_gripper_pub is not None and self.arm_ok and not self.robot_command_lockout:
            self.robot_gripper_pub.publish(ExampleBool(data=transition))

    def _publish_debug(self, target: TargetPose) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in target.position)
        # Debug output must be a valid Pose quaternion, not RPY stuffed into xyz.
        from .rotation_utils import matrix_to_quaternion, rpy_to_matrix
        q = matrix_to_quaternion(rpy_to_matrix(*target.rpy))
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = (float(v) for v in q)
        self.debug_target_pub.publish(msg)

        diagnostics = self.mapping.latest_diagnostics
        if diagnostics is None:
            return
        stamp = msg.header.stamp
        self._publish_vector(self.human_delta_pub, stamp, "camera", diagnostics.human_delta_camera)
        self._publish_vector(self.raw_offset_pub, stamp, "base_link", diagnostics.raw_robot_offset)
        self._publish_vector(self.deadbanded_offset_pub, stamp, "base_link", diagnostics.deadbanded_robot_offset)
        self._publish_vector(self.limited_offset_pub, stamp, "base_link", diagnostics.limited_robot_offset)
        self._publish_vector(self.filtered_offset_pub, stamp, "base_link", diagnostics.filtered_robot_offset)
        self._publish_vector(self.motion_limited_offset_pub, stamp, "base_link", diagnostics.motion_limited_robot_offset)
        hud = {
            "state": self.state_machine.state.value,
            "in_flight": self.in_flight_target is not None,
            "human_delta": [float(value) for value in diagnostics.human_delta_camera],
            "target_delta": [float(value) for value in diagnostics.motion_limited_robot_offset],
            "clipped": bool(np.any(np.abs(diagnostics.deadbanded_robot_offset - diagnostics.limited_robot_offset) > 1e-9)),
            "motion_limited": bool(np.any(np.abs(diagnostics.filtered_robot_offset - diagnostics.motion_limited_robot_offset) > 1e-9)),
            "enabled_axes": [bool(value != 0.0) for value in self.mapping.config.position_gain],
        }
        self.hud_pub.publish(String(data=json.dumps(hud, separators=(",", ":"))))

    @staticmethod
    def _publish_vector(publisher, stamp, frame_id: str, values: np.ndarray) -> None:
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.vector.x, msg.vector.y, msg.vector.z = (float(value) for value in values)
        publisher.publish(msg)

    def _publish_state(self) -> None:
        if self.state_machine.state == self.last_published_state:
            return
        self.last_published_state = self.state_machine.state
        self.state_pub.publish(String(data=self.state_machine.state.value))

    def _tick(self) -> None:
        now = time.monotonic()
        target = self.mapping.target(now)
        if target is None:
            latest_time = self.mapping.latest_human_time
            if (
                self.source_enabled
                and latest_time is not None
                and now - latest_time > self.mapping.config.stale_timeout_sec
            ):
                self.state_machine.stale()
                self.mapping.invalidate_input()
                self.source_enabled = False
                self.robot_command_lockout = True
                self.in_flight_target = None
                self.in_flight_since = None
                self.command_gate.reset()
                self._publish_state()
                self.get_logger().error("Human target became stale; explicit disable/re-enable required")
            return
        self.command_in_flight_pub.publish(Bool(data=self.in_flight_target is not None))
        self._publish_debug(target)
        if self.robot_pose_pub is None or self.robot_command_lockout or not self.arm_ok:
            return
        if self.in_flight_target is not None:
            if self.in_flight_since is not None and now - self.in_flight_since > self.command_timeout:
                self.robot_command_lockout = True
                self.get_logger().error("Robot target timed out; command output locked until disable/re-enable or recalibrate")
            return
        if self.robot_pose_pub.get_subscription_count() < 1 or self.latest_measured_position is None:
            return
        if not self.command_gate.update(target.position - self.latest_measured_position):
            return
        command_position = limit_vector_step(
            self.latest_measured_position,
            target.position,
            self.max_command_step,
        )
        command_target = TargetPose(command_position, target.rpy.copy())
        msg = PosCmd()
        msg.x, msg.y, msg.z = (float(v) for v in command_target.position)
        msg.roll, msg.pitch, msg.yaw = (float(v) for v in command_target.rpy)
        msg.gripper, msg.mode1, msg.mode2 = -1.0, 0, 0
        self.robot_pose_pub.publish(msg)
        self.in_flight_target, self.in_flight_since = command_target, now
        self.get_logger().info(f"Sent one target xyz=[{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]")


def _wrapped_angle_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopMapperNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
