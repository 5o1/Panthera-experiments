"""Verify the lightweight FK/IK backend against GenericSystem."""

from __future__ import annotations

import time
from typing import Optional

from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .kinematics import SerialChain
from .rotation_utils import matrix_to_quaternion
from .cartesian_trajectory_backend import DEFAULT_URDF


class MockCartesianProbe(Node):
    def __init__(self) -> None:
        super().__init__("mock_cartesian_probe")
        self.chain = SerialChain.from_urdf(DEFAULT_URDF, "base_link", "link6")
        self.positions: Optional[np.ndarray] = None
        self.backend_status = ""
        self.target_pub = self.create_publisher(PoseStamped, "/teleop/debug_target", 1)
        self.reset_client = self.create_client(Trigger, "/teleop/reset_backend")
        self.create_subscription(JointState, "/joint_states", self._state, 10)
        self.create_subscription(String, "/teleop/backend_status", self._status, 10)

    def _state(self, message: JointState) -> None:
        by_name = dict(zip(message.name, message.position))
        if all(name in by_name for name in self.chain.joint_names):
            self.positions = np.array([float(by_name[name]) for name in self.chain.joint_names])

    def _status(self, message: String) -> None:
        self.backend_status = message.data

    def spin_until(self, predicate, timeout: float, description: str) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if predicate():
                return
        raise TimeoutError(description)

    def run(self) -> None:
        self.spin_until(lambda: self.positions is not None, 4.0, "no joint feedback")
        assert self.positions is not None
        target_q = self.positions.copy()
        target_q[0] = min(target_q[0] + 0.005, self.chain.upper[0] - 0.001)
        target_transform = self.chain.forward(target_q)
        message = PoseStamped()
        message.header.frame_id = "base_link"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (float(value) for value in target_transform[:3, 3])
        quaternion = matrix_to_quaternion(target_transform[:3, :3])
        message.pose.orientation.x, message.pose.orientation.y, message.pose.orientation.z, message.pose.orientation.w = (float(value) for value in quaternion)
        publish_until = time.monotonic() + 0.35
        while time.monotonic() < publish_until:
            message.header.stamp = self.get_clock().now().to_msg()
            self.target_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.02)

        def pose_reached() -> bool:
            if self.positions is None:
                return False
            current = self.chain.forward(self.positions)
            return float(np.linalg.norm(current[:3, 3] - target_transform[:3, 3])) < 0.002

        self.spin_until(pose_reached, 4.0, "Cartesian target was not reached")
        self.spin_until(lambda: self.backend_status == "STALE_HOLD", 2.0, "backend did not enter stale hold")
        assert self.positions is not None
        held = self.positions.copy()
        hold_until = time.monotonic() + 0.4
        while time.monotonic() < hold_until:
            rclpy.spin_once(self, timeout_sec=0.02)
        assert self.positions is not None
        if float(np.max(np.abs(self.positions - held))) > 1e-3:
            raise RuntimeError("joints moved after Cartesian target became stale")

        # A persistently unreachable target must latch after the configured
        # consecutive-failure threshold. It must stay locked until the
        # operator invokes the explicit reset service.
        unreachable = PoseStamped()
        unreachable.header.frame_id = "base_link"
        unreachable.pose.position.x = 10.0
        unreachable.pose.position.y = 10.0
        unreachable.pose.position.z = 10.0
        unreachable.pose.orientation.w = 1.0
        publish_until = time.monotonic() + 0.25
        while time.monotonic() < publish_until:
            unreachable.header.stamp = self.get_clock().now().to_msg()
            self.target_pub.publish(unreachable)
            rclpy.spin_once(self, timeout_sec=0.02)
        self.spin_until(
            lambda: self.backend_status.startswith("FAULT_LOCK_IK_FAILED"),
            2.0,
            "backend did not latch repeated IK failures",
        )
        if not self.reset_client.wait_for_service(timeout_sec=2.0):
            raise TimeoutError("backend reset service unavailable")
        reset_future = self.reset_client.call_async(Trigger.Request())
        self.spin_until(reset_future.done, 2.0, "backend reset service timed out")
        reset_result = reset_future.result()
        if reset_result is None or not reset_result.success:
            raise RuntimeError("backend failure lock did not reset")

        assert self.positions is not None
        current_transform = self.chain.forward(self.positions)
        recovery = PoseStamped()
        recovery.header.frame_id = "base_link"
        recovery.pose.position.x, recovery.pose.position.y, recovery.pose.position.z = (
            float(value) for value in current_transform[:3, 3]
        )
        recovery_q = matrix_to_quaternion(current_transform[:3, :3])
        (
            recovery.pose.orientation.x,
            recovery.pose.orientation.y,
            recovery.pose.orientation.z,
            recovery.pose.orientation.w,
        ) = (float(value) for value in recovery_q)
        self.target_pub.publish(recovery)
        self.spin_until(
            lambda: self.backend_status == "AT_TARGET",
            2.0,
            "backend did not accept a fresh target after explicit reset",
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockCartesianProbe()
    try:
        node.run()
        print("PASS: lightweight FK/IK backend reached target, stale-held, fault-locked, and reset")
    except Exception as exc:
        node.get_logger().error(f"Mock Cartesian probe failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
