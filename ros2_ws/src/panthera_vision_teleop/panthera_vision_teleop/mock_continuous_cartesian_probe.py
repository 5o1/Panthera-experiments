"""Exercise the latest-only streaming backend against GenericSystem."""

from __future__ import annotations

import time
from typing import Optional

from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from .cartesian_trajectory_backend import DEFAULT_URDF
from .kinematics import SerialChain
from .rotation_utils import matrix_to_quaternion


class MockContinuousCartesianProbe(Node):
    def __init__(self) -> None:
        super().__init__("mock_continuous_cartesian_probe")
        self.chain = SerialChain.from_urdf(DEFAULT_URDF, "base_link", "link6")
        self.positions: Optional[np.ndarray] = None
        self.status = ""
        self.target_pub = self.create_publisher(PoseStamped, "/teleop/debug_target", 1)
        self.enabled_pub = self.create_publisher(Bool, "/teleop/enabled", 1)
        self.create_subscription(JointState, "/joint_states", self._feedback, 10)
        self.create_subscription(String, "/teleop/backend_status", self._status, 10)

    def _feedback(self, message: JointState) -> None:
        values = dict(zip(message.name, message.position))
        if all(name in values for name in self.chain.joint_names):
            self.positions = np.array(
                [float(values[name]) for name in self.chain.joint_names]
            )

    def _status(self, message: String) -> None:
        self.status = message.data

    def spin_until(self, predicate, timeout: float, description: str) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if predicate():
                return
        raise TimeoutError(f"{description}; last backend status={self.status}")

    def run(self) -> None:
        self.spin_until(lambda: self.positions is not None, 5.0, "no joint feedback")
        self.spin_until(
            lambda: self.enabled_pub.get_subscription_count() > 0
            and self.target_pub.get_subscription_count() > 0,
            5.0,
            "backend subscriptions unavailable",
        )
        assert self.positions is not None
        start = self.positions.copy()
        target_q = start.copy()
        target_q[0] = min(target_q[0] + 0.04, self.chain.upper[0] - 0.001)
        transform = self.chain.forward(target_q)
        target = PoseStamped()
        target.header.frame_id = "base_link"
        target.pose.position.x, target.pose.position.y, target.pose.position.z = (
            float(value) for value in transform[:3, 3]
        )
        quaternion = matrix_to_quaternion(transform[:3, :3])
        (
            target.pose.orientation.x,
            target.pose.orientation.y,
            target.pose.orientation.z,
            target.pose.orientation.w,
        ) = (float(value) for value in quaternion)

        self.enabled_pub.publish(Bool(data=True))
        publish_until = time.monotonic() + 1.0
        while time.monotonic() < publish_until:
            target.header.stamp = self.get_clock().now().to_msg()
            self.target_pub.publish(target)
            rclpy.spin_once(self, timeout_sec=0.02)

        def reached() -> bool:
            return self.positions is not None and float(
                np.max(np.abs(self.positions - target_q))
            ) < 0.01

        self.spin_until(reached, 4.0, "streaming target not reached")
        self.spin_until(
            lambda: self.status == "TARGET_STALE_HOLD",
            2.0,
            "backend did not stale-hold",
        )
        assert self.positions is not None
        held = self.positions.copy()
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        assert self.positions is not None
        if float(np.max(np.abs(self.positions - held))) > 1e-3:
            raise RuntimeError("robot moved after the latest target became stale")
        self.enabled_pub.publish(Bool(data=False))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockContinuousCartesianProbe()
    try:
        node.run()
        print("PASS: latest-only 50 Hz Cartesian stream reached target and stale-held")
    except Exception as exc:
        node.get_logger().error(f"Continuous mock probe failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
