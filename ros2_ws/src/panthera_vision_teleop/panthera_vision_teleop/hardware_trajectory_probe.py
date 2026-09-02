"""Staged real-hardware trajectory checks. Run only through the guarded script."""

from __future__ import annotations

import argparse
import time
from typing import Optional

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint


JOINTS = [f"joint{index}" for index in range(1, 7)]
LOWER = [-2.4, -0.1, -0.1, -1.6, -1.7, -2.5]
UPPER = [2.4, 3.2, 4.0, 1.6, 1.7, 2.5]


class HardwareTrajectoryProbe(Node):
    def __init__(self) -> None:
        super().__init__("hardware_trajectory_probe")
        self.client = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        self.positions: Optional[list[float]] = None
        self.marker_pub = self.create_publisher(String, "/teleop/experiment_marker", 10)
        self.create_subscription(JointState, "/joint_states", self._state, 10)

    def marker(self, value: str) -> None:
        deadline = time.monotonic() + 3.0
        while self.marker_pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=max(0.0, deadline - time.monotonic()))
        if self.marker_pub.get_subscription_count() == 0:
            raise RuntimeError("rosbag marker subscriber is not ready; refusing hardware motion")
        self.marker_pub.publish(String(data=f"hardware:{value}"))

    def _state(self, message: JointState) -> None:
        by_name = dict(zip(message.name, message.position))
        if all(name in by_name for name in JOINTS):
            self.positions = [float(by_name[name]) for name in JOINTS]

    def wait(self, future, timeout: float):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        if not future.done():
            raise TimeoutError("ROS future timed out")
        return future.result()

    def feedback(self) -> list[float]:
        deadline = time.monotonic() + 5.0
        while self.positions is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.positions is None:
            raise TimeoutError("no named joint feedback")
        return self.positions.copy()

    @staticmethod
    def goal(positions: list[float], duration: float):
        value = FollowJointTrajectory.Goal()
        value.trajectory.joint_names = JOINTS
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        value.trajectory.points = [point]
        return value

    def execute(self, positions: list[float], duration: float, result_timeout: float = 12.0) -> None:
        handle = self.wait(self.client.send_goal_async(self.goal(positions, duration)), 4.0)
        if handle is None or not handle.accepted:
            raise RuntimeError("trajectory rejected")
        result = self.wait(handle.get_result_async(), result_timeout)
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"trajectory ended with status {result.status}")

    def run(self, stage: str) -> None:
        self.marker(f"{stage}:START")
        if not self.client.wait_for_server(timeout_sec=8.0):
            raise TimeoutError("arm trajectory controller unavailable")
        start = self.feedback()
        if not all(low <= value <= high for value, low, high in zip(start, LOWER, UPPER)):
            raise RuntimeError(
                f"initial joint feedback is invalid or outside audited limits: {start}"
            )
        if stage == "hold":
            self.execute(start, 1.0)
            after = self.feedback()
            if max(abs(a - b) for a, b in zip(start, after)) > 0.01:
                raise RuntimeError("hold test moved more than 0.01 rad")
            self.marker(f"{stage}:PASS")
            return
        direction = 1.0 if start[0] + 0.02 <= UPPER[0] else -1.0
        target = start.copy()
        target[0] += direction * 0.005
        if not all(low <= value <= high for value, low, high in zip(target, LOWER, UPPER)):
            raise RuntimeError("micro target exceeds audited limits")
        if stage == "micro":
            self.execute(target, 3.0)
            self.execute(start, 3.0)
            self.marker(f"{stage}:PASS")
            return
        if stage == "cancel":
            target[0] = start[0] + direction * 0.015
            if not LOWER[0] <= target[0] <= UPPER[0]:
                raise RuntimeError("cancel target exceeds audited joint1 limit")
            handle = self.wait(self.client.send_goal_async(self.goal(target, 6.0)), 4.0)
            if handle is None or not handle.accepted:
                raise RuntimeError("cancel-test trajectory rejected")
            cancel_at = time.monotonic() + 0.5
            while time.monotonic() < cancel_at:
                rclpy.spin_once(self, timeout_sec=0.02)
            canceled = self.wait(handle.cancel_goal_async(), 3.0)
            if not canceled.goals_canceling:
                raise RuntimeError("cancel request rejected")
            result = self.wait(handle.get_result_async(), 4.0)
            if result.status != GoalStatus.STATUS_CANCELED:
                raise RuntimeError(f"cancel result status {result.status}")
            held = self.feedback()
            hold_until = time.monotonic() + 0.5
            while time.monotonic() < hold_until:
                rclpy.spin_once(self, timeout_sec=0.02)
            if max(abs(a - b) for a, b in zip(held, self.feedback())) > 0.003:
                raise RuntimeError("arm continued after cancel")
            self.execute(start, 3.0)
            self.marker(f"{stage}:PASS")
            return
        raise ValueError(f"unknown stage {stage}")


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("hold", "micro", "cancel"))
    parsed, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = HardwareTrajectoryProbe()
    try:
        node.run(parsed.stage)
        print(f"PASS: real hardware stage {parsed.stage}")
    except Exception:
        if node.marker_pub.get_subscription_count() > 0:
            node.marker(f"{parsed.stage}:FAIL")
        raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
