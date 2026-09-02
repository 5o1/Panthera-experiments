"""Move to one named six-joint pose with a bounded cubic trajectory."""

from __future__ import annotations

import argparse
import math
import time
from typing import Optional

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from .cartesian_trajectory_backend import DEFAULT_URDF
from .kinematics import SerialChain


def six_values(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("expected six finite comma-separated radians")
    return values


class JointTrajectoryMove(Node):
    def __init__(self) -> None:
        super().__init__("joint_trajectory_move")
        self.chain = SerialChain.from_urdf(DEFAULT_URDF, "base_link", "link6")
        self.positions: Optional[np.ndarray] = None
        self.client = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        self.create_subscription(JointState, "/joint_states", self._feedback, 10)

    def _feedback(self, message: JointState) -> None:
        by_name = dict(zip(message.name, message.position))
        if all(name in by_name for name in self.chain.joint_names):
            value = np.array(
                [float(by_name[name]) for name in self.chain.joint_names]
            )
            if np.all(np.isfinite(value)):
                self.positions = value

    def wait_future(self, future, timeout: float):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        if not future.done():
            raise TimeoutError("trajectory service timed out")
        return future.result()

    def feedback(self, timeout: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout
        while self.positions is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.positions is None:
            raise TimeoutError("no named six-joint feedback")
        return self.positions.copy()

    def goal(
        self,
        start: np.ndarray,
        target: np.ndarray,
        max_velocity: float,
        max_acceleration: float,
        sample_rate: float = 50.0,
    ) -> tuple[FollowJointTrajectory.Goal, float]:
        delta = target - start
        largest = float(np.max(np.abs(delta)))
        duration = max(
            0.5,
            1.5 * largest / max_velocity,
            math.sqrt(6.0 * largest / max_acceleration),
        )
        count = max(2, math.ceil(duration * sample_rate))
        duration = count / sample_rate
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.chain.joint_names)
        for index in range(1, count + 1):
            elapsed = index / sample_rate
            u = elapsed / duration
            scale = 3.0 * u * u - 2.0 * u * u * u
            scale_velocity = (6.0 * u - 6.0 * u * u) / duration
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in start + delta * scale]
            point.velocities = [float(value) for value in delta * scale_velocity]
            point.time_from_start.sec = int(elapsed)
            point.time_from_start.nanosec = int((elapsed - int(elapsed)) * 1e9)
            goal.trajectory.points.append(point)
        return goal, duration

    def execute(
        self,
        target: list[float],
        max_velocity: float,
        max_acceleration: float,
        tolerance: float,
    ) -> None:
        if not self.client.wait_for_server(timeout_sec=8.0):
            raise TimeoutError("arm trajectory controller is unavailable")
        start = self.feedback()
        target_array = np.asarray(target, dtype=float)
        if np.any(target_array < self.chain.lower) or np.any(target_array > self.chain.upper):
            raise ValueError("target exceeds URDF joint limits")
        goal, duration = self.goal(
            start, target_array, max_velocity, max_acceleration
        )
        handle = self.wait_future(self.client.send_goal_async(goal), 4.0)
        if handle is None or not handle.accepted:
            raise RuntimeError("joint trajectory was rejected")
        result = self.wait_future(handle.get_result_async(), duration + 8.0)
        if result is None or result.status != GoalStatus.STATUS_SUCCEEDED:
            status = None if result is None else result.status
            raise RuntimeError(f"joint trajectory failed with status {status}")

        stable = 0
        deadline = time.monotonic() + 5.0
        last_error = math.inf
        while stable < 5 and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.positions is None:
                stable = 0
                continue
            last_error = float(np.max(np.abs(self.positions - target_array)))
            stable = stable + 1 if last_error <= tolerance else 0
        if stable < 5:
            raise RuntimeError(
                f"joint feedback did not settle: max_error={last_error:.6f} rad"
            )
        print(
            f"Joint target reached: duration={duration:.3f}s, "
            f"max_error={last_error:.6f}rad"
        )


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=six_values, required=True)
    parser.add_argument("--max-velocity", type=float, default=0.6)
    parser.add_argument("--max-acceleration", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parsed, ros_args = parser.parse_known_args(args)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (parsed.max_velocity, parsed.max_acceleration, parsed.tolerance)
    ):
        parser.error("motion limits and tolerance must be positive and finite")
    if parsed.max_velocity > 1.0:
        parser.error("--max-velocity exceeds the Host/hardware 1.0 rad/s limit")

    rclpy.init(args=ros_args)
    node = JointTrajectoryMove()
    try:
        node.execute(
            parsed.target,
            parsed.max_velocity,
            parsed.max_acceleration,
            parsed.tolerance,
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
