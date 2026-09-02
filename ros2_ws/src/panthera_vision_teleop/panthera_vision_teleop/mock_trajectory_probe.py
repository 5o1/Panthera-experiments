"""Probe FollowJointTrajectory success and cancellation on mock hardware only."""

from __future__ import annotations

import math
import time
from typing import Optional

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]


class MockTrajectoryProbe(Node):
    """Validate the standard controller contract without loading a robot SDK."""

    def __init__(self) -> None:
        super().__init__("mock_trajectory_probe")
        self.client = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self.latest_positions: Optional[list[float]] = None
        self.latest_velocities: Optional[list[float]] = None
        self.create_subscription(JointState, "/joint_states", self._joint_state, 10)

    def _joint_state(self, message: JointState) -> None:
        positions_by_name = dict(zip(message.name, message.position))
        velocities_by_name = dict(zip(message.name, message.velocity))
        if not all(name in positions_by_name for name in JOINT_NAMES):
            return
        self.latest_positions = [float(positions_by_name[name]) for name in JOINT_NAMES]
        self.latest_velocities = [float(velocities_by_name.get(name, 0.0)) for name in JOINT_NAMES]

    def _wait_future(self, future, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        if not future.done():
            raise TimeoutError("ROS future timed out")
        return future.result()

    def _wait_joint_state(self, timeout_sec: float = 3.0) -> None:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and self.latest_positions is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        if self.latest_positions is None:
            raise TimeoutError("no /joint_states feedback")

    def _goal(self, positions: list[float], duration_sec: float):
        if len(positions) != len(JOINT_NAMES):
            raise ValueError("expected six joint positions")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        whole_seconds = int(duration_sec)
        point.time_from_start.sec = whole_seconds
        point.time_from_start.nanosec = int((duration_sec - whole_seconds) * 1.0e9)
        goal.trajectory.points = [point]
        return goal

    def run(self) -> None:
        if not self.client.wait_for_server(timeout_sec=5.0):
            raise TimeoutError("FollowJointTrajectory action server unavailable")
        self._wait_joint_state()

        # The controller configuration disallows partial joint goals. This
        # catches a common name/count mismatch before any useful trajectory.
        partial_goal = self._goal([0.0] * 6, 0.5)
        partial_goal.trajectory.joint_names = JOINT_NAMES[:-1]
        partial_goal.trajectory.points[0].positions = [0.0] * 5
        partial_handle = self._wait_future(self.client.send_goal_async(partial_goal), 3.0)
        if partial_handle is not None and partial_handle.accepted:
            partial_result = self._wait_future(partial_handle.get_result_async(), 3.0)
            if partial_result.status == GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError("partial joint trajectory unexpectedly succeeded")

        # A short goal must complete successfully and reach the commanded state.
        success_goal = self._goal([0.0, 0.1, 0.1, 0.0, 0.0, 0.0], 0.6)
        success_handle = self._wait_future(self.client.send_goal_async(success_goal), 3.0)
        if success_handle is None or not success_handle.accepted:
            raise RuntimeError("short trajectory goal was rejected")
        success_result = self._wait_future(success_handle.get_result_async(), 3.0)
        if success_result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"short trajectory status was {success_result.status}, expected SUCCEEDED")
        self._wait_joint_state()
        assert self.latest_positions is not None
        if max(abs(actual - expected) for actual, expected in zip(
            self.latest_positions, success_goal.trajectory.points[0].positions
        )) > 1.0e-3:
            raise RuntimeError("mock feedback did not reach the short trajectory target")

        # Cancel a long motion, then prove the mock controller holds rather than
        # continuing toward the obsolete target.
        cancel_goal = self._goal([0.0, 0.8, 0.8, 0.0, 0.0, 0.0], 5.0)
        cancel_handle = self._wait_future(self.client.send_goal_async(cancel_goal), 3.0)
        if cancel_handle is None or not cancel_handle.accepted:
            raise RuntimeError("long trajectory goal was rejected")
        cancel_deadline = time.monotonic() + 0.35
        while time.monotonic() < cancel_deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        cancel_response = self._wait_future(cancel_handle.cancel_goal_async(), 3.0)
        if not cancel_response.goals_canceling:
            raise RuntimeError("controller did not accept trajectory cancellation")
        canceled_result = self._wait_future(cancel_handle.get_result_async(), 3.0)
        if canceled_result.status != GoalStatus.STATUS_CANCELED:
            raise RuntimeError(
                f"long trajectory status was {canceled_result.status}, expected CANCELED"
            )

        rclpy.spin_once(self, timeout_sec=0.1)
        assert self.latest_positions is not None
        held_position = self.latest_positions.copy()
        hold_deadline = time.monotonic() + 0.4
        while time.monotonic() < hold_deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        assert self.latest_positions is not None
        if max(abs(after - before) for after, before in zip(
            self.latest_positions, held_position
        )) > 1.0e-3:
            raise RuntimeError("mock joints continued moving after cancellation")
        if self.latest_velocities is not None and any(
            math.isfinite(value) and abs(value) > 1.0e-2 for value in self.latest_velocities
        ):
            raise RuntimeError("mock joint velocity remained nonzero after cancellation")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockTrajectoryProbe()
    try:
        node.run()
        print("PASS: trajectory success, named feedback, cancel, and post-cancel hold")
    except Exception as exc:
        node.get_logger().error(f"Mock trajectory probe failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
