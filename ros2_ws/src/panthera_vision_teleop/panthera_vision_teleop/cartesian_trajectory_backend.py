"""Latest-only Cartesian target to short joint-trajectory backend."""

from __future__ import annotations

import math
import time
from typing import Optional

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from .joint_command_guard import ConsecutiveFailureLatch
from .kinematics import SerialChain
from .rotation_utils import quaternion_to_matrix


DEFAULT_URDF = "/home/assaneko/Panthera_HT_ROS2/install/panthera_ht_ros_description/share/panthera_ht_ros_description/urdf/panthera_ht_ros_description.urdf"


class CartesianTrajectoryBackend(Node):
    """Solve only the newest target and send bounded short action segments."""

    def __init__(self) -> None:
        super().__init__("cartesian_trajectory_backend")
        self.declare_parameter("enabled", False)
        self.declare_parameter("urdf_path", DEFAULT_URDF)
        self.declare_parameter("target_topic", "/teleop/debug_target")
        self.declare_parameter("stale_timeout_sec", 0.5)
        self.declare_parameter("max_joint_step_rad", 0.01)
        self.declare_parameter("segment_duration_sec", 0.25)
        self.declare_parameter("goal_timeout_sec", 2.0)
        self.declare_parameter("max_tracking_error_rad", 0.02)
        self.declare_parameter("failure_lock_count", 3)
        self.enabled = bool(self.get_parameter("enabled").value)
        self.stale_timeout = float(self.get_parameter("stale_timeout_sec").value)
        self.max_joint_step = float(self.get_parameter("max_joint_step_rad").value)
        self.segment_duration = float(self.get_parameter("segment_duration_sec").value)
        self.goal_timeout = float(self.get_parameter("goal_timeout_sec").value)
        self.max_tracking_error = float(self.get_parameter("max_tracking_error_rad").value)
        failure_lock_count = int(self.get_parameter("failure_lock_count").value)
        if min(
            self.stale_timeout,
            self.max_joint_step,
            self.segment_duration,
            self.goal_timeout,
            self.max_tracking_error,
        ) <= 0.0:
            raise ValueError("backend time and step parameters must be positive")
        self.failure_latch = ConsecutiveFailureLatch(failure_lock_count)
        self.chain = SerialChain.from_urdf(
            str(self.get_parameter("urdf_path").value), "base_link", "link6"
        )
        self.client = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        self.status_pub = self.create_publisher(String, "/teleop/backend_status", 1)
        self.create_subscription(
            PoseStamped, str(self.get_parameter("target_topic").value), self._target, 1
        )
        self.create_subscription(JointState, "/joint_states", self._feedback, 10)
        self.create_service(Trigger, "/teleop/reset_backend", self._reset_backend)
        self.latest_target: Optional[np.ndarray] = None
        self.latest_target_time: Optional[float] = None
        self.latest_positions: Optional[np.ndarray] = None
        self.goal_handle = None
        self.send_future = None
        self.cancel_requested = False
        self.cancel_reason = ""
        self.goal_started_at: Optional[float] = None
        self.last_command_positions: Optional[np.ndarray] = None
        self.last_status = ""
        self.timer = self.create_timer(0.02, self._tick)
        self._status("WAITING" if self.enabled else "DISABLED")

    def _status(self, value: str) -> None:
        if value == self.last_status:
            return
        self.last_status = value
        self.status_pub.publish(String(data=value))

    def _feedback(self, message: JointState) -> None:
        by_name = dict(zip(message.name, message.position))
        if all(name in by_name for name in self.chain.joint_names):
            self.latest_positions = np.array(
                [float(by_name[name]) for name in self.chain.joint_names]
            )

    def _reset_backend(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.goal_handle is not None or (self.send_future is not None and not self.send_future.done()):
            response.success = False
            response.message = "cannot reset while a goal is active"
            return response
        if self.latest_positions is None:
            response.success = False
            response.message = "cannot reset without valid joint feedback"
            return response
        self.failure_latch.reset()
        self.latest_target = None
        self.latest_target_time = None
        self.last_command_positions = None
        self._status("RESET_WAITING_TARGET")
        response.success = True
        response.message = "backend failure lock cleared; publish a fresh target"
        return response

    def _failure(self, reason: str) -> None:
        if self.failure_latch.failure(reason):
            self._status(f"FAULT_LOCK_{reason}")
        else:
            self._status(f"{reason}_{self.failure_latch.count}/{self.failure_latch.limit}")

    def _target(self, message: PoseStamped) -> None:
        if message.header.frame_id not in ("", "base_link"):
            self._status("REJECTED_FRAME")
            return
        p, q = message.pose.position, message.pose.orientation
        try:
            rotation = quaternion_to_matrix([q.x, q.y, q.z, q.w])
        except ValueError:
            self._status("REJECTED_TARGET")
            return
        target = np.eye(4)
        target[:3, :3] = rotation
        target[:3, 3] = [p.x, p.y, p.z]
        if not np.all(np.isfinite(target)):
            self._status("REJECTED_TARGET")
            return
        self.latest_target = target
        self.latest_target_time = time.monotonic()

    def _tick(self) -> None:
        if not self.enabled:
            return
        if self.failure_latch.locked:
            self._status(f"FAULT_LOCK_{self.failure_latch.reason}")
            return
        now = time.monotonic()
        if (
            self.goal_handle is not None
            and self.goal_started_at is not None
            and now - self.goal_started_at > self.goal_timeout
            and not self.cancel_requested
        ):
            self.cancel_requested = True
            self.cancel_reason = "TIMEOUT"
            self._failure("GOAL_TIMEOUT")
            self.goal_handle.cancel_goal_async()
            if not self.failure_latch.locked:
                self._status("CANCELING_TIMEOUT")
            return
        stale = self.latest_target_time is None or now - self.latest_target_time > self.stale_timeout
        if stale:
            if self.goal_handle is not None and not self.cancel_requested:
                self.cancel_requested = True
                self.cancel_reason = "STALE"
                self.goal_handle.cancel_goal_async()
                self._status("CANCELING_STALE")
            elif self.goal_handle is None:
                self._status("STALE_HOLD")
            return
        if self.goal_handle is not None or (self.send_future is not None and not self.send_future.done()):
            return
        if self.latest_positions is None or self.latest_target is None:
            self._status("WAITING")
            return
        if not self.client.server_is_ready():
            self._status("WAITING_ACTION")
            return
        try:
            desired = self.chain.inverse(self.latest_target, self.latest_positions)
        except ValueError:
            self._failure("IK_FAILED")
            return
        delta = np.clip(
            desired - self.latest_positions, -self.max_joint_step, self.max_joint_step
        )
        if float(np.max(np.abs(delta))) < 1e-4:
            self._status("AT_TARGET")
            return
        command = np.clip(self.latest_positions + delta, self.chain.lower, self.chain.upper)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.chain.joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in command]
        self.last_command_positions = command.copy()
        point.time_from_start.sec = int(self.segment_duration)
        point.time_from_start.nanosec = int(
            (self.segment_duration - int(self.segment_duration)) * 1e9
        )
        goal.trajectory.points = [point]
        self.send_future = self.client.send_goal_async(goal)
        self.send_future.add_done_callback(self._goal_response)
        self._status("SENDING")

    def _goal_response(self, future) -> None:
        try:
            handle = future.result()
        except Exception:
            handle = None
        self.send_future = None
        if handle is None or not handle.accepted:
            self._failure("GOAL_REJECTED")
            return
        self.goal_handle = handle
        self.cancel_requested = False
        self.cancel_reason = ""
        self.goal_started_at = time.monotonic()
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._goal_result)
        self._status("EXECUTING")

    def _goal_result(self, future) -> None:
        try:
            wrapped = future.result()
        except Exception:
            wrapped = None
        cancel_reason = self.cancel_reason
        self.goal_handle = None
        self.cancel_requested = False
        self.cancel_reason = ""
        self.goal_started_at = None
        if wrapped is None:
            self._failure("GOAL_ERROR")
        elif wrapped.status == GoalStatus.STATUS_SUCCEEDED:
            tracking_error = math.inf
            if self.latest_positions is not None and self.last_command_positions is not None:
                tracking_error = float(np.max(np.abs(self.latest_positions - self.last_command_positions)))
            if tracking_error > self.max_tracking_error:
                self._failure("TRACKING_ERROR")
            else:
                self.failure_latch.success()
                self._status("SEGMENT_DONE")
        elif wrapped.status == GoalStatus.STATUS_CANCELED:
            if cancel_reason == "STALE":
                self._status("STALE_HOLD")
            elif cancel_reason == "TIMEOUT":
                if self.failure_latch.locked:
                    self._status("FAULT_LOCK_GOAL_TIMEOUT")
                else:
                    self._status("TIMEOUT_HOLD")
            else:
                self._failure("UNEXPECTED_CANCEL")
        else:
            self._failure(f"GOAL_STATUS_{wrapped.status}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CartesianTrajectoryBackend()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
