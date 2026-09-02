"""Latest-only Cartesian teleoperation backend for Panthera ros2_control."""

from __future__ import annotations

import time
from typing import Optional

from geometry_msgs.msg import PoseStamped
import numpy as np
from panthera_interfaces.msg import EndPoseEuler
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .cartesian_trajectory_backend import DEFAULT_URDF
from .joint_command_guard import ConsecutiveFailureLatch, JointMotionLimiter
from .kinematics import SerialChain
from .rotation_utils import matrix_to_rpy, quaternion_to_matrix


class ContinuousCartesianBackend(Node):
    """Convert the newest Cartesian target into a bounded 50 Hz joint stream.

    ros2_control performs the 200 Hz hardware loop.  This node never waits for a
    point to finish and never queues old camera targets.
    """

    def __init__(self) -> None:
        super().__init__("continuous_cartesian_backend")
        defaults = {
            "enabled": False,
            "urdf_path": DEFAULT_URDF,
            "target_topic": "/teleop/debug_target",
            "joint_state_topic": "/joint_states",
            "command_topic": "/arm_controller/joint_trajectory",
            "control_rate_hz": 50.0,
            "target_timeout_sec": 0.5,
            "feedback_timeout_sec": 0.25,
            # Host's normal position command speed and application limits.
            "max_joint_velocity_radps": [0.6] * 6,
            "hardware_velocity_limit_radps": [1.0] * 6,
            "max_joint_acceleration_radps2": [2.0] * 6,
            "max_tracking_error_rad": 0.20,
            "failure_lock_count": 3,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.enabled = bool(self.get_parameter("enabled").value)
        rate = float(self.get_parameter("control_rate_hz").value)
        self.target_timeout = float(self.get_parameter("target_timeout_sec").value)
        self.feedback_timeout = float(self.get_parameter("feedback_timeout_sec").value)
        self.max_tracking_error = float(self.get_parameter("max_tracking_error_rad").value)
        if min(rate, self.target_timeout, self.feedback_timeout, self.max_tracking_error) <= 0.0:
            raise ValueError("backend rates, timeouts and tracking error must be positive")

        self.chain = SerialChain.from_urdf(
            str(self.get_parameter("urdf_path").value), "base_link", "link6"
        )
        velocity = np.asarray(
            self.get_parameter("max_joint_velocity_radps").value, dtype=float
        )
        hardware_velocity = np.asarray(
            self.get_parameter("hardware_velocity_limit_radps").value, dtype=float
        )
        acceleration = np.asarray(
            self.get_parameter("max_joint_acceleration_radps2").value, dtype=float
        )
        expected = (len(self.chain.joint_names),)
        if velocity.shape != expected or hardware_velocity.shape != expected:
            raise ValueError("joint velocity parameters must match the six-joint chain")
        if acceleration.shape != expected:
            raise ValueError("joint acceleration parameters must match the six-joint chain")
        if np.any(velocity > hardware_velocity):
            raise ValueError("normal command velocity exceeds the hardware velocity limit")
        self.motion = JointMotionLimiter(
            self.chain.lower, self.chain.upper, velocity, acceleration
        )
        self.failure_latch = ConsecutiveFailureLatch(
            int(self.get_parameter("failure_lock_count").value)
        )

        self.command_period = 1.0 / rate
        self.command_pub = self.create_publisher(
            JointTrajectory, str(self.get_parameter("command_topic").value), 1
        )
        self.pose_pub = self.create_publisher(EndPoseEuler, "/end_pose_euler", 10)
        self.status_pub = self.create_publisher(String, "/teleop/backend_status", 10)
        self.create_subscription(
            PoseStamped, str(self.get_parameter("target_topic").value), self._target, 1
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._feedback,
            10,
        )
        self.create_subscription(Bool, "/teleop/enabled", self._source_enabled, 1)
        self.create_service(Trigger, "/teleop/reset_backend", self._reset_backend)
        self.timer = self.create_timer(self.command_period, self._tick)

        self.teleop_enabled = False
        self.latest_target: Optional[np.ndarray] = None
        self.latest_target_time: Optional[float] = None
        self.latest_positions: Optional[np.ndarray] = None
        self.latest_feedback_time: Optional[float] = None
        self.last_command: Optional[np.ndarray] = None
        self.hold_sent = False
        self.last_status = ""
        self._status("WAITING" if self.enabled else "DISABLED")

    def _status(self, value: str) -> None:
        if value == self.last_status:
            return
        self.last_status = value
        self.status_pub.publish(String(data=value))

    def _failure(self, reason: str) -> None:
        if self.failure_latch.failure(reason):
            self._status(f"FAULT_LOCK_{reason}")
        else:
            self._status(f"{reason}_{self.failure_latch.count}/{self.failure_latch.limit}")
        self._hold()

    def _source_enabled(self, message: Bool) -> None:
        next_value = bool(message.data)
        if self.teleop_enabled and not next_value:
            self._hold()
        self.teleop_enabled = next_value
        if next_value:
            self.hold_sent = False
        else:
            self.latest_target = None
            self.latest_target_time = None
            self._status("SOURCE_DISABLED")

    def _target(self, message: PoseStamped) -> None:
        if message.header.frame_id not in ("", "base_link"):
            self._failure("REJECTED_FRAME")
            return
        p, q = message.pose.position, message.pose.orientation
        try:
            rotation = quaternion_to_matrix([q.x, q.y, q.z, q.w])
        except ValueError:
            self._failure("REJECTED_TARGET")
            return
        target = np.eye(4)
        target[:3, :3] = rotation
        target[:3, 3] = [p.x, p.y, p.z]
        if not np.all(np.isfinite(target)):
            self._failure("REJECTED_TARGET")
            return
        self.latest_target = target
        self.latest_target_time = time.monotonic()
        self.hold_sent = False

    def _feedback(self, message: JointState) -> None:
        by_name = dict(zip(message.name, message.position))
        if not all(name in by_name for name in self.chain.joint_names):
            return
        positions = np.array(
            [float(by_name[name]) for name in self.chain.joint_names], dtype=float
        )
        if not np.all(np.isfinite(positions)):
            self._failure("INVALID_FEEDBACK")
            return
        self.latest_positions = positions
        self.latest_feedback_time = time.monotonic()
        transform = self.chain.forward(positions)
        rpy = matrix_to_rpy(transform[:3, :3])
        pose = EndPoseEuler()
        pose.header = message.header
        pose.header.frame_id = "base_link"
        pose.x, pose.y, pose.z = (float(value) for value in transform[:3, 3])
        pose.roll, pose.pitch, pose.yaw = (float(value) for value in rpy)
        self.pose_pub.publish(pose)

    def _publish(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        message = JointTrajectory()
        message.header.stamp = self.get_clock().now().to_msg()
        message.joint_names = list(self.chain.joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        # These derivatives are consumed only by JointTrajectoryController's
        # interpolation.  A separate controller owns the hardware velocity
        # interfaces because the Panthera SDK interprets them as positive
        # speed caps rather than signed derivatives.
        point.velocities = [float(value) for value in velocities]
        point.time_from_start.sec = int(self.command_period)
        point.time_from_start.nanosec = int(
            (self.command_period - int(self.command_period)) * 1e9
        )
        message.points = [point]
        self.command_pub.publish(message)
        self.last_command = positions.copy()

    def _hold(self) -> None:
        if self.hold_sent:
            return
        position = self.latest_positions if self.latest_positions is not None else self.last_command
        if position is None or self.command_pub.get_subscription_count() < 1:
            return
        now = time.monotonic()
        try:
            self.motion.reset(position, now)
        except ValueError:
            return
        self._publish(position, np.zeros(len(position)))
        self.hold_sent = True

    def _reset_backend(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        if self.teleop_enabled:
            response.success = False
            response.message = "disable teleoperation before resetting the backend"
            return response
        if self.latest_positions is None:
            response.success = False
            response.message = "cannot reset without valid joint feedback"
            return response
        self.failure_latch.reset()
        self.motion.reset(self.latest_positions, time.monotonic())
        self.latest_target = None
        self.latest_target_time = None
        self.hold_sent = False
        self._hold()
        self._status("RESET_WAITING_TARGET")
        response.success = True
        response.message = "backend failure lock cleared; enable and publish a fresh target"
        return response

    def _tick(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if self.failure_latch.locked:
            self._hold()
            self._status(f"FAULT_LOCK_{self.failure_latch.reason}")
            return
        if self.command_pub.get_subscription_count() < 1:
            self._status("WAITING_CONTROLLER")
            return
        if (
            self.latest_positions is None
            or self.latest_feedback_time is None
            or now - self.latest_feedback_time > self.feedback_timeout
        ):
            self._hold()
            self._status("FEEDBACK_STALE_HOLD")
            return
        if not self.teleop_enabled:
            self._hold()
            self._status("SOURCE_DISABLED")
            return
        if (
            self.latest_target is None
            or self.latest_target_time is None
            or now - self.latest_target_time > self.target_timeout
        ):
            self._hold()
            self._status("TARGET_STALE_HOLD")
            return
        if self.last_command is not None:
            tracking_error = float(
                np.max(np.abs(self.latest_positions - self.last_command))
            )
            if tracking_error > self.max_tracking_error:
                self._failure("TRACKING_ERROR")
                return
        try:
            desired = self.chain.inverse(self.latest_target, self.latest_positions)
            if self.motion.position is None:
                self.motion.reset(self.latest_positions, now)
            command, velocity = self.motion.update(desired, now)
        except ValueError:
            self._failure("IK_FAILED")
            return
        self._publish(command, velocity)
        self.hold_sent = False
        self.failure_latch.success()
        self._status("TRACKING")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ContinuousCartesianBackend()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
