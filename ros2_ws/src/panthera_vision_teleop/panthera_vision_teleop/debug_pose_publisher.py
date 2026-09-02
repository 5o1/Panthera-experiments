"""Deterministic synthetic inputs for testing without camera or hardware."""

import math
import signal

from geometry_msgs.msg import PoseStamped
import numpy as np
from panthera_interfaces.msg import ArmStatus, EndPoseEuler, PosCmd
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Bool, Float32, String


class DebugPosePublisher(Node):
    """Publish a reproducible human motion plus optional fake robot feedback."""

    def __init__(self) -> None:
        super().__init__("debug_pose_publisher")
        self.declare_parameter("scenario", "xyz_cycle")
        self.declare_parameter("rate_hz", 15.0)
        self.declare_parameter("amplitude_m", 0.12)
        self.declare_parameter("publish_human", True)
        self.declare_parameter("publish_robot_feedback", True)
        self.declare_parameter("follow_robot_commands", False)
        self.declare_parameter("fake_robot_speed_m_s", 0.08)
        self.scenario = str(self.get_parameter("scenario").value)
        self.rate = float(self.get_parameter("rate_hz").value)
        self.amplitude = float(self.get_parameter("amplitude_m").value)
        self.publish_human = bool(self.get_parameter("publish_human").value)
        self.publish_robot_feedback = bool(self.get_parameter("publish_robot_feedback").value)
        self.follow_robot_commands = bool(self.get_parameter("follow_robot_commands").value)
        self.fake_robot_speed = float(self.get_parameter("fake_robot_speed_m_s").value)
        if self.rate <= 0.0:
            raise ValueError("rate_hz must be positive")
        if self.fake_robot_speed <= 0.0:
            raise ValueError("fake_robot_speed_m_s must be positive")

        self.pose_pub = self.create_publisher(PoseStamped, "/teleop/human_pose", 1)
        self.pose_score_pub = self.create_publisher(Float32, "/teleop/pose_score", 1)
        self.gesture_pub = self.create_publisher(String, "/teleop/gesture", 1)
        self.gesture_score_pub = self.create_publisher(Float32, "/teleop/gesture_score", 1)
        self.enabled_pub = self.create_publisher(Bool, "/teleop/enabled", 1)
        self.recalibrate_pub = self.create_publisher(Bool, "/teleop/recalibrate", 1)
        self.end_pose_pub = self.create_publisher(EndPoseEuler, "/end_pose_euler", 1)
        self.arm_status_pub = self.create_publisher(ArmStatus, "/arm_status", 1)
        self.fake_position = np.array([0.2642584594, 0.0, 0.4314376485], dtype=float)
        self.fake_rpy = np.array([0.0, -0.2424, 0.0], dtype=float)
        self.fake_target_position = self.fake_position.copy()
        self.fake_target_rpy = self.fake_rpy.copy()
        self.pos_cmd_sub = None
        if self.publish_robot_feedback and self.follow_robot_commands:
            self.pos_cmd_sub = self.create_subscription(PosCmd, "/pos_cmd", self._pos_cmd, 1)
        self.tick_count = 0
        self.timer = self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().warning(
            f"SYNTHETIC scenario={self.scenario}; data is not from a camera or robot"
        )

    def _tick(self) -> None:
        elapsed = self.tick_count / self.rate
        self.tick_count += 1
        if self.publish_robot_feedback:
            self._publish_fake_robot()
        if not self.publish_human:
            return

        enabled = not (self.scenario == "dropout" and 4.0 <= elapsed % 8.0 < 6.0)
        phase = 2.0 * math.pi * elapsed / 8.0
        dx = dy = dz = 0.0
        wave = self.amplitude * math.sin(phase)
        if self.scenario == "x":
            dx = wave
        elif self.scenario == "y":
            dy = wave
        elif self.scenario == "z":
            dz = wave
        elif self.scenario in ("xyz_cycle", "gesture", "dropout"):
            segment = int(elapsed // 4.0) % 3
            dx, dy, dz = (wave, 0.0, 0.0) if segment == 0 else ((0.0, wave, 0.0) if segment == 1 else (0.0, 0.0, wave))

        self.pose_score_pub.publish(Float32(data=0.99))
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "camera"
        pose.pose.position.x = 0.35 + dx
        pose.pose.position.y = 0.12 + dy
        pose.pose.position.z = -0.10 + dz
        pose.pose.orientation.w = 1.0
        if enabled:
            self.pose_pub.publish(pose)

        gesture = "Unknown"
        if self.scenario in ("gesture", "xyz_cycle"):
            gesture = "Open_Palm" if int(elapsed // 3.0) % 2 == 0 else "Closed_Fist"
        self.gesture_pub.publish(String(data=gesture))
        self.gesture_score_pub.publish(Float32(data=0.95 if gesture != "Unknown" else 0.0))
        self.enabled_pub.publish(Bool(data=enabled))
        self.recalibrate_pub.publish(Bool(data=self.tick_count == 2))

    def _publish_fake_robot(self) -> None:
        delta = self.fake_target_position - self.fake_position
        distance = float(np.linalg.norm(delta))
        max_step = self.fake_robot_speed / self.rate
        moving = distance > 1.0e-6
        if distance <= max_step:
            self.fake_position = self.fake_target_position.copy()
            self.fake_rpy = self.fake_target_rpy.copy()
            moving = False
        elif moving:
            self.fake_position += delta * (max_step / distance)

        pose = EndPoseEuler()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "base_link"
        pose.x, pose.y, pose.z = (float(value) for value in self.fake_position)
        pose.roll, pose.pitch, pose.yaw = (float(value) for value in self.fake_rpy)
        self.end_pose_pub.publish(pose)
        status = ArmStatus()
        status.header = pose.header
        status.arm_enabled = True
        status.motion_status = 1 if moving else 0
        status.error_message = ""
        status.motor_modes = [0] * 6
        status.motor_faults = [0] * 6
        status.joint_at_limit = [False] * 6
        status.gripper_position = 0.0
        status.gripper_fault = 0
        self.arm_status_pub.publish(status)

    def _pos_cmd(self, msg: PosCmd) -> None:
        values = np.array([msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw], dtype=float)
        if not np.all(np.isfinite(values)):
            self.get_logger().error("Fake robot rejected a non-finite /pos_cmd")
            return
        self.fake_target_position = values[:3]
        self.fake_target_rpy = values[3:]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugPosePublisher()
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
