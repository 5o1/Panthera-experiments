"""Continuously supply Panthera's positive posVelMaxTorque speed caps."""

from __future__ import annotations

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class JointVelocityCapNode(Node):
    """Publish the Host application's normal 0.6 rad/s cap for six joints."""

    def __init__(self) -> None:
        super().__init__("joint_velocity_cap_node")
        self.declare_parameter("velocity_cap_radps", 0.6)
        self.declare_parameter("hardware_limit_radps", 1.0)
        self.declare_parameter(
            "command_topic", "/arm_velocity_limit_controller/commands"
        )
        cap = float(self.get_parameter("velocity_cap_radps").value)
        hardware_limit = float(self.get_parameter("hardware_limit_radps").value)
        if not all(math.isfinite(value) and value > 0.0 for value in (cap, hardware_limit)):
            raise ValueError("velocity caps must be positive and finite")
        if cap > hardware_limit:
            raise ValueError("velocity cap exceeds the Panthera hardware limit")
        self.message = Float64MultiArray(data=[cap] * 6)
        self.publisher = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("command_topic").value), 1
        )
        self.timer = self.create_timer(0.1, self._publish)
        self._publish()

    def _publish(self) -> None:
        self.publisher.publish(self.message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointVelocityCapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
