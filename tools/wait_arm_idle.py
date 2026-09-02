#!/usr/bin/env python3
"""Wait for a fault-free idle Panthera status using ROS events, not polling sleeps."""

import argparse
import time

from panthera_interfaces.msg import ArmStatus
import rclpy
from rclpy.node import Node


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()
    rclpy.init()
    node = Node("wait_panthera_arm_idle")
    result = {"idle": False, "fault": ""}

    def status(message: ArmStatus) -> None:
        if message.error_message or any(message.motor_faults) or any(message.joint_at_limit):
            result["fault"] = message.error_message or "motor fault or joint limit"
        elif message.arm_enabled and message.motion_status == 0:
            result["idle"] = True

    node.create_subscription(ArmStatus, "/arm_status", status, 10)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and not result["idle"] and not result["fault"] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=max(0.0, deadline - time.monotonic()))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if result["fault"]:
        raise SystemExit(f"arm fault while waiting for idle: {result['fault']}")
    if not result["idle"]:
        raise SystemExit("timed out waiting for fault-free idle arm status")
    print("Panthera arm is idle and fault-free")


if __name__ == "__main__":
    main()
