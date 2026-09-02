#!/usr/bin/env python3
"""Wait for a typed ROS service through graph events, without shell sleeps."""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_service


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("type", help="for example panthera_interfaces/srv/MoveToJoint")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout <= 0.0:
        raise SystemExit("timeout must be positive")

    rclpy.init()
    node = Node("wait_for_typed_ros_service")
    client = node.create_client(get_service(args.type), args.name)
    try:
        available = client.wait_for_service(timeout_sec=args.timeout)
    finally:
        node.destroy_client(client)
        node.destroy_node()
        rclpy.shutdown()
    if not available:
        raise SystemExit(f"timed out waiting for {args.name} ({args.type})")
    print(f"ROS service ready: {args.name} ({args.type})")


if __name__ == "__main__":
    main()
