#!/usr/bin/env python3
"""Verify that live Panthera joint feedback remains within a target tolerance."""

from __future__ import annotations

import argparse
import math
import time

from panthera_interfaces.msg import ArmStatus
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def parse_target(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("target must contain six finite comma-separated radians")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=parse_target, required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--stable-samples", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance <= 0.0:
        parser.error("--tolerance must be positive and finite")
    if args.stable_samples <= 0 or args.timeout <= 0.0:
        parser.error("--stable-samples and --timeout must be positive")

    rclpy.init()
    node = Node("verify_panthera_joint_target")
    result: dict[str, object] = {"stable": 0, "last": None, "fault": ""}

    def status(message: ArmStatus) -> None:
        if message.error_message or any(message.motor_faults) or any(message.joint_at_limit):
            result["fault"] = message.error_message or "motor fault or joint limit"

    def joints(message: JointState) -> None:
        if len(message.position) < 6:
            result["stable"] = 0
            return
        position = [float(value) for value in message.position[:6]]
        result["last"] = position
        error = max(abs(actual - target) for actual, target in zip(position, args.target))
        result["stable"] = int(result["stable"]) + 1 if error <= args.tolerance else 0

    node.create_subscription(ArmStatus, "/arm_status", status, 10)
    node.create_subscription(JointState, "/joint_states_single", joints, 10)
    deadline = time.monotonic() + args.timeout
    try:
        while (
            rclpy.ok()
            and int(result["stable"]) < args.stable_samples
            and not result["fault"]
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=max(0.0, deadline - time.monotonic()))
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if result["fault"]:
        raise SystemExit(f"arm fault while verifying joint target: {result['fault']}")
    if int(result["stable"]) < args.stable_samples:
        raise SystemExit(
            f"joint target verification failed: target={args.target}, "
            f"last={result['last']}, tolerance={args.tolerance} rad"
        )
    position = result["last"]
    assert isinstance(position, list)
    error = max(abs(actual - target) for actual, target in zip(position, args.target))
    print(
        f"Joint target verified for {args.stable_samples} samples: "
        f"max_error={error:.6f} rad <= {args.tolerance:.6f} rad; position={position}"
    )


if __name__ == "__main__":
    main()
