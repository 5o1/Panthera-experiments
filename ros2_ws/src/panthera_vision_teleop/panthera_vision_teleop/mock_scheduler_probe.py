"""Observe that the direct /pos_cmd scheduler never queues while fake feedback moves."""

import time

from panthera_interfaces.msg import ArmStatus, PosCmd
import rclpy
from rclpy.node import Node


class MockSchedulerProbe(Node):
    def __init__(self) -> None:
        super().__init__("mock_scheduler_probe")
        self.motion_status = None
        self.command_times = []
        self.error = ""
        self.create_subscription(ArmStatus, "/arm_status", self._status, 10)
        self.create_subscription(PosCmd, "/pos_cmd", self._command, 10)

    def _status(self, message: ArmStatus) -> None:
        self.motion_status = int(message.motion_status)

    def _command(self, _message: PosCmd) -> None:
        now = time.monotonic()
        if self.motion_status != 0:
            self.error = f"command arrived while motion_status={self.motion_status}"
        if self.command_times and now - self.command_times[-1] < 0.2:
            self.error = "commands arrived as a rapid FIFO burst"
        self.command_times.append(now)

    def run(self) -> None:
        deadline = time.monotonic() + 10.0
        while rclpy.ok() and time.monotonic() < deadline and not self.error:
            rclpy.spin_once(self, timeout_sec=max(0.0, deadline - time.monotonic()))
            if len(self.command_times) >= 3:
                break
        if self.error:
            raise RuntimeError(self.error)
        if len(self.command_times) < 2:
            raise RuntimeError(f"received only {len(self.command_times)} commands")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockSchedulerProbe()
    try:
        node.run()
        print("PASS: direct scheduler sent only while idle and produced no rapid FIFO burst")
    except Exception as exc:
        node.get_logger().error(f"Mock scheduler probe failed: {exc}")
        raise SystemExit(1) from exc
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
