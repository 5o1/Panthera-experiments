"""ROS-independent guard for a future latest-only continuous joint backend."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


class ConsecutiveFailureLatch:
    """Latch after repeated backend failures; clear only by explicit reset."""

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("failure limit must be a positive integer")
        self.limit = limit
        self.count = 0
        self.locked = False
        self.reason = ""

    def failure(self, reason: str) -> bool:
        if not reason:
            raise ValueError("failure reason must not be empty")
        if self.locked:
            return True
        self.count += 1
        self.reason = reason
        if self.count >= self.limit:
            self.locked = True
        return self.locked

    def success(self) -> None:
        if not self.locked:
            self.count = 0
            self.reason = ""

    def reset(self) -> None:
        self.count = 0
        self.locked = False
        self.reason = ""


class JointCommandGuard:
    """Validate joint order, limits, rate continuity and command freshness."""

    def __init__(
        self,
        joint_names: Sequence[str],
        lower: Sequence[float],
        upper: Sequence[float],
        max_velocity: Sequence[float],
        max_acceleration: Sequence[float],
        stale_timeout_sec: float,
    ) -> None:
        self.joint_names = tuple(joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        size = len(self.joint_names)
        self.lower = self._array(lower, size, "lower")
        self.upper = self._array(upper, size, "upper")
        self.max_velocity = self._array(max_velocity, size, "max_velocity")
        self.max_acceleration = self._array(max_acceleration, size, "max_acceleration")
        if np.any(self.lower >= self.upper):
            raise ValueError("every lower limit must be below its upper limit")
        if np.any(self.max_velocity <= 0.0) or np.any(self.max_acceleration <= 0.0):
            raise ValueError("rate limits must be positive")
        if not math.isfinite(stale_timeout_sec) or stale_timeout_sec <= 0.0:
            raise ValueError("stale_timeout_sec must be positive and finite")
        self.stale_timeout_sec = stale_timeout_sec
        self.last_position: Optional[np.ndarray] = None
        self.last_velocity: Optional[np.ndarray] = None
        self.last_command_time: Optional[float] = None
        self.hold_emitted = False

    @staticmethod
    def _array(values: Sequence[float], size: int, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain {size} finite values")
        return result.copy()

    def seed_feedback(self, names: Sequence[str], positions: Sequence[float], now: float) -> None:
        self._validate_names(names)
        position = self._array(positions, len(self.joint_names), "positions")
        self._validate_limits(position)
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        self.last_position = position
        self.last_velocity = np.zeros(len(self.joint_names))
        self.last_command_time = now
        self.hold_emitted = False

    def accept(self, names: Sequence[str], positions: Sequence[float], now: float) -> np.ndarray:
        self._validate_names(names)
        position = self._array(positions, len(self.joint_names), "positions")
        self._validate_limits(position)
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if self.last_position is not None and self.last_command_time is not None:
            dt = now - self.last_command_time
            if dt <= 0.0:
                raise ValueError("command timestamps must increase")
            velocity = (position - self.last_position) / dt
            if np.any(np.abs(velocity) > self.max_velocity + 1e-12):
                raise ValueError("joint velocity limit exceeded")
            if self.last_velocity is not None:
                acceleration = (velocity - self.last_velocity) / dt
                if np.any(np.abs(acceleration) > self.max_acceleration + 1e-12):
                    raise ValueError("joint acceleration limit exceeded")
            self.last_velocity = velocity
        else:
            self.last_velocity = np.zeros(len(self.joint_names))
        self.last_position = position
        self.last_command_time = now
        self.hold_emitted = False
        return position.copy()

    def stale_hold(self, now: float) -> Optional[np.ndarray]:
        """Return one hold point when the accepted-command stream becomes stale."""
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if self.last_command_time is None or self.last_position is None:
            return None
        if now - self.last_command_time <= self.stale_timeout_sec or self.hold_emitted:
            return None
        self.hold_emitted = True
        self.last_velocity = np.zeros(len(self.joint_names))
        return self.last_position.copy()

    def _validate_names(self, names: Sequence[str]) -> None:
        if tuple(names) != self.joint_names:
            raise ValueError(f"joint order must be exactly {self.joint_names}")

    def _validate_limits(self, position: np.ndarray) -> None:
        if np.any(position < self.lower) or np.any(position > self.upper):
            raise ValueError("joint position limit exceeded")


class JointMotionLimiter:
    """Generate a continuous joint target with velocity and acceleration bounds."""

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        max_velocity: Sequence[float],
        max_acceleration: Sequence[float],
    ) -> None:
        size = len(lower)
        if size < 1:
            raise ValueError("joint limits must not be empty")
        self.lower = JointCommandGuard._array(lower, size, "lower")
        self.upper = JointCommandGuard._array(upper, size, "upper")
        self.max_velocity = JointCommandGuard._array(max_velocity, size, "max_velocity")
        self.max_acceleration = JointCommandGuard._array(
            max_acceleration, size, "max_acceleration"
        )
        if np.any(self.lower >= self.upper):
            raise ValueError("every lower limit must be below its upper limit")
        if np.any(self.max_velocity <= 0.0) or np.any(self.max_acceleration <= 0.0):
            raise ValueError("rate limits must be positive")
        self.position: Optional[np.ndarray] = None
        self.velocity = np.zeros(size)
        self.last_time: Optional[float] = None

    def reset(self, position: Sequence[float], now: float) -> None:
        value = JointCommandGuard._array(position, len(self.lower), "position")
        if np.any(value < self.lower) or np.any(value > self.upper):
            raise ValueError("joint position limit exceeded")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        self.position = value
        self.velocity = np.zeros(len(self.lower))
        self.last_time = now

    def update(self, desired: Sequence[float], now: float) -> tuple[np.ndarray, np.ndarray]:
        target = JointCommandGuard._array(desired, len(self.lower), "desired")
        if np.any(target < self.lower) or np.any(target > self.upper):
            raise ValueError("joint position limit exceeded")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if self.position is None or self.last_time is None:
            self.reset(target, now)
            return target.copy(), self.velocity.copy()
        dt = now - self.last_time
        if dt < 0.0:
            raise ValueError("now must not move backwards")
        if dt == 0.0:
            return self.position.copy(), self.velocity.copy()

        error = target - self.position
        braking_speed = np.sqrt(2.0 * self.max_acceleration * np.abs(error))
        requested_velocity = np.sign(error) * np.minimum(self.max_velocity, braking_speed)
        velocity_delta = np.clip(
            requested_velocity - self.velocity,
            -self.max_acceleration * dt,
            self.max_acceleration * dt,
        )
        next_velocity = self.velocity + velocity_delta
        step = next_velocity * dt
        remaining = target - (self.position + step)
        reached = (np.sign(error) != np.sign(remaining)) | (np.abs(step) >= np.abs(error))
        next_position = self.position + step
        next_position[reached] = target[reached]
        next_velocity[reached] = 0.0
        self.position = np.clip(next_position, self.lower, self.upper)
        self.velocity = next_velocity
        self.last_time = now
        return self.position.copy(), self.velocity.copy()
