"""ROS-independent calibration, retargeting and gesture state machines."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, Sequence

import numpy as np

from .rotation_utils import (
    interpolate_rotation,
    matrix_to_rpy,
    quaternion_to_matrix,
    rotation_from_vector,
    rotation_vector,
    rpy_to_matrix,
)


def _array(values: Sequence[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return result.copy()


@dataclass(frozen=True)
class RobotPose:
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class TargetPose:
    position: np.ndarray
    rpy: np.ndarray


@dataclass(frozen=True)
class MappingDiagnostics:
    """Intermediate position-mapping values for recording and tuning."""

    human_delta_camera: np.ndarray
    raw_robot_offset: np.ndarray
    deadbanded_robot_offset: np.ndarray
    limited_robot_offset: np.ndarray
    filtered_robot_offset: np.ndarray
    motion_limited_robot_offset: np.ndarray


@dataclass
class MappingConfig:
    # MediaPipe world-landmark depth from one RGB camera was visibly noisy on
    # the real arm.  The teaching-demo default therefore disables robot X
    # (camera depth) while retaining lateral Y and vertical Z control.
    position_gain: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.25, 0.12]))
    max_position_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.05, 0.04]))
    # Ignore small calibrated offsets caused by landmark jitter.  Subtracting
    # the deadband (rather than merely thresholding) keeps target motion
    # continuous at its boundary.
    position_deadband_m: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.002, 0.002]))
    max_position_velocity_mps: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.04, 0.03]))
    max_position_acceleration_mps2: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.15, 0.12]))
    # camera x(right), y(down), z(away) -> robot x(forward), y(left), z(up)
    position_axis_matrix: np.ndarray = field(default_factory=lambda: np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]]))
    position_alpha: float = 0.35
    position_alpha_axes: Optional[np.ndarray] = None
    rotation_alpha: float = 0.25
    max_rotation_rad: float = math.radians(20.0)
    orientation_axes: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    max_angular_velocity_radps: float = math.radians(10.0)
    max_angular_acceleration_radps2: float = math.radians(30.0)
    stale_timeout_sec: float = 0.5
    enable_orientation: bool = False

    def __post_init__(self) -> None:
        self.position_gain = _array(self.position_gain, (3,), "position_gain")
        self.max_position_offset = _array(self.max_position_offset, (3,), "max_position_offset")
        self.position_deadband_m = _array(self.position_deadband_m, (3,), "position_deadband_m")
        self.max_position_velocity_mps = _array(self.max_position_velocity_mps, (3,), "max_position_velocity_mps")
        self.max_position_acceleration_mps2 = _array(self.max_position_acceleration_mps2, (3,), "max_position_acceleration_mps2")
        self.position_axis_matrix = _array(self.position_axis_matrix, (3, 3), "position_axis_matrix")
        if self.position_alpha_axes is None:
            self.position_alpha_axes = np.full(3, self.position_alpha)
        else:
            self.position_alpha_axes = _array(self.position_alpha_axes, (3,), "position_alpha_axes")
        self.orientation_axes = _array(self.orientation_axes, (3,), "orientation_axes")
        if np.any(self.max_position_offset < 0.0):
            raise ValueError("max_position_offset must be non-negative")
        if np.any(self.position_deadband_m < 0.0):
            raise ValueError("position_deadband_m must be non-negative")
        if np.any(self.max_position_velocity_mps <= 0.0):
            raise ValueError("max_position_velocity_mps must be positive")
        if np.any(self.max_position_acceleration_mps2 <= 0.0):
            raise ValueError("max_position_acceleration_mps2 must be positive")
        if not 0.0 < self.position_alpha <= 1.0 or not 0.0 < self.rotation_alpha <= 1.0:
            raise ValueError("filter alphas must be in (0, 1]")
        if np.any(self.position_alpha_axes <= 0.0) or np.any(self.position_alpha_axes > 1.0):
            raise ValueError("position_alpha_axes must be in (0, 1]")
        if self.stale_timeout_sec <= 0.0:
            raise ValueError("stale_timeout_sec must be positive")
        if np.any((self.orientation_axes != 0.0) & (self.orientation_axes != 1.0)):
            raise ValueError("orientation_axes entries must be 0 or 1")
        if self.max_rotation_rad < 0.0:
            raise ValueError("max_rotation_rad must be non-negative")
        if self.max_angular_velocity_radps <= 0.0 or self.max_angular_acceleration_radps2 <= 0.0:
            raise ValueError("angular motion limits must be positive")
        if not np.allclose(self.position_axis_matrix.T @ self.position_axis_matrix, np.eye(3), atol=1e-6) or not math.isclose(float(np.linalg.det(self.position_axis_matrix)), 1.0, abs_tol=1e-6):
            raise ValueError("position_axis_matrix must be a proper rotation")


class CartesianMotionLimiter:
    """Apply independent Cartesian velocity and acceleration limits."""

    def __init__(self, max_velocity: Sequence[float], max_acceleration: Sequence[float]) -> None:
        self.max_velocity = _array(max_velocity, (3,), "max_velocity")
        self.max_acceleration = _array(max_acceleration, (3,), "max_acceleration")
        if np.any(self.max_velocity <= 0.0) or np.any(self.max_acceleration <= 0.0):
            raise ValueError("motion limits must be positive")
        self.position: Optional[np.ndarray] = None
        self.velocity = np.zeros(3)
        self.last_time: Optional[float] = None

    def reset(self, position: Sequence[float], now: float) -> None:
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        self.position = _array(position, (3,), "position")
        self.velocity = np.zeros(3)
        self.last_time = now

    def clear(self) -> None:
        self.position = None
        self.velocity = np.zeros(3)
        self.last_time = None

    def update(self, desired: Sequence[float], now: float) -> np.ndarray:
        desired_v = _array(desired, (3,), "desired")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if self.position is None or self.last_time is None:
            self.reset(desired_v, now)
            return desired_v.copy()
        dt = now - self.last_time
        if dt < 0.0:
            raise ValueError("now must not move backwards")
        if dt == 0.0:
            return self.position.copy()

        error = desired_v - self.position
        braking_speed = np.sqrt(2.0 * self.max_acceleration * np.abs(error))
        requested_velocity = np.sign(error) * np.minimum(self.max_velocity, braking_speed)
        velocity_delta = np.clip(
            requested_velocity - self.velocity,
            -self.max_acceleration * dt,
            self.max_acceleration * dt,
        )
        next_velocity = self.velocity + velocity_delta
        step = next_velocity * dt
        remaining = desired_v - (self.position + step)
        reached = (np.sign(error) != np.sign(remaining)) | (np.abs(step) >= np.abs(error))
        next_position = self.position + step
        next_position[reached] = desired_v[reached]
        next_velocity[reached] = 0.0
        self.position = next_position
        self.velocity = next_velocity
        self.last_time = now
        return next_position.copy()


class PerAxisCommandGate:
    """Use separate enter/exit errors to avoid command chatter per axis."""

    def __init__(self, enter_delta: Sequence[float], exit_delta: Sequence[float]) -> None:
        self.enter_delta = _array(enter_delta, (3,), "enter_delta")
        self.exit_delta = _array(exit_delta, (3,), "exit_delta")
        if np.any(self.enter_delta < 0.0) or np.any(self.exit_delta < 0.0):
            raise ValueError("command gate deltas must be non-negative")
        if np.any(self.exit_delta > self.enter_delta):
            raise ValueError("exit_delta must not exceed enter_delta")
        self.active = np.zeros(3, dtype=bool)

    def reset(self) -> None:
        self.active[:] = False

    def update(self, error: Sequence[float]) -> bool:
        magnitude = np.abs(_array(error, (3,), "error"))
        self.active = np.where(
            self.active,
            magnitude > self.exit_delta,
            magnitude >= self.enter_delta,
        )
        return bool(np.any(self.active))


class RotationMotionLimiter:
    """Limit angular velocity and acceleration in rotation-vector space."""

    def __init__(self, max_velocity: float, max_acceleration: float) -> None:
        if not all(math.isfinite(value) and value > 0.0 for value in (max_velocity, max_acceleration)):
            raise ValueError("angular limits must be positive and finite")
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.rotation: Optional[np.ndarray] = None
        self.angular_velocity = np.zeros(3)
        self.last_time: Optional[float] = None

    def reset(self, rotation: Sequence[Sequence[float]], now: float) -> None:
        value = _array(rotation, (3, 3), "rotation")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        self.rotation = value
        self.angular_velocity = np.zeros(3)
        self.last_time = now

    def clear(self) -> None:
        self.rotation = None
        self.angular_velocity = np.zeros(3)
        self.last_time = None

    def update(self, desired: Sequence[Sequence[float]], now: float) -> np.ndarray:
        desired_v = _array(desired, (3, 3), "desired rotation")
        if self.rotation is None or self.last_time is None:
            self.reset(desired_v, now)
            return desired_v.copy()
        dt = now - self.last_time
        if not math.isfinite(now) or dt < 0.0:
            raise ValueError("now must be finite and monotonic")
        if dt == 0.0:
            return self.rotation.copy()
        error_vector = rotation_vector(desired_v @ self.rotation.T)
        error_angle = float(np.linalg.norm(error_vector))
        if error_angle <= 1.0e-10:
            requested_velocity = np.zeros(3)
        else:
            braking_speed = min(self.max_velocity, math.sqrt(2.0 * self.max_acceleration * error_angle))
            requested_velocity = error_vector * (braking_speed / error_angle)
        velocity_delta = requested_velocity - self.angular_velocity
        delta_norm = float(np.linalg.norm(velocity_delta))
        max_delta = self.max_acceleration * dt
        if delta_norm > max_delta:
            velocity_delta *= max_delta / delta_norm
        next_velocity = self.angular_velocity + velocity_delta
        speed = float(np.linalg.norm(next_velocity))
        if speed > self.max_velocity:
            next_velocity *= self.max_velocity / speed
        step_vector = next_velocity * dt
        moving_toward = float(np.dot(step_vector, error_vector)) > 0.0
        if moving_toward and float(np.linalg.norm(step_vector)) >= error_angle:
            next_rotation = desired_v
            next_velocity = np.zeros(3)
        else:
            next_rotation = rotation_from_vector(step_vector) @ self.rotation
        self.rotation = next_rotation
        self.angular_velocity = next_velocity
        self.last_time = now
        return next_rotation.copy()


class TeleopMapping:
    """Map relative human-arm motion around calibration to robot targets."""

    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self.enabled = False
        self.calibrated = False
        self.latest_human_position: Optional[np.ndarray] = None
        self.latest_human_rotation: Optional[np.ndarray] = None
        self.latest_human_time: Optional[float] = None
        self.latest_robot_pose: Optional[RobotPose] = None
        self.human_zero_position: Optional[np.ndarray] = None
        self.human_zero_rotation: Optional[np.ndarray] = None
        self.robot_zero: Optional[RobotPose] = None
        self.filtered_offset: Optional[np.ndarray] = None
        self.filtered_rotation: Optional[np.ndarray] = None
        self.latest_diagnostics: Optional[MappingDiagnostics] = None
        self.motion_limiter = CartesianMotionLimiter(
            config.max_position_velocity_mps,
            config.max_position_acceleration_mps2,
        )
        self.rotation_limiter = RotationMotionLimiter(
            config.max_angular_velocity_radps,
            config.max_angular_acceleration_radps2,
        )

    def update_robot(self, xyz: Sequence[float], rpy: Sequence[float]) -> None:
        position, angles = _array(xyz, (3,), "robot xyz"), _array(rpy, (3,), "robot rpy")
        self.latest_robot_pose = RobotPose(position, rpy_to_matrix(*angles))

    def update_human(self, position: Sequence[float], quaternion_xyzw: Sequence[float], received_monotonic: float) -> None:
        self.latest_human_position = _array(position, (3,), "human position")
        self.latest_human_rotation = quaternion_to_matrix(quaternion_xyzw)
        if not math.isfinite(received_monotonic):
            raise ValueError("received_monotonic must be finite")
        self.latest_human_time = received_monotonic
        if self.enabled and not self.calibrated:
            self.calibrate()

    def set_enabled(self, enabled: bool) -> bool:
        rising = enabled and not self.enabled
        self.enabled = enabled
        if not enabled:
            self.calibrated = False
            self.filtered_offset = self.filtered_rotation = None
            self.latest_diagnostics = None
            self.motion_limiter.clear()
            self.rotation_limiter.clear()
        elif rising:
            self.calibrated = False
            self.calibrate()
        return self.calibrated

    def calibrate(self) -> bool:
        """Capture simultaneous human and measured robot zero poses."""
        if self.latest_human_position is None or self.latest_human_rotation is None or self.latest_robot_pose is None:
            self.calibrated = False
            return False
        self.human_zero_position = self.latest_human_position.copy()
        self.human_zero_rotation = self.latest_human_rotation.copy()
        self.robot_zero = RobotPose(self.latest_robot_pose.position.copy(), self.latest_robot_pose.rotation.copy())
        self.filtered_offset = np.zeros(3)
        self.filtered_rotation = self.robot_zero.rotation.copy()
        self.latest_diagnostics = MappingDiagnostics(
            human_delta_camera=np.zeros(3),
            raw_robot_offset=np.zeros(3),
            deadbanded_robot_offset=np.zeros(3),
            limited_robot_offset=np.zeros(3),
            filtered_robot_offset=np.zeros(3),
            motion_limited_robot_offset=np.zeros(3),
        )
        assert self.latest_human_time is not None
        self.motion_limiter.reset(self.robot_zero.position, self.latest_human_time)
        self.rotation_limiter.reset(self.robot_zero.rotation, self.latest_human_time)
        self.calibrated = True
        return True

    def invalidate_input(self) -> None:
        self.enabled = self.calibrated = False
        self.filtered_offset = self.filtered_rotation = None
        self.latest_diagnostics = None
        self.motion_limiter.clear()
        self.rotation_limiter.clear()

    def target(self, now_monotonic: float) -> Optional[TargetPose]:
        """Return a safe target, or None when disabled, stale or uncalibrated."""
        if (not self.enabled or not self.calibrated or self.latest_human_time is None or now_monotonic - self.latest_human_time > self.config.stale_timeout_sec or self.latest_human_position is None or self.latest_human_rotation is None or self.human_zero_position is None or self.human_zero_rotation is None or self.robot_zero is None):
            return None
        human_delta = self.latest_human_position - self.human_zero_position
        raw_offset = self.config.position_gain * (self.config.position_axis_matrix @ human_delta)
        deadbanded = np.sign(raw_offset) * np.maximum(
            np.abs(raw_offset) - self.config.position_deadband_m,
            0.0,
        )
        limited = np.clip(deadbanded, -self.config.max_position_offset, self.config.max_position_offset)
        assert self.filtered_offset is not None and self.filtered_rotation is not None
        self.filtered_offset = self.config.position_alpha_axes * limited + (1.0 - self.config.position_alpha_axes) * self.filtered_offset
        desired_position = self.robot_zero.position + self.filtered_offset
        motion_limited_position = self.motion_limiter.update(desired_position, now_monotonic)
        motion_limited_offset = motion_limited_position - self.robot_zero.position
        self.latest_diagnostics = MappingDiagnostics(
            human_delta_camera=human_delta.copy(),
            raw_robot_offset=raw_offset.copy(),
            deadbanded_robot_offset=deadbanded.copy(),
            limited_robot_offset=limited.copy(),
            filtered_robot_offset=self.filtered_offset.copy(),
            motion_limited_robot_offset=motion_limited_offset.copy(),
        )
        if self.config.enable_orientation:
            relative_human = self.latest_human_rotation @ self.human_zero_rotation.T
            relative_robot = self.config.position_axis_matrix @ relative_human @ self.config.position_axis_matrix.T
            relative_rpy = np.clip(
                matrix_to_rpy(relative_robot) * self.config.orientation_axes,
                -self.config.max_rotation_rad,
                self.config.max_rotation_rad,
            )
            target_rotation = rpy_to_matrix(*relative_rpy) @ self.robot_zero.rotation
            self.filtered_rotation = interpolate_rotation(self.filtered_rotation, target_rotation, self.config.rotation_alpha)
            output_rotation = self.rotation_limiter.update(self.filtered_rotation, now_monotonic)
        else:
            self.filtered_rotation = self.robot_zero.rotation.copy()
            output_rotation = self.robot_zero.rotation.copy()
        return TargetPose(motion_limited_position, matrix_to_rpy(output_rotation))


def limit_vector_step(
    current: Sequence[float],
    desired: Sequence[float],
    max_step: float,
) -> np.ndarray:
    """Limit one commanded Cartesian displacement to ``max_step`` metres."""
    current_v = _array(current, (3,), "current")
    desired_v = _array(desired, (3,), "desired")
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("max_step must be positive and finite")
    delta = desired_v - current_v
    distance = float(np.linalg.norm(delta))
    if distance <= max_step:
        return desired_v
    return current_v + delta * (max_step / distance)


class GestureDebouncer:
    """Emit a gripper state only after a confident stable transition."""

    def __init__(self, open_labels: Sequence[str], close_labels: Sequence[str], score_min: float, stable_frames: int) -> None:
        if not 0.0 <= score_min <= 1.0 or stable_frames < 1:
            raise ValueError("invalid gesture debounce settings")
        self.open_labels, self.close_labels = set(open_labels), set(close_labels)
        self.score_min, self.stable_frames = score_min, stable_frames
        self.candidate: Optional[bool] = None
        self.candidate_count = 0
        self.current: Optional[bool] = None

    def update(self, label: str, score: float, enabled: bool) -> Optional[bool]:
        desired: Optional[bool]
        if not enabled or not math.isfinite(score) or score < self.score_min:
            desired = None
        elif label in self.open_labels:
            desired = True
        elif label in self.close_labels:
            desired = False
        else:
            desired = None
        if desired is None:
            self.candidate, self.candidate_count = None, 0
            return None
        if desired != self.candidate:
            self.candidate, self.candidate_count = desired, 1
        else:
            self.candidate_count += 1
        if self.candidate_count < self.stable_frames or desired == self.current:
            return None
        self.current = desired
        return desired


class PinchHysteresis:
    """Convert a normalized thumb/index distance into timed gripper states."""

    def __init__(
        self,
        close_ratio: float,
        open_ratio: float,
        stable_sec: float,
        cooldown_sec: float,
    ) -> None:
        values = (close_ratio, open_ratio, stable_sec, cooldown_sec)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pinch settings must be finite")
        if close_ratio < 0.0 or open_ratio <= close_ratio:
            raise ValueError("open_ratio must exceed non-negative close_ratio")
        if stable_sec < 0.0 or cooldown_sec < 0.0:
            raise ValueError("pinch times must be non-negative")
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.stable_sec = stable_sec
        self.cooldown_sec = cooldown_sec
        self.candidate: Optional[bool] = None
        self.candidate_since: Optional[float] = None
        self.current: Optional[bool] = None
        self.last_transition_time: Optional[float] = None

    def reset_candidate(self) -> None:
        self.candidate = None
        self.candidate_since = None

    def update(self, ratio: float, now: float, enabled: bool) -> Optional[bool]:
        if not math.isfinite(ratio) or not math.isfinite(now):
            self.reset_candidate()
            return None
        if not enabled:
            self.reset_candidate()
            return None
        desired: Optional[bool]
        if ratio <= self.close_ratio:
            desired = False
        elif ratio >= self.open_ratio:
            desired = True
        else:
            desired = None
        if desired is None or desired == self.current:
            self.reset_candidate()
            return None
        if desired != self.candidate:
            self.candidate = desired
            self.candidate_since = now
            return None
        assert self.candidate_since is not None
        if now - self.candidate_since < self.stable_sec:
            return None
        if self.last_transition_time is not None and now - self.last_transition_time < self.cooldown_sec:
            return None
        self.current = desired
        self.last_transition_time = now
        self.reset_candidate()
        return desired


class TeleopState(str, Enum):
    DISABLED = "DISABLED"
    CALIBRATING = "CALIBRATING"
    TRACKING = "TRACKING"
    STALE_LOCK = "STALE_LOCK"
    FAULT_LOCK = "FAULT_LOCK"


class TeleopStateMachine:
    """Explicitly require disable/re-enable after stale input or a fault."""

    def __init__(self) -> None:
        self.state = TeleopState.DISABLED
        self.disabled_seen = True

    def source_enable(self, enabled: bool, inputs_ready: bool) -> TeleopState:
        if not enabled:
            self.disabled_seen = True
            self.state = TeleopState.DISABLED
            return self.state
        if self.state in (TeleopState.STALE_LOCK, TeleopState.FAULT_LOCK):
            if not self.disabled_seen:
                return self.state
        self.disabled_seen = False
        self.state = TeleopState.TRACKING if inputs_ready else TeleopState.CALIBRATING
        return self.state

    def inputs_ready(self) -> TeleopState:
        if self.state == TeleopState.CALIBRATING:
            self.state = TeleopState.TRACKING
        return self.state

    def stale(self) -> TeleopState:
        self.state = TeleopState.STALE_LOCK
        self.disabled_seen = False
        return self.state

    def fault(self) -> TeleopState:
        self.state = TeleopState.FAULT_LOCK
        self.disabled_seen = False
        return self.state


class DepthClutch:
    """Update monocular depth only while its explicit clutch is active."""

    def __init__(self) -> None:
        self.held_value: Optional[float] = None

    def reset(self) -> None:
        self.held_value = None

    def update(self, observed_depth: float, active: bool) -> float:
        if not math.isfinite(observed_depth):
            raise ValueError("observed_depth must be finite")
        if self.held_value is None or active:
            self.held_value = observed_depth
        return self.held_value
