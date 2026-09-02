"""Validation for the camera-process to ROS WebSocket protocol."""

from dataclasses import dataclass
import json
import math
from typing import Any, Optional


class ProtocolError(ValueError):
    """Raised when a packet must not enter the ROS graph."""


@dataclass(frozen=True)
class VisionPacket:
    seq: int
    source_time_ms: int
    enabled: bool
    recalibrate: bool
    pose_valid: bool
    hand_valid: bool
    shoulder: tuple[float, float, float]
    elbow: tuple[float, float, float]
    wrist: tuple[float, float, float]
    pose_score: float
    gesture: str
    gesture_score: float
    depth_world_m: float
    image_reach_ratio: float
    image_arm_scale: float
    pinch_ratio: float
    depth_clutch: bool
    orientation_valid: bool
    experiment_marker: str


@dataclass
class TransportLagGuard:
    """Detect source-frame backlog without assuming synchronized clock epochs."""

    max_extra_delay_ms: float
    minimum_clock_offset_ms: Optional[float] = None
    latest_extra_delay_ms: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_extra_delay_ms) or self.max_extra_delay_ms <= 0.0:
            raise ValueError("max_extra_delay_ms must be positive and finite")

    def accept(self, source_time_ms: int, arrival_time_ms: float) -> bool:
        """Return false when a frame is delayed beyond the best seen offset."""
        if source_time_ms < 0 or not math.isfinite(arrival_time_ms):
            raise ValueError("timestamps must be finite and non-negative")
        offset = arrival_time_ms - source_time_ms
        if self.minimum_clock_offset_ms is None or offset < self.minimum_clock_offset_ms:
            self.minimum_clock_offset_ms = offset
        self.latest_extra_delay_ms = max(0.0, offset - self.minimum_clock_offset_ms)
        return self.latest_extra_delay_ms <= self.max_extra_delay_ms


@dataclass
class EnableRearmLatch:
    """Require an explicit disabled-to-enabled transition after a fault."""

    locked: bool = False
    disabled_seen: bool = False

    def force_lock(self) -> None:
        self.locked = True
        self.disabled_seen = False

    def update(self, source_enabled: bool, pose_usable: bool) -> bool:
        if not pose_usable:
            self.locked = True
            self.disabled_seen = not source_enabled
            return False
        if not source_enabled:
            self.disabled_seen = True
            return False
        if self.locked:
            if not self.disabled_seen:
                return False
            self.locked = False
            self.disabled_seen = False
        return True


def latch_enable_on_pose_loss(
    enabled: bool,
    pose_valid: bool,
    pose_score: float,
    pose_score_min: float,
) -> tuple[bool, bool]:
    """Latch teleoperation off when tracking becomes unusable.

    Returns ``(enabled_after_check, newly_latched)``.  Re-enabling is an
    explicit operator action; a later valid pose never clears this latch by
    itself.
    """
    if not 0.0 <= pose_score_min <= 1.0:
        raise ValueError("pose_score_min must be in [0, 1]")
    usable = pose_valid and math.isfinite(pose_score) and pose_score >= pose_score_min
    if enabled and not usable:
        return False, True
    return enabled, False


def _bool(data: dict[str, Any], name: str) -> bool:
    value = data.get(name)
    if type(value) is not bool:
        raise ProtocolError(f"{name} must be a boolean")
    return value


def _int(data: dict[str, Any], name: str) -> int:
    value = data.get(name)
    if type(value) is not int or value < 0:
        raise ProtocolError(f"{name} must be a non-negative integer")
    return value


def _score(data: dict[str, Any], name: str) -> float:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ProtocolError(f"{name} must be finite and in [0, 1]")
    return result


def _optional_finite(data: dict[str, Any], name: str, default: float) -> float:
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite")
    return result


def _optional_bool(data: dict[str, Any], name: str, default: bool) -> bool:
    value = data.get(name, default)
    if type(value) is not bool:
        raise ProtocolError(f"{name} must be a boolean")
    return value


def _optional_string(data: dict[str, Any], name: str, default: str, max_length: int = 160) -> str:
    value = data.get(name, default)
    if not isinstance(value, str) or len(value) > max_length:
        raise ProtocolError(f"{name} must be a string of at most {max_length} characters")
    return value


def _vector3(data: dict[str, Any], name: str) -> tuple[float, float, float]:
    value = data.get(name)
    if not isinstance(value, list) or len(value) != 3:
        raise ProtocolError(f"{name} must be a 3-element array")
    result: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ProtocolError(f"{name} components must be numeric")
        converted = float(component)
        if not math.isfinite(converted):
            raise ProtocolError(f"{name} components must be finite")
        result.append(converted)
    return result[0], result[1], result[2]


def parse_packet(
    raw: str,
    previous_seq: Optional[int] = None,
    previous_source_time_ms: Optional[int] = None,
) -> VisionPacket:
    """Parse protocol v1 and enforce sequence monotonicity within a connection."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolError("message is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ProtocolError("top-level JSON value must be an object")
    if data.get("v") != 1:
        raise ProtocolError("unsupported protocol version")
    seq = _int(data, "seq")
    if previous_seq is not None and seq <= previous_seq:
        raise ProtocolError(f"seq must increase (received {seq} after {previous_seq})")
    source_time_ms = _int(data, "source_time_ms")
    if previous_source_time_ms is not None and source_time_ms <= previous_source_time_ms:
        raise ProtocolError(
            "source_time_ms must increase "
            f"(received {source_time_ms} after {previous_source_time_ms})"
        )
    gesture = data.get("gesture")
    if not isinstance(gesture, str) or len(gesture) > 80:
        raise ProtocolError("gesture must be a string of at most 80 characters")
    return VisionPacket(
        seq=seq,
        source_time_ms=source_time_ms,
        enabled=_bool(data, "enabled"),
        recalibrate=_bool(data, "recalibrate"),
        pose_valid=_bool(data, "pose_valid"),
        hand_valid=_bool(data, "hand_valid"),
        shoulder=_vector3(data, "shoulder"),
        elbow=_vector3(data, "elbow"),
        wrist=_vector3(data, "wrist"),
        pose_score=_score(data, "pose_score"),
        gesture=gesture,
        gesture_score=_score(data, "gesture_score"),
        depth_world_m=_optional_finite(data, "depth_world_m", 0.0),
        image_reach_ratio=_optional_finite(data, "image_reach_ratio", 0.0),
        image_arm_scale=_optional_finite(data, "image_arm_scale", 0.0),
        pinch_ratio=_optional_finite(data, "pinch_ratio", -1.0),
        depth_clutch=_optional_bool(data, "depth_clutch", False),
        orientation_valid=_optional_bool(data, "orientation_valid", True),
        experiment_marker=_optional_string(data, "experiment_marker", ""),
    )
