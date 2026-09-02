"""ROS-independent summary calculations for teleoperation recordings."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


VECTOR_TOPICS = {
    "/teleop/human_pose": ("x", "y", "z"),
    "/teleop/debug_target": ("x", "y", "z"),
    "/teleop/diagnostics/human_delta_camera": ("x", "y", "z"),
    "/teleop/diagnostics/raw_robot_offset": ("x", "y", "z"),
    "/teleop/diagnostics/deadbanded_robot_offset": ("x", "y", "z"),
    "/teleop/diagnostics/limited_robot_offset": ("x", "y", "z"),
    "/teleop/diagnostics/filtered_robot_offset": ("x", "y", "z"),
    "/teleop/diagnostics/motion_limited_robot_offset": ("x", "y", "z"),
    "/teleop/depth_features": ("world_depth", "image_reach", "image_arm_scale"),
    "/end_pose_euler": ("x", "y", "z"),
    "/pos_cmd": ("x", "y", "z"),
}

SCALAR_TOPICS = {
    "/teleop/pose_score": "score",
    "/teleop/gesture_score": "score",
    "/teleop/pinch_ratio": "ratio",
    "/teleop/transport_extra_lag_ms": "ms",
}


def _array(samples: Sequence[Sequence[float]]) -> np.ndarray:
    if len(samples) == 0:
        return np.empty((0, 3), dtype=float)
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("vector samples must have shape (N, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("vector samples must be finite")
    return values


def vector_summary(samples: Sequence[Sequence[float]], axis_names: Sequence[str]) -> dict:
    """Return per-axis range/noise statistics for a sequence of 3-vectors."""
    values = _array(samples)
    if len(axis_names) != 3:
        raise ValueError("axis_names must contain three labels")
    result: dict[str, object] = {"count": int(len(values))}
    if len(values) == 0:
        result["axes"] = {}
        return result
    axes = {}
    for index, name in enumerate(axis_names):
        column = values[:, index]
        axes[str(name)] = {
            "min": float(np.min(column)),
            "max": float(np.max(column)),
            "peak_to_peak": float(np.ptp(column)),
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
        }
    result["axes"] = axes
    result["path_length"] = float(np.sum(np.linalg.norm(np.diff(values, axis=0), axis=1))) if len(values) > 1 else 0.0
    return result


def scalar_summary(samples: Sequence[float]) -> dict:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scalar samples must be a finite 1-D sequence")
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "peak_to_peak": float(np.ptp(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
    }


def recording_phase_intervals(markers: Sequence[tuple[float, str]]) -> dict[str, tuple[float, float]]:
    """Find first/last timestamps for each frame-marked RECORDING phase."""
    intervals: dict[str, tuple[float, float]] = {}
    for timestamp, marker in markers:
        parts = marker.split(":", 2)
        if len(parts) != 3 or parts[1] != "RECORDING" or parts[0] == "none":
            continue
        name = parts[0]
        if name in intervals:
            intervals[name] = (intervals[name][0], float(timestamp))
        else:
            intervals[name] = (float(timestamp), float(timestamp))
    return intervals


def timing_summary(timestamps: Sequence[float]) -> dict:
    values = np.asarray(timestamps, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("timestamps must be a finite 1-D sequence")
    if len(values) < 2:
        return {"count": int(len(values)), "duration_sec": 0.0}
    intervals = np.diff(values)
    if np.any(intervals <= 0.0):
        raise ValueError("timestamps must strictly increase")
    duration = float(values[-1] - values[0])
    return {
        "count": int(len(values)),
        "duration_sec": duration,
        "mean_rate_hz": float((len(values) - 1) / duration),
        "median_interval_ms": float(np.median(intervals) * 1000.0),
        "p95_interval_ms": float(np.percentile(intervals, 95) * 1000.0),
        "max_interval_ms": float(np.max(intervals) * 1000.0),
    }


def summarize_recording(samples: Mapping[str, Sequence]) -> dict:
    """Summarize known vector topics and mapping-limit activation rates."""
    summary: dict[str, object] = {"topics": {}}
    topic_result: dict[str, object] = summary["topics"]  # type: ignore[assignment]
    for topic, axes in VECTOR_TOPICS.items():
        if topic in samples:
            topic_result[topic] = vector_summary(samples[topic], axes)
    for topic in SCALAR_TOPICS:
        if topic in samples:
            topic_result[topic] = scalar_summary(samples[topic])

    pairs = (
        ("clipped_fraction", "/teleop/diagnostics/deadbanded_robot_offset", "/teleop/diagnostics/limited_robot_offset"),
        ("motion_limited_fraction", "/teleop/diagnostics/filtered_robot_offset", "/teleop/diagnostics/motion_limited_robot_offset"),
    )
    for name, before_topic, after_topic in pairs:
        if before_topic not in samples or after_topic not in samples:
            continue
        before, after = _array(samples[before_topic]), _array(samples[after_topic])
        count = min(len(before), len(after))
        if count:
            changed = np.any(np.abs(before[:count] - after[:count]) > 1e-9, axis=1)
            summary[name] = float(np.mean(changed))
    return summary
