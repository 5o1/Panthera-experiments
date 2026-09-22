"""Deterministic constrained head-camera sampling for Panthera datasets.

The sampler is intentionally independent from SAPIEN.  It accepts explicit
world-space keypoints, samples a front-hemisphere overhead camera, and rejects
the candidate unless every keypoint projects into the configured central
image region.  This makes the visibility contract independently auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np

POSE_MODES = ("randomized", "fixed")


@dataclass(frozen=True)
class CameraRandomizationSpec:
    enabled: bool = True
    # ``randomized`` samples a new constrained pose per episode; ``fixed``
    # replays one configured world pose for every episode. Fixed mode requires
    # all four ``fixed_*`` values and ignores the sampling ranges except as
    # bounds the fixed pose must satisfy.
    pose_mode: str = "randomized"
    fixed_azimuth_deg: float | None = None
    fixed_height_above_table_m: float | None = None
    fixed_horizontal_distance_m: float | None = None
    fixed_look_at_xyz_m: tuple[float, float, float] | None = None
    image_width_px: int = 320
    image_height_px: int = 240
    vertical_fov_deg: float = 75.0
    central_region_fraction: float = 0.80
    minimum_height_above_table_m: float = 0.55
    maximum_height_above_table_m: float = 0.95
    minimum_horizontal_distance_m: float = 0.65
    maximum_horizontal_distance_m: float = 1.10
    minimum_azimuth_deg: float = 25.0
    maximum_azimuth_deg: float = 155.0
    minimum_downward_angle_deg: float = 28.0
    target_xy_jitter_m: float = 0.06
    target_height_min_m: float = 0.08
    target_height_max_m: float = 0.22
    maximum_sampling_attempts: int = 512


def camera_randomization_spec(
    config: Mapping[str, object] | None = None,
) -> CameraRandomizationSpec:
    values = dict(config or {})
    allowed = set(CameraRandomizationSpec.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown camera_randomization keys: {unknown}")
    spec = CameraRandomizationSpec(**{name: values[name] for name in values})
    numeric = np.asarray(
        [
            spec.vertical_fov_deg,
            spec.central_region_fraction,
            spec.minimum_height_above_table_m,
            spec.maximum_height_above_table_m,
            spec.minimum_horizontal_distance_m,
            spec.maximum_horizontal_distance_m,
            spec.minimum_azimuth_deg,
            spec.maximum_azimuth_deg,
            spec.minimum_downward_angle_deg,
            spec.target_xy_jitter_m,
            spec.target_height_min_m,
            spec.target_height_max_m,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("camera randomization values must be finite")
    if spec.image_width_px <= 0 or spec.image_height_px <= 0:
        raise ValueError("camera image dimensions must be positive")
    if not 1.0 < spec.vertical_fov_deg < 179.0:
        raise ValueError("vertical_fov_deg must be in (1, 179)")
    if not 0.0 < spec.central_region_fraction < 1.0:
        raise ValueError("central_region_fraction must be in (0, 1)")
    if not 0.0 < spec.minimum_height_above_table_m <= spec.maximum_height_above_table_m:
        raise ValueError("invalid camera height range")
    if not 0.0 < spec.minimum_horizontal_distance_m <= spec.maximum_horizontal_distance_m:
        raise ValueError("invalid horizontal distance range")
    if not spec.minimum_azimuth_deg < spec.maximum_azimuth_deg:
        raise ValueError("invalid camera azimuth range")
    if not 0.0 < spec.minimum_downward_angle_deg < 90.0:
        raise ValueError("minimum_downward_angle_deg must be in (0, 90)")
    if spec.target_xy_jitter_m < 0.0:
        raise ValueError("target_xy_jitter_m must be nonnegative")
    if not 0.0 <= spec.target_height_min_m <= spec.target_height_max_m:
        raise ValueError("invalid camera target-height range")
    if spec.maximum_sampling_attempts <= 0:
        raise ValueError("maximum_sampling_attempts must be positive")
    if spec.pose_mode not in POSE_MODES:
        raise ValueError(f"pose_mode must be one of {POSE_MODES}")
    if spec.pose_mode == "randomized":
        stray = [
            name
            for name in (
                "fixed_azimuth_deg",
                "fixed_height_above_table_m",
                "fixed_horizontal_distance_m",
                "fixed_look_at_xyz_m",
            )
            if getattr(spec, name) is not None
        ]
        if stray:
            raise ValueError(
                f"pose_mode is randomized but fixed pose values are set: {stray}"
            )
        return spec
    missing = [
        name
        for name in (
            "fixed_azimuth_deg",
            "fixed_height_above_table_m",
            "fixed_horizontal_distance_m",
            "fixed_look_at_xyz_m",
        )
        if getattr(spec, name) is None
    ]
    if missing:
        raise ValueError(f"fixed camera pose requires {missing}")
    look_at = np.asarray(spec.fixed_look_at_xyz_m, dtype=float)
    if look_at.shape != (3,) or not np.all(np.isfinite(look_at)):
        raise ValueError("fixed_look_at_xyz_m must be one finite 3-vector")
    spec = replace(
        spec,
        fixed_azimuth_deg=float(spec.fixed_azimuth_deg),
        fixed_height_above_table_m=float(spec.fixed_height_above_table_m),
        fixed_horizontal_distance_m=float(spec.fixed_horizontal_distance_m),
        fixed_look_at_xyz_m=tuple(float(value) for value in look_at),
    )
    for name, value, lower, upper in (
        (
            "fixed_azimuth_deg",
            spec.fixed_azimuth_deg,
            spec.minimum_azimuth_deg,
            spec.maximum_azimuth_deg,
        ),
        (
            "fixed_height_above_table_m",
            spec.fixed_height_above_table_m,
            spec.minimum_height_above_table_m,
            spec.maximum_height_above_table_m,
        ),
        (
            "fixed_horizontal_distance_m",
            spec.fixed_horizontal_distance_m,
            spec.minimum_horizontal_distance_m,
            spec.maximum_horizontal_distance_m,
        ),
    ):
        if not lower - 1.0e-9 <= value <= upper + 1.0e-9:
            raise ValueError(
                f"{name}={value} must lie inside the configured range [{lower}, {upper}]"
            )
    return spec


def project_world_points(
    points_xyz: Sequence[Sequence[float]],
    position_xyz: Sequence[float],
    forward_xyz: Sequence[float],
    left_xyz: Sequence[float],
    vertical_fov_deg: float,
    image_width_px: int,
    image_height_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to normalized image UV and return camera depths.

    UV uses the conventional image range ``[0, 1]`` from left/top to
    right/bottom.  Only margins matter to the contract, so the horizontal sign
    follows SAPIEN's camera basis but is not otherwise semantically important.
    """
    points = np.asarray(points_xyz, dtype=float)
    position = np.asarray(position_xyz, dtype=float)
    forward = np.asarray(forward_xyz, dtype=float)
    left = np.asarray(left_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_xyz must have shape (N, 3)")
    if position.shape != (3,) or forward.shape != (3,) or left.shape != (3,):
        raise ValueError("camera vectors must have shape (3,)")
    if not np.all(np.isfinite(np.concatenate((points.ravel(), position, forward, left)))):
        raise ValueError("camera projection inputs must be finite")
    forward /= np.linalg.norm(forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    up /= np.linalg.norm(up)
    relative = points - position
    depth = relative @ forward
    horizontal = relative @ left
    vertical = relative @ up
    tan_y = math.tan(math.radians(vertical_fov_deg) / 2.0)
    tan_x = tan_y * (float(image_width_px) / float(image_height_px))
    ndc_x = horizontal / (depth * tan_x)
    ndc_y = vertical / (depth * tan_y)
    uv = np.column_stack(((1.0 - ndc_x) / 2.0, (1.0 - ndc_y) / 2.0))
    return uv, depth


def sample_constrained_camera(
    seed: int,
    table_height_m: float,
    task_center_xyz: Sequence[float],
    keypoint_names: Sequence[str],
    keypoints_xyz: Sequence[Sequence[float]],
    config: Mapping[str, object] | None = None,
    sampling_stream: int = 0,
) -> dict:
    """Return one deterministic overhead camera satisfying all constraints.

    ``pose_mode: randomized`` samples a fresh pose per episode from the seeded
    stream. ``pose_mode: fixed`` ignores both the seed and ``task_center_xyz``
    for the pose itself and replays ``fixed_*``, so every episode shares one
    world pose. The visibility gates still apply; a failing fixed pose raises
    with the specific reason instead of resampling.
    """
    if seed < 0:
        raise ValueError("episode seed must be nonnegative")
    if sampling_stream < 0:
        raise ValueError("sampling_stream must be nonnegative")
    spec = camera_randomization_spec(config)
    fixed_pose = spec.pose_mode == "fixed"
    center = np.asarray(task_center_xyz, dtype=float)
    points = np.asarray(keypoints_xyz, dtype=float)
    names = list(keypoint_names)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("task_center_xyz must be one finite 3-vector")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(names):
        raise ValueError("keypoint names and coordinates must have matching length")
    if len(set(names)) != len(names):
        raise ValueError("keypoint names must be unique")
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x43414D45, sampling_stream])
    )
    half_fraction = spec.central_region_fraction / 2.0
    lower_uv = 0.5 - half_fraction
    upper_uv = 0.5 + half_fraction

    def reject(reason: str) -> None:
        """Raise for a fixed pose; fall through so randomized sampling retries."""
        if fixed_pose:
            raise RuntimeError(f"fixed camera pose rejected: {reason}")

    maximum_attempts = 1 if fixed_pose else spec.maximum_sampling_attempts
    for attempt in range(1, maximum_attempts + 1):
        if fixed_pose:
            azimuth = math.radians(float(spec.fixed_azimuth_deg))
            horizontal_distance = float(spec.fixed_horizontal_distance_m)
            height_above_table = float(spec.fixed_height_above_table_m)
            target = np.asarray(spec.fixed_look_at_xyz_m, dtype=float)
        else:
            azimuth = math.radians(
                float(rng.uniform(spec.minimum_azimuth_deg, spec.maximum_azimuth_deg))
            )
            horizontal_distance = float(
                rng.uniform(
                    spec.minimum_horizontal_distance_m,
                    spec.maximum_horizontal_distance_m,
                )
            )
            height_above_table = float(
                rng.uniform(
                    spec.minimum_height_above_table_m,
                    spec.maximum_height_above_table_m,
                )
            )
            target = center.copy()
            target[:2] += rng.uniform(
                -spec.target_xy_jitter_m, spec.target_xy_jitter_m, size=2
            )
            target[2] = table_height_m + float(
                rng.uniform(spec.target_height_min_m, spec.target_height_max_m)
            )
        position = np.array(
            [
                target[0] + horizontal_distance * math.cos(azimuth),
                target[1] + horizontal_distance * math.sin(azimuth),
                table_height_m + height_above_table,
            ],
            dtype=float,
        )
        forward = target - position
        forward /= np.linalg.norm(forward)
        downward_angle_deg = math.degrees(math.asin(float(-forward[2])))
        if downward_angle_deg < spec.minimum_downward_angle_deg:
            reject(
                f"downward angle {downward_angle_deg:.3f} deg is below the "
                f"{spec.minimum_downward_angle_deg} deg minimum"
            )
            continue
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left_norm = float(np.linalg.norm(left))
        if left_norm <= 1.0e-8:
            reject("camera frame is degenerate: forward is parallel to world Z")
            continue
        left /= left_norm
        up = np.cross(forward, left)
        up /= np.linalg.norm(up)
        uv, depth = project_world_points(
            points,
            position,
            forward,
            left,
            spec.vertical_fov_deg,
            spec.image_width_px,
            spec.image_height_px,
        )
        if np.any(depth <= 0.10):
            reject("at least one keypoint is closer than 0.10 m to the camera")
            continue
        margins = np.minimum(uv, 1.0 - uv)
        if np.any(uv < lower_uv) or np.any(uv > upper_uv):
            reject("at least one keypoint leaves the central image region")
            continue
        minimum_border_margin = float(np.min(margins))
        minimum_central_margin = float(
            np.min(np.minimum(uv - lower_uv, upper_uv - uv))
        )
        return {
            "schema_version": 1,
            "pose_mode": spec.pose_mode,
            "sampling_stream": sampling_stream,
            "sampling_attempt": attempt,
            "position_xyz_m": position.tolist(),
            "forward_xyz": forward.tolist(),
            "left_xyz": left.tolist(),
            "up_xyz": up.tolist(),
            "look_at_xyz_m": target.tolist(),
            "height_above_table_m": height_above_table,
            "horizontal_distance_m": horizontal_distance,
            "azimuth_deg": math.degrees(azimuth),
            "downward_angle_deg": downward_angle_deg,
            "image_width_px": spec.image_width_px,
            "image_height_px": spec.image_height_px,
            "vertical_fov_deg": spec.vertical_fov_deg,
            "central_region_fraction": spec.central_region_fraction,
            "central_region_uv": [lower_uv, lower_uv, upper_uv, upper_uv],
            "minimum_border_margin_fraction": minimum_border_margin,
            "minimum_central_margin_fraction": minimum_central_margin,
            "keypoint_names": names,
            "keypoints_world_xyz_m": points.tolist(),
            "keypoints_uv": uv.tolist(),
            "keypoint_depths_m": depth.tolist(),
            "constraints": asdict(spec),
        }
    if fixed_pose:
        raise RuntimeError("fixed camera pose did not satisfy the visibility contract")
    raise RuntimeError(
        f"could not sample a visible overhead camera in {spec.maximum_sampling_attempts} attempts"
    )
