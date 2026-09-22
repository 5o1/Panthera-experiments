"""Deterministic, stratified scene sampling for the Panthera v2 dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class WorkspaceSpec:
    """Safe tabletop subset of 75% of the Panthera nominal horizontal reach."""

    robot_base_x_m: float = 0.0
    robot_base_y_m: float = -0.35
    nominal_reach_radius_m: float = 0.46
    reach_fraction: float = 0.75
    minimum_radius_m: float = 0.22
    minimum_angle_deg: float = 5.0
    maximum_angle_deg: float = 175.0
    radial_bins: int = 3
    angular_bins: int = 12
    minimum_object_separation_m: float = 0.16

    @property
    def maximum_radius_m(self) -> float:
        return self.nominal_reach_radius_m * self.reach_fraction


def workspace_spec(config: Mapping[str, object] | None = None) -> WorkspaceSpec:
    """Create and validate a workspace specification from task YAML values."""
    values = dict(config or {})
    allowed = set(WorkspaceSpec.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown workspace keys: {unknown}")
    spec = WorkspaceSpec(**{name: values[name] for name in values})
    numeric = np.asarray(
        [
            spec.robot_base_x_m,
            spec.robot_base_y_m,
            spec.nominal_reach_radius_m,
            spec.reach_fraction,
            spec.minimum_radius_m,
            spec.minimum_angle_deg,
            spec.maximum_angle_deg,
            spec.minimum_object_separation_m,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("workspace values must be finite")
    if spec.nominal_reach_radius_m <= 0.0:
        raise ValueError("nominal reach radius must be positive")
    if not 0.0 < spec.reach_fraction <= 1.0:
        raise ValueError("reach fraction must be in (0, 1]")
    if not 0.0 <= spec.minimum_radius_m < spec.maximum_radius_m:
        raise ValueError("minimum radius must be below maximum radius")
    if not 0.0 <= spec.minimum_angle_deg < spec.maximum_angle_deg <= 180.0:
        raise ValueError("workspace angle range must lie in [0, 180]")
    if spec.radial_bins <= 0 or spec.angular_bins <= 0:
        raise ValueError("workspace bin counts must be positive")
    if spec.minimum_object_separation_m <= 0.0:
        raise ValueError("minimum object separation must be positive")
    return spec


def _stratified_point(
    rng: np.random.Generator,
    spec: WorkspaceSpec,
    radial_bin: int,
    angular_bin: int,
) -> np.ndarray:
    """Sample uniformly by area inside one polar workspace cell."""
    radial_edges = np.linspace(
        spec.minimum_radius_m, spec.maximum_radius_m, spec.radial_bins + 1
    )
    angle_edges = np.deg2rad(
        np.linspace(
            spec.minimum_angle_deg,
            spec.maximum_angle_deg,
            spec.angular_bins + 1,
        )
    )
    low_radius, high_radius = radial_edges[radial_bin : radial_bin + 2]
    radius = math.sqrt(
        float(rng.uniform(low_radius * low_radius, high_radius * high_radius))
    )
    angle = float(rng.uniform(*angle_edges[angular_bin : angular_bin + 2]))
    return np.asarray(
        [
            spec.robot_base_x_m + radius * math.cos(angle),
            spec.robot_base_y_m + radius * math.sin(angle),
        ],
        dtype=float,
    )


def sample_scene(seed: int, config: Mapping[str, object] | None = None) -> dict:
    """Return one balanced posture and two non-overlapping workspace positions."""
    if seed < 0:
        raise ValueError("episode seed must be nonnegative")
    spec = workspace_spec(config)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x50414E54]))

    # Consecutive blocks of 36 seeds cover every 12x3 polar cell once for the
    # cylinder. The socket uses coprime permutations, so its placement is not a
    # fixed offset from the cylinder even though the whole scene is replayable.
    cell_count = spec.radial_bins * spec.angular_bins
    cell_index = seed % cell_count
    cylinder_radial_bin = cell_index // spec.angular_bins
    cylinder_angular_bin = cell_index % spec.angular_bins
    cylinder_xy = _stratified_point(
        rng, spec, cylinder_radial_bin, cylinder_angular_bin
    )

    # 25 is coprime with the 36-cell default grid.  Before separation repair,
    # each block of 36 seeds therefore also covers every socket cell exactly
    # once instead of collapsing onto half of the cells.
    socket_cell_index = (seed * 25 + 7) % cell_count
    socket_radial_bin = socket_cell_index // spec.angular_bins
    socket_angular_bin = socket_cell_index % spec.angular_bins
    for attempt in range(spec.angular_bins * spec.radial_bins):
        socket_xy = _stratified_point(
            rng, spec, socket_radial_bin, socket_angular_bin
        )
        if float(np.linalg.norm(socket_xy - cylinder_xy)) >= spec.minimum_object_separation_m:
            break
        socket_angular_bin = (socket_angular_bin + 5) % spec.angular_bins
        if attempt % spec.angular_bins == spec.angular_bins - 1:
            socket_radial_bin = (socket_radial_bin + 1) % spec.radial_bins
    else:
        raise RuntimeError("could not sample non-overlapping cylinder and socket")

    posture = "upright" if seed % 2 == 0 else "lying"
    lying_angle_bin = (seed // 2) % 8
    if posture == "lying":
        low = lying_angle_bin * math.pi / 8.0
        high = (lying_angle_bin + 1) * math.pi / 8.0
        cylinder_angle_rad = float(rng.uniform(low, high))
    else:
        cylinder_angle_rad = 0.0

    return {
        "schema_version": 1,
        "episode_seed": seed,
        "workspace": {
            **asdict(spec),
            "maximum_radius_m": spec.maximum_radius_m,
        },
        "cylinder_initial_xy_m": cylinder_xy.tolist(),
        "socket_target_xy_m": socket_xy.tolist(),
        "cylinder_radial_bin": cylinder_radial_bin,
        "cylinder_angular_bin": cylinder_angular_bin,
        "socket_radial_bin": socket_radial_bin,
        "socket_angular_bin": socket_angular_bin,
        "object_separation_m": float(np.linalg.norm(socket_xy - cylinder_xy)),
        "cylinder_posture": posture,
        "lying_angle_bin": lying_angle_bin if posture == "lying" else None,
        "cylinder_angle_rad": cylinder_angle_rad,
    }
