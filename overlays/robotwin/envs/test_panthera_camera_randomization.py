#!/usr/bin/env python3
"""Acceptance test for deterministic constrained Panthera camera sampling."""

from __future__ import annotations

import numpy as np

from envs.panthera_camera_randomization import (
    camera_randomization_spec,
    project_world_points,
    sample_constrained_camera,
)


def expect_error(callback, expected: str, kind: type[Exception]) -> None:
    try:
        callback()
    except kind as error:
        assert expected in str(error), (expected, str(error))
        return
    raise AssertionError(f"expected {kind.__name__} containing {expected!r}")


def main() -> int:
    table_height = 0.74
    radius = 0.46 * 0.75
    names = []
    points = []
    for x_index, x_value in enumerate((-radius, radius)):
        for y_index, y_value in enumerate((-0.35, -0.35 + radius)):
            for z_index, z_value in enumerate((table_height, table_height + 0.46)):
                names.append(f"robot_task_envelope_{x_index}{y_index}{z_index}")
                points.append([x_value, y_value, z_value])
    names.extend(("cylinder_center", "socket_center"))
    points.extend(([-0.25, -0.10, 0.80], [0.25, -0.08, 0.80]))
    points_array = np.asarray(points, dtype=float)
    center = (np.min(points_array, axis=0) + np.max(points_array, axis=0)) / 2.0
    samples = [
        sample_constrained_camera(
            seed,
            table_height,
            center,
            names,
            points,
        )
        for seed in range(1280)
    ]
    assert samples == [
        sample_constrained_camera(
            seed,
            table_height,
            center,
            names,
            points,
        )
        for seed in range(1280)
    ]
    positions = set()
    for sample in samples:
        uv, depth = project_world_points(
            points,
            sample["position_xyz_m"],
            sample["forward_xyz"],
            sample["left_xyz"],
            sample["vertical_fov_deg"],
            sample["image_width_px"],
            sample["image_height_px"],
        )
        lower_u, lower_v, upper_u, upper_v = sample["central_region_uv"]
        assert np.all(depth > 0.10)
        assert np.all(uv[:, 0] >= lower_u)
        assert np.all(uv[:, 0] <= upper_u)
        assert np.all(uv[:, 1] >= lower_v)
        assert np.all(uv[:, 1] <= upper_v)
        assert sample["height_above_table_m"] >= 0.55
        assert sample["downward_angle_deg"] >= 28.0
        positions.add(tuple(np.round(sample["position_xyz_m"], 4)))
    azimuths = [sample["azimuth_deg"] for sample in samples]
    heights = [sample["height_above_table_m"] for sample in samples]
    assert len(positions) >= 1000
    assert max(azimuths) - min(azimuths) >= 100.0
    assert max(heights) - min(heights) >= 0.30
    print(
        "camera sampler acceptance passed: "
        f"1280 poses, azimuth {min(azimuths):.1f}..{max(azimuths):.1f} deg, "
        f"height {min(heights):.3f}..{max(heights):.3f} m"
    )
    fixed_pose_acceptance()
    return 0


def fixed_pose_acceptance() -> None:
    """Fixed mode must replay one world pose for every seed and object layout."""
    table_height = 0.74
    radius = 0.46 * 0.75
    names = []
    points = []
    for x_index, x_value in enumerate((-radius, radius)):
        for y_index, y_value in enumerate((-0.35, -0.35 + radius)):
            for z_index, z_value in enumerate((table_height, table_height + 0.46)):
                names.append(f"robot_task_envelope_{x_index}{y_index}{z_index}")
                points.append([x_value, y_value, z_value])
    names.extend(("cylinder_center", "socket_center"))
    points.extend(([-0.25, -0.10, 0.80], [0.25, -0.08, 0.80]))
    points_array = np.asarray(points, dtype=float)
    center = (np.min(points_array, axis=0) + np.max(points_array, axis=0)) / 2.0

    reference = sample_constrained_camera(0, table_height, center, names, points)
    config = {
        "pose_mode": "fixed",
        "fixed_azimuth_deg": reference["azimuth_deg"],
        "fixed_height_above_table_m": reference["height_above_table_m"],
        "fixed_horizontal_distance_m": reference["horizontal_distance_m"],
        "fixed_look_at_xyz_m": reference["look_at_xyz_m"],
    }
    samples = [
        sample_constrained_camera(seed, table_height, center, names, points, config)
        for seed in range(4)
    ]
    for sample in samples:
        assert sample["pose_mode"] == "fixed"
        assert sample["constraints"]["pose_mode"] == "fixed"
        assert sample["position_xyz_m"] == reference["position_xyz_m"]
        assert sample["forward_xyz"] == reference["forward_xyz"]
        assert sample["left_xyz"] == reference["left_xyz"]
        assert sample["azimuth_deg"] == reference["azimuth_deg"]
        assert sample["height_above_table_m"] == reference["height_above_table_m"]

    # Object layout must no longer move the camera.
    for offset in ((0.30, 0.20, 0.10), (-0.20, -0.15, -0.05)):
        moved = sample_constrained_camera(
            3, table_height, center + np.asarray(offset), names, points, config
        )
        assert moved["position_xyz_m"] == reference["position_xyz_m"]
        assert moved["look_at_xyz_m"] == reference["look_at_xyz_m"]

    # The visibility contract still applies to a fixed pose.
    expect_error(
        lambda: sample_constrained_camera(
            0, table_height, center, names, points, dict(config, fixed_look_at_xyz_m=[0.0, 0.0, 0.0])
        ),
        "leaves the central image region",
        RuntimeError,
    )

    # Configuration guard rails.
    expect_error(
        lambda: camera_randomization_spec(
            {"pose_mode": "fixed", "fixed_azimuth_deg": 90.0}
        ),
        "fixed camera pose requires",
        ValueError,
    )
    expect_error(
        lambda: camera_randomization_spec(
            {
                "pose_mode": "fixed",
                "fixed_azimuth_deg": 90.0,
                "fixed_height_above_table_m": 0.70,
                "fixed_horizontal_distance_m": 0.20,
                "fixed_look_at_xyz_m": [0.0, -0.09, 0.89],
            }
        ),
        "must lie inside the configured range",
        ValueError,
    )
    expect_error(
        lambda: camera_randomization_spec(
            {"pose_mode": "randomized", "fixed_azimuth_deg": 90.0}
        ),
        "fixed pose values are set",
        ValueError,
    )
    print(
        "camera fixed-pose acceptance passed: "
        "one shared world pose across seeds and object layouts"
    )


if __name__ == "__main__":
    raise SystemExit(main())
