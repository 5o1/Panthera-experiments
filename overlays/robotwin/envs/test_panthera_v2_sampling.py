#!/usr/bin/env python3
"""Dependency-light acceptance tests for the v2 scene sampler."""

from __future__ import annotations

import math

import numpy as np

from envs.panthera_v2_sampling import sample_scene


def main() -> int:
    cases = [sample_scene(seed) for seed in range(72)]
    assert cases == [sample_scene(seed) for seed in range(72)]
    assert sum(case["cylinder_posture"] == "upright" for case in cases[:64]) == 32
    assert sum(case["cylinder_posture"] == "lying" for case in cases[:64]) == 32
    lying_bins = {
        index: sum(case["lying_angle_bin"] == index for case in cases[:64])
        for index in range(8)
    }
    assert set(lying_bins.values()) == {4}, lying_bins
    assert {
        (case["cylinder_radial_bin"], case["cylinder_angular_bin"])
        for case in cases[:36]
    } == {(radius, angle) for radius in range(3) for angle in range(12)}
    assert {
        (case["socket_radial_bin"], case["socket_angular_bin"])
        for case in cases[:36]
    } == {(radius, angle) for radius in range(3) for angle in range(12)}
    cylinder_x = [case["cylinder_initial_xy_m"][0] for case in cases[:36]]
    socket_x = [case["socket_target_xy_m"][0] for case in cases[:36]]
    assert sum(value < 0.0 for value in cylinder_x) == 18
    assert sum(value > 0.0 for value in cylinder_x) == 18
    assert sum(value < 0.0 for value in socket_x) >= 14
    assert sum(value > 0.0 for value in socket_x) >= 14
    for case in cases:
        workspace = case["workspace"]
        base = np.asarray(
            [workspace["robot_base_x_m"], workspace["robot_base_y_m"]]
        )
        for key in ("cylinder_initial_xy_m", "socket_target_xy_m"):
            radius = float(np.linalg.norm(np.asarray(case[key]) - base))
            assert workspace["minimum_radius_m"] <= radius <= workspace["maximum_radius_m"]
        assert math.isclose(workspace["reach_fraction"], 0.75)
        assert math.isclose(
            workspace["minimum_angle_deg"] + workspace["maximum_angle_deg"],
            180.0,
        )
        assert case["object_separation_m"] >= workspace["minimum_object_separation_m"]
    print("panthera v2 sampler acceptance passed: 72 deterministic scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
