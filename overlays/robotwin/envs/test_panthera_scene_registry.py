"""Checks for the seed-to-scene registry.

The bug this guards against is silent: a scene rebuilt from a seed alone looks
perfectly valid, it is simply a different scene, and the run still reports a
success rate.  So the tests here are about rejecting scenes that cannot be the
recorded one, and about the seed mapping surviving a round trip.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from panthera_scene_registry import (
    load_registry,
    registry_from_scene_info,
    validate_scene,
    write_registry,
)

TABLE_HEIGHT_M = 0.74
CYLINDER_RADIUS_M = 0.0275
CYLINDER_HALF_HEIGHT_M = 0.06
WORKSPACE = {"robot_base_x_m": 0.0, "robot_base_y_m": -0.35}


def upright_scene() -> dict:
    return {
        "cylinder_initial_xy_m": [0.12, -0.07],
        "socket_target_xy_m": [-0.02, -0.10],
        "groove_target_xy_m": [-0.02, -0.10],
        "cylinder_center_z_m": TABLE_HEIGHT_M + CYLINDER_HALF_HEIGHT_M + 0.002,
        "cylinder_quaternion_wxyz": [math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0],
        "cylinder_axis_world": [0.0, 0.0, 1.0],
        "cylinder_posture": "upright",
        "cylinder_angle_rad": 0.0,
        "workspace": dict(WORKSPACE),
    }


def lying_scene(angle: float = 0.26497400563875556) -> dict:
    return {
        "cylinder_initial_xy_m": [-0.29, -0.30],
        "socket_target_xy_m": [-0.018, -0.100],
        "groove_target_xy_m": [-0.018, -0.100],
        "cylinder_center_z_m": TABLE_HEIGHT_M + CYLINDER_RADIUS_M + 0.002,
        "cylinder_quaternion_wxyz": [
            math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)
        ],
        "cylinder_axis_world": [math.cos(angle), math.sin(angle), 0.0],
        "cylinder_posture": "lying",
        "cylinder_angle_rad": angle,
        "workspace": dict(WORKSPACE),
    }


def heights() -> dict:
    return {
        "upright_center_z_m": TABLE_HEIGHT_M + CYLINDER_HALF_HEIGHT_M + 0.002,
        "lying_center_z_m": TABLE_HEIGHT_M + CYLINDER_RADIUS_M + 0.002,
    }


def test_recorded_scenes_validate():
    validate_scene(upright_scene(), **heights())
    validate_scene(lying_scene(), **heights())


def test_missing_key_is_named():
    scene = upright_scene()
    del scene["cylinder_axis_world"]
    with pytest.raises(ValueError, match="cylinder_axis_world"):
        validate_scene(scene)


@pytest.mark.parametrize("posture", ["standing", "", "Upright"])
def test_unknown_posture_rejected(posture):
    scene = upright_scene()
    scene["cylinder_posture"] = posture
    with pytest.raises(ValueError, match="cylinder_posture"):
        validate_scene(scene)


def test_upright_with_nonzero_angle_rejected():
    scene = upright_scene()
    scene["cylinder_angle_rad"] = 0.1
    with pytest.raises(ValueError, match="zero angle"):
        validate_scene(scene)


def test_quaternion_from_the_wrong_sector_rejected():
    """The exact failure seen in practice: right position, 56 deg of yaw off."""
    scene = lying_scene()
    wrong = 1.2478
    scene["cylinder_quaternion_wxyz"] = [
        math.cos(wrong / 2.0), 0.0, 0.0, math.sin(wrong / 2.0)
    ]
    with pytest.raises(ValueError, match="disagrees with cylinder_angle_rad"):
        validate_scene(scene)


def test_axis_disagreeing_with_angle_rejected():
    scene = lying_scene()
    scene["cylinder_axis_world"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="cylinder_axis_world"):
        validate_scene(scene)


def test_non_unit_quaternion_rejected():
    scene = upright_scene()
    scene["cylinder_quaternion_wxyz"] = [1.0, 0.0, -1.0, 0.0]
    with pytest.raises(ValueError, match="unit"):
        validate_scene(scene)


def test_groove_must_mirror_socket():
    scene = upright_scene()
    scene["groove_target_xy_m"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="groove_target_xy_m"):
        validate_scene(scene)


def test_height_checked_only_when_constants_given():
    scene = lying_scene()
    scene["cylinder_center_z_m"] = TABLE_HEIGHT_M + CYLINDER_HALF_HEIGHT_M + 0.002
    validate_scene(scene)
    with pytest.raises(ValueError, match="cylinder_center_z_m"):
        validate_scene(scene, **heights())


def test_registry_from_scene_info_keys_by_seed():
    scene_info = {
        "episode_0": {"panthera_episode": {
            "episode_seed": 10000002, "realized_geometry": upright_scene()}},
        "episode_700": {"panthera_episode": {
            "episode_seed": 10870007, "realized_geometry": lying_scene()}},
        "dataset_note": {"unrelated": True},
    }
    registry = registry_from_scene_info(scene_info)
    assert sorted(registry) == [10000002, 10870007]
    assert registry[10870007]["cylinder_posture"] == "lying"


def test_registry_rejects_duplicate_seeds():
    scene_info = {
        "episode_0": {"panthera_episode": {
            "episode_seed": 7, "realized_geometry": upright_scene()}},
        "episode_1": {"panthera_episode": {
            "episode_seed": 7, "realized_geometry": lying_scene()}},
    }
    with pytest.raises(ValueError, match="more than one episode"):
        registry_from_scene_info(scene_info)


def test_registry_can_select_episodes():
    scene_info = {
        f"episode_{index}": {"panthera_episode": {
            "episode_seed": 100 + index, "realized_geometry": upright_scene()}}
        for index in range(4)
    }
    registry = registry_from_scene_info(scene_info, episodes=[1, 3])
    assert sorted(registry) == [101, 103]


def test_round_trip_preserves_seeds_and_geometry(tmp_path: Path):
    registry = {10000002: upright_scene(), 10870007: lying_scene()}
    path = tmp_path / "registry.json"
    write_registry(registry, path, source="unit test")
    restored = load_registry(path)
    assert restored == registry
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    # Seeds survive as integers even though JSON keys are strings.
    assert sorted(payload["scenes"]) == ["10000002", "10870007"]


def test_load_rejects_a_corrupted_scene(tmp_path: Path):
    path = tmp_path / "registry.json"
    scene = lying_scene()
    scene["cylinder_posture"] = "upright"
    path.write_text(json.dumps({"scenes": {"5": scene}}), encoding="utf-8")
    with pytest.raises(ValueError, match="zero angle"):
        load_registry(path)


def test_empty_registry_rejected(tmp_path: Path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"scenes": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no scenes"):
        load_registry(path)
