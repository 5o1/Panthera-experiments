"""Address a task scene by data instead of by seed.

The collector overrode the seed's own posture and lying angle per shard, and
wrote only the outcome into ``scene_info.json``.  A seed therefore does not
identify a scene: rebuilding a dataset episode from its seed alone reproduced a
different posture for 57 of 112 episodes in the fixed-camera training subset,
and a lying angle a pi/8 sector away for the rest.  Every consumer that wanted a
recorded scene back had to reconstruct the forcing flags itself and verify the
result by hand.

A registry removes the reconstruction step: it maps a seed straight to the
``realized_geometry`` the episode was recorded with, which is exactly what
``_sample_task_scene`` returns, so the environment can hand it through
unchanged.  Sampling stays the default; a registry only pins the seeds it lists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

# The keys ``load_actors`` and the grasp routes read off ``realized_geometry``.
REQUIRED_KEYS = (
    "cylinder_initial_xy_m",
    "socket_target_xy_m",
    "groove_target_xy_m",
    "cylinder_center_z_m",
    "cylinder_quaternion_wxyz",
    "cylinder_axis_world",
    "cylinder_posture",
    "cylinder_angle_rad",
    "workspace",
)
POSTURES = ("upright", "lying")
ORIENTATION_TOLERANCE_DEG = 1e-3
POSITION_TOLERANCE_M = 1e-9


def _as_vector(value, length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (length,):
        raise ValueError(f"{name} must have {length} values, got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def validate_scene(
    scene: Mapping,
    upright_center_z_m: float | None = None,
    lying_center_z_m: float | None = None,
) -> None:
    """Reject a scene that cannot be the one the episode was recorded with.

    The checks are the ones that caught real mistakes: a posture that disagrees
    with the angle, and a quaternion or axis that disagrees with either.  Height
    is only checked when the caller supplies the task's constants, because the
    registry itself carries no geometry.
    """
    missing = [key for key in REQUIRED_KEYS if key not in scene]
    if missing:
        raise ValueError(f"scene is missing {', '.join(missing)}")

    posture = str(scene["cylinder_posture"])
    if posture not in POSTURES:
        raise ValueError(f"cylinder_posture must be one of {POSTURES}, got {posture}")

    angle = float(scene["cylinder_angle_rad"])
    if not math.isfinite(angle):
        raise ValueError("cylinder_angle_rad must be finite")
    if posture == "upright" and angle != 0.0:
        raise ValueError("an upright cylinder must record a zero angle")

    _as_vector(scene["cylinder_initial_xy_m"], 2, "cylinder_initial_xy_m")
    socket = _as_vector(scene["socket_target_xy_m"], 2, "socket_target_xy_m")
    groove = _as_vector(scene["groove_target_xy_m"], 2, "groove_target_xy_m")
    if float(np.abs(socket - groove).max()) > POSITION_TOLERANCE_M:
        raise ValueError("groove_target_xy_m must mirror socket_target_xy_m")

    quaternion = _as_vector(scene["cylinder_quaternion_wxyz"], 4, "cylinder_quaternion_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(f"cylinder_quaternion_wxyz must be unit, norm is {norm}")
    axis = _as_vector(scene["cylinder_axis_world"], 3, "cylinder_axis_world")

    if posture == "upright":
        expected_quaternion = np.array([math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0])
        expected_axis = np.array([0.0, 0.0, 1.0])
    else:
        expected_quaternion = np.array(
            [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]
        )
        expected_axis = np.array([math.cos(angle), math.sin(angle), 0.0])
    alignment = abs(float(np.dot(expected_quaternion, quaternion)))
    error_deg = math.degrees(2.0 * math.acos(min(max(alignment, -1.0), 1.0)))
    if error_deg > ORIENTATION_TOLERANCE_DEG:
        raise ValueError(
            f"cylinder_quaternion_wxyz disagrees with cylinder_angle_rad by "
            f"{error_deg:.6f} deg"
        )
    if float(np.abs(axis - expected_axis).max()) > 1e-6:
        raise ValueError("cylinder_axis_world disagrees with cylinder_angle_rad")

    if posture == "upright" and upright_center_z_m is not None:
        expected_z = upright_center_z_m
    elif posture == "lying" and lying_center_z_m is not None:
        expected_z = lying_center_z_m
    else:
        expected_z = None
    if expected_z is not None:
        actual_z = float(scene["cylinder_center_z_m"])
        if abs(actual_z - expected_z) > 1e-9:
            raise ValueError(
                f"cylinder_center_z_m is {actual_z} but a {posture} cylinder "
                f"spawns at {expected_z}"
            )

    workspace = scene["workspace"]
    if not isinstance(workspace, Mapping) or not workspace:
        raise ValueError("workspace must be a non-empty mapping")


def registry_from_scene_info(
    scene_info: Mapping, episodes: Iterable[int] | None = None
) -> dict[int, dict]:
    """Map each episode's seed to the geometry it was recorded with."""
    wanted = None if episodes is None else {int(value) for value in episodes}
    registry: dict[int, dict] = {}
    for key, value in scene_info.items():
        if not key.startswith("episode_"):
            continue
        episode_id = int(key.split("_", 1)[1])
        if wanted is not None and episode_id not in wanted:
            continue
        episode = value.get("panthera_episode")
        if episode is None:
            raise ValueError(f"{key} has no panthera_episode block")
        seed = int(episode["episode_seed"])
        if seed in registry:
            raise ValueError(f"seed {seed} is claimed by more than one episode")
        scene = dict(episode["realized_geometry"])
        validate_scene(scene)
        registry[seed] = scene
    if not registry:
        raise ValueError("scene_info produced an empty registry")
    return registry


def load_registry(path: str | Path) -> dict[int, dict]:
    """Read a registry file written by :func:`write_registry`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scenes = payload.get("scenes", payload)
    registry: dict[int, dict] = {}
    for seed, scene in scenes.items():
        validate_scene(scene)
        registry[int(seed)] = dict(scene)
    if not registry:
        raise ValueError(f"{path} contains no scenes")
    return registry


def write_registry(
    registry: Mapping[int, Mapping], path: str | Path, source: str | None = None
) -> None:
    """Write a registry, keyed by seed as a string so JSON round-trips."""
    payload = {
        "schema_version": 1,
        "source": source,
        "scenes": {str(seed): scene for seed, scene in sorted(registry.items())},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episode", type=int, action="append",
        help="restrict to these episode ids; omit for every episode",
    )
    cli = parser.parse_args()

    scene_info = json.loads(cli.scene_info.read_text(encoding="utf-8"))
    registry = registry_from_scene_info(scene_info, cli.episode)
    write_registry(registry, cli.output, source=str(cli.scene_info))
    postures: dict[str, int] = {}
    for scene in registry.values():
        posture = str(scene["cylinder_posture"])
        postures[posture] = postures.get(posture, 0) + 1
    print(f"写入 {cli.output}：{len(registry)} 个场景 {postures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
