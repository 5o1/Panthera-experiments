"""One-shot migration: give an already-collected dataset its snapshot.

Three 1280-episode datasets exist and cost days of machine time, so they are not
re-collected to gain a snapshot.  Everything the snapshot needs is already
present, just in the wrong shape: the task configuration sits in
``scene_info.json`` repeated once per episode, and the action budget can be
measured from the recorded trajectories.  This reads that and writes
``dataset.json`` and ``scenes.json``, marking the result ``backfilled`` so a
reader can tell it was reconstructed rather than written at collection time.

New collections write their snapshot directly and do not use this.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dataset import SCENE_KEYS, SCENES_FILE, DATASET_FILE, DatasetError, _validate_scene

# Fields of scene_info that genuinely vary per episode; everything else is the
# task configuration, repeated 1280 times.
EPISODE_LEVEL = {
    "episode_seed", "collection_shard", "shard_episode_id", "cylinder_posture",
    "realized_geometry", "continuous_motion_audit", "motion_settle_audit",
    "joint_smoothing_audit", "joint_retime_audit", "grasp_route_selection_audit",
    "camera_randomization", "task_randomization", "initial_cylinder_axis_world",
    "cylinder_axis_world", "grasp_quaternion_wxyz", "grasp_approach_axis_world",
    "finger_closing_axis_world", "lying_grasp_axis_offset_m",
    "lying_grasp_physical_axis_offset_m", "final_settle_simulation_steps",
}

# scene_info records physics under these names; the declaration uses shorter ones.
PHYSICS_ALIASES = {
    "cylinder_linear_damping": "cylinder_linear_damping",
    "cylinder_angular_damping": "cylinder_angular_damping",
    "cylinder_solver_position_iterations": "solver_position_iterations",
    "cylinder_solver_velocity_iterations": "solver_velocity_iterations",
    "cylinder_max_depenetration_velocity_mps": "max_depenetration_velocity_mps",
}


def _grid_action_count(hdf5_path: Path, sample_period: int) -> int:
    import h5py
    import numpy as np

    with h5py.File(hdf5_path, "r") as data:
        steps = np.asarray(data["timing/simulation_step_index"], dtype=np.int64)
    return int(np.sum(steps % sample_period == 0))


def _assets(first: Mapping[str, Any]) -> dict:
    """Describe the assets from what the collection recorded, not from a guess.

    Only ``scene_profile`` is in the data, and it varies across the collections
    being backfilled -- the schema 5 pilot recorded
    ``panthera_phone_randomized_cylinder_socket_v2``, which predates the
    symmetric embodiment entirely. Naming that embodiment anyway would put a
    value that decides how the dataset is interpreted somewhere nobody would
    think to question it, which is the defect this whole structure exists to
    prevent. So the profile is claimed only when the recorded name carries it,
    and is null otherwise.
    """
    scene_profile = first.get("scene_profile")
    embodiment = None
    if scene_profile and "symmetric" in scene_profile:
        embodiment = "panthera_phone_symmetric"
    return {
        "scene_profile": scene_profile,
        "embodiment_profile": embodiment,
        "object": "panthera_cylinder",
        "note": "digests are recorded by new collections; a backfill cannot "
                "know which asset revision was used. embodiment_profile is "
                "null when scene_profile does not name one -- the collection "
                "recorded no embodiment, so none is asserted.",
    }


def build_manifest(root: Path, measure_budget: bool = True) -> tuple[dict, dict]:
    """Reconstruct the dataset snapshot and the scene table from scene_info."""
    from config import PARAMETERS, resolve

    root = Path(root)
    scene_info = json.loads((root / "scene_info.json").read_text(encoding="utf-8"))
    keys = sorted(k for k in scene_info if k.startswith("episode_"))
    if not keys:
        raise DatasetError(f"{root} has no episodes")

    first = scene_info[keys[0]]["panthera_episode"]
    sample_period = int(first["sample_period_physics_steps"])
    physics_timestep = float(first["physics_timestep_s"])

    scenes: dict[str, dict] = {}
    episodes: list[dict] = []
    budgets: list[int] = []
    seeds_seen: set[int] = set()
    for key in keys:
        episode_id = int(key.split("_", 1)[1])
        episode = scene_info[key]["panthera_episode"]
        seed = int(episode["episode_seed"])
        if seed in seeds_seen:
            raise DatasetError(f"seed {seed} is claimed by more than one episode")
        seeds_seen.add(seed)
        geometry = {k: episode["realized_geometry"][k] for k in SCENE_KEYS}
        _validate_scene(geometry, f"episode {episode_id}")
        scenes[str(seed)] = geometry
        posture = str(episode["cylinder_posture"])
        entry = {"episode_id": episode_id, "seed": seed, "posture": posture}
        if measure_budget:
            count = _grid_action_count(root / "data" / f"episode{episode_id}.hdf5",
                                       sample_period)
            budgets.append(count)
            entry["recorded_actions"] = count
        episodes.append(entry)

    # The environment variables are not recorded anywhere, so the physics that is
    # recorded is cross-checked against the declaration and anything it does not
    # cover is left at its default and marked as such.
    recorded_physics = first.get("physics_parameters", {})
    live = resolve({})
    values = dict(live.values)
    sources = {name: "default" for name in values}
    for recorded_name, declared_name in PHYSICS_ALIASES.items():
        if recorded_name in recorded_physics:
            values[declared_name] = recorded_physics[recorded_name]
            sources[declared_name] = "dataset"
    for name in ("lying_grasp_axis_offset_m", "direct_release_bottom_clearance_m"):
        for entry_key in keys:
            candidate = scene_info[entry_key]["panthera_episode"].get(name)
            if candidate:
                values[name] = candidate
                sources[name] = "dataset"
                break

    task_level = {
        name: value for name, value in first.items() if name not in EPISODE_LEVEL
    }
    manifest = {
        "schema_version": 1,
        "name": root.name,
        "contract": {
            "action_dim": int(first["action_dimension"]),
            "action_chunk": 25,
            "sample_period_physics_steps": sample_period,
            "physics_timestep_s": physics_timestep,
            "required_action_budget": max(budgets) if budgets else 0,
        },
        "task": {
            "name": root.parent.name,
            "schema_version": first.get("schema_version"),
            "scene_profile": first.get("scene_profile"),
            "config": task_level,
        },
        "sim": {
            "schema_version": 1,
            "geometry": live.geometry,
            "values": values,
            "sources": sources,
        },
        "assets": _assets(first),
        "upstream": {
            "robot_source_commit": first.get("robot_source_commit"),
            "robotwin_source_commit": first.get("robotwin_source_commit"),
        },
        "episodes": episodes,
        "posture_counts": dict(Counter(e["posture"] for e in episodes)),
        "provenance": {
            "backfilled": True,
            "backfilled_at": datetime.now(timezone.utc).isoformat(),
            "reconstructed_from": "scene_info.json",
            "caveat": "environment-variable overrides were not recorded at "
                      "collection; anything scene_info does not carry is the "
                      "declared default and is marked source=default",
        },
    }
    scene_table = {"schema_version": 1, "scenes": scenes}
    return manifest, scene_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--no-measure-budget", action="store_true",
                        help="skip reading every HDF5 to measure the action budget")
    parser.add_argument("--force", action="store_true")
    cli = parser.parse_args()

    root = cli.dataset_root.resolve()
    target = root / DATASET_FILE
    if target.exists() and not cli.force:
        raise SystemExit(f"{target} already exists; pass --force to rewrite")

    manifest, scenes = build_manifest(root, measure_budget=not cli.no_measure_budget)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    (root / SCENES_FILE).write_text(
        json.dumps(scenes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"写入 {target}")
    print(f"  episode {len(manifest['episodes'])}  姿态 {manifest['posture_counts']}")
    print(f"  动作预算需求 {manifest['contract']['required_action_budget']}")
    print(f"写入 {root / SCENES_FILE}：{len(scenes['scenes'])} 个场景")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
