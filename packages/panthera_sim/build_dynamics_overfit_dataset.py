#!/usr/bin/env python3
"""Replay one recorded episode and attach 28-D dynamics proprioception.

The source RGB, absolute actions and measured positions remain byte-for-byte
from the approved demonstration.  RoboTwin replays its 50 Hz drive targets in
the recorded scene to recover velocity, acceleration and transmitted joint
load, which older HDF5 files did not store.  A qpos parity gate refuses the
derived dataset when that replay no longer follows the recording closely.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

from dataset import DATASET_FILE, SCENES_FILE, SCENE_INFO_FILE, open_dataset
from executor import DenseExecutor, ExecutorConfig
from replay import grid_indices
from robotwin_env import build_task, inside, task_args


STATE_BLOCKS = ("qpos", "qvel", "qacc", "effort")
STATE_WIDTH = 28


def _copy_selected_hdf5(source: Path, target: Path, indices: np.ndarray) -> None:
    """Copy an episode while retaining only the global 50 Hz sample grid."""
    with h5py.File(source, "r") as old, h5py.File(target, "w") as new:
        for key, value in old.attrs.items():
            new.attrs[key] = value

        source_rows = int(old["timing/simulation_step_index"].shape[0])

        def copy(name: str, item) -> None:
            if not isinstance(item, h5py.Dataset):
                new.require_group(name)
                return
            parent, _, leaf = name.rpartition("/")
            group = new.require_group(parent) if parent else new
            values = item[...]
            if item.ndim > 0 and item.shape[0] == source_rows:
                values = values[indices]
            dataset = group.create_dataset(leaf, data=values)
            for attr, attr_value in item.attrs.items():
                dataset.attrs[attr] = attr_value

        old.visititems(copy)


def _write_manifest(source, target: Path, episode_id: int, rows: int) -> None:
    episode = source.episode(episode_id)
    manifest = json.loads(json.dumps(source.manifest))
    manifest["name"] = target.name
    manifest["contract"] = dict(manifest["contract"])
    manifest["contract"].update(
        {
            "proprio_dim": STATE_WIDTH,
            "proprio_order": [
                f"{block}/joint1..joint6,gripper" for block in STATE_BLOCKS
            ],
            "effort_semantics": (
                "PhysX incoming joint-frame load: torque-x for revolute arm "
                "joints and force-x for the prismatic gripper"
            ),
            "required_action_budget": rows,
        }
    )
    for record in manifest["episodes"]:
        if int(record["episode_id"]) == episode_id:
            record["recorded_actions"] = rows
    manifest["provenance"] = {
        "backfilled": False,
        "derived_at": datetime.now(timezone.utc).isoformat(),
        "derived_from_root": str(source.root),
        "derived_from_digest": source.digest(),
        "method": "recorded RGB/qpos/actions plus deterministic scene replay dynamics",
    }
    manifest["subset"] = {
        "source_root": str(source.root),
        "source_digest": source.digest(),
        "contract": f"episode {episode_id} with 28-D dynamics proprioception",
        "validate_on_train": True,
        "train": [episode_id],
        "val": [episode_id],
    }
    (target / DATASET_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    scenes = {str(episode.seed): dict(episode.scene)}
    (target / SCENES_FILE).write_text(
        json.dumps({"schema_version": 1, "scenes": scenes}, indent=2) + "\n",
        encoding="utf-8",
    )
    record = {f"episode_{episode_id}": {"panthera_episode": dict(
        source.episode_record(episode_id)
    )}}
    (target / SCENE_INFO_FILE).write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-name", default="place_randomized_cylinder_in_socket")
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--max-qpos-median-rad", type=float, default=0.02)
    parser.add_argument("--max-qpos-p99-rad", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    source = open_dataset(args.source_root.resolve())
    episode = source.episode(args.episode)
    target = args.target_root.resolve()
    if target.exists():
        raise SystemExit(f"target already exists: {target}")
    if args.max_qpos_median_rad <= 0 or args.max_qpos_p99_rad <= 0:
        raise SystemExit("qpos parity limits must be positive")

    selected = grid_indices(episode.hdf5_path)
    with h5py.File(episode.hdf5_path, "r") as handle:
        commands = np.asarray(handle["joint_action/vector"], dtype=np.float64)[selected]
        recorded_position = np.asarray(
            handle["observation/robot_state/vector"], dtype=np.float64
        )[selected]

    target.mkdir(parents=True)
    (target / "data").mkdir()
    (target / "instructions").mkdir()
    output_hdf5 = target / "data" / f"episode{args.episode}.hdf5"
    try:
        _copy_selected_hdf5(episode.hdf5_path, output_hdf5, selected)
        shutil.copy2(
            episode.instruction_path,
            target / "instructions" / f"episode{args.episode}.json",
        )

        root = args.robotwin_root.resolve()
        base = task_args(root, args.task_config, args.task_name)
        dynamics = []
        replay_position = []
        task = None
        with inside(root):
            try:
                task, _ = build_task(
                    root, args.task_config, args.task_name, episode,
                    source.scenes_path, len(commands) + 16, base,
                    camera=source.camera(args.episode),
                    camera_pose=source.camera_pose(args.episode),
                )
                executor = DenseExecutor(task, ExecutorConfig(final_settle=False))
                for command in commands:
                    executor.step(command)
                    state = task._actual_robot_state()
                    dynamics.append(
                        np.asarray(state["dynamics_vector"], dtype=np.float64)
                    )
                    replay_position.append(
                        np.asarray(state["vector"], dtype=np.float64)
                    )
            finally:
                if task is not None:
                    with contextlib.suppress(Exception):
                        task.close_env()

        dynamics_array = np.asarray(dynamics, dtype=np.float64)
        replay_array = np.asarray(replay_position, dtype=np.float64)
        if dynamics_array.shape != (len(commands), STATE_WIDTH):
            raise RuntimeError(f"unexpected dynamics shape {dynamics_array.shape}")
        if not np.all(np.isfinite(dynamics_array)):
            raise RuntimeError("replayed dynamics contain NaN or infinity")

        error = np.abs(replay_array[:, :6] - recorded_position[:, :6])
        median = float(np.median(error))
        p99 = float(np.quantile(error, 0.99))
        if median > args.max_qpos_median_rad or p99 > args.max_qpos_p99_rad:
            raise RuntimeError(
                "replay does not track the recorded qpos closely enough: "
                f"median={median:.6f} rad, p99={p99:.6f} rad"
            )

        # Preserve the position component that belongs to the recorded image;
        # only the three previously unavailable dynamics blocks come from replay.
        dynamics_array[:, :7] = recorded_position
        with h5py.File(output_hdf5, "a") as handle:
            state = handle["observation/robot_state"]
            state.create_dataset("dynamics_vector", data=dynamics_array)
            for offset, block in enumerate(STATE_BLOCKS):
                values = dynamics_array[:, offset * 7 : (offset + 1) * 7]
                state.create_dataset(block, data=values)

        _write_manifest(source, target, args.episode, len(commands))
        audit = {
            "schema_version": 1,
            "episode": args.episode,
            "rows": len(commands),
            "proprio_dim": STATE_WIDTH,
            "qpos_replay_error_rad": {"median": median, "p99": p99},
            "block_abs_max": {
                block: float(np.max(np.abs(
                    dynamics_array[:, index * 7 : (index + 1) * 7]
                )))
                for index, block in enumerate(STATE_BLOCKS)
            },
            "source_digest": source.digest(),
        }
        (target / "dynamics-audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        open_dataset(target)
        print(json.dumps(audit, indent=2))
        return 0
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
