#!/usr/bin/env python3
"""Replay recorded expert actions through the evaluation executor.

This is the ceiling every policy number is measured against: an episode the
expert cannot reproduce is one where a perfect policy would also be scored a
failure.  It reads the dataset's own snapshot -- its scenes, its contract, its
action budget -- so a run cannot quietly disagree with the data it is checking.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from dataset import open_dataset
from executor import SAMPLE_PERIOD_STEPS, DenseExecutor, ExecutorConfig
from robotwin_env import SceneMismatch, build_task, inside, task_args

DEFAULT_TASK = "place_randomized_cylinder_in_socket"


def grid_indices(hdf5_path: Path) -> np.ndarray:
    """Return the HDF5 rows retained by the global 50 Hz sample grid."""
    import h5py

    with h5py.File(hdf5_path, "r") as data:
        steps = np.asarray(data["timing/simulation_step_index"], dtype=np.int64)
    selected: dict[int, int] = {}
    for index, step in enumerate(steps.tolist()):
        if step % SAMPLE_PERIOD_STEPS == 0:
            selected[int(step)] = index
    return np.asarray([selected[step] for step in sorted(selected)], dtype=np.int64)


def grid_actions(hdf5_path: Path, alignment: str = "next") -> np.ndarray:
    """The 50 Hz targets, under the alignment the policy is trained on.

    ``next`` labels the observation at sample i with the action at i+1, which is
    the training convention and also leads the position drive by one grid period.
    ``current`` is what the oracle had in the drive at sample i; replaying under
    it scores 37/40 against 39/40, because the lead happens to cancel the
    controller's lag.
    """
    import h5py

    if alignment not in {"next", "current"}:
        raise ValueError("alignment must be 'next' or 'current'")
    with h5py.File(hdf5_path, "r") as data:
        actions = np.asarray(data["joint_action/vector"], dtype=np.float64)
    index = grid_indices(hdf5_path)
    if alignment == "current":
        return actions[index]
    return actions[np.concatenate((index[1:], index[-1:]))]


_WORKER: dict = {}


def _init_worker(config: dict, devices) -> None:
    import os
    import sys

    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices.get()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    dataset = open_dataset(config["dataset_root"])
    _WORKER.update(
        {
            "config": config,
            "dataset": dataset,
            "base_args": task_args(
                config["robotwin_root"], config["task_config"], config["task_name"]
            ),
        }
    )


def _run(episode_id: int) -> dict:
    config = _WORKER["config"]
    dataset = _WORKER["dataset"]
    episode = dataset.episode(episode_id)
    targets = grid_actions(episode.hdf5_path, config["action_alignment"])
    case = {
        "episode": episode_id,
        "seed": episode.seed,
        "posture": episode.posture,
        "recorded_actions": int(len(targets)),
    }
    root = Path(config["robotwin_root"])
    with inside(root):
        task = None
        try:
            task, case["scene"] = build_task(
                root, config["task_config"], config["task_name"], episode,
                dataset.scenes_path, len(targets), _WORKER["base_args"],
                camera=dataset.camera(episode_id),
                camera_pose=dataset.camera_pose(episode_id),
            )
            executor = DenseExecutor(
                task,
                ExecutorConfig(**config["executor"]),
                trace=config["trace"],
            )
            for target in targets:
                executor.step(target)
                if executor.succeeded():
                    break
            case.update(executor.finish())
        except SceneMismatch as error:
            case.update({"success": False, "error": str(error)})
        except Exception as error:  # a failed episode must not stop the sweep
            case.update({"success": False, "error": f"{type(error).__name__}: {error}"})
        finally:
            if task is not None:
                with contextlib.suppress(Exception):
                    task.close_env()
    return case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--episode", type=int, action="append")
    parser.add_argument("--sample", type=int, default=0,
                        help="stratified sample size; 0 replays every episode")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-alignment", choices=("next", "current"), default="next")
    parser.add_argument("--velocity-mode", choices=("zero", "finite_difference"),
                        default="finite_difference")
    parser.add_argument("--velocity-gain", type=float, default=1.0)
    parser.add_argument("--speed-limit-radps", type=float, default=1000.0)
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--converge-tolerance-rad", type=float, default=0.0)
    parser.add_argument("--no-final-settle", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    dataset = open_dataset(cli.dataset_root)
    episodes = cli.episode or dataset.stratified_sample(cli.sample)
    config = {
        "dataset_root": str(Path(cli.dataset_root).resolve()),
        "robotwin_root": str(Path(cli.robotwin_root).resolve()),
        "task_config": cli.task_config,
        "task_name": cli.task_name,
        "action_alignment": cli.action_alignment,
        "trace": cli.trace,
        "executor": {
            "velocity_mode": cli.velocity_mode,
            "velocity_gain": cli.velocity_gain,
            "speed_limit_radps": cli.speed_limit_radps,
            "interpolate": cli.interpolate,
            "converge_tolerance_rad": cli.converge_tolerance_rad,
            "final_settle": not cli.no_final_settle,
        },
    }

    context = mp.get_context("spawn")
    devices = None
    if cli.gpus:
        ids = [value.strip() for value in cli.gpus.split(",") if value.strip()]
        devices = context.Queue()
        for index in range(cli.workers):
            devices.put(ids[index % len(ids)])

    cases: list[dict] = []
    with context.Pool(
        processes=min(cli.workers, len(episodes)),
        initializer=_init_worker,
        initargs=(config, devices),
    ) as pool:
        for case in pool.imap_unordered(_run, episodes):
            cases.append(case)
            print(
                f"  [{len(cases):4d}/{len(episodes)}] ep{case['episode']:5d} "
                f"{case['posture']:8s} success={case.get('success')} "
                f"{case.get('error', '')}",
                flush=True,
            )
    cases.sort(key=lambda case: case["episode"])

    by_posture = {
        posture: {
            "n": sum(1 for c in cases if c["posture"] == posture),
            "success": sum(1 for c in cases if c["posture"] == posture and c.get("success")),
        }
        for posture in sorted({c["posture"] for c in cases})
    }
    summary = {
        "dataset": dataset.name,
        "dataset_digest": dataset.digest(),
        "executor": config["executor"],
        "action_alignment": cli.action_alignment,
        "total": len(cases),
        "success": sum(1 for c in cases if c.get("success")),
        "by_posture": by_posture,
        "cases": cases,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"  专家自复现 {summary['success']}/{summary['total']}")
    for posture, value in by_posture.items():
        print(f"  {posture:8s} {value['success']}/{value['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
