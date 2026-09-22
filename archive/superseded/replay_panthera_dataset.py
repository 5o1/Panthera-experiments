#!/usr/bin/env python3
"""Replay recorded Panthera action chunks without a scripted object attachment.

This is a semantic gate between demonstration collection and VLA training.  It
uses RoboTwin's real evaluation action path (`gen_sparse_reward_data`) in a
fresh scene, so a passing result proves that the recorded 7-D or legacy 14-D targets can
complete the task without calling the scripted oracle or creating its
attach-on-grasp constraint.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image
import yaml


DEFAULT_TASK_NAME = "place_cylinder_in_groove"
SAMPLE_PERIOD_STEPS = 5
ACTION_CHUNK = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--diagnostic-dir", type=Path)
    return parser.parse_args()


def _task_args(robotwin_root: Path, config_name: str, task_name: str) -> dict:
    """Resolve one RoboTwin task config into the direct task setup contract."""
    if Path(config_name).name != config_name:
        raise ValueError("task config must be a filename")
    with (robotwin_root / "task_config" / config_name).open(
        "r", encoding="utf-8"
    ) as stream:
        args = yaml.safe_load(stream)
    with (robotwin_root / "task_config" / "_embodiment_config.yml").open(
        "r", encoding="utf-8"
    ) as stream:
        registry = yaml.safe_load(stream)

    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        robot_names = (embodiment[0], embodiment[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment[0])
    elif len(embodiment) == 3:
        robot_names = (embodiment[0], embodiment[1])
        args["embodiment_dis"] = float(embodiment[2])
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment[0]}+{embodiment[1]}"
    else:
        raise ValueError("embodiment must contain one robot or [left, right, separation]")
    for side, name in zip(("left", "right"), robot_names, strict=True):
        robot_root = (robotwin_root / registry[name]["file_path"]).resolve()
        args[f"{side}_robot_file"] = str(robot_root)
        with (robot_root / "config.yml").open("r", encoding="utf-8") as stream:
            args[f"{side}_embodiment_config"] = yaml.safe_load(stream)
    if not task_name.isidentifier():
        raise ValueError("task name must be a Python identifier")
    args.update(
        {
            "task_name": task_name,
            "need_plan": False,
            "render_freq": 0,
            "save_data": False,
            "eval_mode": True,
            "eval_video_log": False,
            "collect_data": False,
        }
    )
    return args


def _grid_actions(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one next-sample-aligned action sequence on the exact 50 Hz grid."""
    with h5py.File(hdf5_path, "r") as data:
        steps = np.asarray(data["timing/simulation_step_index"], dtype=np.int64)
        actions = np.asarray(data["joint_action/vector"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] not in (7, 14) or len(steps) != len(actions):
        raise ValueError(f"invalid action/timing contract: {hdf5_path}")
    if np.any(np.diff(steps) < 0) or not np.all(np.isfinite(actions)):
        raise ValueError(f"invalid action values or clock: {hdf5_path}")

    # Keep the last capture for a duplicate stage-boundary step, matching the
    # RLDS adapter.  The observation at one grid sample is trained against the
    # following target, so replay uses the same next-grid alignment.
    selected: dict[int, int] = {}
    for index, step in enumerate(steps.tolist()):
        if step % SAMPLE_PERIOD_STEPS == 0:
            selected[int(step)] = index
    ordered_steps = np.asarray(sorted(selected), dtype=np.int64)
    if len(ordered_steps) < ACTION_CHUNK + 1:
        raise ValueError(f"episode is too short: {hdf5_path}")
    if not np.all(np.diff(ordered_steps) == SAMPLE_PERIOD_STEPS):
        raise ValueError(f"episode grid is discontinuous: {hdf5_path}")
    indices = np.asarray([selected[int(step)] for step in ordered_steps], dtype=np.int64)
    next_indices = np.concatenate((indices[1:], indices[-1:]))
    return actions[next_indices], ordered_steps


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    cli = parse_args()
    robotwin_root = cli.robotwin_root.resolve()
    dataset_root = cli.dataset_root.resolve()
    os.environ["ASSETS_PATH"] = str(robotwin_root)
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(robotwin_root))
    os.chdir(robotwin_root)

    from sapien.render import clear_cache

    task_module = importlib.import_module(f"envs.{cli.task_name}")
    task_class = getattr(task_module, cli.task_name)

    scene_info = json.loads(
        (dataset_root / "scene_info.json").read_text(encoding="utf-8")
    )
    base_args = _task_args(robotwin_root, cli.task_config, cli.task_name)
    cases: list[dict] = []
    for episode_id in cli.episode:
        metadata = scene_info[f"episode_{episode_id}"]["panthera_episode"]
        episode_seed = int(metadata["episode_seed"])
        hdf5_path = dataset_root / "data" / f"episode{episode_id}.hdf5"
        actions, grid_steps = _grid_actions(hdf5_path)
        task = task_class()
        case: dict = {
            "episode_id": episode_id,
            "episode_seed": episode_seed,
            "grid_action_count": int(len(actions)),
            "grid_step_min": int(grid_steps[0]),
            "grid_step_max": int(grid_steps[-1]),
            "action_dimension": int(actions.shape[1]),
            "success": False,
            "oracle_attachment_created": False,
        }
        try:
            task_args = copy.deepcopy(base_args)
            task_args["step_lim"] = int(len(actions))
            task.setup_demo(
                now_ep_num=episode_seed,
                seed=episode_seed,
                **task_args,
            )
            initial_position = np.asarray(task.cylinder.get_pose().p, dtype=float)
            target_position = np.asarray(task.groove_target_pose.p, dtype=float)
            chunk_telemetry: list[dict] = []
            for start in range(0, len(actions), ACTION_CHUNK):
                chunk = actions[start : start + ACTION_CHUNK]
                reward, termination, truncation, info = task.gen_sparse_reward_data(
                    chunk, action_type="qpos"
                )
                cylinder_position = np.asarray(task.cylinder.get_pose().p, dtype=float)
                contacts = len(
                    task.get_gripper_actor_contact_position("panthera_cylinder")
                )
                chunk_index = start // ACTION_CHUNK
                chunk_telemetry.append(
                    {
                        "chunk_index": chunk_index,
                        "action_start": start,
                        "action_stop": start + len(chunk),
                        "cylinder_position_m": cylinder_position.tolist(),
                        "cylinder_displacement_m": float(
                            np.linalg.norm(cylinder_position - initial_position)
                        ),
                        "distance_to_target_m": float(
                            np.linalg.norm(cylinder_position - target_position)
                        ),
                        "gripper_contact_points": contacts,
                        "gripper": float(task.robot.get_left_gripper_val()),
                        "tcp": list(task.robot.get_left_tcp_pose()),
                    }
                )
                if cli.diagnostic_dir is not None:
                    frame = task.get_obs()["observation"]["head_camera"]["rgb"]
                    directory = cli.diagnostic_dir / f"episode-{episode_id}"
                    directory.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                        directory / f"chunk-{chunk_index:03d}.jpg",
                        quality=90,
                    )
                if bool(np.asarray(info.get("success", False)).any()):
                    break
                if bool(np.asarray(truncation).any()):
                    break
            case["oracle_attachment_created"] = getattr(task, "grasp_drive", None) is not None
            case["chunk_telemetry"] = chunk_telemetry
            case["max_gripper_contact_points"] = max(
                item["gripper_contact_points"] for item in chunk_telemetry
            )
            case["max_cylinder_displacement_m"] = max(
                item["cylinder_displacement_m"] for item in chunk_telemetry
            )
            case["min_distance_to_target_m"] = min(
                item["distance_to_target_m"] for item in chunk_telemetry
            )
            case["metrics"] = task.success_metrics()
            case["success"] = bool(task.check_success())
            case["take_action_count"] = int(task.take_action_cnt)
            case["reward"] = np.asarray(reward).tolist()
            case["termination"] = np.asarray(termination).tolist()
            case["truncation"] = np.asarray(truncation).tolist()
            if case["oracle_attachment_created"]:
                raise RuntimeError("replay unexpectedly created the oracle D6 attachment")
        except Exception as error:
            case["error"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                task.close_env(clear_cache=True)
            except Exception as error:
                case["cleanup_error"] = f"{type(error).__name__}: {error}"
            finally:
                del task
                gc.collect()
                clear_cache()
        cases.append(case)
        print(
            f"[replay] episode={episode_id} seed={episode_seed} "
            f"success={case['success']} actions={case['grid_action_count']}",
            flush=True,
        )

    passed = sum(case["success"] and not case.get("error") for case in cases)
    summary = {
        "status": "passed" if passed == len(cases) else "failed",
        "task": cli.task_name,
        "action_path": "Base_Task.gen_sparse_reward_data",
        "action_chunk": ACTION_CHUNK,
        "oracle_attachment_allowed": False,
        "passed": passed,
        "total": len(cases),
        "cases": cases,
    }
    _atomic_json(cli.summary, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
