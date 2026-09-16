#!/usr/bin/env python3
"""Run a bounded multi-seed scripted-oracle acceptance without wall-clock sleeps."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
import yaml


DEFAULT_TASK_NAME = "place_cylinder_in_groove"
ROBOTWIN_COMMIT = "0008ae6800df9f75fc8de7098bacb01735fd8fd2"


def parse_args() -> argparse.Namespace:
    """Parse fixed-scope smoke inputs; all seeds run in one invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path(os.environ.get("ROBOTWIN_PATH", "/data/lyy/panthera-vla/RoboTwin")),
    )
    parser.add_argument(
        "--task-name",
        default=DEFAULT_TASK_NAME,
        help="RoboTwin task module and class name",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/lyy/panthera-vla/results/cylinder-oracle-smoke"),
    )
    parser.add_argument(
        "--task-config",
        default="panthera_cylinder_oracle.yml",
        help="RoboTwin task_config filename to load",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    return parser.parse_args()


def _load_task_args(robotwin_root: Path, task_config: str, task_name: str) -> dict:
    """Resolve the same embodiment arguments used by RoboTwin collect_data.py."""
    if Path(task_config).name != task_config:
        raise ValueError("task config must be a filename, not a path")
    config_path = robotwin_root / "task_config" / task_config
    with config_path.open("r", encoding="utf-8") as stream:
        task_args = yaml.safe_load(stream)

    embodiment = task_args["embodiment"]
    with (robotwin_root / "task_config" / "_embodiment_config.yml").open(
        "r", encoding="utf-8"
    ) as stream:
        embodiment_registry = yaml.safe_load(stream)

    def robot_file(name: str) -> str:
        relative_path = embodiment_registry[name]["file_path"]
        return str((robotwin_root / relative_path).resolve())

    if not task_name.isidentifier():
        raise ValueError("task name must be a Python identifier")
    task_args["task_name"] = task_name
    if len(embodiment) == 1:
        task_args["left_robot_file"] = robot_file(embodiment[0])
        task_args["right_robot_file"] = robot_file(embodiment[0])
        task_args["dual_arm_embodied"] = True
        task_args["embodiment_name"] = str(embodiment[0])
    elif len(embodiment) == 3:
        task_args["left_robot_file"] = robot_file(embodiment[0])
        task_args["right_robot_file"] = robot_file(embodiment[1])
        task_args["embodiment_dis"] = float(embodiment[2])
        task_args["dual_arm_embodied"] = False
        task_args["embodiment_name"] = f"{embodiment[0]}+{embodiment[1]}"
    else:
        raise ValueError("embodiment must contain one robot or [left, right, separation]")

    for side in ("left", "right"):
        robot_root = Path(task_args[f"{side}_robot_file"])
        with (robot_root / "config.yml").open("r", encoding="utf-8") as stream:
            task_args[f"{side}_embodiment_config"] = yaml.safe_load(stream)

    task_args.update(
        {
            "need_plan": True,
            "render_freq": 0,
            "save_data": False,
            "eval_mode": False,
        }
    )
    return task_args


def _label_frame(stage: str, frame: np.ndarray) -> Image.Image:
    """Add a compact stage header without changing the captured RGB pixels."""
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    result = Image.new("RGB", (image.width, image.height + 24), "white")
    result.paste(image, (0, 24))
    ImageDraw.Draw(result).text((6, 5), stage, fill="black")
    return result


def _save_frames(seed_root: Path, frames: list[tuple[str, np.ndarray]]) -> list[str]:
    """Save all milestone frames plus one horizontal contact sheet."""
    seed_root.mkdir(parents=True, exist_ok=True)
    labelled_frames: list[Image.Image] = []
    stage_names: list[str] = []
    for index, (stage, frame) in enumerate(frames):
        labelled = _label_frame(stage, frame)
        labelled.save(seed_root / f"{index:02d}-{stage}.png")
        labelled_frames.append(labelled)
        stage_names.append(stage)

    if labelled_frames:
        sheet = Image.new(
            "RGB",
            (labelled_frames[0].width * len(labelled_frames), labelled_frames[0].height),
            "white",
        )
        for index, frame in enumerate(labelled_frames):
            sheet.paste(frame, (index * frame.width, 0))
        sheet.save(seed_root / "contact-sheet.png")
    return stage_names


def _atomic_json(path: Path, payload: object):
    """Write JSON through a sibling temporary file so partial results are never accepted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_path, path)


def _plan_attempts(task) -> list[dict[str, object]]:
    """Summarize each paired planner result without serializing large trajectories."""
    left_results = getattr(task, "left_joint_path", [])
    right_results = getattr(task, "right_joint_path", [])
    attempts: list[dict[str, object]] = []
    for index in range(max(len(left_results), len(right_results))):
        attempt: dict[str, object] = {"index": index}
        for side, results in (("left", left_results), ("right", right_results)):
            if index >= len(results):
                attempt[side] = {"status": "missing"}
                continue
            result = results[index]
            if not isinstance(result, dict):
                attempt[side] = {"status": type(result).__name__}
                continue
            position = result.get("position")
            attempt[side] = {
                "status": str(result.get("status", "unknown")),
                "trajectory_steps": int(position.shape[0]) if position is not None else 0,
            }
        attempts.append(attempt)
    return attempts


def main() -> int:
    """Run every requested seed, always clean up, and fail unless all seeds pass."""
    cli = parse_args()
    robotwin_root = cli.robotwin_root.resolve()
    output_root = cli.output_root.resolve()
    os.environ["ASSETS_PATH"] = str(robotwin_root)
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(robotwin_root))
    os.chdir(robotwin_root)

    from sapien.render import clear_cache

    task_module = importlib.import_module(f"envs.{cli.task_name}")
    task_class = getattr(task_module, cli.task_name)

    task_args = _load_task_args(robotwin_root, cli.task_config, cli.task_name)
    output_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []

    for seed in cli.seeds:
        print(f"[oracle] seed={seed} start", flush=True)
        task = task_class()
        started = time.monotonic()
        case: dict = {"seed": seed, "success": False, "plan_success": False}
        try:
            task.setup_demo(now_ep_num=seed, seed=seed, **task_args)
            task.capture_oracle_stage("initial")
            task.play_once()
            case["plan_success"] = bool(task.plan_success)
            case["plan_attempts"] = _plan_attempts(task)
            case["telemetry"] = getattr(task, "oracle_telemetry", [])
            case["placement_debug"] = getattr(task, "oracle_placement_debug", None)
            case["metrics"] = task.success_metrics()
            case["continuous_motion_audit"] = getattr(
                task, "continuous_motion_audit", []
            )
            case["motion_settle_audit"] = getattr(task, "motion_settle_audit", [])
            case["joint_smoothing_audit"] = getattr(task, "joint_smoothing_audit", [])
            case["joint_retime_audit"] = getattr(task, "joint_retime_audit", [])
            case["grasp_route_selection_audit"] = getattr(
                task, "grasp_route_selection_audit", []
            )
            case["selected_grasp_axis_sign"] = getattr(
                task, "lying_grasp_physical_axis_sign", None
            )
            case["selected_reorientation_radial_mode"] = getattr(
                task, "reorientation_radial_mode", None
            )
            case["success"] = bool(task.plan_success and task.check_success())
            case["stages"] = _save_frames(output_root / f"seed-{seed}", task.oracle_frames)
        except Exception as error:  # preserve all evidence and continue the bounded seed set
            case["error"] = f"{type(error).__name__}: {error}"
            case["traceback"] = traceback.format_exc()
            case["plan_attempts"] = _plan_attempts(task)
            case["telemetry"] = getattr(task, "oracle_telemetry", [])
            case["placement_debug"] = getattr(task, "oracle_placement_debug", None)
            case["motion_settle_audit"] = getattr(task, "motion_settle_audit", [])
            case["joint_smoothing_audit"] = getattr(task, "joint_smoothing_audit", [])
            case["grasp_route_selection_audit"] = getattr(
                task, "grasp_route_selection_audit", []
            )
            if getattr(task, "oracle_frames", None):
                case["stages"] = _save_frames(
                    output_root / f"seed-{seed}", task.oracle_frames
                )
        finally:
            case["elapsed_seconds"] = round(time.monotonic() - started, 3)
            cases.append(case)
            try:
                task.close_env(clear_cache=True)
            except Exception:
                case["cleanup_error"] = traceback.format_exc()
            del task
            gc.collect()
            clear_cache()
        print(
            f"[oracle] seed={seed} success={case['success']} "
            f"plan_success={case['plan_success']}",
            flush=True,
        )

    passed = sum(bool(case["success"]) for case in cases)
    summary = {
        "task": cli.task_name,
        "task_config": cli.task_config,
        "embodiment": task_args["embodiment_name"],
        "robotwin_commit": ROBOTWIN_COMMIT,
        "seeds": cli.seeds,
        "passed": passed,
        "total": len(cases),
        "all_passed": passed == len(cases),
        "cases": cases,
    }
    _atomic_json(output_root / "summary.json", summary)
    print(f"[oracle] passed={passed}/{len(cases)} output={output_root}", flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
