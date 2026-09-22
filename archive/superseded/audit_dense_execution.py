#!/usr/bin/env python3
"""Replay recorded 50 Hz actions without TOPP, to qualify the dense executor.

``gen_sparse_reward_data`` re-times every action chunk with TOPP, which discards
the timing already implied by the 50 Hz grid and re-derives it from path
geometry.  Measured on this dataset, adding jitter of the same magnitude as the
expert's own per-step motion inflates a 20-step chunk from ~153 to ~762 physics
steps, so a policy's nominal 0.4 s of action takes seconds to execute.

The dense path -- ``set_arm_joints`` plus ``_step_scene`` once per physics step,
which is what the oracle itself used to record this data -- has no such
behaviour.  Before building a policy adapter on it, the expert's own actions must
be shown to still succeed when replayed through it.

Velocity is the one free choice, since the dataset stores positions only:
``zero`` commands no feedforward, ``finite_difference`` derives it from the
target sequence.  Both are reported so the controller's sensitivity is visible.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

SAMPLE_PERIOD_STEPS = 5
PHYSICS_DT = 1.0 / 250.0
# The oracle holds the object at 0.3 and pregrasps at 0.9, so anything below the
# midpoint is unambiguously a closed gripper.
GRIPPER_CLOSED_OPENING = 0.5
DEFAULT_TASK_NAME = "place_randomized_cylinder_in_socket"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument(
        "--scene-registry",
        type=Path,
        required=True,
        help="seed-to-geometry registry built by panthera_scene_registry.py; "
             "a seed alone does not identify a recorded scene",
    )
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--velocity-mode",
        choices=("zero", "finite_difference", "both"),
        default="both",
    )
    parser.add_argument("--gripper-ratchet", action="store_true",
                        help="never let the gripper drift open while holding")
    parser.add_argument("--ratchet-margin", type=float, default=0.15,
                        help="opening above the held value that counts as intent")
    parser.add_argument("--ratchet-hold-steps", type=int, default=3,
                        help="consecutive intent frames required before releasing")
    parser.add_argument("--trace-object", action="store_true",
                        help="record the cylinder pose at every grid step")
    parser.add_argument(
        "--velocity-gain",
        type=float,
        default=1.0,
        help="scale on the velocity feedforward; a position drive lags in "
             "proportion to speed over stiffness, and this is the one executor "
             "knob that cancels it without altering the recorded timing",
    )
    parser.add_argument(
        "--converge-tolerance-rad",
        type=float,
        default=0.0,
        help="hold each target until every joint is within this of it, instead "
             "of advancing after a fixed grid period; 0 keeps the fixed period",
    )
    parser.add_argument(
        "--converge-max-steps",
        type=int,
        default=25,
        help="physics steps a single target may be held while converging",
    )
    parser.add_argument(
        "--action-alignment",
        choices=("next", "current"),
        default="next",
        help="which grid sample's action to command at each step",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="episodes to replay concurrently; each worker holds its own SAPIEN "
             "scene and costs roughly 2.3 GB of GPU memory, so size this against "
             "free memory rather than core count",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated GPU ids the workers are pinned to, round robin; "
             "defaults to whatever CUDA_VISIBLE_DEVICES already selects",
    )
    parser.add_argument(
        "--no-final-settle",
        action="store_true",
        help="judge the instant the action stream ends, without letting the "
             "cylinder come to rest as the oracle did",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="ramp linearly between grid targets instead of holding each one; "
             "the recorded 50 Hz samples are every fifth step of a 250 Hz TOPP "
             "command, so a first-order hold is the closer reconstruction",
    )
    parser.add_argument(
        "--speed-limit-radps",
        type=float,
        default=0.601,
        help="per-step clamp that replaces the limit guarantee TOPP provided",
    )
    return parser.parse_args()


def _task_args(robotwin_root: Path, config_name: str, task_name: str) -> dict:
    if Path(config_name).name != config_name:
        raise ValueError("task config must be a filename")
    with (robotwin_root / "task_config" / config_name).open("r", encoding="utf-8") as s:
        args = yaml.safe_load(s)
    with (robotwin_root / "task_config" / "_embodiment_config.yml").open(
        "r", encoding="utf-8"
    ) as s:
        registry = yaml.safe_load(s)
    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        robot_names = (embodiment[0], embodiment[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment[0])
    else:
        robot_names = (embodiment[0], embodiment[1])
        args["embodiment_dis"] = float(embodiment[2])
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment[0]}+{embodiment[1]}"
    for side, name in zip(("left", "right"), robot_names):
        robot_root = (robotwin_root / registry[name]["file_path"]).resolve()
        args[f"{side}_robot_file"] = str(robot_root)
        with (robot_root / "config.yml").open("r", encoding="utf-8") as s:
            args[f"{side}_embodiment_config"] = yaml.safe_load(s)
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


def _use_scene_registry(args: dict, registry_path: str) -> dict:
    """Rebuild the recorded scene from the registry rather than from the seed.

    Collection overrode the seed's posture and lying angle per shard and kept
    only the outcome, so a seed does not identify a scene.  Pointing the task at
    a registry replaces every caller's private reconstruction of those flags
    with one lookup, and the task validates the entry before placing anything.
    """
    randomization = args.setdefault("task_randomization", {})
    randomization["scene_registry"] = registry_path
    randomization["scene_registry_required"] = True
    # The forcing flags are collection-side sharding controls; leaving them set
    # would re-sample an angle inside the forced sector instead of using the
    # recorded one.
    randomization.pop("forced_posture", None)
    randomization.pop("forced_lying_angle_bin", None)
    return args


def _verify_scene(task, metadata: dict) -> dict:
    """Confirm the rebuilt scene is the one the actions were recorded against."""
    geometry = metadata["realized_geometry"]
    expected_quaternion = np.asarray(
        geometry["cylinder_quaternion_wxyz"], dtype=np.float64
    )
    pose = task.cylinder.get_pose()
    actual_quaternion = np.asarray(pose.q, dtype=np.float64)
    alignment = abs(float(np.dot(expected_quaternion, actual_quaternion)))
    orientation_error_deg = float(
        np.degrees(2.0 * np.arccos(np.clip(alignment, -1.0, 1.0)))
    )
    expected_xy = np.asarray(geometry["cylinder_initial_xy_m"], dtype=np.float64)
    position_error_m = float(
        np.linalg.norm(np.asarray(pose.p, dtype=np.float64)[:2] - expected_xy)
    )
    return {
        "orientation_error_deg": orientation_error_deg,
        "position_error_m": position_error_m,
        "matches_recording": bool(
            orientation_error_deg < 1.0 and position_error_m < 1e-3
        ),
    }


def _grid_actions(hdf5_path: Path, alignment: str = "next") -> np.ndarray:
    """Return the 50 Hz commands, under one of two alignment conventions.

    ``next`` is what the policy is trained on: the action at grid sample i+1 is
    the label for the observation at sample i.  ``current`` is what the oracle
    actually had in the drive at sample i.  The two differ by one grid period,
    so replaying under ``next`` runs the whole command stream 20 ms early.
    """
    if alignment not in {"next", "current"}:
        raise ValueError("alignment must be 'next' or 'current'")
    with h5py.File(hdf5_path, "r") as data:
        steps = np.asarray(data["timing/simulation_step_index"], dtype=np.int64)
        actions = np.asarray(data["joint_action/vector"], dtype=np.float64)
    selected: dict[int, int] = {}
    for index, step in enumerate(steps.tolist()):
        if step % SAMPLE_PERIOD_STEPS == 0:
            selected[int(step)] = index
    index = np.asarray([selected[s] for s in sorted(selected)], dtype=np.int64)
    if alignment == "current":
        return actions[index]
    following = np.concatenate((index[1:], index[-1:]))
    return actions[following]


class GripperRatchet:
    """Let the gripper close freely but never drift open while holding.

    A larger value is a wider opening.  Arm jitter is absorbed by the controller
    and the task tolerates centimetres, but a single frame of extra opening drops
    the object and nothing recovers from that -- the two dimensions do not
    deserve the same treatment.  Release must therefore be distinguishable from
    jitter, which it is: jitter is small and transient, an intended release is
    large and sustained.
    """

    def __init__(self, margin: float, hold_steps: int, contact_threshold: int = 2):
        self.margin = margin
        self.hold_steps = hold_steps
        self.contact_threshold = contact_threshold
        self.holding = False
        self.held_value: float | None = None
        self.release_votes = 0
        self.engaged_at: int | None = None
        self.suppressed = 0

    def __call__(self, commanded: float, contacts: int, row: int) -> float:
        if not self.holding:
            if contacts >= self.contact_threshold:
                self.holding = True
                self.held_value = commanded
                self.engaged_at = row
            return commanded
        if commanded > self.held_value + self.margin:
            self.release_votes += 1
            if self.release_votes >= self.hold_steps:
                self.holding = False
                return commanded
        else:
            self.release_votes = 0
        if commanded < self.held_value:
            self.held_value = commanded  # ratchet only tightens
        elif commanded > self.held_value:
            self.suppressed += 1
        return self.held_value


def _execute(
    task, targets: np.ndarray, velocity_mode: str, speed_limit: float,
    ratchet: GripperRatchet | None, trace_object: bool = False,
    interpolate: bool = False, settle: bool = True,
    converge_tolerance: float = 0.0, converge_max_steps: int = 25,
    velocity_gain: float = 1.0,
) -> dict:
    """Hold each 50 Hz target for one grid period, stepping physics directly."""
    clamped = 0
    max_lag = 0.0
    trace: list[dict] = []
    # ``play_once`` tightens the cylinder's damping and contact solving right
    # before it opens the gripper, and those values are only computed at setup --
    # never applied by replaying actions.  Without it the cylinder keeps the
    # loose defaults and rattles in the socket indefinitely, failing the two
    # at-rest criteria however long it is left to settle.
    release_configured = False
    gripper_was_closed = False
    release_row: int | None = None
    physics_steps = 0
    previous = np.asarray(task.robot.get_left_arm_jointState()[:6], dtype=np.float64)
    last_gripper = float(targets[0][6])
    for row, target in enumerate(targets):
        arm = np.asarray(target[:6], dtype=np.float64)
        span = arm - previous
        limit = speed_limit * SAMPLE_PERIOD_STEPS * PHYSICS_DT
        norm = float(np.abs(span).max())
        if norm > limit:
            arm = previous + span * (limit / norm)
            clamped += 1
        # The limiter carries its own state, so once it falls behind it never
        # catches up: counting clamped steps alone would let a badly lagging
        # replay pass for a faithful one.
        max_lag = max(max_lag, float(np.abs(arm - target[:6]).max()))
        velocity = (
            np.zeros(6)
            if velocity_mode == "zero"
            else velocity_gain * (arm - previous) / (SAMPLE_PERIOD_STEPS * PHYSICS_DT)
        )
        gripper = float(target[6])
        if gripper < GRIPPER_CLOSED_OPENING:
            gripper_was_closed = True
        elif gripper_was_closed and not release_configured:
            task._configure_release_contact_solver()
            release_configured = True
            release_row = row
        if ratchet is not None:
            contacts = len(task.get_gripper_actor_contact_position("panthera_cylinder"))
            gripper = ratchet(gripper, contacts, row)
        # A fixed five-step hold advances whether or not the arm arrived, so the
        # controller runs a standing lag through every motion.  That lag is what
        # leaves the cylinder released centimetres above its seat.  Converging
        # instead spends the steps where they are needed and costs nothing once
        # the arm is already there.
        budget = converge_max_steps if converge_tolerance > 0.0 else SAMPLE_PERIOD_STEPS
        held = 0
        while held < budget:
            if interpolate:
                alpha = min((held + 1) / SAMPLE_PERIOD_STEPS, 1.0)
                step_arm = previous + (arm - previous) * alpha
                step_gripper = last_gripper + (gripper - last_gripper) * alpha
            else:
                step_arm, step_gripper = arm, gripper
            task.robot.set_arm_joints(step_arm, velocity, "left")
            task.robot.set_gripper(step_gripper, "left")
            task._step_scene()
            held += 1
            if converge_tolerance > 0.0 and held >= SAMPLE_PERIOD_STEPS:
                measured = np.asarray(
                    task._actual_robot_state()["arm_qpos"], dtype=np.float64
                )
                if float(np.abs(measured - arm).max()) <= converge_tolerance:
                    break
        physics_steps += held
        last_gripper = gripper
        if trace_object:
            pose = task.cylinder.get_pose()
            contacts = len(
                task.get_gripper_actor_contact_position("panthera_cylinder")
            )
            # ``get_left_arm_jointState`` returns drive targets and
            # ``get_left_gripper_val`` echoes the last command, so both are
            # tautological here.  ``_actual_robot_state`` is the same measurement
            # the recorder wrote into the dataset, which makes the replay and the
            # recording directly comparable.
            measured = task._actual_robot_state()
            achieved = np.asarray(measured["arm_qpos"], dtype=np.float64)
            trace.append(
                {
                    "row": row,
                    "xyz": [float(v) for v in pose.p],
                    "quat": [float(v) for v in pose.q],
                    "gripper_cmd": float(gripper),
                    "gripper_qpos": float(measured["gripper_qpos"]),
                    "gripper_target": float(
                        task.robot.get_normal_real_gripper_val()[0]
                    ),
                    "contacts": int(contacts),
                    "arm_cmd": [float(v) for v in arm],
                    "arm_qpos": [float(v) for v in achieved],
                    "track_err_rad": float(np.abs(achieved - arm).max()),
                }
            )
        previous = arm
        if task.check_success():
            break
    # Success requires the cylinder to be at rest, and the oracle only judged it
    # after ``_advance_until_inserted_stable``.  Judging the instant the action
    # stream ends fails an episode that is correctly inserted but still rocking.
    if settle and not release_configured:
        # A stream that never reopened the gripper still finished its route.
        task._configure_release_contact_solver()
        release_configured = True
    settled_steps = 0
    if settle and not task.check_success():
        before = int(getattr(task, "simulation_step_count", 0))
        task._advance_until_inserted_stable()
        settled_steps = int(getattr(task, "simulation_step_count", 0)) - before
    result = {
        "success": bool(task.check_success()),
        "final_settle_steps": settled_steps,
        "release_solver_row": release_row,
        "physics_steps": physics_steps,
        "executed_targets": row + 1,
        "clamped": clamped,
        "max_speed_limit_lag_rad": max_lag,
    }
    if trace_object:
        result["object_trace"] = trace
    if ratchet is not None:
        result.update(
            {
                "ratchet_engaged_at": ratchet.engaged_at,
                "ratchet_suppressed_openings": ratchet.suppressed,
                "ratchet_still_holding": ratchet.holding,
            }
        )
    return result


_WORKER: dict = {}


def _init_worker(config: dict, devices) -> None:
    """Pin one GPU and build this process's own copy of the task environment.

    curobo initialises CUDA on import, so the device has to be chosen before the
    task module is imported -- which is why every worker sets up from scratch
    instead of inheriting a parent that already imported it.
    """
    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices.get()
    os.environ["ASSETS_PATH"] = config["robotwin_root"]
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, config["robotwin_root"])
    os.chdir(config["robotwin_root"])
    module = importlib.import_module(f"envs.{config['task_name']}")
    _WORKER["task_class"] = getattr(module, config["task_name"])
    _WORKER["config"] = config
    _WORKER["scene"] = json.loads(
        (Path(config["dataset_root"]) / "scene_info.json").read_text(encoding="utf-8")
    )


def _run_job(job: tuple[int, str]) -> dict:
    """Replay one episode under one velocity mode and report it as a case."""
    episode_id, mode = job
    config = _WORKER["config"]
    cli = config["options"]
    dataset_root = Path(config["dataset_root"])
    metadata = _WORKER["scene"][f"episode_{episode_id}"]["panthera_episode"]
    seed = int(metadata["episode_seed"])
    targets = _grid_actions(
        dataset_root / "data" / f"episode{episode_id}.hdf5", cli["action_alignment"]
    )

    task = _WORKER["task_class"]()
    case = {
        "episode": episode_id,
        "seed": seed,
        "velocity_mode": mode,
        "interpolate": bool(cli["interpolate"]),
        "action_alignment": cli["action_alignment"],
        "final_settle": not cli["no_final_settle"],
        "posture": metadata["realized_geometry"]["cylinder_posture"],
        "gripper_ratchet": bool(cli["gripper_ratchet"]),
        "grid_targets": int(len(targets)),
    }
    try:
        args = copy.deepcopy(config["base_args"])
        args["step_lim"] = int(len(targets))
        _use_scene_registry(args, config["scene_registry"])
        task.setup_demo(now_ep_num=seed, seed=seed, **args)
        case["scene"] = _verify_scene(task, metadata)
        if not case["scene"]["matches_recording"]:
            raise ValueError(
                "rebuilt scene does not match the recording: "
                f"orientation off by "
                f"{case['scene']['orientation_error_deg']:.2f} deg, "
                f"position off by "
                f"{case['scene']['position_error_m'] * 1000:.2f} mm"
            )
        ratchet = (
            GripperRatchet(cli["ratchet_margin"], cli["ratchet_hold_steps"])
            if cli["gripper_ratchet"] else None
        )
        case.update(
            _execute(
                task, targets, mode, cli["speed_limit_radps"], ratchet,
                cli["trace_object"], cli["interpolate"],
                not cli["no_final_settle"],
                cli["converge_tolerance_rad"], cli["converge_max_steps"],
                cli["velocity_gain"],
            )
        )
        case["metrics"] = {
            key: (float(value) if not isinstance(value, bool) else value)
            for key, value in task.success_metrics().items()
        }
    except Exception as error:
        case["error"] = f"{type(error).__name__}: {error}"
        case["success"] = False
    finally:
        try:
            task.close_env()
        except Exception:
            pass
    return case


def main() -> int:
    cli = parse_args()
    robotwin_root = cli.robotwin_root.resolve()
    dataset_root = cli.dataset_root.resolve()
    if cli.workers < 1:
        raise ValueError("--workers must be at least one")
    modes = (
        ("zero", "finite_difference")
        if cli.velocity_mode == "both"
        else (cli.velocity_mode,)
    )
    config = {
        "robotwin_root": str(robotwin_root),
        "dataset_root": str(dataset_root),
        "task_name": cli.task_name,
        "base_args": _task_args(robotwin_root, cli.task_config, cli.task_name),
        "scene_registry": str(cli.scene_registry.resolve()),
        "options": {
            "velocity_mode": cli.velocity_mode,
            "speed_limit_radps": cli.speed_limit_radps,
            "gripper_ratchet": cli.gripper_ratchet,
            "ratchet_margin": cli.ratchet_margin,
            "ratchet_hold_steps": cli.ratchet_hold_steps,
            "trace_object": cli.trace_object,
            "interpolate": cli.interpolate,
            "no_final_settle": cli.no_final_settle,
            "action_alignment": cli.action_alignment,
            "converge_tolerance_rad": cli.converge_tolerance_rad,
            "converge_max_steps": cli.converge_max_steps,
            "velocity_gain": cli.velocity_gain,
        },
    }

    jobs = [(episode_id, mode) for episode_id in cli.episode for mode in modes]
    devices = None
    if cli.gpus:
        gpu_ids = [value.strip() for value in cli.gpus.split(",") if value.strip()]
        if not gpu_ids:
            raise ValueError("--gpus must name at least one device")
        devices = mp.get_context("spawn").Queue()
        for index in range(cli.workers):
            devices.put(gpu_ids[index % len(gpu_ids)])

    cases: list[dict] = []
    context = mp.get_context("spawn")
    with context.Pool(
        processes=min(cli.workers, len(jobs)),
        initializer=_init_worker,
        initargs=(config, devices),
    ) as pool:
        for case in pool.imap_unordered(_run_job, jobs):
            cases.append(case)
            print(
                f"  [{len(cases):3d}/{len(jobs)}] ep{case['episode']:5d} "
                f"{case['posture']:8s} {case['velocity_mode']:17s} "
                f"success={case.get('success')} "
                f"clamped={case.get('clamped')} "
                f"lag={case.get('max_speed_limit_lag_rad', 0.0):.4f} "
                f"err={case.get('error', '')}",
                flush=True,
            )
    cases.sort(key=lambda case: (case["episode"], case["velocity_mode"]))

    summary = {
        "speed_limit_radps": cli.speed_limit_radps,
        "by_mode": {
            mode: {
                "n": sum(1 for c in cases if c["velocity_mode"] == mode),
                "success": sum(
                    1 for c in cases if c["velocity_mode"] == mode and c.get("success")
                ),
            }
            for mode in modes
        },
        "by_posture": {
            posture: {
                "n": sum(1 for c in cases if c["posture"] == posture),
                "success": sum(
                    1 for c in cases if c["posture"] == posture and c.get("success")
                ),
            }
            for posture in sorted({c["posture"] for c in cases})
        },
        "cases": cases,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print()
    for mode, value in summary["by_mode"].items():
        print(f"  {mode:17s} {value['success']}/{value['n']}")
    for posture, value in summary["by_posture"].items():
        print(f"  {posture:17s} {value['success']}/{value['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
