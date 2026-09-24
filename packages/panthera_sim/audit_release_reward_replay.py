#!/usr/bin/env python3
"""Audit R1/R2 reward ordering with three real PhysX release replays.

Every case rebuilds episode 2 from the recorded scene and camera, applies the
same expert warm-start prefix, then executes one suffix through the production
50 Hz action executor.  R2 components also contain the terminal task term, so
the same physical rollout yields both the sparse R1 and shaped R2 returns.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import h5py
import numpy as np

DEFAULT_TASK = "place_randomized_cylinder_in_socket"
DEFAULT_CONFIG = "panthera_phone_cylinder_socket_v2_pilot.yml"


def build_case_actions(
    actions: np.ndarray, prefix_actions: int
) -> dict[str, np.ndarray]:
    """Return expert, immediate-open and never-open suffixes."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"actions must have shape [N, 7], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("actions must be finite")
    if not 1 <= prefix_actions < len(values):
        raise ValueError(
            f"prefix_actions must be in [1, {len(values) - 1}], got {prefix_actions}"
        )
    expert = values[prefix_actions:].copy()
    immediate_open = expert.copy()
    immediate_open[:, 6] = 1.0
    never_open = expert.copy()
    closed_target = float(np.clip(values[prefix_actions - 1, 6], 0.0, 0.49))
    never_open[:, 6] = closed_target
    return {
        "expert_suffix": expert,
        "immediate_open": immediate_open,
        "never_open": never_open,
    }


def gate_results(cases: dict[str, dict]) -> tuple[bool, list[str]]:
    """Require expert success and strict R1/R2 ordering over both failures."""
    reasons: list[str] = []
    required = {"expert_suffix", "immediate_open", "never_open"}
    missing = required.difference(cases)
    if missing:
        return False, [f"missing cases: {sorted(missing)}"]
    expert = cases["expert_suffix"]
    failures = [cases["immediate_open"], cases["never_open"]]
    if expert.get("error"):
        reasons.append(f"expert replay errored: {expert['error']}")
    if not expert.get("success", False):
        reasons.append("expert suffix did not reach stable insertion")
    for failure in failures:
        if failure.get("error"):
            reasons.append(f"{failure['case']} errored: {failure['error']}")
        if failure.get("success", False):
            reasons.append(f"{failure['case']} unexpectedly succeeded")
    for reward_name in ("r1_return", "r2_return"):
        expert_return = expert.get(reward_name)
        failure_returns = [case.get(reward_name) for case in failures]
        if expert_return is None or any(value is None for value in failure_returns):
            reasons.append(f"{reward_name} is missing")
        elif not all(expert_return > value for value in failure_returns):
            reasons.append(
                f"{reward_name} does not rank expert strictly above failures: "
                f"expert={expert_return}, failures={failure_returns}"
            )
    return not reasons, reasons


def _load_actions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "joint_action/vector" not in handle:
            raise ValueError(f"{path} has no joint_action/vector")
        return np.asarray(handle["joint_action/vector"], dtype=np.float64)


def _run_case(
    *,
    name: str,
    actions: np.ndarray,
    robotwin_root: Path,
    task_config: str,
    task_name: str,
    episode,
    scenes_path: Path,
    camera: dict,
    camera_pose: np.ndarray,
    base_args: dict,
) -> dict:
    from robotwin_env import build_task, inside

    record: dict = {"case": name, "actions_requested": int(len(actions))}
    task = None
    with inside(robotwin_root):
        try:
            task, scene_report = build_task(
                robotwin_root,
                task_config,
                task_name,
                episode,
                scenes_path,
                len(actions),
                base_args,
                camera=camera,
                camera_pose=camera_pose,
            )
            initial = task.policy_release_effect_state(update_settle=False)
            reward, termination, truncation, info = task.gen_sparse_reward_data(
                actions
            )
            final = task.policy_release_effect_state(update_settle=False)
            components = info.get("reward_components", [])
            r2_return = float(sum(info.get("step_rewards", [])))
            r1_return = float(sum(float(item["task"]) for item in components))
            record.update(
                {
                    "scene": scene_report,
                    "warm_start": dict(task.rl_warm_start_audit),
                    "initial_effect": asdict(initial),
                    "final_effect": asdict(final),
                    "actions_executed": int(info.get("executed_steps", 0)),
                    "success": bool(info.get("success", False)),
                    "hard_failure": bool(info.get("hard_failure", False)),
                    "termination": int(np.asarray(termination).reshape(-1)[0]),
                    "truncation": int(np.asarray(truncation).reshape(-1)[0]),
                    "r1_return": r1_return,
                    "r2_return": r2_return,
                    "reported_reward": float(np.asarray(reward).reshape(-1)[0]),
                    "reward_component_count": len(components),
                }
            )
            if not np.isclose(record["reported_reward"], r2_return, atol=1.0e-6):
                raise RuntimeError(
                    "reported R2 reward disagrees with action-aligned step rewards"
                )
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            if task is not None:
                with contextlib.suppress(Exception):
                    task.close_env()
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task-config", default=DEFAULT_CONFIG)
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--prefix-actions", type=int, default=780)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    from dataset import open_dataset
    from robotwin_env import task_args

    cli = parse_args()
    dataset = open_dataset(cli.dataset_root)
    episode = dataset.episode(cli.episode)
    all_actions = _load_actions(episode.hdf5_path)
    cases_actions = build_case_actions(all_actions, cli.prefix_actions)
    robotwin_root = cli.robotwin_root.resolve()
    base_args = task_args(robotwin_root, cli.task_config, cli.task_name)
    base_args["rl_reward_variant"] = "r2"
    warm_start = base_args.setdefault("task_randomization", {}).setdefault(
        "rl_warm_start", {}
    )
    warm_start.update(
        {
            "enabled": True,
            "trajectory_path": str(episode.hdf5_path.resolve()),
            "prefix_actions": cli.prefix_actions,
        }
    )

    cases: dict[str, dict] = {}
    for name, actions in cases_actions.items():
        print(f"运行 PhysX reward replay：{name}", flush=True)
        cases[name] = _run_case(
            name=name,
            actions=actions,
            robotwin_root=robotwin_root,
            task_config=cli.task_config,
            task_name=cli.task_name,
            episode=episode,
            scenes_path=dataset.scenes_path,
            camera=dataset.camera(cli.episode),
            camera_pose=dataset.camera_pose(cli.episode),
            base_args=base_args,
        )
    passed, reasons = gate_results(cases)
    payload = {
        "schema_version": 1,
        "dataset_root": str(cli.dataset_root.resolve()),
        "dataset_digest": dataset.digest(),
        "episode": cli.episode,
        "prefix_actions": cli.prefix_actions,
        "task_name": cli.task_name,
        "task_config": cli.task_config,
        "cases": cases,
        "gate_passed": passed,
        "gate_reasons": reasons,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, case in cases.items():
        print(
            f"  {name}: success={case.get('success')} "
            f"R1={case.get('r1_return')} R2={case.get('r2_return')} "
            f"error={case.get('error', '')}",
            flush=True,
        )
    if not passed:
        for reason in reasons:
            print(f"门禁失败：{reason}", flush=True)
        return 1
    print(f"奖励物理回放排序门禁通过：{cli.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
