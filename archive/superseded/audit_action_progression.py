#!/usr/bin/env python3
"""Audit action imbalance and closed-loop progress against expert episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


SAMPLE_PERIOD_STEPS = 5


def grid_indices(steps: np.ndarray) -> np.ndarray:
    """Match the RLDS adapter's last-capture-per-global-grid selection."""
    selected: dict[int, int] = {}
    for index, step in enumerate(steps.tolist()):
        if step % SAMPLE_PERIOD_STEPS == 0:
            selected[int(step)] = index
    return np.asarray([selected[step] for step in sorted(selected)], dtype=np.int64)


def fractions(values: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, float]:
    """Return the fraction of scalar magnitudes under each threshold."""
    return {
        f"le_{threshold:g}": float(np.mean(values <= threshold))
        for threshold in thresholds
    }


def summarize_dataset(dataset: Path) -> tuple[dict, dict[int, dict]]:
    """Summarize next-action magnitudes and retain per-seed expert arrays."""
    scene = json.loads((dataset / "scene_info.json").read_text())
    joint_leads: list[np.ndarray] = []
    gripper_leads: list[np.ndarray] = []
    joint_steps: list[np.ndarray] = []
    gripper_steps: list[np.ndarray] = []
    gripper_targets: list[np.ndarray] = []
    phase_joint_leads: list[list[np.ndarray]] = [[] for _ in range(10)]
    experts: dict[int, dict] = {}

    for path in sorted((dataset / "data").glob("episode*.hdf5")):
        episode_id = int(path.stem.removeprefix("episode"))
        seed = int(scene[f"episode_{episode_id}"]["panthera_episode"]["episode_seed"])
        with h5py.File(path, "r") as data:
            indices = grid_indices(np.asarray(data["timing/simulation_step_index"]))
            states = np.asarray(data["observation/robot_state/vector"])[indices]
            raw_actions = np.asarray(data["joint_action/vector"])
            actions = raw_actions[indices]
        targets = actions[np.minimum(np.arange(len(indices)) + 1, len(indices) - 1)]
        joint_lead = np.linalg.norm(targets[:, :6] - states[:, :6], axis=1)
        gripper_lead = np.abs(targets[:, 6] - states[:, 6])
        joint_step = np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1)
        gripper_step = np.abs(np.diff(actions[:, 6], axis=0))
        joint_leads.append(joint_lead)
        gripper_leads.append(gripper_lead)
        joint_steps.append(joint_step)
        gripper_steps.append(gripper_step)
        gripper_targets.append(targets[:, 6])
        for phase in range(10):
            start = phase * len(targets) // 10
            stop = (phase + 1) * len(targets) // 10
            phase_joint_leads[phase].append(joint_lead[start:stop])
        experts[seed] = {"states": states, "actions": actions}

    joint_lead_all = np.concatenate(joint_leads)
    gripper_lead_all = np.concatenate(gripper_leads)
    joint_step_all = np.concatenate(joint_steps)
    gripper_step_all = np.concatenate(gripper_steps)
    gripper_all = np.concatenate(gripper_targets)
    summary = {
        "episodes": len(experts),
        "samples": int(joint_lead_all.size),
        "joint_next_target_norm_rad": {
            "mean": float(joint_lead_all.mean()),
            "median": float(np.median(joint_lead_all)),
            "p90": float(np.quantile(joint_lead_all, 0.9)),
            "fractions": fractions(joint_lead_all, (1e-4, 1e-3, 1e-2, 5e-2)),
        },
        "gripper_next_target_abs": {
            "mean": float(gripper_lead_all.mean()),
            "median": float(np.median(gripper_lead_all)),
            "fractions": fractions(gripper_lead_all, (1e-4, 1e-3, 1e-2)),
        },
        "successive_joint_action_norm_rad": {
            "mean": float(joint_step_all.mean()),
            "median": float(np.median(joint_step_all)),
            "p90": float(np.quantile(joint_step_all, 0.9)),
            "fractions": fractions(joint_step_all, (1e-4, 1e-3, 1e-2, 5e-2)),
        },
        "successive_gripper_action_abs": {
            "mean": float(gripper_step_all.mean()),
            "fractions": fractions(gripper_step_all, (1e-4, 1e-3, 1e-2)),
        },
        "gripper_target_fraction": {
            "open_ge_0_8": float(np.mean(gripper_all >= 0.8)),
            "closed_le_0_5": float(np.mean(gripper_all <= 0.5)),
            "middle": float(np.mean((gripper_all > 0.5) & (gripper_all < 0.8))),
        },
        "joint_lead_mean_by_episode_decile": [
            float(np.concatenate(values).mean()) for values in phase_joint_leads
        ],
    }
    return summary, experts


def summarize_traces(
    trace_dir: Path, experts: dict[int, dict], action_chunk: int
) -> list[dict]:
    """Map each policy chunk to the nearest measured expert state."""
    summaries = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        seed = int(rows[0]["episode_seed"])
        if seed not in experts:
            summaries.append(
                {
                    "file": path.name,
                    "seed": seed,
                    "chunks": len(rows),
                    "expert_available": False,
                }
            )
            continue
        expert = experts[seed]
        states = expert["states"]
        actions = expert["actions"]
        nearest = []
        first_errors = []
        last_errors = []
        for row in rows:
            before = np.asarray(row["before_state"], dtype=np.float64)
            index = int(np.argmin(np.linalg.norm(states - before, axis=1)))
            nearest.append(index)
            first_target = actions[min(index + 1, len(actions) - 1)]
            last_target = actions[min(index + action_chunk, len(actions) - 1)]
            first_errors.append(np.mean(np.abs(np.asarray(row["action_first"]) - first_target)))
            last_errors.append(np.mean(np.abs(np.asarray(row["action_last"]) - last_target)))
        summaries.append(
            {
                "file": path.name,
                "seed": seed,
                "chunks": len(rows),
                "expert_available": True,
                "nearest_expert_index_first_20": nearest[:20],
                "nearest_expert_index_last_20": nearest[-20:],
                "nearest_expert_index_min": min(nearest),
                "nearest_expert_index_max": max(nearest),
                "nearest_expert_index_unique": len(set(nearest)),
                "nearest_expert_index_final": nearest[-1],
                "policy_vs_expert_first_action_mae": float(np.mean(first_errors)),
                "policy_vs_expert_last_action_mae": float(np.mean(last_errors)),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--action-chunk", type=int, default=5)
    args = parser.parse_args()
    if args.action_chunk <= 0:
        parser.error("--action-chunk must be positive")
    dataset_summary, experts = summarize_dataset(args.dataset)
    result = {"dataset": dataset_summary, "action_chunk": args.action_chunk}
    if args.trace_dir:
        result["traces"] = summarize_traces(
            args.trace_dir, experts, args.action_chunk
        )
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
