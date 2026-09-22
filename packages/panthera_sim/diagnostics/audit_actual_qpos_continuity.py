#!/usr/bin/env python3
"""Gate recorded episodes on actual-qpos joint velocity continuity.

The merged dataset audit bounds peak velocity and acceleration on the *retimed
plan*, and checks actual qpos only for near-zero-speed stalls.  Neither detects
a velocity discontinuity in what was executed, which is how the schema-9 pilot
passed a plan-only audit while still looking jerky.

This probe uses the same window and interior definition as that audit -- the
route segment from ``continuous_motion_audit`` and quintic geometric progress
0.05..0.95 -- and adds the missing measure: the per-joint velocity change
between consecutive samples.  A bounded-acceleration profile cannot change
velocity by more than ``a_limit * dt`` in one sample, so the ratio of the
observed jump to that budget is the discontinuity in units of the contract.

Jumps are only evaluated across regularly spaced samples.  The recorder inserts
extra diagnostic frames at stage boundaries, and a finite difference taken
across an irregular gap measures the sampling, not the trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


INTERIOR_LOW, INTERIOR_HIGH = 0.05, 0.95
DT_REGULARITY_TOLERANCE = 0.25


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=24)
    parser.add_argument("--acceleration-limit", type=float, default=2.002)
    parser.add_argument("--max-jump-ratio", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _episode_metrics(
    source_root: Path, episode_id: int, metadata: dict, acceleration_limit: float
) -> dict:
    motion_audit = metadata.get("continuous_motion_audit", [])
    if len(motion_audit) != 1:
        raise ValueError(f"episode {episode_id} lacks one continuous motion audit")
    route = motion_audit[0]

    with h5py.File(source_root / "data" / f"episode{episode_id}.hdf5", "r") as episode:
        arm_qpos = np.asarray(episode["observation/robot_state/arm_qpos"], dtype=np.float64)
        step_index = np.asarray(episode["timing/simulation_step_index"])
        simulation_time = np.asarray(episode["timing/simulation_time_s"], dtype=np.float64)

    route_mask = (step_index >= int(route["start_simulation_step"])) & (
        step_index <= int(route["end_simulation_step"])
    )
    route_qpos = arm_qpos[route_mask]
    route_time = simulation_time[route_mask]
    if len(route_qpos) < 4:
        raise ValueError(f"episode {episode_id} route has too few samples")

    sample_dt = np.diff(route_time)
    sample_mid_time = (route_time[:-1] + route_time[1:]) / 2.0
    phase = (sample_mid_time - route_time[0]) / (route_time[-1] - route_time[0])
    progress = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    interior = (
        (sample_dt > 1.0e-9)
        & (progress >= INTERIOR_LOW)
        & (progress <= INTERIOR_HIGH)
    )
    if np.count_nonzero(interior) < 3:
        raise ValueError(f"episode {episode_id} has too few interior samples")

    nominal_dt = float(np.median(sample_dt[interior]))
    regular = np.abs(sample_dt - nominal_dt) <= DT_REGULARITY_TOLERANCE * nominal_dt
    jump_mask = interior[:-1] & interior[1:] & regular[:-1] & regular[1:]

    # Stage-boundary frames can repeat a timestamp; those samples are excluded by
    # both masks below, so keep the division itself from producing warnings.
    safe_dt = np.where(sample_dt > 1.0e-9, sample_dt, np.nan)
    velocity = np.diff(route_qpos, axis=0) / safe_dt[:, None]
    jump = np.abs(np.diff(velocity, axis=0))
    budget = acceleration_limit * sample_dt[1:][:, None]
    ratio = jump / np.maximum(budget, 1e-12)

    if not np.any(jump_mask):
        raise ValueError(f"episode {episode_id} has no regularly spaced interior pair")
    selected_ratio = ratio[jump_mask]
    selected_jump = jump[jump_mask]
    return {
        "episode": episode_id,
        "route_samples": int(len(route_qpos)),
        "nominal_dt_s": nominal_dt,
        "evaluated_pairs": int(np.count_nonzero(jump_mask)),
        "skipped_irregular_pairs": int(
            np.count_nonzero(interior[:-1] & interior[1:] & ~(regular[:-1] & regular[1:]))
        ),
        "max_interior_joint_speed_radps": float(
            np.abs(velocity[interior]).max()
        ),
        "max_velocity_jump_radps": float(selected_jump.max()),
        "max_jump_over_budget_ratio": float(selected_ratio.max()),
        "jump_violations": int(np.count_nonzero(selected_ratio > 1.0)),
        "jump_violation_fraction": float(np.mean(selected_ratio > 1.0)),
    }


def main() -> int:
    args = _parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset import open_dataset

    dataset = open_dataset(args.source_root)
    available = dataset.episode_ids()
    if not available:
        raise SystemExit(f"{args.source_root} contains no episodes")
    chosen = np.unique(
        np.linspace(0, len(available) - 1, args.episode_count, dtype=np.int64)
    )
    records = [
        _episode_metrics(
            args.source_root,
            available[index],
            dataset.episode_record(available[index]),
            args.acceleration_limit,
        )
        for index in chosen.tolist()
    ]

    worst_ratio = max(record["max_jump_over_budget_ratio"] for record in records)
    passed = worst_ratio <= args.max_jump_ratio
    summary = {
        "status": "passed" if passed else "failed",
        "source_root": str(args.source_root),
        "dataset_episodes": len(available),
        "episodes_audited": len(records),
        "acceleration_limit_radps2": args.acceleration_limit,
        "max_jump_ratio_gate": args.max_jump_ratio,
        "interior_definition": (
            "quintic geometric progress 0.05..0.95 within the recorded route"
        ),
        "worst_jump_over_budget_ratio": worst_ratio,
        "worst_max_velocity_jump_radps": max(
            record["max_velocity_jump_radps"] for record in records
        ),
        "worst_max_interior_joint_speed_radps": max(
            record["max_interior_joint_speed_radps"] for record in records
        ),
        "total_jump_violations": sum(record["jump_violations"] for record in records),
        "total_evaluated_pairs": sum(record["evaluated_pairs"] for record in records),
        "total_skipped_irregular_pairs": sum(
            record["skipped_irregular_pairs"] for record in records
        ),
        "per_episode": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for key, value in summary.items():
        if key != "per_episode":
            print(f"{key}: {value}")
    if not passed:
        raise SystemExit(
            f"actual qpos velocity jump {worst_ratio:.3f}x exceeds the "
            f"{args.max_jump_ratio}x budget gate"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
