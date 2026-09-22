#!/usr/bin/env python3
"""Compare two policy rollout reports only when their evaluation contract matches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


FAIRNESS_FIELDS = (
    "dataset_digest",
    "max_actions",
    "execution_horizon",
    "temporal_ensemble",
)


def _episode_ids(report: dict) -> list[int]:
    return sorted(int(case["episode"]) for case in report.get("cases", []))


def _latency(report: dict) -> float | None:
    values = []
    for case in report.get("cases", []):
        value = case.get("inference_timing", {}).get("amortized_ms_per_action", {}).get("median")
        if value is not None:
            values.append(float(value))
    return None if not values else float(median(values))


def compare(left: dict, right: dict) -> dict:
    """Return a paired summary, rejecting apples-to-oranges inputs."""
    mismatches = {
        field: {"left": left.get(field), "right": right.get(field)}
        for field in FAIRNESS_FIELDS
        if left.get(field) != right.get(field)
    }
    left_episodes, right_episodes = _episode_ids(left), _episode_ids(right)
    if left_episodes != right_episodes:
        mismatches["episodes"] = {"left": left_episodes, "right": right_episodes}
    if mismatches:
        raise ValueError(f"evaluation contracts differ: {json.dumps(mismatches, sort_keys=True)}")

    def summary(report: dict) -> dict:
        total = int(report.get("total", len(report.get("cases", []))))
        success = int(report.get("success", 0))
        gate_success = int(report.get("gate_success", 0))
        return {
            "backend": report.get("policy_backend"),
            "model": report.get("model"),
            "success": success,
            "gate_success": gate_success,
            "total": total,
            "success_rate": success / total if total else 0.0,
            "gate_success_rate": gate_success / total if total else 0.0,
            "median_amortized_ms_per_action": _latency(report),
            "by_posture": report.get("by_posture", {}),
        }

    return {
        "comparison_contract": {
            **{field: left.get(field) for field in FAIRNESS_FIELDS},
            "episodes": left_episodes,
        },
        "policies": [summary(left), summary(right)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    result = compare(
        json.loads(cli.left.read_text(encoding="utf-8")),
        json.loads(cli.right.read_text(encoding="utf-8")),
    )
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for policy in result["policies"]:
        latency = policy["median_amortized_ms_per_action"]
        latency_text = "n/a" if latency is None else f"{latency:.2f} ms/action"
        print(
            f"{policy['backend']}: gate={policy['gate_success']}/{policy['total']} "
            f"raw={policy['success']}/{policy['total']} latency={latency_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
