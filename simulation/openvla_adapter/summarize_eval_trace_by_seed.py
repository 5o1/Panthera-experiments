#!/usr/bin/env python3
"""Summarize RoboTwin closed-loop JSONL traces by episode seed."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def metrics_pass(metrics: dict[str, Any]) -> bool:
    """Apply the task's complete physical insertion acceptance contract."""
    return bool(
        float(metrics["x_error_m"]) <= 0.010
        and float(metrics["y_error_m"]) <= 0.010
        and float(metrics["height_error_m"]) <= 0.012
        and float(metrics["axis_error_deg"]) <= 5.0
        and float(metrics["insertion_depth_m"]) >= 0.030
        and float(metrics["linear_speed_mps"]) <= 0.025
        and float(metrics["angular_speed_radps"]) <= 0.35
        and bool(metrics["gripper_open"])
        and float(metrics["gripper_contact_impulse_ns"]) <= 1.0e-6
    )


def longest_true_run(values: list[bool]) -> int:
    """Return the longest consecutive run of true values."""
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def summarize(paths: list[Path]) -> dict[str, Any]:
    """Load one or more worker traces and aggregate their rows by seed."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sources: dict[int, set[str]] = defaultdict(set)
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                seed = int(row["episode_seed"])
                grouped[seed].append(row)
                sources[seed].add(str(path))

    episodes = []
    for seed in sorted(grouped):
        rows = grouped[seed]
        passed = [metrics_pass(row["success_metrics"]) for row in rows]
        attempted = [
            row.get("terminal_insertion_assist", {})
            for row in rows
            if row.get("terminal_insertion_assist", {}).get("attempted")
        ]
        tail_run = 0
        for value in reversed(passed):
            if not value:
                break
            tail_run += 1
        episodes.append(
            {
                "seed": seed,
                "rows": len(rows),
                "success_once": any(passed),
                "success_at_end": passed[-1],
                "passing_rows": sum(passed),
                "first_passing_row": passed.index(True) if any(passed) else None,
                "longest_passing_run": longest_true_run(passed),
                "passing_tail_rows": tail_run,
                "assist_attempts": len(attempted),
                "assist_succeeded": bool(attempted) and all(
                    item.get("succeeded") for item in attempted
                ),
                "assist_stage_results": (
                    attempted[-1].get("stage_results") if attempted else None
                ),
                "final_metrics": rows[-1]["success_metrics"],
                "source_files": sorted(sources[seed]),
            }
        )

    total = len(episodes)
    once = sum(item["success_once"] for item in episodes)
    at_end = sum(item["success_at_end"] for item in episodes)
    return {
        "trace_files": [str(path) for path in paths],
        "trajectory_count": total,
        "success_once_count": once,
        "success_once_rate": once / total if total else 0.0,
        "success_at_end_count": at_end,
        "success_at_end_rate": at_end / total if total else 0.0,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="print only aggregate counts and one status line per seed",
    )
    args = parser.parse_args()
    summary = summarize(args.traces)
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if args.compact:
        compact = {
            "trajectory_count": summary["trajectory_count"],
            "success_once_count": summary["success_once_count"],
            "success_once_rate": summary["success_once_rate"],
            "success_at_end_count": summary["success_at_end_count"],
            "success_at_end_rate": summary["success_at_end_rate"],
            "episodes": [
                {
                    "seed": episode["seed"],
                    "rows": episode["rows"],
                    "success_once": episode["success_once"],
                    "success_at_end": episode["success_at_end"],
                    "passing_tail_rows": episode["passing_tail_rows"],
                    "assist_attempts": episode["assist_attempts"],
                    "assist_succeeded": episode["assist_succeeded"],
                }
                for episode in summary["episodes"]
            ],
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
