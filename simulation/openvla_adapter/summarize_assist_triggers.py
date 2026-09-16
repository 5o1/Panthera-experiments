#!/usr/bin/env python3
"""Print compact, per-seed diagnostics around terminal-assist triggers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _round_vector(values: list[float], digits: int = 5) -> list[float]:
    return [round(float(value), digits) for value in values]


def _compact_stage_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Keep only physical quantities needed to locate a dropped object."""
    result: dict[str, Any] = {}
    for stage, snapshot in diagnostics.items():
        metrics = snapshot["success_metrics"]
        result[stage] = {
            "cylinder_position_m": _round_vector(
                snapshot["cylinder_position_m"]
            ),
            "ee_position_m": _round_vector(snapshot["ee_position_m"]),
            "gripper_opening": round(float(snapshot["gripper_opening"]), 5),
            "axis_error_deg": round(float(metrics["axis_error_deg"]), 3),
            "linear_speed_mps": round(float(metrics["linear_speed_mps"]), 5),
            "angular_speed_radps": round(
                float(metrics["angular_speed_radps"]), 5
            ),
        }
    return result


def summarize(paths: list[Path], history_rows: int) -> dict[str, Any]:
    """Summarize only the rows needed to audit each assist trigger."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seed_source: dict[int, str] = {}
    seed_source_index: dict[int, int] = {}
    for path in paths:
        ordered_seeds: list[int] = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    seed = int(row["episode_seed"])
                    grouped[seed].append(row)
                    if seed not in ordered_seeds:
                        ordered_seeds.append(seed)
        for source_index, seed in enumerate(ordered_seeds):
            seed_source[seed] = str(path)
            seed_source_index[seed] = source_index

    episodes: list[dict[str, Any]] = []
    for seed in sorted(grouped):
        rows = grouped[seed]
        trigger_index = next(
            (
                index
                for index, row in enumerate(rows)
                if row.get("terminal_insertion_assist", {}).get("attempted")
            ),
            None,
        )
        if trigger_index is None:
            episodes.append(
                {
                    "seed": seed,
                    "source": seed_source[seed],
                    "source_episode_index": seed_source_index[seed],
                    "rows": len(rows),
                    "triggered": False,
                    "final_metrics": rows[-1]["success_metrics"],
                }
            )
            continue

        trigger = rows[trigger_index]
        assist = trigger["terminal_insertion_assist"]
        start = max(0, trigger_index - history_rows)
        history = []
        for row in rows[start : trigger_index + 1]:
            metrics = row["success_metrics"]
            history.append(
                {
                    "chunk": int(row["chunk_index"]),
                    "gripper_before": round(float(row["before_state"][6]), 5),
                    "gripper_action_min": round(float(row["action_min"][6]), 5),
                    "gripper_action_max": round(float(row["action_max"][6]), 5),
                    "cylinder_position_m": _round_vector(
                        row["before_cylinder_position_m"]
                    ),
                    "entry_error_m": _round_vector(
                        row["terminal_insertion_assist"]["target_entry_error_m"]
                    ),
                    "proxy_steps": int(
                        row["terminal_insertion_assist"][
                            "grasp_proxy_consecutive_control_steps"
                        ]
                    ),
                    "object_axis_error_deg": round(
                        float(metrics["axis_error_deg"]), 3
                    ),
                    "object_height_error_m": round(
                        float(metrics["height_error_m"]), 5
                    ),
                    "object_x_error_m": round(float(metrics["x_error_m"]), 5),
                    "object_y_error_m": round(float(metrics["y_error_m"]), 5),
                }
            )
        episodes.append(
            {
                "seed": seed,
                "source": seed_source[seed],
                "source_episode_index": seed_source_index[seed],
                "rows": len(rows),
                "triggered": True,
                "trigger_chunk": int(trigger["chunk_index"]),
                "stage_results": assist.get("stage_results"),
                "stage_diagnostics": _compact_stage_diagnostics(
                    assist.get("stage_diagnostics", {})
                ),
                "history": history,
                "final_metrics": rows[-1]["success_metrics"],
            }
        )
    return {"episode_count": len(episodes), "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--history-rows", type=int, default=4)
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="include only this seed; repeat to select multiple seeds",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "omit trigger history and stage snapshots; keep only fields "
            "needed for live multi-episode monitoring"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.traces, max(0, args.history_rows))
    if args.seeds:
        selected = set(args.seeds)
        result["episodes"] = [
            episode
            for episode in result["episodes"]
            if episode["seed"] in selected
        ]
        result["episode_count"] = len(result["episodes"])
    if args.compact:
        compact_episodes = []
        for episode in result["episodes"]:
            compact = {
                key: episode[key]
                for key in (
                    "seed",
                    "source_episode_index",
                    "rows",
                    "triggered",
                    "trigger_chunk",
                    "stage_results",
                    "final_metrics",
                )
                if key in episode
            }
            compact_episodes.append(compact)
        result["episodes"] = compact_episodes
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
