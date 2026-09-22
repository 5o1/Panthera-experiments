#!/usr/bin/env python3
"""Audit when a deployable gripper-proprio grasp proxy first becomes true."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _rounded_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "axis_error_deg": round(float(metrics["axis_error_deg"]), 3),
        "height_error_m": round(float(metrics["height_error_m"]), 5),
        "x_error_m": round(float(metrics["x_error_m"]), 5),
        "y_error_m": round(float(metrics["y_error_m"]), 5),
    }


def _state_before_row(
    rows: list[dict[str, Any]], index: int
) -> tuple[dict[str, Any], list[float]]:
    """Return object metrics and position aligned with the row's pre-action EE."""
    if index > 0:
        previous = rows[index - 1]
        return (
            previous["success_metrics"],
            previous["after_cylinder_position_m"],
        )
    row = rows[index]
    return row["success_metrics"], row["before_cylinder_position_m"]


def summarize(
    paths: list[Path], ee_height_thresholds: list[float]
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    grouped[int(row["episode_seed"])].append(row)

    episodes = []
    for seed in sorted(grouped):
        rows = grouped[seed]
        close_band_run = 0
        stable_close_band_run = 0
        max_close_band_run = 0
        max_stable_close_band_run = 0
        previous_opening: float | None = None
        for row in rows:
            opening = float(row["before_state"][6])
            commanded_close = float(row["action_max"][6]) <= 0.50
            in_contact_band = 0.60 <= opening < 0.75
            if commanded_close and in_contact_band:
                close_band_run += 1
                if (
                    previous_opening is not None
                    and abs(opening - previous_opening) <= 0.005
                ):
                    stable_close_band_run += 1
                else:
                    stable_close_band_run = 1
                max_close_band_run = max(max_close_band_run, close_band_run)
                max_stable_close_band_run = max(
                    max_stable_close_band_run, stable_close_band_run
                )
                previous_opening = opening
            else:
                close_band_run = 0
                stable_close_band_run = 0
                previous_opening = None
        grasp_index = next(
            (
                index
                for index, row in enumerate(rows)
                if row.get("terminal_insertion_assist", {}).get(
                    "policy_grasp_seen"
                )
            ),
            None,
        )
        assist_index = next(
            (
                index
                for index, row in enumerate(rows)
                if row.get("terminal_insertion_assist", {}).get("attempted")
            ),
            None,
        )
        release_indices = [
            index
            for index, row in enumerate(rows)
            if row.get("terminal_insertion_assist", {}).get(
                "release_requested"
            )
        ]
        episode: dict[str, Any] = {
            "seed": seed,
            "rows": len(rows),
            "grasp_proxy_seen": grasp_index is not None,
            "assist_attempted": assist_index is not None,
            "release_request_count": len(release_indices),
            "max_close_band_rows": max_close_band_run,
            "max_stable_close_band_rows": max_stable_close_band_run,
        }
        if grasp_index is not None:
            row = rows[grasp_index]
            before_metrics, _ = _state_before_row(rows, grasp_index)
            assist = row["terminal_insertion_assist"]
            episode["first_grasp_chunk"] = int(row["chunk_index"])
            episode["gripper_opening"] = round(float(row["before_state"][6]), 5)
            episode["entry_error_m"] = [
                round(float(value), 5)
                for value in assist["target_entry_error_m"]
            ]
            episode["object_metrics_before_handoff"] = _rounded_metrics(
                before_metrics
            )
            height_candidates: dict[str, Any] = {}
            for threshold in ee_height_thresholds:
                candidate_index = next(
                    (
                        index
                        for index in range(grasp_index, len(rows))
                        if float(
                            rows[index]["terminal_insertion_assist"][
                                "current_ee_position_m"
                            ][2]
                        )
                        >= threshold
                    ),
                    None,
                )
                key = f"{threshold:.5f}"
                if candidate_index is None:
                    height_candidates[key] = None
                    continue
                candidate = rows[candidate_index]
                metrics, cylinder_position = _state_before_row(
                    rows, candidate_index
                )
                ee_position = candidate["terminal_insertion_assist"][
                    "current_ee_position_m"
                ]
                height_candidates[key] = {
                    "chunk": int(candidate["chunk_index"]),
                    "rows_after_grasp": candidate_index - grasp_index,
                    "ee_position_m": [
                        round(float(value), 5) for value in ee_position
                    ],
                    "cylinder_position_m": [
                        round(float(value), 5) for value in cylinder_position
                    ],
                    "object_metrics": _rounded_metrics(metrics),
                }
            episode["height_handoff_candidates"] = height_candidates
        if release_indices:
            release_index = release_indices[0]
            row = rows[release_index]
            metrics, cylinder_position = _state_before_row(
                rows, release_index
            )
            assist = row["terminal_insertion_assist"]
            episode["first_release_request"] = {
                "chunk": int(row["chunk_index"]),
                "row_index": release_index,
                "entry_error_m": [
                    round(float(value), 5)
                    for value in assist["target_entry_error_m"]
                ],
                "calibrated_target_entry": bool(
                    assist["calibrated_target_entry"]
                ),
                "ee_position_m": [
                    round(float(value), 5)
                    for value in assist["current_ee_position_m"]
                ],
                "cylinder_position_m": [
                    round(float(value), 5) for value in cylinder_position
                ],
                "object_metrics": _rounded_metrics(metrics),
            }
        if assist_index is not None:
            episode["assist_chunk"] = int(rows[assist_index]["chunk_index"])
        episodes.append(episode)
    return {
        "episode_count": len(episodes),
        "grasp_proxy_count": sum(
            episode["grasp_proxy_seen"] for episode in episodes
        ),
        "assist_attempt_count": sum(
            episode["assist_attempted"] for episode in episodes
        ),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument(
        "--ee-height-threshold",
        action="append",
        type=float,
        default=[],
        dest="ee_height_thresholds",
        help="audit the first post-grasp row at or above this EE world-Z",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    thresholds = sorted(set(args.ee_height_thresholds))
    result = summarize(args.traces, thresholds)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
