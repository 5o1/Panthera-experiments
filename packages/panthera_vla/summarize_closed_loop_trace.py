#!/usr/bin/env python3
"""Summarize Panthera closed-loop JSONL traces without simulator imports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "x_error_m",
    "y_error_m",
    "height_error_m",
    "axis_error_deg",
    "insertion_depth_m",
    "linear_speed_mps",
    "angular_speed_radps",
    "gripper_contact_impulse_ns",
)


def distance(a: list[float], b: list[float]) -> float:
    """Return Euclidean distance between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def summarize(path: Path, open_threshold: float) -> dict[str, Any]:
    """Extract phase transitions and best/final task metrics from one trace."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"empty trace: {path}")

    positions = [row["after_cylinder_position_m"] for row in rows]
    path_length = distance(rows[0]["before_cylinder_position_m"], positions[0])
    path_length += sum(distance(a, b) for a, b in zip(positions, positions[1:]))
    start_position = rows[0]["before_cylinder_position_m"]
    final_position = positions[-1]

    contact_indices = [
        index
        for index, row in enumerate(rows)
        if row["success_metrics"]["gripper_contact_impulse_ns"] > 1.0e-5
    ]
    release_indices = [
        index
        for index, row in enumerate(rows)
        if index > 0
        and float(row["after_state"][6]) >= open_threshold
        and float(rows[index - 1]["after_state"][6]) < open_threshold
    ]
    closed_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if float(row["after_state"][6]) < open_threshold
    ]

    best_metrics = {}
    for key in METRIC_KEYS:
        values = [float(row["success_metrics"][key]) for row in rows]
        if key == "insertion_depth_m":
            index = max(range(len(values)), key=values.__getitem__)
        else:
            index = min(range(len(values)), key=values.__getitem__)
        best_metrics[key] = {"value": values[index], "chunk": index}

    best_closed_alignment = None
    if closed_rows:
        index, row = min(
            closed_rows,
            key=lambda item: math.hypot(
                item[1]["success_metrics"]["x_error_m"],
                item[1]["success_metrics"]["y_error_m"],
            ),
        )
        best_closed_alignment = {
            "chunk": index,
            "xy_error_m": math.hypot(
                row["success_metrics"]["x_error_m"],
                row["success_metrics"]["y_error_m"],
            ),
            "metrics": row["success_metrics"],
        }

    near_socket_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if math.hypot(
            row["success_metrics"]["x_error_m"],
            row["success_metrics"]["y_error_m"],
        )
        <= 0.020
    ]
    best_near_socket_height = None
    if near_socket_rows:
        index, row = min(
            near_socket_rows,
            key=lambda item: item[1]["success_metrics"]["height_error_m"],
        )
        best_near_socket_height = {
            "chunk": index,
            "metrics": row["success_metrics"],
        }

    return {
        "file": str(path),
        "episode_seed": int(rows[0]["episode_seed"]),
        "chunks": len(rows),
        "object_net_displacement_m": distance(start_position, final_position),
        "object_path_length_m": path_length,
        "contact_chunk_range": (
            [contact_indices[0], contact_indices[-1]] if contact_indices else None
        ),
        "release_transition_chunks": release_indices,
        "best_metrics": best_metrics,
        "best_closed_alignment": best_closed_alignment,
        "best_near_socket_height": best_near_socket_height,
        "final_metrics": rows[-1]["success_metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--open-threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.open_threshold <= 1.0:
        parser.error("--open-threshold must be in [0, 1]")

    result = {
        "open_threshold": args.open_threshold,
        "traces": [summarize(path, args.open_threshold) for path in args.traces],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
