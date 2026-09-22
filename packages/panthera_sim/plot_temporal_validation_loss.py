#!/usr/bin/env python3
"""Plot deterministic validation L1 along one recorded trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CHUNK_KEYS = (
    ("normalized_chunk_l1", "all 25x7"),
    ("normalized_chunk_joint_l1", "joints 25x6"),
    ("normalized_chunk_gripper_l1", "gripper 25x1"),
)
CURRENT_KEYS = (
    ("normalized_current_l1", "all current"),
    ("normalized_current_joint_l1", "joints current"),
    ("normalized_current_gripper_l1", "gripper current"),
)


def _window_summary(records: list[dict], transitions: list[dict], chunk: int) -> dict:
    rows = np.asarray([row["row"] for row in records], dtype=np.int64)
    result = {}
    for transition in transitions:
        center = int(transition["row"])
        mask = (rows >= center - chunk + 1) & (rows <= center + chunk - 1)
        name = f"{transition['kind']}_row_{center}"
        result[name] = {
            "row_start": int(max(0, center - chunk + 1)),
            "row_end": int(center + chunk - 1),
            **{
                key: {
                    "mean": float(np.mean([records[i][key] for i in np.flatnonzero(mask)])),
                    "max": float(np.max([records[i][key] for i in np.flatnonzero(mask)])),
                }
                for key, _ in CHUNK_KEYS + CURRENT_KEYS
            },
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--title", default="temporal validation loss")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    validation = report["cases"][0]["offline_validation"]
    records = validation.get("per_frame")
    if not records:
        raise SystemExit("report has no offline_validation.per_frame trace")
    transitions = validation.get("expert_gripper_transitions", [])
    action_chunk = int(validation["action_chunk"])
    rows = np.asarray([record["row"] for record in records])

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    figure, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for key, label in CHUNK_KEYS:
        axes[0].plot(rows, [record[key] for record in records], label=label, linewidth=1.0)
    axes[0].set_ylabel("normalized L1")
    axes[0].set_title("25-step action chunk")
    axes[0].legend(loc="upper right", ncol=3)

    for key, label in CURRENT_KEYS:
        axes[1].plot(rows, [record[key] for record in records], label=label, linewidth=1.0)
    axes[1].set_ylabel("normalized L1")
    axes[1].set_title("current action")
    axes[1].legend(loc="upper right", ncol=3)

    axes[2].plot(
        rows, [record["target_current_gripper"] for record in records],
        label="expert gripper", linewidth=1.5,
    )
    axes[2].plot(
        rows, [record["predicted_current_gripper"] for record in records],
        label="predicted gripper", linewidth=1.0,
    )
    axes[2].set_ylabel("physical opening")
    axes[2].set_xlabel("trajectory time step (50 Hz)")
    axes[2].set_title("current gripper target")
    axes[2].legend(loc="upper right")

    colors = {"close": "tab:orange", "open": "tab:red"}
    for transition in transitions:
        row = int(transition["row"])
        kind = str(transition["kind"])
        for axis in axes:
            axis.axvspan(
                max(0, row - action_chunk + 1), row,
                color=colors.get(kind, "gray"), alpha=0.08,
            )
            axis.axvline(row, color=colors.get(kind, "gray"), linestyle="--", linewidth=1.2)
        axes[0].text(row + 3, axes[0].get_ylim()[1] * 0.92, f"{kind} @{row}")

    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle(args.title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)

    summary = {
        "source": str(args.report),
        "frames": len(records),
        "action_chunk": action_chunk,
        "expert_gripper_transitions": transitions,
        "global": {
            key: {
                "mean": float(np.mean([record[key] for record in records])),
                "max": float(np.max([record[key] for record in records])),
            }
            for key, _ in CHUNK_KEYS + CURRENT_KEYS
        },
        "transition_windows": _window_summary(records, transitions, action_chunk),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
