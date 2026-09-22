#!/usr/bin/env python3
"""Reconstruct newest and temporal-ensemble gripper commands from a rollout trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def reconstruct(
    prediction_trace: list[dict],
    executed_actions: int,
    *,
    coefficient: float = 0.0,
    history_horizon: int = 64,
    threshold: float = 0.5,
) -> dict:
    """Replay :class:`rollout.TemporalEnsemble` for the gripper dimension."""
    if coefficient < 0 or history_horizon <= 0 or executed_actions <= 0:
        raise ValueError("coefficient must be nonnegative and horizons must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    queries: dict[int, np.ndarray] = {}
    for query in prediction_trace:
        row = int(query["row"])
        chunk = np.asarray(query["actions"], dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise ValueError(f"trace row {row} has action shape {chunk.shape}, expected [N, 7]")
        if not np.all(np.isfinite(chunk)):
            raise ValueError(f"trace row {row} contains NaN or infinity")
        if row in queries:
            raise ValueError(f"duplicate prediction query at row {row}")
        queries[row] = chunk

    active: list[tuple[int, np.ndarray]] = []
    rows = []
    for row in range(executed_actions):
        if row in queries:
            active.append((row, queries[row]))
            if len(active) > history_horizon:
                del active[: len(active) - history_horizon]

        covering = []
        for age, (start, chunk) in enumerate(reversed(active)):
            offset = row - start
            if 0 <= offset < len(chunk):
                covering.append(
                    {
                        "start": start,
                        "offset": offset,
                        "age": age,
                        "value": float(chunk[offset, 6]),
                        "weight": math.exp(-coefficient * age),
                    }
                )
        if not covering:
            continue
        weights = np.asarray([item["weight"] for item in covering], dtype=np.float64)
        values = np.asarray([item["value"] for item in covering], dtype=np.float64)
        newest = covering[0]["value"]
        ensemble = float(values @ (weights / weights.sum()))
        rows.append(
            {
                "row": row,
                "covering_predictions": len(covering),
                "newest": newest,
                "ensemble": ensemble,
                "newest_open": bool(newest >= threshold),
                "ensemble_open": bool(ensemble >= threshold),
            }
        )

    newest_open_rows = [item["row"] for item in rows if item["newest_open"]]
    ensemble_open_rows = [item["row"] for item in rows if item["ensemble_open"]]
    suppressed = [
        item["row"]
        for item in rows
        if item["newest_open"] and not item["ensemble_open"]
    ]
    return {
        "coefficient": coefficient,
        "history_horizon": history_horizon,
        "threshold": threshold,
        "executed_actions": executed_actions,
        "reconstructed_rows": len(rows),
        "first_newest_open_row": newest_open_rows[0] if newest_open_rows else None,
        "first_ensemble_open_row": ensemble_open_rows[0] if ensemble_open_rows else None,
        "newest_open_rows": len(newest_open_rows),
        "ensemble_open_rows": len(ensemble_open_rows),
        "suppressed_open_rows": len(suppressed),
        "first_suppressed_open_row": suppressed[0] if suppressed else None,
        "raw_open_was_suppressed": bool(suppressed),
        "rows": rows,
    }


def _case(payload: dict, episode: int | None) -> dict:
    if "prediction_trace" in payload:
        return payload
    cases = payload.get("cases", [])
    if episode is None and len(cases) == 1:
        return cases[0]
    matches = [case for case in cases if int(case.get("episode", -1)) == episode]
    if len(matches) != 1:
        raise ValueError("select exactly one case with --episode")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--coefficient", type=float, default=0.0)
    parser.add_argument("--history-horizon", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    payload = json.loads(cli.trace.read_text(encoding="utf-8"))
    case = _case(payload, cli.episode)
    report = reconstruct(
        case["prediction_trace"],
        int(case["executed_actions"]),
        coefficient=cli.coefficient,
        history_horizon=cli.history_horizon,
        threshold=cli.threshold,
    )
    report["episode"] = case.get("episode")
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "newest first-open:", report["first_newest_open_row"],
        "ensemble first-open:", report["first_ensemble_open_row"],
        "suppressed rows:", report["suppressed_open_rows"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
