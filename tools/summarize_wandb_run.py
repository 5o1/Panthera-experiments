#!/usr/bin/env python3
"""Print a compact JSON summary of scalar history in an offline W&B run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def scalar(value: str) -> int | float | bool | str | None:
    """Decode a W&B JSON value and retain only scalar values."""
    try:
        decoded: Any = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, (int, float, bool, str)) else None


def item_key(item: Any) -> str:
    """Return the flat or nested metric key used by the W&B record format."""
    if item.key:
        return item.key
    return ".".join(item.nested_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_file", type=Path)
    parser.add_argument("--tail", type=int, default=8)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="show only history rows containing a VLA validation metric",
    )
    args = parser.parse_args()

    store = DataStore()
    store.open_for_scan(str(args.run_file))
    history: list[dict[str, int | float | bool | str]] = []
    summary: dict[str, int | float | bool | str] = {}
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.HasField("history"):
            row = {
                item_key(item): value
                for item in record.history.item
                if item_key(item) and (value := scalar(item.value_json)) is not None
            }
            if row:
                history.append(row)
        if record.HasField("summary"):
            summary.update(
                {
                    item_key(item): value
                    for item in record.summary.update
                    if item_key(item) and (value := scalar(item.value_json)) is not None
                }
            )

    selected_history = history
    if args.validation_only:
        selected_history = [
            row
            for row in history
            if any(key.startswith("VLA Val/") for key in row)
        ]

    print(
        json.dumps(
            {
                "run_file": str(args.run_file),
                "history_rows": len(history),
                "keys": sorted({key for row in history for key in row}),
                "tail": selected_history[-args.tail :],
                "summary": summary,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
