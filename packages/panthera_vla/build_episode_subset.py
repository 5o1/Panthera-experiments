#!/usr/bin/env python3
"""Build a smaller dataset from a larger one, as a dataset in its own right.

Training throughput, not dataset size, is the binding constraint: one epoch over
the 1280-episode set costs about 48 GPU-hours, so a run that fits in a few hours
covers a few percent of it.  A subset buys back epochs at the cost of diversity,
which is the right trade while establishing whether the data can teach the task.

The subset is symlinks plus its own snapshot, so it costs no disk, keeps the
parent's episode numbering, and can be opened by the same access layer as the
parent -- including by the single-trajectory overfit gate, where the subset is
one episode and the validation split is that same episode.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "panthera_sim"))

from dataset import DATASET_FILE, SCENES_FILE, SCENE_INFO_FILE, open_dataset  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--upright", type=int, default=64)
    parser.add_argument("--lying-per-bin", type=int, default=8)
    parser.add_argument("--validation", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--episode", type=int, action="append",
        help="take exactly these episodes and skip the balanced selection; "
             "used by the overfit gate, where train and eval are one trajectory",
    )
    parser.add_argument(
        "--validate-on-train", action="store_true",
        help="validate on the training episodes themselves; an overfit run has "
             "nothing to hold out, and a held-out split would stop training on "
             "a signal that does not apply to the question being asked",
    )
    return parser.parse_args()


def _select(dataset, args: argparse.Namespace) -> list[int]:
    """Greedily balance side and spatial-cell coverage inside each stratum."""
    rng = random.Random(args.seed)
    rows = []
    for episode in dataset.episodes():
        scene = episode.scene
        workspace = scene["workspace"]
        side = (
            "left"
            if scene["cylinder_initial_xy_m"][0] < workspace["robot_base_x_m"]
            else "right"
        )
        rows.append(
            {
                "episode": episode.episode_id,
                "posture": episode.posture,
                "bin": scene.get("lying_angle_bin"),
                "side": side,
                "cells": [
                    (scene.get("cylinder_radial_bin"), scene.get("cylinder_angular_bin")),
                    (scene.get("socket_radial_bin"), scene.get("socket_angular_bin")),
                ],
            }
        )

    chosen: list[dict] = []
    side_counts: Counter = Counter()
    seen_cells: set = set()
    picked: set[int] = set()

    def take(candidates: list[dict]) -> None:
        if not candidates:
            raise SystemExit("stratum is empty; cannot honour the requested balance")
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda row: (
                side_counts[row["side"]],
                -sum(cell not in seen_cells for cell in row["cells"]),
                row["episode"],
            )
        )
        row = candidates[0]
        chosen.append(row)
        side_counts[row["side"]] += 1
        seen_cells.update(row["cells"])
        picked.add(row["episode"])

    for angle_bin in range(8):
        for _ in range(args.lying_per_bin):
            take([r for r in rows if r["posture"] == "lying"
                  and r["bin"] == angle_bin and r["episode"] not in picked])
    for _ in range(args.upright):
        take([r for r in rows if r["posture"] == "upright"
              and r["episode"] not in picked])
    return sorted(row["episode"] for row in chosen)


def _episode_records(dataset, episodes: list[int]) -> dict:
    """Keep the camera/audit record needed to reproduce each selected episode."""
    return {
        f"episode_{episode_id}": {
            "panthera_episode": dict(dataset.episode_record(episode_id))
        }
        for episode_id in episodes
    }


def main() -> int:
    args = _parse_args()
    source = args.source_root.resolve()
    target = args.target_root.resolve()
    dataset = open_dataset(source)

    if args.episode:
        wanted = sorted(set(args.episode))
        known = set(dataset.episode_ids())
        missing = [value for value in wanted if value not in known]
        if missing:
            raise SystemExit(f"episodes not in the source dataset: {missing}")
        episodes = wanted
    else:
        episodes = _select(dataset, args)

    if target.exists():
        raise SystemExit(f"target already exists: {target}")
    (target / "data").mkdir(parents=True)
    (target / "instructions").mkdir()

    scenes = {}
    entries = []
    for episode_id in episodes:
        episode = dataset.episode(episode_id)
        scenes[str(episode.seed)] = dict(episode.scene)
        source_entry = next(
            entry for entry in dataset.manifest["episodes"]
            if int(entry["episode_id"]) == episode_id
        )
        entries.append(dict(source_entry))
        for origin, name in (
            (episode.hdf5_path, f"data/episode{episode_id}.hdf5"),
            (episode.instruction_path, f"instructions/episode{episode_id}.json"),
        ):
            if not origin.exists():
                raise SystemExit(f"missing source file: {origin}")
            os.symlink(origin.resolve(), target / name)

    if args.validate_on_train:
        train, validation = list(episodes), list(episodes)
    else:
        if args.validation >= len(episodes):
            raise SystemExit(
                f"--validation {args.validation} leaves no training episodes out "
                f"of {len(episodes)}; use --validate-on-train for an overfit set"
            )
        step = len(episodes) / args.validation
        validation = sorted(
            {episodes[int(index * step)] for index in range(args.validation)}
        )
        train = [episode for episode in episodes if episode not in validation]

    manifest = dict(dataset.manifest)
    manifest["name"] = target.name
    manifest["episodes"] = entries
    manifest["posture_counts"] = dict(Counter(e["posture"] for e in entries))
    manifest["contract"] = dict(manifest["contract"])
    manifest["contract"]["required_action_budget"] = max(
        (e.get("recorded_actions", 0) for e in dataset.manifest["episodes"]
         if e["episode_id"] in set(episodes)),
        default=manifest["contract"]["required_action_budget"],
    )
    manifest["subset"] = {
        "source_root": str(source),
        "source_digest": dataset.digest(),
        "selection_seed": args.seed,
        "contract": (
            f"explicit episodes {episodes}" if args.episode
            else f"{args.upright} upright plus {args.lying_per_bin} lying per angle "
                 "bin; greedy side/cell balance"
        ),
        "validate_on_train": bool(args.validate_on_train),
        "train": train,
        "val": validation,
    }
    (target / DATASET_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (target / SCENES_FILE).write_text(
        json.dumps({"schema_version": 1, "scenes": scenes}, indent=2) + "\n",
        encoding="utf-8",
    )
    (target / SCENE_INFO_FILE).write_text(
        json.dumps(_episode_records(dataset, episodes), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    subset = open_dataset(target)
    print(f"写入 {target}")
    print(f"  episode {len(episodes)}  姿态 {subset.posture_counts()}")
    print(f"  train {len(train)}  val {len(validation)}"
          f"{'（同一批，过拟合用）' if args.validate_on_train else ''}")
    print(f"  动作预算需求 {subset.contract.required_action_budget}")
    print(f"  源数据集摘要 {dataset.digest()[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
