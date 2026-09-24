#!/usr/bin/env python3
"""Export C0/C1/C2 RLinf TensorBoard scalars to CSV, JSON, and PNG.

RLinf's terminal table is asynchronous and W&B is intentionally offline, so
the final console summary only exposes the last update.  TensorBoard events are
the authoritative per-update record.  This utility preserves the complete
learning curve and, when available, the independent checkpoint-evaluation
metrics in ordinary reviewable files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from tensorboard.backend.event_processing import event_accumulator


VARIANTS = ("c0", "c1", "c2")
EVAL_VARIANTS = ("r0", *VARIANTS)
TRAIN_TAGS = (
    "env/return",
    "env/reward",
    "env/success_once",
    "rollout/rewards",
    "train/actor/policy_loss",
    "train/actor/total_loss",
    "train/actor/grad_norm",
    "train/actor/chunk_overlap_loss",
    "train/actor/auxiliary_loss",
    "train/actor/approx_kl",
    "train/critic/value_loss",
)
EVAL_TAGS = (
    "eval/episode_len",
    "eval/num_trajectories",
    "eval/return",
    "eval/reward",
    "eval/success_once",
    "eval/panthera_grasped_once",
    "eval/panthera_target_grasped_once",
    "eval/panthera_valid_release_once",
    "eval/panthera_invalid_release_once",
    "eval/panthera_hard_failure_once",
    "eval/panthera_target_still_grasped_at_end",
    "eval/panthera_valid_release_at_end",
    "eval/panthera_release_delay_actions",
    "eval/panthera_release_delay_censored",
    "eval/panthera_arm_delta_l1_mean",
    "eval/panthera_arm_second_difference_l1_mean",
    "eval/panthera_arm_second_difference_l1_max",
    "eval/panthera_chunk_boundary_second_difference_l1_mean",
    "eval/panthera_chunk_boundary_second_difference_l1_max",
    "eval/panthera_gripper_delta_l1_mean",
    "eval/panthera_target_xy_error_m",
    "eval/panthera_target_height_error_m",
    "eval/panthera_target_axis_error_deg",
)


def _event_files(root: Path) -> list[Path]:
    return sorted(root.glob("tensorboard/events.out.tfevents.*"))


def _read_scalars(root: Path) -> dict[str, dict[int, float]]:
    """Merge event files, keeping the newest wall-time value per tag/step."""
    newest: dict[tuple[str, int], tuple[float, float]] = {}
    for path in _event_files(root):
        accumulator = event_accumulator.EventAccumulator(
            str(path), size_guidance={event_accumulator.SCALARS: 0}
        )
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                key = (tag, int(event.step))
                candidate = (float(event.wall_time), float(event.value))
                if key not in newest or candidate[0] >= newest[key][0]:
                    newest[key] = candidate
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for (tag, step), (_, value) in newest.items():
        result[tag][step] = value
    return dict(result)


def _variant_rows(
    root: Path,
    variants: Iterable[str],
    *,
    tags: Iterable[str],
    allow_incomplete: bool,
) -> tuple[list[dict[str, float | int | str]], dict[str, dict[str, dict[int, float]]]]:
    rows: list[dict[str, float | int | str]] = []
    all_scalars: dict[str, dict[str, dict[int, float]]] = {}
    for variant in variants:
        variant_root = root / variant
        files = _event_files(variant_root)
        if not files:
            if allow_incomplete:
                continue
            raise FileNotFoundError(f"no TensorBoard event for {variant}: {variant_root}")
        scalars = _read_scalars(variant_root)
        all_scalars[variant] = scalars
        steps = sorted({step for values in scalars.values() for step in values})
        for step in steps:
            row: dict[str, float | int | str] = {
                "variant": variant,
                "raw_step": step,
                "update": step + 1,
            }
            for tag in tags:
                if step in scalars.get(tag, {}):
                    row[tag] = scalars[tag][step]
            rows.append(row)
    return rows, all_scalars


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["variant", "raw_step", "update"]
    extras = sorted({key for row in rows for key in row if key not in fields})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + extras)
        writer.writeheader()
        writer.writerows(rows)


def _summary(scalars: dict[str, dict[str, dict[int, float]]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for variant, tags in scalars.items():
        variant_summary: dict[str, object] = {}
        for tag, values_by_step in sorted(tags.items()):
            if not values_by_step:
                continue
            ordered = sorted(values_by_step.items())
            finite = [value for _, value in ordered if math.isfinite(value)]
            variant_summary[tag] = {
                "count": len(ordered),
                "latest_step": ordered[-1][0],
                "latest": ordered[-1][1],
                "minimum": min(finite) if finite else None,
                "maximum": max(finite) if finite else None,
                "mean": sum(finite) / len(finite) if finite else None,
                "all_finite": len(finite) == len(ordered),
            }
        result[variant] = variant_summary
    return result


def _plot(
    path: Path,
    scalars: dict[str, dict[str, dict[int, float]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("env/return", "Environment return"),
        ("env/success_once", "Stable success"),
        ("train/actor/policy_loss", "PPO policy loss"),
        ("train/actor/grad_norm", "Actor gradient norm"),
        ("train/actor/chunk_overlap_loss", "Chunk overlap loss"),
        ("train/critic/value_loss", "Critic value loss"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (tag, title) in zip(axes.flat, panels, strict=True):
        for variant in VARIANTS:
            values = scalars.get(variant, {}).get(tag, {})
            if not values:
                continue
            ordered = sorted(values.items())
            axis.plot(
                [step + 1 for step, _ in ordered],
                [value for _, value in ordered],
                marker="o",
                label=variant.upper(),
            )
        axis.set_title(title)
        axis.set_xlabel("PPO update")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.suptitle("Panthera R2 release-arena short-run comparison")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_evaluation(
    path: Path,
    scalars: dict[str, dict[str, dict[int, float]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("eval/success_once", "Stable success"),
        ("eval/return", "Environment return"),
        ("eval/reward", "Mean action reward"),
        ("eval/episode_len", "Episode length"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, (tag, title) in zip(axes.flat, panels, strict=True):
        labels: list[str] = []
        values: list[float] = []
        for variant in EVAL_VARIANTS:
            by_step = scalars.get(variant, {}).get(tag, {})
            if not by_step:
                continue
            labels.append(variant.upper())
            values.append(sorted(by_step.items())[-1][1])
        axis.bar(labels, values)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("Panthera deterministic step-10 release-arena evaluation")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_evaluation_panels(
    path: Path,
    scalars: dict[str, dict[str, dict[int, float]]],
    panels: tuple[tuple[str, str], ...],
    *,
    title: str,
) -> None:
    """Plot final deterministic-evaluation scalars for every available variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 4
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 3.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (tag, panel_title) in zip(axes.flat, panels):
        labels: list[str] = []
        values: list[float] = []
        for variant in EVAL_VARIANTS:
            by_step = scalars.get(variant, {}).get(tag, {})
            if not by_step:
                continue
            labels.append(variant.upper())
            values.append(sorted(by_step.items())[-1][1])
        axis.bar(labels, values)
        axis.set_title(panel_title)
        axis.grid(True, axis="y", alpha=0.3)
    for axis in tuple(axes.flat)[len(panels):]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output = (args.output_dir or run_root / "summary").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, scalars = _variant_rows(
        run_root,
        VARIANTS,
        tags=TRAIN_TAGS,
        allow_incomplete=args.allow_incomplete,
    )
    if not rows:
        raise RuntimeError(f"no scalar rows found below {run_root}")
    _write_csv(output / "training_metrics.csv", rows)
    (output / "training_summary.json").write_text(
        json.dumps(_summary(scalars), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(output / "training_curves.png", scalars)
    if args.eval_root is not None:
        eval_rows, eval_scalars = _variant_rows(
            args.eval_root.resolve(),
            EVAL_VARIANTS,
            tags=EVAL_TAGS,
            allow_incomplete=args.allow_incomplete,
        )
        if not eval_rows:
            raise RuntimeError(f"no evaluation scalar rows found below {args.eval_root}")
        _write_csv(output / "evaluation_metrics.csv", eval_rows)
        (output / "evaluation_summary.json").write_text(
            json.dumps(_summary(eval_scalars), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _plot_evaluation(output / "evaluation_comparison.png", eval_scalars)
        _plot_evaluation_panels(
            output / "evaluation_failure_modes.png",
            eval_scalars,
            (
                ("eval/success_once", "Stable success"),
                ("eval/panthera_grasped_once", "Reliable grasp reached"),
                ("eval/panthera_target_grasped_once", "Target reached while held"),
                ("eval/panthera_valid_release_once", "Valid release reached"),
                ("eval/panthera_invalid_release_once", "Invalid release / loss"),
                (
                    "eval/panthera_target_still_grasped_at_end",
                    "At target, still held at end",
                ),
                ("eval/panthera_hard_failure_once", "Hard failure"),
                ("eval/panthera_release_delay_actions", "Release delay (actions)"),
            ),
            title="Panthera release-arena outcome decomposition",
        )
        _plot_evaluation_panels(
            output / "evaluation_motion_quality.png",
            eval_scalars,
            (
                ("eval/panthera_arm_delta_l1_mean", "Arm target delta mean"),
                (
                    "eval/panthera_arm_second_difference_l1_mean",
                    "Arm second difference mean",
                ),
                (
                    "eval/panthera_arm_second_difference_l1_max",
                    "Arm second difference max",
                ),
                (
                    "eval/panthera_chunk_boundary_second_difference_l1_mean",
                    "Boundary second difference mean",
                ),
                (
                    "eval/panthera_chunk_boundary_second_difference_l1_max",
                    "Boundary second difference max",
                ),
                ("eval/panthera_gripper_delta_l1_mean", "Gripper target delta mean"),
                ("eval/panthera_target_xy_error_m", "Final target XY error (m)"),
                (
                    "eval/panthera_target_axis_error_deg",
                    "Final target axis error (deg)",
                ),
            ),
            title="Panthera release-arena motion and final alignment",
        )
    print(f"wrote {len(rows)} per-update rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
