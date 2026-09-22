#!/usr/bin/env python3
"""Render a recorded command stream back into video.

A rollout is judged from a success flag and a handful of scalars, and twice in
the 2026-09-19 investigation those scalars hid what one look at the image made
obvious -- the evaluation camera was 47.6 cm from the recorded one, pointing
from the other side of the table (``docs/20``).  Replaying the trace is exact:
the commands are the same, the scene is restored from the dataset, so the video
is the rollout rather than an impression of it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np

from dataset import open_dataset
from executor import DenseExecutor, ExecutorConfig
from robotwin_env import build_task, inside


def label(frame: np.ndarray, lines) -> np.ndarray:
    import cv2

    out = np.ascontiguousarray(frame)
    for row, text in enumerate(lines):
        y = 14 + row * 14
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def targets_from(source: Path, dataset, episode_id: int):
    if source.suffix == ".json":
        report = json.loads(source.read_text(encoding="utf-8"))
        case = report["cases"][0]
        trace = case["trace"]
        if not trace:
            raise SystemExit(f"{source} has no trace; rerun that rollout with --trace")
        targets = np.asarray(
            [list(row["arm_cmd"]) + [row["gripper_cmd"]] for row in trace],
            dtype=np.float64,
        )
        return targets, report, case
    from replay import grid_actions

    targets = np.asarray(grid_actions(dataset.episode(episode_id).hdf5_path), dtype=np.float64)
    return targets, None, None


def inference_labels(case: dict | None, actions: int, control_period_ms: float) -> list[list[str]]:
    """Map each executed action to latency labels from the query that produced it."""
    labels: list[list[str]] = [[] for _ in range(actions)]
    if case is None:
        return labels
    queries = case.get("prediction_trace", [])
    if not queries or any("inference_ms" not in query for query in queries):
        return labels
    starts = [int(query["row"]) for query in queries]
    ends = starts[1:] + [actions]
    for query, start, end in zip(queries, starts, ends):
        span = max(1, end - start)
        query_ms = float(query["inference_ms"])
        action_ms = query_ms / span
        load = action_ms / control_period_ms
        status = "OK" if load <= 1.0 else "LATE"
        lines = [
            f"policy {query_ms:.1f} ms/query | {action_ms:.1f} ms/action",
            f"50Hz load {load:.2f}x {status} | budget {control_period_ms:.1f}ms",
            "frame-ready real-table lower bound",
            "sim render / physics / video excluded",
        ]
        for row in range(max(0, start), min(actions, end)):
            labels[row] = lines
    return labels


def render(args) -> int:
    dataset = open_dataset(Path(args.dataset_root))
    episode = dataset.episode(args.episode)
    targets, report, case = targets_from(Path(args.source), dataset, args.episode)
    control_period_ms = 1000.0 / dataset.contract.control_hz
    timing_labels = inference_labels(case, len(targets), control_period_ms)
    root = Path(args.robotwin_root)
    frames = []
    with inside(root):
        task = None
        try:
            task, _ = build_task(
                root, args.task_config, args.task_name, episode, dataset.scenes_path,
                len(targets) + 16, None,
                camera=dataset.camera(args.episode),
                camera_pose=dataset.camera_pose(args.episode),
            )
            executor = DenseExecutor(task, ExecutorConfig())
            for index, target in enumerate(targets):
                executor.step(target)
                if index % args.stride:
                    continue
                picture = np.asarray(
                    task.get_obs()["observation"]["head_camera"]["rgb"], dtype=np.uint8
                )
                cylinder = task.cylinder.get_pose()
                frames.append(label(picture, [
                    args.title,
                    *args.annotation,
                    *timing_labels[index],
                    f"step {index}/{len(targets)}  grip {target[6]:.2f}",
                    f"cyl z {cylinder.p[2] * 100:.1f}cm",
                ]))
        finally:
            if task is not None:
                with contextlib.suppress(Exception):
                    task.close_env()
    # RoboTwin ships this writer, and its ``is_rgb`` argument is the channel
    # convention written down.  Hand-rolling an encoder here is how the same
    # convention was lost on the read side (see ``parity.recorded_frames``).
    with inside(root):
        from envs.utils.images_to_video import images_to_video

        images_to_video(np.stack(frames), str(Path(args.output).resolve()),
                        fps=float(args.fps), is_rgb=True)
    print(f"{args.output}  {len(frames)} 帧  来自 {len(targets)} 步")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-name", default="place_randomized_cylinder_in_socket")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--source", required=True,
        help="a rollout json written with --trace, or 'expert' for the recording",
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--annotation", action="append", default=[])
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--output", required=True)
    return render(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
