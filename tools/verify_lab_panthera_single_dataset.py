#!/usr/bin/env python3
"""Audit every single-Panthera HDF5, instruction and MP4 before training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np


def _probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream: {path}")
    return streams[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=128)
    parser.add_argument("--expected-schema-version", type=int, default=3)
    parser.add_argument("--expected-scene-profile", default="")
    parser.add_argument("--dataset-name", default="panthera_single_cylinder_sft_v1")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="bounded parallel episode readers/ffprobe workers",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.dataset_root.resolve()
    scene = json.loads((root / "scene_info.json").read_text(encoding="utf-8"))
    if len(scene) != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} scene records")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    def audit_episode(episode_id: int) -> tuple[int, int]:
        hdf5_path = root / "data" / f"episode{episode_id}.hdf5"
        video_path = root / "video" / f"episode{episode_id}.mp4"
        instruction_path = root / "instructions" / f"episode{episode_id}.json"
        for path in (hdf5_path, video_path, instruction_path):
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"missing or empty episode artifact: {path}")

        metadata = scene[f"episode_{episode_id}"]["panthera_episode"]
        if (
            metadata.get("schema_version") != args.expected_schema_version
            or metadata.get("robot_count") != 1
            or metadata.get("action_dimension") != 7
            or metadata.get("attach_on_grasp")
        ):
            raise ValueError(f"episode {episode_id} is not the single-arm schema")
        if (
            args.expected_scene_profile
            and metadata.get("scene_profile") != args.expected_scene_profile
        ):
            raise ValueError(f"episode {episode_id} scene profile mismatch")

        instructions = json.loads(instruction_path.read_text(encoding="utf-8"))
        texts = instructions.get("seen", []) + instructions.get("unseen", [])
        if not texts or any(
            phrase in text.lower()
            for text in texts
            for phrase in ("both arm", "both gripper", "dual-arm", "two arm")
        ):
            raise ValueError(f"episode {episode_id} has invalid task language")

        with h5py.File(hdf5_path, "r") as data:
            actions = np.asarray(data["joint_action/vector"])
            states = np.asarray(data["observation/robot_state/vector"])
            if actions.ndim != 2 or actions.shape[1] != 7 or states.shape != actions.shape:
                raise ValueError(f"episode {episode_id} state/action shape mismatch")
            frame_count = int(actions.shape[0])

        stream = _probe_video(video_path)
        expected_stream = {
            "codec_name": "h264",
            "width": 320,
            "height": 240,
            "r_frame_rate": "30/1",
        }
        for key, expected in expected_stream.items():
            if stream.get(key) != expected:
                raise ValueError(
                    f"episode {episode_id} video {key}={stream.get(key)!r}, expected {expected!r}"
                )
        video_frames = int(stream.get("nb_read_frames", -1))
        if video_frames != frame_count:
            raise ValueError(
                f"episode {episode_id} video/HDF5 mismatch: {video_frames} != {frame_count}"
            )
        return frame_count, video_path.stat().st_size

    # HDF5 handles remain episode-local.  ffprobe is a subprocess, so bounded
    # threads overlap independent disk reads without sharing decoder state.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        episode_results = list(
            executor.map(audit_episode, range(args.expected_episodes))
        )
    frame_counts = [result[0] for result in episode_results]
    video_sizes = [result[1] for result in episode_results]

    summary = {
        "status": "passed",
        "dataset": args.dataset_name,
        "task_schema_version": args.expected_schema_version,
        "scene_profile": args.expected_scene_profile or None,
        "robot_count": 1,
        "action_dimension": 7,
        "episodes": args.expected_episodes,
        "hdf5_count": len(frame_counts),
        "video_count": len(video_sizes),
        "frame_count_min": min(frame_counts),
        "frame_count_max": max(frame_counts),
        "video_bytes_min": min(video_sizes),
        "video_bytes_total": sum(video_sizes),
        "video_codec": "h264",
        "video_size": [320, 240],
        "video_fps": 30,
        "all_video_frame_counts_match_hdf5": True,
        "single_arm_language_only": True,
        "parallel_workers": args.workers,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
