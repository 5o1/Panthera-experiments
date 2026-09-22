"""Bounded process-pool writer for RoboTwin episode caches.

Simulation remains in the collector process.  Only complete, immutable episode
cache directories cross the process boundary.  A completion marker is written
after both HDF5 and MP4 outputs have been validated and atomically committed.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import shutil
from typing import Deque

import cv2
import h5py

from .pkl2hdf5 import process_folder_to_hdf5_video


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def _validate_outputs(hdf5_path: Path, video_path: Path) -> int:
    if not hdf5_path.is_file() or hdf5_path.stat().st_size == 0:
        raise RuntimeError(f"empty HDF5 output: {hdf5_path}")
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"empty video output: {video_path}")
    with h5py.File(hdf5_path, "r") as episode:
        actions = episode["joint_action/vector"]
        rgb = episode["observation/head_camera/rgb"]
        frame_count = int(actions.shape[0])
        if frame_count <= 0 or int(rgb.shape[0]) != frame_count:
            raise RuntimeError("HDF5 action/RGB frame count mismatch")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot decode video output: {video_path}")
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if video_frames != frame_count:
        raise RuntimeError(
            f"video/HDF5 frame count mismatch: video={video_frames}, hdf5={frame_count}"
        )
    return frame_count


def _write_episode_job(
    episode_id: int,
    cache_path_text: str,
    output_root_text: str,
) -> dict:
    cache_path = Path(cache_path_text)
    output_root = Path(output_root_text)
    data_root = output_root / "data"
    video_root = output_root / "video"
    commit_root = output_root / ".episode_commits"
    data_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    commit_root.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{episode_id}"
    partial_hdf5 = data_root / f".episode{episode_id}.{token}.partial.hdf5"
    partial_video = video_root / f".episode{episode_id}.{token}.partial.mp4"
    final_hdf5 = data_root / f"episode{episode_id}.hdf5"
    final_video = video_root / f"episode{episode_id}.mp4"
    final_marker = commit_root / f"episode{episode_id}.json"
    partial_marker = commit_root / f".episode{episode_id}.{token}.partial.json"
    for path in (partial_hdf5, partial_video, partial_marker):
        path.unlink(missing_ok=True)
    try:
        process_folder_to_hdf5_video(
            str(cache_path), str(partial_hdf5), str(partial_video)
        )
        frame_count = _validate_outputs(partial_hdf5, partial_video)
        os.replace(partial_hdf5, final_hdf5)
        os.replace(partial_video, final_video)
        marker = {
            "schema_version": 1,
            "episode": episode_id,
            "frames": frame_count,
            "hdf5_bytes": final_hdf5.stat().st_size,
            "video_bytes": final_video.stat().st_size,
            "writer_pid": os.getpid(),
            "committed_at": datetime.now(timezone.utc).isoformat(),
        }
        partial_marker.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        os.replace(partial_marker, final_marker)
        shutil.rmtree(cache_path)
        return marker
    except BaseException:
        for path in (partial_hdf5, partial_video, partial_marker):
            path.unlink(missing_ok=True)
        raise


def episode_is_committed(output_root: str, episode_id: int) -> bool:
    root = Path(output_root)
    marker_path = root / ".episode_commits" / f"episode{episode_id}.json"
    hdf5_path = root / "data" / f"episode{episode_id}.hdf5"
    video_path = root / "video" / f"episode{episode_id}.mp4"
    if not marker_path.is_file() or not hdf5_path.is_file() or not video_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        return (
            int(marker["episode"]) == episode_id
            and int(marker["hdf5_bytes"]) == hdf5_path.stat().st_size
            and int(marker["video_bytes"]) == video_path.stat().st_size
            and int(marker["frames"]) > 0
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


@dataclass
class _Pending:
    episode_id: int
    cache_bytes: int
    future: Future


class AsyncEpisodeWriter:
    """Bounded asynchronous episode converter with exception propagation."""

    def __init__(
        self,
        output_root: str,
        workers: int = 1,
        max_pending_episodes: int = 2,
        max_pending_bytes: int = 2 * 1024**3,
    ) -> None:
        if workers <= 0 or max_pending_episodes <= 0 or max_pending_bytes <= 0:
            raise ValueError("writer bounds must be positive")
        self.output_root = str(Path(output_root))
        self.max_pending_episodes = max_pending_episodes
        self.max_pending_bytes = max_pending_bytes
        self.pending_bytes = 0
        self.pending: Deque[_Pending] = deque()
        self.executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self.closed = False

    def _finish_oldest(self) -> dict:
        pending = self.pending.popleft()
        try:
            result = pending.future.result()
        finally:
            self.pending_bytes -= pending.cache_bytes
        print(
            f"async writer committed episode {pending.episode_id} "
            f"({result['frames']} frames)",
            flush=True,
        )
        return result

    def poll(self) -> None:
        while self.pending and self.pending[0].future.done():
            self._finish_oldest()

    def submit(self, episode_id: int, cache_path: str) -> None:
        if self.closed:
            raise RuntimeError("async writer is closed")
        path = Path(cache_path)
        cache_bytes = _directory_size(path)
        if cache_bytes <= 0:
            raise RuntimeError(f"episode cache is empty: {path}")
        while self.pending and (
            len(self.pending) >= self.max_pending_episodes
            or self.pending_bytes + cache_bytes > self.max_pending_bytes
        ):
            self._finish_oldest()
        if cache_bytes > self.max_pending_bytes:
            raise RuntimeError(
                f"episode cache ({cache_bytes} bytes) exceeds writer byte bound"
            )
        future = self.executor.submit(
            _write_episode_job,
            episode_id,
            str(path),
            self.output_root,
        )
        self.pending.append(_Pending(episode_id, cache_bytes, future))
        self.pending_bytes += cache_bytes

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            while self.pending:
                self._finish_oldest()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "AsyncEpisodeWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
