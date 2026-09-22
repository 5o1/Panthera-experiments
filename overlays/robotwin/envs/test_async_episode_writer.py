#!/usr/bin/env python3
"""Integration test for bounded, atomic asynchronous episode conversion."""

from __future__ import annotations

import pickle
from pathlib import Path
import tempfile

import h5py
import numpy as np

from envs.utils.async_episode_writer import AsyncEpisodeWriter, episode_is_committed


def write_cache(root: Path, episode_id: int, valid: bool = True) -> Path:
    cache = root / ".cache" / f"episode{episode_id}"
    cache.mkdir(parents=True)
    for frame in range(4):
        if valid:
            item = {
                "observation": {
                    "head_camera": {
                        "rgb": np.full((48, 64, 3), frame * 30, dtype=np.uint8)
                    },
                    "robot_state": {
                        "vector": np.full(7, frame, dtype=np.float32),
                    },
                },
                "joint_action": {
                    "vector": np.full(7, frame + 1, dtype=np.float32),
                },
            }
        else:
            item = {"malformed": np.asarray([frame])}
        with (cache / f"{frame}.pkl").open("wb") as handle:
            pickle.dump(item, handle)
    return cache


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="panthera-async-writer-") as temp:
        root = Path(temp)
        writer = AsyncEpisodeWriter(
            str(root), workers=1, max_pending_episodes=1, max_pending_bytes=64 * 1024**2
        )
        writer.submit(0, str(write_cache(root, 0)))
        writer.submit(1, str(write_cache(root, 1)))
        writer.close()
        for episode_id in (0, 1):
            assert episode_is_committed(str(root), episode_id)
            assert not (root / ".cache" / f"episode{episode_id}").exists()
            with h5py.File(root / "data" / f"episode{episode_id}.hdf5", "r") as data:
                assert data["joint_action/vector"].shape == (4, 7)

        failed_root = root / "failed"
        failed_writer = AsyncEpisodeWriter(
            str(failed_root),
            workers=1,
            max_pending_episodes=1,
            max_pending_bytes=64 * 1024**2,
        )
        failed_writer.submit(2, str(write_cache(failed_root, 2, valid=False)))
        try:
            failed_writer.close()
        except (KeyError, RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("writer failure was not propagated")
        assert not episode_is_committed(str(failed_root), 2)
        assert not list((failed_root / "data").glob("*.partial.hdf5"))
        assert not list((failed_root / "video").glob("*.partial.mp4"))
    print("async episode writer acceptance passed: bounded queue, atomic commit, failure propagation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
