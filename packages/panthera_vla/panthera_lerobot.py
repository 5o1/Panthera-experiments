#!/usr/bin/env python3
"""Convert a snapshotted Panthera dataset to local LeRobot format.

This reuses :mod:`panthera_rlds` for the 50 Hz grid, next-action alignment and
RGB repair.  Consequently OpenVLA-OFT and π0.5 see the same observations and
physical action labels; only their model-specific transforms differ.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping

import numpy as np
from PIL import Image

_SIM_PACKAGE = Path(__file__).resolve().parents[1] / "panthera_sim"
if str(_SIM_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_SIM_PACKAGE))

from dataset import open_dataset


FEATURES = {
    "image": {
        "dtype": "image",
        "shape": (240, 320, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
    },
    "actions": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
    },
}


def _decode_rgb(encoded: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def lerobot_frame(step: Mapping[str, object]) -> dict:
    """Map one already-aligned RLDS-style step to LeRobot feature names."""
    observation = step["observation"]
    if not isinstance(observation, Mapping):
        raise ValueError("step observation must be a mapping")
    image = _decode_rgb(observation["image"])
    state = np.asarray(observation["state"], dtype=np.float32)
    action = np.asarray(step["action"], dtype=np.float32)
    if image.shape != FEATURES["image"]["shape"]:
        raise ValueError(f"unexpected image shape: {image.shape}")
    if state.shape != (7,) or action.shape != (7,):
        raise ValueError(f"expected 7-D state/action, got {state.shape}/{action.shape}")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError("state/action contains NaN or infinity")
    instruction = step["language_instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("step has no language instruction")
    return {
        "image": image,
        "state": state,
        "actions": action,
        "task": instruction,
    }


def convert(
    source_root: Path,
    output_root: Path,
    repo_id: str,
    *,
    episode_ids: Iterable[int] | None = None,
    image_writer_threads: int = 8,
    image_writer_processes: int = 4,
) -> dict:
    """Create a non-overwriting LeRobot dataset and return its provenance."""
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite LeRobot output: {output_root}; choose a new path"
        )
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must look like 'owner/dataset'")

    source = open_dataset(source_root)
    if source.contract.action_dim != 7:
        raise ValueError(f"source action dimension is {source.contract.action_dim}, expected 7")
    fps = source.contract.control_hz
    if not np.isclose(fps, round(fps)):
        raise ValueError(f"LeRobot requires integral fps, got {fps}")

    # The RLDS module owns the temporal alignment.  Its constants are normally
    # configured by the OpenVLA launcher; derive the same values from this
    # dataset snapshot so a standalone LeRobot conversion cannot silently fall
    # back to the historical 5-step/schema-3 defaults.
    schema_version = source.task_config.get("schema_version")
    if schema_version is None:
        raise ValueError("source snapshot has no task schema_version")
    os.environ["PANTHERA_ACTION_CHUNK"] = str(source.contract.action_chunk)
    os.environ["PANTHERA_RLDS_SCHEMA_VERSION"] = str(schema_version)
    scene_profile = source.task_config.get("scene_profile")
    if scene_profile:
        os.environ["PANTHERA_RLDS_SCENE_PROFILE"] = str(scene_profile)

    # Import it only for a real
    # conversion so light-weight contract tests do not need TensorFlow/TFDS.
    from panthera_rlds import discover_episodes, iter_aligned_steps

    episodes = discover_episodes(source_root)
    selected = None if episode_ids is None else {int(value) for value in episode_ids}
    if selected is not None:
        available = {episode.episode_id for episode in episodes}
        missing = selected - available
        if missing:
            raise ValueError(f"episodes not found: {sorted(missing)}")
        episodes = [episode for episode in episodes if episode.episode_id in selected]
    if not episodes:
        raise ValueError("no episodes selected")

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "LeRobot is not installed; run this converter in the pinned OpenPI environment"
        ) from error

    writer = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        robot_type="panthera",
        fps=int(round(fps)),
        features=FEATURES,
        image_writer_threads=int(image_writer_threads),
        image_writer_processes=int(image_writer_processes),
    )
    total_frames = 0
    mapping = []
    for lerobot_index, episode in enumerate(episodes):
        episode_frames = 0
        for step in iter_aligned_steps(episode):
            writer.add_frame(lerobot_frame(step))
            episode_frames += 1
        writer.save_episode()
        total_frames += episode_frames
        mapping.append(
            {
                "lerobot_episode_index": lerobot_index,
                "panthera_episode_id": episode.episode_id,
                "frames": episode_frames,
            }
        )
    finalize = getattr(writer, "finalize", None)
    if callable(finalize):
        finalize()

    provenance = {
        "schema_version": 1,
        "source_root": str(source_root),
        "source_name": source.name,
        "source_digest": source.digest(),
        "repo_id": repo_id,
        "fps": int(round(fps)),
        "episodes": len(episodes),
        "frames": total_frames,
        "action_dim": 7,
        "action_horizon": source.contract.action_chunk,
        "action_semantics": "absolute_joint_position_plus_absolute_gripper",
        "alignment": "observation_at_t_to_next_50hz_drive_target",
        "camera_views": ["image"],
        "episode_mapping": mapping,
    }
    (output_root / "panthera_conversion.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episode", type=int, action="append")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--image-writer-processes", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    report = convert(
        cli.source_root,
        cli.output_root,
        cli.repo_id,
        episode_ids=cli.episode,
        image_writer_threads=cli.image_writer_threads,
        image_writer_processes=cli.image_writer_processes,
    )
    print(
        f"LeRobot conversion complete: {report['episodes']} episodes, "
        f"{report['frames']} frames -> {cli.output_root.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
