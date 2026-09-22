#!/usr/bin/env python3
"""Compare a trained OpenVLA-OFT L1 head against the copy-proprio baseline.

Absolute-joint supervision lets a policy score well by echoing its own proprio
input, because consecutive 50 Hz targets barely move.  Chunk L1 alone therefore
cannot distinguish a policy that predicts motion from one that predicts "stay
put".  For every sampled observation this probe reports, per chunk position:

  model_l1  mean |predicted - target|
  copy_l1   mean |proprio - target|, the score of echoing proprio forever
  skill     1 - model_l1 / copy_l1; 0 means no better than echoing, 1 is exact
  ratio     |predicted - proprio| / |target - proprio|, predicted motion size
  cosine    direction agreement between predicted and true displacement

Arm joints and the gripper are reported separately because they carry different
units and the gripper otherwise dominates an averaged L1.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


SAMPLE_PERIOD_STEPS = 5
ACTION_DIMENSION = 7
ARM_DIMENSIONS = 6
REPORT_POSITIONS = (1, 5, 15, 25)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-chunk", type=int, default=25)
    parser.add_argument("--samples-per-episode", type=int, default=8)
    return parser.parse_args()


def _grid_indices(simulation_steps: np.ndarray, action_chunk: int) -> np.ndarray:
    selected: dict[int, int] = {}
    for index, step in enumerate(simulation_steps.tolist()):
        if int(step) % SAMPLE_PERIOD_STEPS == 0:
            selected[int(step)] = index
    ordered_steps = sorted(selected)
    if len(ordered_steps) <= action_chunk:
        raise ValueError("episode is too short for one action chunk")
    gaps = np.diff(np.asarray(ordered_steps, dtype=np.int64))
    if not np.all(gaps == SAMPLE_PERIOD_STEPS):
        raise ValueError("episode has a discontinuous global sample grid")
    return np.asarray([selected[step] for step in ordered_steps], dtype=np.int64)


def _sample_positions(grid_size: int, action_chunk: int, count: int) -> np.ndarray:
    last_start = grid_size - action_chunk - 1
    if last_start < 0:
        raise ValueError("not enough grid samples for one chunk")
    return np.unique(np.linspace(0, last_start, count, dtype=np.int64))


def _load_episode(source_root: Path, episode_id: int, action_chunk: int, count: int):
    """Return the observations and next-grid supervision for one episode."""
    from rlinf.envs.utils import center_crop_image

    episode_path = source_root / "data" / f"episode{episode_id}.hdf5"
    instruction_path = source_root / "instructions" / f"episode{episode_id}.json"
    instructions = json.loads(instruction_path.read_text(encoding="utf-8"))["seen"]
    task_description = instructions[episode_id % len(instructions)]

    with h5py.File(episode_path, "r") as data:
        simulation_steps = np.asarray(
            data["timing/simulation_step_index"], dtype=np.int64
        )
        grid = _grid_indices(simulation_steps, action_chunk)
        positions = _sample_positions(len(grid), action_chunk, count)
        source_indices = grid[positions]
        states = np.asarray(
            data["observation/robot_state/vector"], dtype=np.float32
        )[source_indices]
        all_actions = np.asarray(data["joint_action/vector"], dtype=np.float32)
        targets = np.stack(
            [
                all_actions[grid[position + 1 : position + 1 + action_chunk]]
                for position in positions
            ]
        )
        encoded = data["observation/head_camera/rgb"]
        images = []
        for source_index in source_indices:
            with Image.open(io.BytesIO(bytes(encoded[int(source_index)]))) as frame:
                rgb = frame.convert("RGB")
            images.append(np.asarray(center_crop_image(rgb), dtype=np.uint8))
    return images, states, targets, task_description


def _slice_metrics(
    predicted: np.ndarray, targets: np.ndarray, states: np.ndarray, columns: slice
) -> dict:
    """Aggregate every metric over one group of action dimensions."""
    prediction = predicted[:, :, columns]
    target = targets[:, :, columns]
    proprio = states[:, None, columns]

    model_error = np.abs(prediction - target)
    copy_error = np.abs(proprio - target)
    predicted_motion = prediction - proprio
    true_motion = target - proprio

    model_l1 = float(model_error.mean())
    copy_l1 = float(copy_error.mean())
    predicted_norm = np.linalg.norm(predicted_motion, axis=2)
    true_norm = np.linalg.norm(true_motion, axis=2)
    valid = (predicted_norm > 1e-9) & (true_norm > 1e-9)
    cosine = np.full(predicted_norm.shape, np.nan)
    if np.any(valid):
        dot = np.sum(predicted_motion * true_motion, axis=2)
        cosine[valid] = dot[valid] / (predicted_norm[valid] * true_norm[valid])
    return {
        "model_l1": model_l1,
        "copy_l1": copy_l1,
        "skill_score": float(1.0 - model_l1 / copy_l1) if copy_l1 > 0 else None,
        "predicted_motion_norm": float(predicted_norm.mean()),
        "true_motion_norm": float(true_norm.mean()),
        "motion_magnitude_ratio": (
            float(predicted_norm.mean() / true_norm.mean())
            if true_norm.mean() > 0
            else None
        ),
        "motion_cosine": (
            float(np.nanmean(cosine)) if np.any(np.isfinite(cosine)) else None
        ),
    }


def _report(predicted: np.ndarray, targets: np.ndarray, states: np.ndarray) -> dict:
    arm = slice(0, ARM_DIMENSIONS)
    gripper = slice(ARM_DIMENSIONS, ACTION_DIMENSION)
    summary = {
        "overall": {
            "arm": _slice_metrics(predicted, targets, states, arm),
            "gripper": _slice_metrics(predicted, targets, states, gripper),
        },
        "by_chunk_position": {},
    }
    for position in REPORT_POSITIONS:
        if position > predicted.shape[1]:
            continue
        index = slice(position - 1, position)
        summary["by_chunk_position"][str(position)] = {
            "arm": _slice_metrics(
                predicted[:, index], targets[:, index], states, arm
            ),
            "gripper": _slice_metrics(
                predicted[:, index], targets[:, index], states, gripper
            ),
        }
    return summary


def main() -> int:
    args = _parse_args()
    platform = os.environ.get("ROBOT_PLATFORM", "").upper()
    if platform != "PANTHERA":
        raise RuntimeError("ROBOT_PLATFORM=PANTHERA is required for the 25x7 contract")

    from rlinf.models.embodiment.openvla_oft.official import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(args.model),
            "action_dim": ACTION_DIMENSION,
            "num_action_chunks": args.action_chunk,
            "add_value_head": False,
            "value_type": "action_level",
            "proprio_dim": ACTION_DIMENSION,
            "use_proprio": True,
            "use_film": False,
            "use_l1_regression": True,
            "num_images_in_input": 1,
            "max_prompt_length": 512,
            "unnorm_key": args.unnorm_key,
        }
    )
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    if not hasattr(model, "action_head"):
        raise RuntimeError("continuous L1 action head was not loaded")
    model = model.to("cuda").eval()

    all_predicted, all_targets, all_states = [], [], []
    per_episode = {}
    for episode_id in args.episode:
        images, states, targets, task_description = _load_episode(
            args.source_root, episode_id, args.action_chunk, args.samples_per_episode
        )
        predicted, _ = model.predict_action_batch(
            env_obs={
                "main_images": images,
                "wrist_images": None,
                "states": states,
                "task_descriptions": [task_description] * len(images),
            },
            do_sample=False,
            temperature=-1.0,
            top_k=-1,
            calulate_logprobs=False,
            calulate_values=False,
        )
        predicted = predicted.float().cpu().numpy()
        expected = (len(images), args.action_chunk, ACTION_DIMENSION)
        if predicted.shape != expected:
            raise RuntimeError(f"predicted {predicted.shape}, expected {expected}")
        if not np.all(np.isfinite(predicted)):
            raise RuntimeError(f"episode {episode_id} produced nonfinite actions")
        per_episode[str(episode_id)] = _report(predicted, targets, states)
        all_predicted.append(predicted)
        all_targets.append(targets)
        all_states.append(states)

    predicted = np.concatenate(all_predicted)
    targets = np.concatenate(all_targets)
    states = np.concatenate(all_states)
    summary = {
        "status": "passed",
        "model": str(args.model),
        "unnorm_key": args.unnorm_key,
        "action_chunk": args.action_chunk,
        "episodes": list(args.episode),
        "sample_count": int(predicted.shape[0]),
        "units": "raw absolute joint targets (radians); gripper is normalized opening",
        "aggregate": _report(predicted, targets, states),
        "per_episode": per_episode,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
