#!/usr/bin/env python3
"""Probe a trained OpenVLA-OFT L1 head on recorded Panthera observations.

This is an inference-only contract check.  It loads several frames in one batch,
reconstructs the same next-grid 5x7 supervision used by the RLDS adapter, and
reports predicted absolute joint targets versus the recorded oracle targets.
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
ACTION_CHUNK = 5
ACTION_DIMENSION = 7


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    return parser.parse_args()


def _grid_indices(simulation_steps: np.ndarray) -> np.ndarray:
    selected: dict[int, int] = {}
    for index, step in enumerate(simulation_steps.tolist()):
        if int(step) % SAMPLE_PERIOD_STEPS == 0:
            selected[int(step)] = index
    ordered_steps = sorted(selected)
    if len(ordered_steps) <= ACTION_CHUNK:
        raise ValueError("episode is too short for one action chunk")
    gaps = np.diff(np.asarray(ordered_steps, dtype=np.int64))
    if not np.all(gaps == SAMPLE_PERIOD_STEPS):
        raise ValueError("episode has a discontinuous global sample grid")
    return np.asarray([selected[step] for step in ordered_steps], dtype=np.int64)


def _decode_rgb(encoded: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _sample_positions(grid_size: int, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("sample-count must be positive")
    last_start = grid_size - ACTION_CHUNK - 1
    if last_start < 0:
        raise ValueError("not enough grid samples")
    return np.unique(np.linspace(0, last_start, count, dtype=np.int64))


def main() -> int:
    args = _parse_args()
    if os.environ.get("ROBOT_PLATFORM", "").upper() != "BRIDGE":
        raise RuntimeError("ROBOT_PLATFORM=BRIDGE is required for the 5x7 contract")

    from rlinf.envs.utils import center_crop_image
    from rlinf.models.embodiment.openvla_oft.official import get_model

    instructions = json.loads(args.instruction.read_text(encoding="utf-8"))["seen"]
    episode_id = int(args.episode.stem.removeprefix("episode"))
    task_description = instructions[episode_id % len(instructions)]

    with h5py.File(args.episode, "r") as data:
        simulation_steps = np.asarray(
            data["timing/simulation_step_index"], dtype=np.int64
        )
        grid = _grid_indices(simulation_steps)
        sample_positions = _sample_positions(len(grid), args.sample_count)
        source_indices = grid[sample_positions]
        states = np.asarray(
            data["observation/robot_state/vector"], dtype=np.float32
        )[source_indices]
        all_actions = np.asarray(data["joint_action/vector"], dtype=np.float32)
        target_chunks = np.stack(
            [
                all_actions[grid[position + 1 : position + 1 + ACTION_CHUNK]]
                for position in sample_positions
            ]
        )
        encoded_images = data["observation/head_camera/rgb"]
        images = []
        for source_index in source_indices:
            image = Image.fromarray(
                _decode_rgb(bytes(encoded_images[int(source_index)]))
            ).convert("RGB")
            images.append(np.asarray(center_crop_image(image), dtype=np.uint8))

    cfg = OmegaConf.create(
        {
            "model_path": str(args.model),
            "action_dim": ACTION_DIMENSION,
            "num_action_chunks": ACTION_CHUNK,
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
    predicted_np = predicted.float().cpu().numpy()
    expected_shape = (len(images), ACTION_CHUNK, ACTION_DIMENSION)
    if predicted_np.shape != expected_shape:
        raise RuntimeError(
            f"predicted shape {predicted_np.shape}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(predicted_np)):
        raise RuntimeError("predicted actions contain NaN or infinity")

    absolute_error = np.abs(predicted_np - target_chunks)
    records = []
    for batch_index, source_index in enumerate(source_indices.tolist()):
        records.append(
            {
                "source_index": int(source_index),
                "simulation_step": int(simulation_steps[source_index]),
                "state": states[batch_index].tolist(),
                "predicted_first_action": predicted_np[batch_index, 0].tolist(),
                "target_first_action": target_chunks[batch_index, 0].tolist(),
                "chunk_mae": float(absolute_error[batch_index].mean()),
                "first_action_mae": float(absolute_error[batch_index, 0].mean()),
            }
        )
    summary = {
        "status": "passed",
        "model": str(args.model),
        "episode": str(args.episode),
        "instruction": task_description,
        "action_contract": "absolute_joint_qpos_5x7",
        "continuous_l1_action_head_loaded": True,
        "sample_count": len(records),
        "mean_absolute_error": float(absolute_error.mean()),
        "maximum_absolute_error": float(absolute_error.max()),
        "predicted_min": predicted_np.min(axis=(0, 1)).tolist(),
        "predicted_max": predicted_np.max(axis=(0, 1)).tolist(),
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
