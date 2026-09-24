#!/usr/bin/env python3
"""Check that evaluation feeds the model what training did.

The standard sanity check for a fine-tuned VLA is two-sided: replay the
demonstrations to validate the collection pipeline, then load the checkpoint
into the inference pipeline and confirm it reproduces the L1 error it reached
in training.  We had the first as a gate and ran the second only as an ad-hoc
probe, which is why an evaluation that placed the head camera 47.6 cm from the
recorded viewpoint went unnoticed through two documented rounds of results
(``docs/20``).  The same model scored a median 2.7 mrad on its recorded frames
and 187.2 mrad through the harness.

Three layers, so a failure says which half is broken:

``render``   the rendered frame against the recorded one, at the same step.
             Catches camera, appearance and asset drift.
``model``    the chunk predicted from the *recorded* frame against the
             recording.  Catches the unnorm key, normalisation, action chunk,
             proprio width -- everything the checkpoint contract pins.
``pipeline`` the chunk predicted from the *rendered* frame.  This is what a
             rollout actually consumes, and it is the product of the other two.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from dataset import open_dataset
from executor import DenseExecutor, ExecutorConfig
from replay import grid_actions, grid_indices
from robotwin_env import build_task, inside

# Two renders of the same scene are not bit-identical: the frames are path
# traced at 32 samples per pixel and the recording was JPEG compressed, so a
# faithful pipeline still lands around 2.7 of 255.  Measured against a camera
# placed by the wrong config it was 16.3, and against a decoder that swapped
# red and blue, 7.0.
RENDER_TOLERANCE = 3.5
# The model reproduced its training chunk to a median 3.8 mrad, and the expert
# advances 5.12 mrad per step; past this the error is the size of the signal.
ACTION_TOLERANCE_MRAD = 10.0


def _proprio_dataset(proprio_dim: int) -> str:
    try:
        return {
            7: "observation/robot_state/vector",
            28: "observation/robot_state/dynamics_vector",
        }[int(proprio_dim)]
    except KeyError as error:
        raise ValueError(
            f"unsupported Panthera proprioception width: {proprio_dim}"
        ) from error


def recorded_frames(path: Path, steps, *, proprio_dim: int = 7):
    """Read recorded frames the way the collector wrote them.

    The frames were encoded by ``cv2.imencode`` inside RoboTwin's
    ``envs/utils/pkl2hdf5.py``, which reads its input as BGR, so the stored
    JPEG holds the simulator's RGB in the other order.  RoboTwin's own reader,
    ``envs/utils/parse_hdf5.parse_img_array``, decodes with ``cv2.imdecode``
    and puts it back; going around that reader is how the convention got lost,
    so this uses the same decoder.  The RLDS conversion re-encodes to ordinary
    RGB, so this is also what training ends up seeing.
    """
    import cv2
    import h5py

    with h5py.File(path, "r") as handle:
        frames = handle["observation/head_camera/rgb"]
        state_path = _proprio_dataset(proprio_dim)
        if state_path not in handle:
            raise ValueError(f"{path} has no {state_path}")
        states = handle[state_path]
        out = []
        for step in steps:
            blob = np.frombuffer(frames[step].tobytes(), dtype=np.uint8)
            picture = cv2.imdecode(blob, cv2.IMREAD_COLOR)
            if picture is None:
                raise ValueError(f"{path} step {step} is not a decodable image")
            out.append((np.ascontiguousarray(picture),
                        np.asarray(states[step], dtype=np.float32)))
    return out


def _normalize_actions(actions: np.ndarray, statistics: dict, *, clip: bool) -> np.ndarray:
    """Map physical actions to the bounds-q99 space used by OpenVLA training."""
    value = np.asarray(actions, dtype=np.float64)
    low = np.asarray(statistics["q01"], dtype=np.float64)
    high = np.asarray(statistics["q99"], dtype=np.float64)
    mask = np.asarray(statistics.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    if value.shape[-1] != low.shape[0] or low.shape != high.shape:
        raise ValueError(
            f"action/statistics shape mismatch: {value.shape[-1]} vs {low.shape}/{high.shape}"
        )
    normalized = np.where(mask, 2.0 * (value - low) / (high - low + 1.0e-8) - 1.0, value)
    if clip:
        normalized = np.where(mask, np.clip(normalized, -1.0, 1.0), normalized)
    constant = np.isclose(low, high)
    return np.where(constant, 0.0, normalized)


def _action_chunks(actions: np.ndarray, starts, action_chunk: int) -> np.ndarray:
    """Build training-style future chunks, padding the tail with the last action."""
    value = np.asarray(actions, dtype=np.float64)
    starts_array = np.asarray(list(starts), dtype=np.int64)
    offsets = np.arange(action_chunk, dtype=np.int64)
    indices = np.minimum(starts_array[:, None] + offsets[None, :], len(value) - 1)
    return value[indices]


def _gripper_transitions(actions: np.ndarray, threshold: float = 0.5) -> list[dict]:
    """Locate diagnostic midpoint crossings, not starts of continuous motion."""
    gripper = np.asarray(actions, dtype=np.float64)[:, 6]
    wide = gripper >= threshold
    rows = np.flatnonzero(wide[1:] != wide[:-1]) + 1
    return [
        {
            "row": int(row),
            "kind": "open" if bool(wide[row]) else "close",
            "before": float(gripper[row - 1]),
            "after": float(gripper[row]),
        }
        for row in rows
    ]


def offline_validation_loss(
    model, episode, *, action_chunk: int = 25, batch_size: int = 2,
    include_per_frame: bool = False, proprio_dim: int = 7,
) -> dict:
    """Recompute deterministic validation L1 on every recorded grid frame.

    The 24-hour overfit run intentionally disabled the trainer's validation
    loop, so it has no historical ``VLA Val/Loss``.  This uses the same
    bounds-q99 action space and 25-step labels as training, but evaluates the
    saved checkpoint after the fact with augmentation disabled.  Keeping the
    result under an explicit ``offline_validation`` key prevents it being
    mistaken for a W&B metric recorded during training.
    """
    if action_chunk <= 0 or batch_size <= 0:
        raise ValueError("action_chunk and batch_size must be positive")

    expert = np.asarray(grid_actions(episode.hdf5_path), dtype=np.float64)
    source_rows = grid_indices(episode.hdf5_path)
    if len(expert) != len(source_rows):
        raise ValueError(f"grid action/frame mismatch: {len(expert)} != {len(source_rows)}")

    statistics = model.get_action_stats(model.unnorm_key)
    instruction = episode.instruction()
    totals = {
        "normalized": 0.0,
        "normalized_current": 0.0,
        "normalized_next": 0.0,
        "normalized_joint": 0.0,
        "normalized_gripper": 0.0,
        "physical": 0.0,
    }
    counts = {
        "all": 0,
        "current": 0,
        "next": 0,
        "joint": 0,
        "gripper": 0,
    }
    per_frame = []

    for begin in range(0, len(expert), batch_size):
        starts = np.arange(begin, min(begin + batch_size, len(expert)), dtype=np.int64)
        recorded = recorded_frames(
            episode.hdf5_path, source_rows[starts], proprio_dim=proprio_dim
        )
        images = [item[0] for item in recorded]
        states = np.stack([item[1] for item in recorded])
        predicted, _ = model.predict_action_batch(
            env_obs={
                "main_images": images,
                "wrist_images": None,
                "states": states,
                "task_descriptions": [instruction] * len(images),
            },
            do_sample=False,
            temperature=-1.0,
            top_k=-1,
            calulate_logprobs=False,
            calulate_values=False,
        )
        if hasattr(predicted, "float"):
            predicted = predicted.float().cpu().numpy()
        predicted = np.asarray(predicted, dtype=np.float64)
        target = _action_chunks(expert, starts, action_chunk)
        expected_shape = (len(starts), action_chunk, 7)
        if predicted.shape != expected_shape or target.shape != expected_shape:
            raise ValueError(
                f"validation chunk shape mismatch: prediction {predicted.shape}, "
                f"target {target.shape}, expected {expected_shape}"
            )

        predicted_norm = _normalize_actions(predicted, statistics, clip=False)
        target_norm = _normalize_actions(target, statistics, clip=True)
        norm_error = np.abs(predicted_norm - target_norm)
        physical_error = np.abs(predicted - target)
        if include_per_frame:
            for offset, start in enumerate(starts):
                error = norm_error[offset]
                per_frame.append(
                    {
                        "row": int(start),
                        "source_row": int(source_rows[start]),
                        "normalized_chunk_l1": float(error.mean()),
                        "normalized_chunk_joint_l1": float(error[:, :6].mean()),
                        "normalized_chunk_gripper_l1": float(error[:, 6].mean()),
                        "normalized_current_l1": float(error[0].mean()),
                        "normalized_current_joint_l1": float(error[0, :6].mean()),
                        "normalized_current_gripper_l1": float(error[0, 6]),
                        "predicted_current_gripper": float(predicted[offset, 0, 6]),
                        "target_current_gripper": float(target[offset, 0, 6]),
                    }
                )
        totals["normalized"] += float(norm_error.sum())
        totals["normalized_current"] += float(norm_error[:, 0, :].sum())
        totals["normalized_next"] += float(norm_error[:, 1:, :].sum())
        totals["normalized_joint"] += float(norm_error[:, :, :6].sum())
        totals["normalized_gripper"] += float(norm_error[:, :, 6:].sum())
        totals["physical"] += float(physical_error.sum())
        counts["all"] += int(norm_error.size)
        counts["current"] += int(norm_error[:, 0, :].size)
        counts["next"] += int(norm_error[:, 1:, :].size)
        counts["joint"] += int(norm_error[:, :, :6].size)
        counts["gripper"] += int(norm_error[:, :, 6:].size)

    result = {
        "name": "offline deterministic validation L1",
        "recorded_during_training": False,
        "normalization": "bounds_q99",
        "frames": int(len(expert)),
        "action_chunk": int(action_chunk),
        "action_dim": 7,
        "batch_size": int(batch_size),
        "normalized_l1": totals["normalized"] / counts["all"],
        "normalized_current_l1": totals["normalized_current"] / counts["current"],
        "normalized_next_l1": totals["normalized_next"] / counts["next"],
        "normalized_joint_l1": totals["normalized_joint"] / counts["joint"],
        "normalized_gripper_l1": totals["normalized_gripper"] / counts["gripper"],
        "physical_action_mae": totals["physical"] / counts["all"],
    }
    if include_per_frame:
        result["expert_gripper_transitions"] = _gripper_transitions(expert)
        result["per_frame"] = per_frame
    return result


def render_at(args, dataset, episode, steps, *, proprio_dim: int = 7):
    """Replay the recording and capture the frame at each requested step."""
    targets = np.asarray(grid_actions(episode.hdf5_path), dtype=np.float64)
    wanted = sorted(set(int(s) for s in steps))
    root = Path(args.robotwin_root)
    captured = {}
    with inside(root):
        task = None
        try:
            task, scene = build_task(
                root, args.task_config, args.task_name, episode, dataset.scenes_path,
                len(targets) + 16, None,
                camera=dataset.camera(episode.episode_id),
                camera_pose=dataset.camera_pose(episode.episode_id),
            )
            executor = DenseExecutor(task, ExecutorConfig())
            for index, target in enumerate(targets):
                if index in wanted:
                    observation = task.get_obs()["observation"]
                    key = {7: "vector", 28: "dynamics_vector"}.get(proprio_dim)
                    if key is None:
                        raise ValueError(
                            f"unsupported Panthera proprioception width: {proprio_dim}"
                        )
                    captured[index] = (
                        np.asarray(observation["head_camera"]["rgb"], dtype=np.uint8),
                        np.asarray(observation["robot_state"][key], dtype=np.float32),
                    )
                executor.step(target)
            return captured, scene
        finally:
            if task is not None:
                with contextlib.suppress(Exception):
                    task.close_env()


def load_policy(args, dataset):
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openvla_oft.official import get_model

    model_root = Path(args.model)
    from checkpoint import open_checkpoint

    checkpoint = open_checkpoint(model_root)
    proprio_dim = checkpoint.policy.proprio_dim
    statistics = json.loads((model_root / "dataset_statistics.json").read_text(encoding="utf-8"))
    unnorm_key = args.unnorm_key or sorted(statistics)[0]
    if unnorm_key not in statistics:
        raise SystemExit(f"{unnorm_key} is not among {sorted(statistics)}")
    config = OmegaConf.create({
        "model_path": str(model_root), "action_dim": 7,
        "num_action_chunks": args.action_chunk, "add_value_head": False,
        "value_type": "action_level", "proprio_dim": proprio_dim,
        "use_proprio": True,
        "use_film": False, "use_l1_regression": True, "num_images_in_input": 1,
        "max_prompt_length": 512, "unnorm_key": unnorm_key,
    })
    return (
        get_model(config, torch_dtype=torch.bfloat16).to("cuda").eval(),
        unnorm_key,
        proprio_dim,
    )


def _openvla_predict(model, image, state, instruction):
    predicted, _ = model.predict_action_batch(
        env_obs={"main_images": [image], "wrist_images": None, "states": state[None, :],
                 "task_descriptions": [instruction]},
        do_sample=False, temperature=-1.0, top_k=-1,
        calulate_logprobs=False, calulate_values=False,
    )
    return predicted.float().cpu().numpy()[0]


def deviation(predict, image, state, instruction, expert, step) -> float:
    chunk = np.asarray(predict(image, state, instruction), dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError(f"policy parity prediction must be [N, 7], got {chunk.shape}")
    target = expert[step: step + len(chunk)]
    count = min(len(chunk), len(target))
    if count == 0:
        return float("nan")
    return float(np.median(np.abs(chunk[:count, :6] - target[:count, :6])) * 1000)


def measure(robotwin_root, task_config, task_name, dataset, episode_id,
            model=None, samples: int = 12, action_chunk: int = 25, predict=None,
            proprio_dim: int = 7) -> dict:
    """Measure the three layers for one episode and return the report.

    This is an invariant, not a diagnostic: whenever the evaluation environment
    reproduces a recorded episode, the policy's behaviour there must stay close
    to that recording, however many episodes it was trained on and however good
    it is at the task.  A policy may be poor and still satisfy it.  If it fails,
    no number measured through that pipeline means anything -- which is how the
    0/18 and 0/12 in ``docs/18`` came to be recorded as policy results.
    """
    episode = dataset.episode(episode_id)
    expert = np.asarray(grid_actions(episode.hdf5_path), dtype=np.float64)
    steps = np.linspace(0, len(expert) - action_chunk - 1, samples).astype(int)

    args = argparse.Namespace(
        robotwin_root=robotwin_root, task_config=task_config, task_name=task_name,
    )
    rendered, scene = render_at(
        args, dataset, episode, steps, proprio_dim=proprio_dim
    )
    recorded = recorded_frames(
        episode.hdf5_path, steps, proprio_dim=proprio_dim
    )

    pixel = [float(np.abs(rendered[int(s)][0].astype(np.int16) - rec[0].astype(np.int16)).mean())
             for s, rec in zip(steps, recorded)]
    report = {
        "episode": int(episode_id),
        "samples": len(steps),
        "scene": scene,
        "render": {"median": float(np.median(pixel)), "max": float(np.max(pixel)),
                   "tolerance": RENDER_TOLERANCE,
                   "passed": bool(np.median(pixel) <= RENDER_TOLERANCE)},
    }
    if predict is None and model is not None:
        predict = lambda image, state, instruction: _openvla_predict(
            model, image, state, instruction
        )
    if predict is not None:
        instruction = episode.instruction()
        on_record, on_render = [], []
        for step, rec in zip(steps, recorded):
            on_record.append(deviation(predict, rec[0], rec[1], instruction, expert, int(step)))
            live = rendered[int(step)]
            on_render.append(deviation(predict, live[0], live[1], instruction, expert, int(step)))
        for name, values in (("model", on_record), ("pipeline", on_render)):
            report[name] = {
                "median_mrad": float(np.median(values)),
                "max_mrad": float(np.max(values)),
                "tolerance_mrad": ACTION_TOLERANCE_MRAD,
                "passed": bool(np.median(values) <= ACTION_TOLERANCE_MRAD),
            }
    report["passed"] = all(
        report[name]["passed"] for name in ("render", "model", "pipeline") if name in report
    )
    return report


def describe(report: dict) -> str:
    lines = []
    for name in ("render", "model", "pipeline"):
        if name not in report:
            continue
        result = report[name]
        value = result.get("median_mrad", result.get("median"))
        unit = "mrad" if "median_mrad" in result else "/255"
        lines.append(f"  {name:9s} 中位 {value:8.2f} {unit:5s} "
                     f"{'通过' if result['passed'] else '未通过'}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-name", default="place_randomized_cylinder_in_socket")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--model", help="a merged checkpoint; omit to check rendering only")
    parser.add_argument("--unnorm-key")
    parser.add_argument("--action-chunk", type=int, default=25)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--output")
    args = parser.parse_args()

    dataset = open_dataset(Path(args.dataset_root))
    model = None
    proprio_dim = 7
    if args.model:
        model, _, proprio_dim = load_policy(args, dataset)
    report = measure(args.robotwin_root, args.task_config, args.task_name, dataset,
                     args.episode, model, args.samples, args.action_chunk,
                     proprio_dim=proprio_dim)
    print(describe(report))
    print(f"结论: {'通过' if report['passed'] else '未通过'}")
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
