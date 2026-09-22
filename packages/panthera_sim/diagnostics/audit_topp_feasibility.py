#!/usr/bin/env python3
"""Measure whether a policy's action chunks survive RoboTwin's TOPP executor.

``gen_sparse_reward_data`` turns each action chunk into a waypoint path and
re-times it with TOPP before execution.  When that call raises, or returns an
empty trajectory, the fallback in ``_base_task.py`` sets ``topp_flag=False`` and
then **skips ``set_arm_joints`` entirely** for those steps -- the arm receives no
command at all and merely drifts.  A policy whose chunks fail TOPP would look
like a policy that cannot act, regardless of what it predicted.

Expert chunks are smooth by construction (Chaikin smoothing plus a global
arc-length time law).  A policy's chunks are not, so the two must be compared on
the same executor before any conclusion is drawn about the policy itself.

Chunks come either from a trained model (``--model``) or, when no GPU is free,
from expert chunks perturbed with Gaussian noise (``--noise-rad``) that stands in
for a policy of a given accuracy.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

SAMPLE_PERIOD_STEPS = 5
DEFAULT_TASK_NAME = "place_randomized_cylinder_in_socket"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-chunk", type=int, default=25)
    parser.add_argument("--execution-horizon", type=int, default=20)
    parser.add_argument(
        "--model", type=Path, help="trained model dir; omit to use perturbed expert chunks"
    )
    parser.add_argument("--unnorm-key", default="panthera_phone_cylinder_socket_v2")
    parser.add_argument(
        "--noise-rad",
        type=float,
        action="append",
        help="stand-in policy error when --model is absent; repeatable",
    )
    parser.add_argument("--chunks-per-episode", type=int, default=12)
    parser.add_argument(
        "--predict-batch",
        type=int,
        default=16,
        help="observations per forward pass when predicting chunks",
    )
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="also evaluate the chunk an ACT-style temporal ensemble would "
             "execute: query every step and average each timestep over the "
             "still-valid predictions that covered it",
    )
    parser.add_argument(
        "--ensemble-m",
        type=float,
        default=0.01,
        help="exponential weight decay per step of prediction age",
    )
    return parser.parse_args()


def _task_args(robotwin_root: Path, config_name: str, task_name: str) -> dict:
    if Path(config_name).name != config_name:
        raise ValueError("task config must be a filename")
    with (robotwin_root / "task_config" / config_name).open("r", encoding="utf-8") as s:
        args = yaml.safe_load(s)
    with (robotwin_root / "task_config" / "_embodiment_config.yml").open(
        "r", encoding="utf-8"
    ) as s:
        registry = yaml.safe_load(s)
    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        robot_names = (embodiment[0], embodiment[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment[0])
    else:
        robot_names = (embodiment[0], embodiment[1])
        args["embodiment_dis"] = float(embodiment[2])
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment[0]}+{embodiment[1]}"
    for side, name in zip(("left", "right"), robot_names):
        robot_root = (robotwin_root / registry[name]["file_path"]).resolve()
        args[f"{side}_robot_file"] = str(robot_root)
        with (robot_root / "config.yml").open("r", encoding="utf-8") as s:
            args[f"{side}_embodiment_config"] = yaml.safe_load(s)
    args.update(
        {
            "task_name": task_name,
            "need_plan": False,
            "render_freq": 0,
            "save_data": False,
            "eval_mode": True,
            "eval_video_log": False,
            "collect_data": False,
        }
    )
    return args


def _grid(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return next-grid-aligned actions and the measured arm state, at 50 Hz."""
    with h5py.File(hdf5_path, "r") as data:
        steps = np.asarray(data["timing/simulation_step_index"], dtype=np.int64)
        actions = np.asarray(data["joint_action/vector"], dtype=np.float64)
        arm_qpos = np.asarray(data["observation/robot_state/arm_qpos"], dtype=np.float64)
    selected: dict[int, int] = {}
    for index, step in enumerate(steps.tolist()):
        if step % SAMPLE_PERIOD_STEPS == 0:
            selected[int(step)] = index
    ordered = sorted(selected)
    idx = np.asarray([selected[s] for s in ordered], dtype=np.int64)
    nxt = np.concatenate((idx[1:], idx[-1:]))
    return actions[nxt], arm_qpos[idx], idx


def _topp_outcome(planner, path: np.ndarray) -> dict:
    """Classify one TOPP call exactly as ``_base_task`` would interpret it."""
    try:
        times, position, velocity, acceleration, duration = planner.TOPP(
            path, 1.0 / 250.0, verbose=False
        )
    except Exception as error:  # the executor catches everything here too
        return {"outcome": "exception", "detail": type(error).__name__, "n_step": 0}
    n_step = int(np.asarray(position).shape[0])
    if n_step == 0:
        return {"outcome": "empty", "detail": "", "n_step": 0}
    return {
        "outcome": "ok",
        "detail": "",
        "n_step": n_step,
        "duration_s": float(duration),
        "peak_speed_radps": float(np.abs(np.asarray(velocity)).max()),
    }


def _temporal_ensemble(
    chunks: dict[int, np.ndarray], start: int, horizon: int, decay: float
) -> np.ndarray | None:
    """Average each timestep over the predictions that are still valid for it.

    Executing only the newest chunk leaves every prediction error in the
    commanded path, and RoboTwin re-times that path with TOPP, which stretches a
    jittery chunk several-fold.  Averaging over the overlapping predictions is
    the standard way action-chunking policies suppress that jitter, at the cost
    of one forward pass per step instead of one per chunk.
    """
    executed = []
    for step in range(start, start + horizon):
        contributions = []
        weights = []
        for origin in range(start, step + 1):
            offset = step - origin
            chunk = chunks.get(origin)
            if chunk is None or offset >= len(chunk):
                continue
            contributions.append(chunk[offset])
            weights.append(math.exp(-decay * offset))
        if not contributions:
            return None
        weight = np.asarray(weights, dtype=np.float64)
        executed.append(
            np.average(np.asarray(contributions, dtype=np.float64), axis=0,
                       weights=weight / weight.sum())
        )
    return np.asarray(executed, dtype=np.float64)


def _chaikin(path: np.ndarray, rounds: int = 4) -> np.ndarray:
    """The same corner-cutting smoothing the oracle applies before retiming."""
    out = path
    for _ in range(rounds):
        if len(out) < 3:
            break
        q = 0.75 * out[:-1] + 0.25 * out[1:]
        r = 0.25 * out[:-1] + 0.75 * out[1:]
        interleaved = np.empty((2 * len(q), out.shape[1]), dtype=out.dtype)
        interleaved[0::2], interleaved[1::2] = q, r
        out = np.vstack((out[:1], interleaved, out[-1:]))
    return out


def _policy_chunks(
    model, hdf5_path: Path, instruction_path: Path, episode_id: int,
    grid_index: np.ndarray, starts: np.ndarray, horizon: int,
    batch_size: int = 16,
) -> dict[int, np.ndarray]:
    """Predict one action chunk per sampled start, in bounded batches.

    A temporal ensemble needs one prediction per step, so the number of starts
    grows with the horizon and a single batch no longer fits a 7B model.
    """
    import io
    from PIL import Image
    from rlinf.envs.utils import center_crop_image

    values = json.loads(instruction_path.read_text(encoding="utf-8"))["seen"]
    instruction = values[episode_id % len(values)]
    with h5py.File(hdf5_path, "r") as data:
        states = np.asarray(
            data["observation/robot_state/vector"], dtype=np.float32
        )[grid_index[starts]]
        encoded = data["observation/head_camera/rgb"]
        images = []
        for source in grid_index[starts].tolist():
            with Image.open(io.BytesIO(bytes(encoded[int(source)]))) as frame:
                rgb = frame.convert("RGB")
            images.append(np.asarray(center_crop_image(rgb), dtype=np.uint8))

    chunks: dict[int, np.ndarray] = {}
    order = starts.tolist()
    for offset in range(0, len(order), batch_size):
        window = slice(offset, offset + batch_size)
        batch_images = images[window]
        predicted, _ = model.predict_action_batch(
            env_obs={
                "main_images": batch_images, "wrist_images": None,
                "states": states[window],
                "task_descriptions": [instruction] * len(batch_images),
            },
            do_sample=False, temperature=-1.0, top_k=-1,
            calulate_logprobs=False, calulate_values=False,
        )
        predicted = predicted.float().cpu().numpy()
        for row, start in enumerate(order[window]):
            chunks[int(start)] = predicted[row, :, :6]
    return chunks


def main() -> int:
    cli = parse_args()
    robotwin_root = cli.robotwin_root.resolve()
    dataset_root = cli.dataset_root.resolve()
    os.environ["ASSETS_PATH"] = str(robotwin_root)
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(robotwin_root))
    os.chdir(robotwin_root)

    task_module = importlib.import_module(f"envs.{cli.task_name}")
    task_class = getattr(task_module, cli.task_name)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset import open_dataset

    dataset = open_dataset(dataset_root)
    base_args = _task_args(robotwin_root, cli.task_config, cli.task_name)

    model = None
    if cli.model is not None:
        from omegaconf import OmegaConf
        import torch
        from rlinf.models.embodiment.openvla_oft.official import get_model

        cfg = OmegaConf.create(
            {
                "model_path": str(cli.model), "action_dim": 7,
                "num_action_chunks": cli.action_chunk, "add_value_head": False,
                "value_type": "action_level", "proprio_dim": 7, "use_proprio": True,
                "use_film": False, "use_l1_regression": True, "num_images_in_input": 1,
                "max_prompt_length": 512, "unnorm_key": cli.unnorm_key,
            }
        )
        model = get_model(cfg, torch_dtype=torch.bfloat16).to("cuda").eval()

    noise_levels = cli.noise_rad or ([] if model is not None else [0.0, 0.003, 0.01, 0.03])
    records: list[dict] = []

    for episode_id in cli.episode:
        seed = dataset.episode(episode_id).seed
        hdf5_path = dataset_root / "data" / f"episode{episode_id}.hdf5"
        actions, arm_qpos, grid_index = _grid(hdf5_path)
        task = task_class()
        try:
            args = copy.deepcopy(base_args)
            args["step_lim"] = int(len(actions))
            task.setup_demo(now_ep_num=seed, seed=seed, **args)
            planner = task.robot.left_mplib_planner

            anchors = np.unique(
                np.linspace(
                    0, max(len(actions) - cli.action_chunk - 1, 0),
                    cli.chunks_per_episode, dtype=np.int64,
                )
            )
            # A temporal ensemble needs every prediction that overlaps the
            # window, which means querying once per step rather than per chunk.
            if cli.temporal_ensemble:
                starts = np.unique(
                    np.concatenate(
                        [
                            np.arange(a, a + cli.execution_horizon, dtype=np.int64)
                            for a in anchors.tolist()
                        ]
                    )
                )
                starts = starts[starts <= len(actions) - cli.action_chunk - 1]
            else:
                starts = anchors
            policy_chunks = None
            if model is not None:
                policy_chunks = _policy_chunks(
                    model, hdf5_path,
                    dataset_root / "instructions" / f"episode{episode_id}.json",
                    episode_id, grid_index, starts, cli.execution_horizon,
                    cli.predict_batch,
                )
            for start in anchors.tolist():
                expert = actions[start : start + cli.execution_horizon, :6]
                current = arm_qpos[start]
                if len(expert) < 2:
                    continue
                variants = {"expert": expert}
                for level in noise_levels:
                    rng = np.random.default_rng(start + int(level * 1e6))
                    variants[f"noise_{level:g}"] = expert + rng.normal(
                        0.0, level, expert.shape
                    )
                if policy_chunks is not None and start in policy_chunks:
                    variants["policy"] = policy_chunks[int(start)][
                        : cli.execution_horizon
                    ]
                    if cli.temporal_ensemble:
                        ensembled = _temporal_ensemble(
                            policy_chunks, int(start), cli.execution_horizon,
                            cli.ensemble_m,
                        )
                        if ensembled is not None:
                            variants["policy_ensembled"] = ensembled
                for name, chunk in variants.items():
                    raw = _topp_outcome(planner, np.vstack((current, chunk)))
                    smoothed = _topp_outcome(
                        planner, _chaikin(np.vstack((current, chunk)))
                    )
                    records.append(
                        {
                            "episode": episode_id, "chunk_start": int(start),
                            "variant": name, "raw": raw, "chaikin": smoothed,
                        }
                    )
        finally:
            try:
                task.close_env()
            except Exception:
                pass

    summary: dict = {"records": records, "by_variant": {}}
    for variant in sorted({r["variant"] for r in records}):
        subset = [r for r in records if r["variant"] == variant]
        for mode in ("raw", "chaikin"):
            outcomes = [r[mode]["outcome"] for r in subset]
            ok = [r[mode] for r in subset if r[mode]["outcome"] == "ok"]
            summary["by_variant"][f"{variant}/{mode}"] = {
                "n": len(subset),
                "ok": outcomes.count("ok"),
                "exception": outcomes.count("exception"),
                "empty": outcomes.count("empty"),
                "fail_rate": round(1 - outcomes.count("ok") / max(len(subset), 1), 4),
                "median_n_step": float(np.median([o["n_step"] for o in ok])) if ok else None,
                "median_duration_s": float(
                    np.median([o["duration_s"] for o in ok])
                ) if ok else None,
            }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for key, value in summary["by_variant"].items():
        print(
            f"  {key:24s} n={value['n']:4d} ok={value['ok']:4d} "
            f"fail={value['fail_rate']:.3f} 步数中位数={value['median_n_step']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
