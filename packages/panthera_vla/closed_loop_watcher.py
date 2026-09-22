#!/usr/bin/env python3
"""Stop an overfit run when its trajectory actually succeeds, not on a clock.

Validation loss does not predict closed-loop success on this task -- the
historical model that grasped reliably scored a worse chunk L1 than the model
that scored zero -- so a run stopped on a loss plateau stops on the wrong
quantity.  A wall clock is worse still: the 4h first attempt would have covered
159 epochs of one trajectory and been cut off mid-descent, with nothing said
about whether the trajectory was reproducible.

This watches the checkpoint a run keeps writing, rolls the policy in the scene
it was fitted to, and signals when it succeeds.  It runs beside the trainer on
its own GPU rather than inside it, so the training forward path is untouched and
a failure here cannot corrupt a run.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import subprocess

import numpy as np
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGES / "panthera_sim"))

SUCCESS_SENTINEL = "CLOSED_LOOP_SUCCESS"
LOG_FILE = "closed-loop-validation.jsonl"
SNAPSHOT_FILES = (
    "lora_adapter/adapter_model.safetensors",
    "lora_adapter/adapter_config.json",
    "action_head--latest_checkpoint.pt",
    "proprio_projector--latest_checkpoint.pt",
    "dataset_statistics.json",
    "training.json",
)


def _checkpoint_stamp(run_dir: Path) -> tuple[tuple[str, int, int], ...] | None:
    """Fingerprint every independently-written part of a deployable checkpoint."""
    paths = [run_dir / name for name in SNAPSHOT_FILES]
    if any(not path.is_file() for path in paths):
        return None
    stamp = []
    for name, path in zip(SNAPSHOT_FILES, paths):
        stat = path.stat()
        stamp.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(stamp)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _promote_success(model: Path, target: Path) -> Path:
    """Preserve the exact merged snapshot that passed the closed-loop gate."""
    if target.exists():
        raise RuntimeError(f"refusing to overwrite successful model: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.replace(target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        temporary = target.with_name(f".{target.name}.copying")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(model, temporary)
        temporary.replace(target)
        shutil.rmtree(model)
    return target


def _merge(run_dir: Path, base_model: Path, into: Path) -> Path:
    """Materialise the current adapter as a merged model the loader accepts.

    The merge runs against a snapshot copy, never the live run directory, so
    the trainer can keep writing.  The copy is verified not to be torn: if the
    adapter changed while it was being read, the snapshot is discarded.
    """
    before = _checkpoint_stamp(run_dir)
    if before is None:
        raise RuntimeError("no adapter to merge yet")
    if into.exists():
        shutil.rmtree(into)
    into.mkdir(parents=True)
    for name in (
        "config.json", "dataset_statistics.json", "preprocessor_config.json",
        "processor_config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
        "added_tokens.json", "configuration_prismatic.py", "modeling_prismatic.py",
        "processing_prismatic.py", "action_head--latest_checkpoint.pt",
        "proprio_projector--latest_checkpoint.pt", "training.json",
        "early-stopping.json",
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, into / name)
    shutil.copytree(run_dir / "lora_adapter", into / "lora_adapter")
    if _checkpoint_stamp(run_dir) != before:
        shutil.rmtree(into)
        raise RuntimeError("the checkpoint was rewritten while being copied")
    result = subprocess.run(
        [sys.executable, str(PACKAGES / "panthera_vla" / "merge_lora_checkpoint.py"),
         "--run-dir", str(into), "--base-model", str(base_model),
         # The guard in that script protects the live run directory; this is a
         # verified snapshot of it, which is the case the guard cannot see.
         "--allow-running-training"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"merge failed: {(result.stderr or result.stdout).strip().splitlines()[-1]}"
        )
    return into


def _rollout(model: Path, cli: argparse.Namespace) -> dict:
    """Run one episode through the same executor the gate will use."""
    from dataset import open_dataset
    from rollout import observe_task, run_episode
    from robotwin_env import task_args

    dataset = open_dataset(cli.dataset_root)
    budget = cli.max_actions or dataset.contract.required_action_budget

    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openvla_oft.official import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(model), "action_dim": 7,
            "num_action_chunks": cli.action_chunk, "add_value_head": False,
            "value_type": "action_level", "proprio_dim": 7, "use_proprio": True,
            "use_film": False, "use_l1_regression": True, "num_images_in_input": 1,
            "max_prompt_length": 512, "unnorm_key": cli.unnorm_key,
        }
    )
    policy = get_model(cfg, torch_dtype=torch.bfloat16).to("cuda").eval()

    def predict(image, state, instruction, step):
        del step
        predicted, _ = policy.predict_action_batch(
            env_obs={
                "main_images": [image], "wrist_images": None,
                "states": state[None, :], "task_descriptions": [instruction],
            },
            do_sample=False, temperature=-1.0, top_k=-1,
            calulate_logprobs=False, calulate_values=False,
        )
        return predicted.float().cpu().numpy()[0]

    # The expert's measured joints make a failing run comparable to another
    # failing run: both score zero, but they differ in how long they stayed on
    # the trajectory that is known to work.
    import h5py

    with h5py.File(dataset.episode(cli.episode).hdf5_path, "r") as data:
        steps = np.asarray(data["timing/simulation_step_index"], dtype=np.int64)
        qpos = np.asarray(data["observation/robot_state/arm_qpos"], dtype=np.float64)
    selected: dict[int, int] = {}
    for index, step in enumerate(steps.tolist()):
        if step % 5 == 0:
            selected[int(step)] = index
    reference = qpos[np.asarray([selected[s] for s in sorted(selected)], dtype=np.int64)]

    case = run_episode(
        cli.robotwin_root, cli.task_config, cli.task_name, dataset, cli.episode,
        predict, cli.execution_horizon, budget,
        base_args=task_args(cli.robotwin_root, cli.task_config, cli.task_name),
        observe=observe_task, reference_qpos=reference,
    )
    del policy
    torch.cuda.empty_cache()
    return case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-name", default="place_randomized_cylinder_in_socket")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--action-chunk", type=int, default=25)
    parser.add_argument("--execution-horizon", type=int, default=20)
    parser.add_argument("--max-actions", type=int, default=0)
    parser.add_argument("--interval-s", type=float, default=60.0,
                        help="how often to check for a newer checkpoint")
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--success-model-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    os.environ.setdefault("ROBOT_PLATFORM", "PANTHERA")
    cli.run_dir.mkdir(parents=True, exist_ok=True)
    cli.scratch = cli.scratch or (cli.run_dir / ".closed-loop-watch")
    cli.success_model_dir = cli.success_model_dir or (cli.run_dir / "closed-loop-best")
    log = cli.run_dir / LOG_FILE
    sentinel = cli.run_dir / SUCCESS_SENTINEL
    seen: tuple[tuple[str, int, int], ...] | None = None
    attempts = 0

    while True:
        if sentinel.exists():
            return 0
        stamp = _checkpoint_stamp(cli.run_dir)
        if stamp is None or stamp == seen:
            time.sleep(min(cli.interval_s, 60.0))
            continue
        attempts += 1
        started = datetime.now(timezone.utc).isoformat()
        try:
            model = _merge(cli.run_dir, cli.base_model, cli.scratch / "merged")
            case = _rollout(model, cli)
            seen = stamp
            record = {
                "attempt": attempts, "started_at": started,
                "checkpoint_stamp": stamp, "success": bool(case.get("success")),
                "executed_actions": case.get("executed_actions"),
                "policy_queries": case.get("policy_queries"),
                "metrics": case.get("metrics", {}), "error": case.get("error"),
                "progress": case.get("progress"),
            }
        except Exception as error:  # a watcher failure must not end a training run
            record = {
                "attempt": attempts, "started_at": started,
                "checkpoint_stamp": stamp, "success": False,
                "error": f"{type(error).__name__}: {error}",
            }
        if record["success"]:
            try:
                promoted = _promote_success(model, cli.success_model_dir)
                record["model"] = str(promoted)
            except Exception as error:
                record["success"] = False
                record["error"] = f"promotion failed: {type(error).__name__}: {error}"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        progress = record.get("progress") or {}
        deviation = (progress.get("expert_deviation") or {})
        print(
            f"[闭环验证 {attempts}] success={record['success']} "
            f"阶段={progress.get('stage')} "
            f"最近夹爪距物体={progress.get('nearest_gripper_to_object_m')}m "
            f"接触步={progress.get('grasp_steps')} "
            f"抬升={progress.get('max_lift_m')}m "
            f"物体逼近槽={progress.get('closed_fraction')} "
            f"偏离中位={deviation.get('median_mrad')}mrad "
            f"首破100mrad={deviation.get('first_step_over_100mrad')} "
            f"{record.get('error') or ''}",
            flush=True,
        )
        if record["success"]:
            _write_json_atomic(sentinel, record)
            print("闭环成功，已写出停止信号。", flush=True)
            return 0
        time.sleep(cli.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
