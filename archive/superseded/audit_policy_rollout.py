#!/usr/bin/env python3
"""Roll a trained policy through the executor that reproduces the expert.

The offline metrics do not separate a policy that works from one that does not:
the historical model that grasped reliably 11/16 scores a *worse* chunk L1 skill
score and the same motion cosine as the model that scores 0/18.  So the only way
left to see what the policy does is to watch it act, in a scene the expert is
known to solve through the same executor.

Everything except the source of the actions is held fixed against
``audit_dense_execution``: the same scene registry, the same dense stepping, the
same success criteria.  A run therefore answers "does this policy diverge from a
trajectory the executor can otherwise complete, and where".
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from audit_dense_execution import _grid_actions
from panthera_closed_loop import ClosedLoopRunner, RolloutConfig

DEFAULT_TASK_NAME = "place_randomized_cylinder_in_socket"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-chunk", type=int, default=25)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=20,
        help="actions consumed from each predicted chunk before re-querying",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=3200,
        help="action budget, in the same 50 Hz units the eval harness counts",
    )
    parser.add_argument(
        "--source",
        choices=("policy", "expert"),
        default="policy",
        help="expert replays the recorded actions through this same loop, as a "
             "control that isolates the policy from the executor",
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    return parser.parse_args()


_WORKER: dict = {}


def _init_worker(config: dict, devices) -> None:
    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices.get()
    os.environ["ASSETS_PATH"] = config["robotwin_root"]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("ROBOT_PLATFORM", "PANTHERA")
    sys.path.insert(0, config["robotwin_root"])
    os.chdir(config["robotwin_root"])
    _WORKER["config"] = config
    _WORKER["runner"] = ClosedLoopRunner(
        RolloutConfig(
            robotwin_root=Path(config["robotwin_root"]),
            dataset_root=Path(config["dataset_root"]),
            scene_registry=Path(config["scene_registry"]),
            task_config=config["task_config"],
            task_name=config["task_name"],
            execution_horizon=config["execution_horizon"],
            max_actions=config["max_actions"],
        )
    )
    if config["source"] == "policy":
        import torch
        from omegaconf import OmegaConf
        from rlinf.models.embodiment.openvla_oft.official import get_model

        cfg = OmegaConf.create(
            {
                "model_path": config["model"], "action_dim": 7,
                "num_action_chunks": config["action_chunk"], "add_value_head": False,
                "value_type": "action_level", "proprio_dim": 7, "use_proprio": True,
                "use_film": False, "use_l1_regression": True, "num_images_in_input": 1,
                "max_prompt_length": 512, "unnorm_key": config["unnorm_key"],
            }
        )
        _WORKER["model"] = get_model(cfg, torch_dtype=torch.bfloat16).to("cuda").eval()


def _run_job(episode_id: int) -> dict:
    config = _WORKER["config"]
    runner = _WORKER["runner"]
    expert = _grid_actions(
        Path(config["dataset_root"]) / "data" / f"episode{episode_id}.hdf5"
    )
    if config["source"] == "expert":
        cursor = {"row": 0}

        def predict(image, state, instruction):
            del image, state, instruction
            start = cursor["row"]
            cursor["row"] = start + config["execution_horizon"]
            window = expert[start : start + config["execution_horizon"]]
            # A stream that has run out still has to return a chunk; repeating
            # the last target holds position rather than ending the episode on a
            # shape error.
            if len(window) == 0:
                return np.repeat(expert[-1][None, :], config["execution_horizon"], axis=0)
            return window
    else:
        model = _WORKER["model"]

        def predict(image, state, instruction):
            predicted, _ = model.predict_action_batch(
                env_obs={
                    "main_images": [image], "wrist_images": None,
                    "states": state[None, :], "task_descriptions": [instruction],
                },
                do_sample=False, temperature=-1.0, top_k=-1,
                calulate_logprobs=False, calulate_values=False,
            )
            return predicted.float().cpu().numpy()[0]

    case = runner.run(episode_id, predict, trace=config["trace"])
    case["source"] = config["source"]
    case["expert_actions"] = int(len(expert))
    return case


def main() -> int:
    cli = parse_args()
    robotwin_root = cli.robotwin_root.resolve()
    if cli.execution_horizon < 1 or cli.execution_horizon > cli.action_chunk:
        raise ValueError("execution horizon must be within the action chunk")
    config = {
        "robotwin_root": str(robotwin_root),
        "dataset_root": str(cli.dataset_root.resolve()),
        "scene_registry": str(cli.scene_registry.resolve()),
        "task_name": cli.task_name,
        "task_config": cli.task_config,
        "model": str(cli.model.resolve()),
        "unnorm_key": cli.unnorm_key,
        "action_chunk": cli.action_chunk,
        "execution_horizon": cli.execution_horizon,
        "max_actions": cli.max_actions,
        "source": cli.source,
        "trace": cli.trace,
    }

    devices = None
    context = mp.get_context("spawn")
    if cli.gpus:
        gpu_ids = [value.strip() for value in cli.gpus.split(",") if value.strip()]
        devices = context.Queue()
        for index in range(cli.workers):
            devices.put(gpu_ids[index % len(gpu_ids)])

    cases: list[dict] = []
    with context.Pool(
        processes=min(cli.workers, len(cli.episode)),
        initializer=_init_worker,
        initargs=(config, devices),
    ) as pool:
        for case in pool.imap_unordered(_run_job, cli.episode):
            cases.append(case)
            print(
                f"  [{len(cases):3d}/{len(cli.episode)}] ep{case['episode']:5d} "
                f"{case['posture']:8s} success={case.get('success')} "
                f"动作={case.get('executed_actions')}/{case['expert_actions']} "
                f"查询={case.get('policy_queries')} err={case.get('error', '')}",
                flush=True,
            )
    cases.sort(key=lambda case: case["episode"])

    summary = {
        "source": cli.source,
        "model": str(cli.model),
        "execution_horizon": cli.execution_horizon,
        "max_actions": cli.max_actions,
        "by_posture": {
            posture: {
                "n": sum(1 for c in cases if c["posture"] == posture),
                "success": sum(
                    1 for c in cases if c["posture"] == posture and c.get("success")
                ),
            }
            for posture in sorted({c["posture"] for c in cases})
        },
        "cases": cases,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print()
    total = sum(1 for c in cases if c.get("success"))
    print(f"  {cli.source:8s} {total}/{len(cases)}")
    for posture, value in summary["by_posture"].items():
        print(f"  {posture:8s} {value['success']}/{value['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
