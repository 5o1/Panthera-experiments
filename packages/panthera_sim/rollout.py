#!/usr/bin/env python3
"""Roll a policy through the same executor the expert replay is measured on.

Holding the executor fixed is the whole point: the replay establishes what the
environment can reproduce, and a policy number only means something against that
ceiling.  The policy enters as a ``predict`` callable, so this module stays free
of OpenVLA and the expert can be run through the identical loop as a control.
"""

from __future__ import annotations

import math

import argparse
import contextlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from dataset import open_dataset
from executor import DenseExecutor, ExecutorConfig
from robotwin_env import SceneMismatch, build_task, inside, task_args

_VLA_PACKAGE = Path(__file__).resolve().parents[1] / "panthera_vla"
if str(_VLA_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_VLA_PACKAGE))

DEFAULT_TASK = "place_randomized_cylinder_in_socket"

# ``step`` is the number of actions already executed.  A policy ignores it; an
# expert stream needs it, because indexing by query count instead drifts from
# the executed count whenever a chunk ends early on success.
Predict = Callable[[np.ndarray, np.ndarray, str, int], np.ndarray]


def classify_completion(
    success: bool, executed_actions: int, expert_action_budget: int
) -> str:
    """Classify success without confusing the expert deadline with a timeout."""
    if expert_action_budget <= 0:
        raise ValueError("expert_action_budget must be positive")
    if executed_actions < 0:
        raise ValueError("executed_actions must be nonnegative")
    if not success:
        return "failure"
    if executed_actions <= expert_action_budget:
        return "on_time_success"
    return "delayed_success"


def summarize_inference_timing(
    queries: list[dict], executed_actions: int, control_hz: float
) -> dict:
    """Report raw and per-executed-action latency against the control deadline."""
    if not queries:
        return {}
    control_period_ms = 1000.0 / float(control_hz)
    starts = [int(query["row"]) for query in queries]
    ends = starts[1:] + [int(executed_actions)]
    spans = np.asarray(
        [max(1, end - start) for start, end in zip(starts, ends)], dtype=np.float64
    )
    query_ms = np.asarray([float(query["inference_ms"]) for query in queries])
    action_ms = query_ms / spans
    realtime_load = action_ms / control_period_ms

    def distribution(values: np.ndarray) -> dict:
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    return {
        "measurement_scope": (
            "frame-ready policy path: image preprocessing, host-to-device input, "
            "model inference, and device-to-host action"
        ),
        "excluded": [
            "simulator rendering",
            "physics stepping",
            "video rendering and encoding",
            "real camera capture, transport, and decode",
            "ROS transport and robot controller",
        ],
        "real_workbench_interpretation": (
            "lower-bound policy latency once a decoded camera frame and robot "
            "state are available in memory"
        ),
        "control_hz": float(control_hz),
        "control_period_ms_per_action": control_period_ms,
        "query_ms": distribution(query_ms),
        "amortized_ms_per_action": distribution(action_ms),
        "realtime_load_x": distribution(realtime_load),
        "deadline_miss_queries": int(np.count_nonzero(realtime_load > 1.0)),
        "queries": int(len(queries)),
    }


class TemporalEnsemble:
    """Average the overlapping chunks that predict the same step.

    The model predicts 25 steps at a time and the harness executed them
    verbatim, so every commanded step carried the model's full per-step output
    noise.  Measured on the single-trajectory overfit model, episode 2: the
    expert holds still for 26.9% of the trajectory -- four settling blocks
    around the gripper closing at step 231, opening at 801, and at each end --
    and moves 5.23 mrad per step in between.  The policy held still for 0.0% of
    its rollout and moved 13.71 mrad per step, peaking at 76.76 against the
    expert's 20.06, for 168.2 rad of joint travel against 10.9.  It never
    settled, so it never reached the state the release is predicted from.

    Averaging K overlapping predictions cuts that noise roughly as sqrt(K), and
    where the underlying prediction is "stay here" the average converges instead
    of dithering.  Weights decay exponentially with the age of the chunk, so a
    fresher observation still dominates.  This is the ACT scheme and it needs no
    retraining.
    """

    def __init__(self, coefficient: float = 0.01, horizon: int = 64):
        if coefficient < 0.0:
            raise ValueError(f"ensemble coefficient must be nonnegative: {coefficient}")
        self.coefficient = float(coefficient)
        self.horizon = int(horizon)
        self._chunks: list[tuple[int, np.ndarray]] = []

    def add(self, row: int, chunk: np.ndarray) -> None:
        self._chunks.append((int(row), np.asarray(chunk, dtype=np.float64)))
        # A chunk can only speak about steps it covers; keeping more of them
        # would only cost memory.
        if len(self._chunks) > self.horizon:
            del self._chunks[: len(self._chunks) - self.horizon]

    def action(self, row: int) -> np.ndarray:
        targets, weights = [], []
        # Newest first, so age 0 is the most recent observation.
        for age, (start, chunk) in enumerate(reversed(self._chunks)):
            offset = row - start
            if 0 <= offset < len(chunk):
                targets.append(chunk[offset])
                weights.append(math.exp(-self.coefficient * age))
        if not targets:
            raise ValueError(f"no chunk covers step {row}")
        weight = np.asarray(weights, dtype=np.float64)
        return np.asarray(targets, dtype=np.float64).T @ (weight / weight.sum())

    def depth(self, row: int) -> int:
        return sum(1 for start, chunk in self._chunks if 0 <= row - start < len(chunk))


def run_episode(
    robotwin_root: Path, task_config: str, task_name: str, dataset, episode_id: int,
    predict: Predict, execution_horizon: int, max_actions: int,
    executor_config: Optional[ExecutorConfig] = None, trace: bool = False,
    base_args=None, ratchet=None, observe=None, reference_qpos=None,
    center_crop: bool = False, ensemble: Optional[TemporalEnsemble] = None,
    relative_chunk: bool = False, proprio_dim: int = 7,
) -> dict:
    """Roll one episode in its recorded scene and report whether it was solved."""
    episode = dataset.episode(episode_id)
    case = {
        "episode": episode_id,
        "seed": episode.seed,
        "posture": episode.posture,
    }
    root = Path(robotwin_root)
    task = None
    with inside(root):
        try:
            task, case["scene"] = build_task(
                root, task_config, task_name, episode, dataset.scenes_path,
                max_actions, base_args,
                camera=dataset.camera(episode_id),
                camera_pose=dataset.camera_pose(episode_id),
            )
            executor = DenseExecutor(
                task, executor_config or ExecutorConfig(), ratchet=ratchet,
                trace=trace, reference_qpos=reference_qpos,
            )
            instruction = episode.instruction()
            reader = observe or (
                lambda env: observe_task(env, center_crop, proprio_dim=proprio_dim)
            )
            queries = 0
            prediction_trace = []
            query_timings = []
            while executor.rows < max_actions and not executor.succeeded():
                # Observation acquisition deliberately stays outside this timer:
                # the metric estimates the real-table policy path after a decoded
                # camera frame and robot state are already available in memory.
                # It therefore excludes simulator rendering and physics work.
                image, state = reader(task)
                inference_started = time.perf_counter()
                chunk = np.asarray(
                    predict(image, state, instruction, executor.rows), dtype=np.float64
                )
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                if chunk.ndim != 2 or chunk.shape[1] != 7:
                    raise ValueError(f"predict must return [N, 7], got {chunk.shape}")
                query_timings.append(
                    {"row": int(executor.rows), "inference_ms": float(inference_ms)}
                )
                if trace:
                    prediction_trace.append(
                        {
                            "row": int(executor.rows),
                            "inference_ms": float(inference_ms),
                            "actions": chunk.tolist(),
                        }
                    )
                queries += 1
                if relative_chunk:
                    # The model keeps predicting the right shape and the wrong
                    # anchor.  Measured at the stall, episode 2: the chunk's
                    # first target sat 65.1 mrad from the arm's actual joints
                    # while the chunk itself only travelled 26 mrad end to end,
                    # so the executor spent the whole chunk chasing an offset
                    # instead of following the motion.  Re-anchoring on the arm
                    # keeps the shape and drops the offset.  The gripper is a
                    # command rather than a pose, so it stays absolute.
                    here = np.asarray(
                        task._actual_robot_state()["arm_qpos"], dtype=np.float64
                    )
                    chunk = chunk.copy()
                    chunk[:, :6] += here - chunk[0, :6]
                if ensemble is None:
                    for target in chunk[:execution_horizon]:
                        executor.step(target)
                        if executor.rows >= max_actions or executor.succeeded():
                            break
                else:
                    ensemble.add(executor.rows, chunk)
                    for _ in range(execution_horizon):
                        executor.step(ensemble.action(executor.rows))
                        if executor.rows >= max_actions or executor.succeeded():
                            break
            case["policy_queries"] = queries
            case["inference_timing"] = summarize_inference_timing(
                query_timings, executor.rows, dataset.contract.control_hz
            )
            if trace:
                case["prediction_trace"] = prediction_trace
            case.update(executor.finish())
        except SceneMismatch as error:
            case.update({"success": False, "error": str(error)})
        except Exception as error:
            case.update({"success": False, "error": f"{type(error).__name__}: {error}"})
        finally:
            if task is not None:
                with contextlib.suppress(Exception):
                    task.close_env()
    return case


def observe_task(
    task, center_crop: bool = False, *, proprio_dim: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Build the observation the model was trained to read.

    The crop is not a preprocessing preference.  Upstream applies it only when
    ``cfg.center_crop`` is set, and its own docstring says it exists "to match
    training data distribution" -- it compensates for the random crops that
    ``--image_aug`` introduces.  This harness cropped unconditionally while
    training ran with ``--image_aug false``, so evaluation fed a 0.9-scale crop
    of a distribution the model had never seen.  The flag therefore comes from
    the checkpoint's observation contract, never from here.
    """
    from PIL import Image

    observation = task.get_obs()
    frame = np.asarray(observation["observation"]["head_camera"]["rgb"])
    picture = Image.fromarray(frame).convert("RGB")
    if center_crop:
        from rlinf.envs.utils import center_crop_image

        picture = center_crop_image(picture)
    image = np.asarray(picture, dtype=np.uint8)
    state_key = {7: "vector", 28: "dynamics_vector"}.get(proprio_dim)
    if state_key is None:
        raise ValueError(f"unsupported Panthera proprioception width: {proprio_dim}")
    robot_state = observation["observation"]["robot_state"]
    if state_key not in robot_state:
        raise ValueError(
            f"environment does not provide {state_key} for {proprio_dim}-D proprioception"
        )
    state = np.asarray(robot_state[state_key], dtype=np.float32)
    if state.shape != (proprio_dim,) or not np.all(np.isfinite(state)):
        raise ValueError(
            f"invalid {proprio_dim}-D proprioception: shape={state.shape}, "
            f"finite={bool(np.all(np.isfinite(state)))}"
        )
    return image, state


_WORKER: dict = {}


def _init_worker(config: dict, devices) -> None:
    import os
    import sys

    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices.get()
    os.environ.setdefault("ROBOT_PLATFORM", config["robot_platform"])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    dataset = open_dataset(config["dataset_root"])
    _WORKER.update(
        {
            "config": config,
            "dataset": dataset,
            "base_args": task_args(
                config["robotwin_root"], config["task_config"], config["task_name"]
            ),
        }
    )
    if config["source"] == "policy":
        from policy_backends import create_backend

        _WORKER["backend"] = create_backend(config)


def _run(episode_id: int) -> dict:
    config = _WORKER["config"]
    dataset = _WORKER["dataset"]
    if config["source"] == "expert":
        from replay import grid_actions

        expert = grid_actions(dataset.episode(episode_id).hdf5_path)

        def predict(image, state, instruction, step):
            del image, state, instruction
            window = expert[step : step + config["execution_horizon"]]
            if len(window) == 0:
                # The stream has run out; holding the last target keeps the
                # episode running to its budget instead of ending on a shape error.
                return np.repeat(expert[-1][None, :], config["execution_horizon"], axis=0)
            return window
    else:
        backend = _WORKER["backend"]

        def predict(image, state, instruction, step):
            del step
            return backend.predict(image, state, instruction)

    offline_validation = None
    if config["offline_validation_loss"] and config["source"] == "policy":
        if config["policy_backend"] != "openvla-oft":
            raise ValueError(
                "--offline-validation-loss currently implements OpenVLA's "
                "bounds-q99 metric only"
            )
        from parity import offline_validation_loss

        offline_validation = offline_validation_loss(
            _WORKER["backend"].native_model, dataset.episode(episode_id),
            action_chunk=config["action_chunk"],
            batch_size=config["offline_validation_batch_size"],
            include_per_frame=config["offline_validation_trace"],
            proprio_dim=config["proprio_dim"],
        )
        print(
            "离线验证 L1（归一化 25x7 全量动作块）："
            f"{offline_validation['normalized_l1']:.8f}",
            flush=True,
        )

    parity_report = None
    if config["parity"] and config["source"] == "policy":
        # An evaluation result is an artefact, and it carries the measurement
        # that says its pipeline was faithful.  Without that the success rate
        # is uninterpretable: the 0/18 and 0/12 in docs/18 were recorded as
        # policy results while the harness was rendering a scene the model had
        # never seen.
        from parity import describe, measure

        report = measure(
            config["robotwin_root"], config["task_config"], config["task_name"],
            dataset, episode_id, None, config["parity_samples"],
            config["action_chunk"], predict=_WORKER["backend"].predict,
            proprio_dim=config["proprio_dim"],
        )
        parity_report = report
        if not report["passed"]:
            message = "推理一致性未通过，评测数字不可解读：\n" + describe(report)
            if not config["allow_parity_failure"]:
                raise SystemExit(
                    message
                    + "\n加 --allow-parity-failure 只生成诊断视频；其成功率不计入门禁"
                )
            print("警告：" + message + "\n继续生成诊断视频，但不计为门禁通过。", flush=True)

    case = run_episode(
        Path(config["robotwin_root"]), config["task_config"], config["task_name"],
        dataset, episode_id, predict, config["execution_horizon"],
        config["max_actions"], ExecutorConfig(**config["executor"]),
        trace=config["trace"], base_args=_WORKER["base_args"],
        center_crop=config["center_crop"],
        ensemble=(
            TemporalEnsemble(config["ensemble_coefficient"])
            if config["ensemble_coefficient"] is not None
            else None
        ),
        relative_chunk=config["relative_chunk"],
        proprio_dim=config["proprio_dim"],
    )
    case["source"] = config["source"]
    if parity_report is not None:
        case["parity"] = parity_report
    if offline_validation is not None:
        case["offline_validation"] = offline_validation
    completion_class = classify_completion(
        bool(case.get("success")),
        int(case.get("executed_actions", 0)),
        int(config["expert_action_budget"]),
    )
    parity_passed = parity_report is None or bool(parity_report.get("passed"))
    case["expert_action_budget"] = int(config["expert_action_budget"])
    case["evaluation_action_budget"] = int(config["max_actions"])
    case["completion_class"] = completion_class
    case["on_time_success"] = completion_class == "on_time_success"
    case["delayed_success"] = completion_class == "delayed_success"
    case["diagnostic_success"] = bool(case.get("success") and parity_passed)
    # The primary gate keeps the historical expert-duration contract.  The
    # longer rollout only diagnoses policies that solve the task more slowly.
    case["gate_success"] = bool(case["on_time_success"] and parity_passed)
    return case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--policy-backend",
        choices=("openvla-oft", "openpi-pi05"),
        default="openvla-oft",
    )
    parser.add_argument(
        "--openpi-repo-id",
        help="LeRobot repo id recorded in the π0.5 checkpoint, e.g. panthera/schema10",
    )
    parser.add_argument("--openpi-lora", action="store_true")
    parser.add_argument("--openpi-device", default=None)
    parser.add_argument(
        "--openpi-training-manifest",
        type=Path,
        help="immutable *.panthera-training.json written by the OpenPI launcher",
    )
    # The model's own contract supplies these; the overrides exist only for a
    # checkpoint that predates training.json, and a run that uses them says so.
    parser.add_argument("--unnorm-key")
    parser.add_argument("--robot-platform")
    parser.add_argument("--action-chunk", type=int)
    parser.add_argument("--allow-dataset-mismatch", action="store_true",
                        help="evaluate on a dataset the model was not trained "
                             "on; the mismatch is recorded in the summary")
    parser.add_argument(
        "--no-parity", action="store_true",
        help=(
            "skip the inference-parity measurement. The resulting success rate "
            "cannot be read as a statement about the policy"
        ),
    )
    parser.add_argument("--parity-samples", type=int, default=8)
    parser.add_argument(
        "--allow-parity-failure", action="store_true",
        help="continue only to produce diagnostic traces/videos when parity fails",
    )
    parser.add_argument(
        "--offline-validation-loss", action="store_true",
        help="recompute full recorded-frame validation L1 for this checkpoint",
    )
    parser.add_argument("--offline-validation-batch-size", type=int, default=2)
    parser.add_argument(
        "--offline-validation-trace", action="store_true",
        help="retain one validation-loss record per 50 Hz trajectory frame",
    )
    parser.add_argument(
        "--relative-chunk", action="store_true",
        help=(
            "re-anchor each predicted chunk on the arm's actual joints, keeping "
            "its shape and dropping its absolute offset"
        ),
    )
    parser.add_argument(
        "--temporal-ensemble", type=float, default=None, metavar="COEFFICIENT",
        help=(
            "average the overlapping action chunks, ACT style, with this "
            "exponential age decay (0.01 is the usual value). The policy never "
            "held still without it: 0.0%% of rollout steps against the expert's "
            "26.9%%, at 13.71 mrad per step against 5.23."
        ),
    )
    parser.add_argument("--execution-horizon", type=int, default=20)
    parser.add_argument(
        "--expert-action-budget",
        type=int,
        default=0,
        help="0 uses the recorded expert trajectory length as the on-time gate",
    )
    parser.add_argument("--max-actions", type=int, default=0,
                        help="0 stops at the expert budget; pass a larger diagnostic timeout")
    parser.add_argument("--source", choices=("policy", "expert"), default="policy")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    dataset = open_dataset(cli.dataset_root)
    expert_budget = cli.expert_action_budget or dataset.contract.required_action_budget
    budget = cli.max_actions or expert_budget
    if expert_budget <= 0:
        raise SystemExit("expert action budget must be positive")
    if budget < expert_budget:
        raise SystemExit(
            f"max actions {budget} is smaller than expert budget {expert_budget}"
        )

    contract = None
    observation = None
    dataset_mismatch = None
    if cli.source == "policy" and cli.policy_backend == "openvla-oft":
        if cli.model is None:
            raise SystemExit("--model is required for a policy rollout")
        try:
            from checkpoint import open_checkpoint

            checkpoint = open_checkpoint(cli.model)
            contract = checkpoint.policy
            observation = checkpoint.observation
            dataset_mismatch = checkpoint.check_dataset(dataset)
        except Exception as error:
            if cli.unnorm_key is None:
                raise SystemExit(
                    f"{cli.model} has no usable contract ({error}); pass "
                    "--unnorm-key and --action-chunk to run it anyway"
                ) from error
            print(f"警告：未能读取模型契约（{error}），改用命令行参数。", flush=True)
        if dataset_mismatch and not cli.allow_dataset_mismatch:
            raise SystemExit(
                f"{dataset_mismatch}\n"
                "pass --allow-dataset-mismatch if that is intended"
            )
        if dataset_mismatch:
            print(f"警告：{dataset_mismatch}", flush=True)
    elif cli.source == "policy":
        if cli.model is None:
            raise SystemExit("--model is required for a policy rollout")
        if not cli.openpi_repo_id:
            raise SystemExit("--openpi-repo-id is required for the OpenPI backend")
        if cli.openpi_training_manifest is None:
            raise SystemExit("--openpi-training-manifest is required for the OpenPI backend")
        training = json.loads(cli.openpi_training_manifest.read_text(encoding="utf-8"))
        mismatches = []
        if training.get("source_dataset_digest") != dataset.digest():
            mismatches.append("source dataset digest differs")
        if training.get("repo_id") != cli.openpi_repo_id:
            mismatches.append("LeRobot repo id differs")
        requested_chunk = cli.action_chunk or 25
        if int(training.get("action_horizon", -1)) != requested_chunk:
            mismatches.append("action horizon differs")
        if bool(training.get("lora")) != bool(cli.openpi_lora):
            mismatches.append("LoRA architecture flag differs")
        dataset_mismatch = "; ".join(mismatches) or None
        if dataset_mismatch and not cli.allow_dataset_mismatch:
            raise SystemExit(
                f"OpenPI training contract mismatch: {dataset_mismatch}\n"
                "pass --allow-dataset-mismatch only for an intentional diagnostic"
            )
    config = {
        "dataset_root": str(Path(cli.dataset_root).resolve()),
        "dataset_digest": dataset.digest(),
        "robotwin_root": str(Path(cli.robotwin_root).resolve()),
        "task_config": cli.task_config,
        "task_name": cli.task_name,
        "model": str(cli.model.resolve()) if cli.model else None,
        "openpi_repo_id": cli.openpi_repo_id,
        "openpi_lora": bool(cli.openpi_lora),
        "openpi_device": cli.openpi_device,
        "unnorm_key": cli.unnorm_key or (contract.unnorm_key if contract else None),
        "robot_platform": (
            cli.robot_platform or (contract.robot_platform if contract else "PANTHERA")
        ),
        "action_chunk": cli.action_chunk or (contract.action_chunk if contract else 25),
        "use_proprio": contract.use_proprio if contract else True,
        "proprio_dim": contract.proprio_dim if contract else 7,
        "use_l1_regression": contract.use_l1_regression if contract else True,
        "num_images_in_input": contract.num_images_in_input if contract else 1,
        # No contract means no evidence the model was trained with crops, and
        # cropping without that evidence is what broke evaluation before.
        "center_crop": bool(observation.center_crop) if observation else False,
        "ensemble_coefficient": cli.temporal_ensemble,
        "relative_chunk": bool(cli.relative_chunk),
        "parity": not cli.no_parity,
        "parity_samples": cli.parity_samples,
        "allow_parity_failure": cli.allow_parity_failure,
        "offline_validation_loss": cli.offline_validation_loss,
        "offline_validation_batch_size": cli.offline_validation_batch_size,
        "offline_validation_trace": cli.offline_validation_trace,
        "execution_horizon": cli.execution_horizon,
        "expert_action_budget": expert_budget,
        "max_actions": budget,
        "source": cli.source,
        "policy_backend": cli.policy_backend if cli.source == "policy" else None,
        "trace": cli.trace,
        "executor": {},
    }

    context = mp.get_context("spawn")
    devices = None
    if cli.gpus:
        ids = [value.strip() for value in cli.gpus.split(",") if value.strip()]
        devices = context.Queue()
        for index in range(cli.workers):
            devices.put(ids[index % len(ids)])

    cases: list[dict] = []
    with context.Pool(
        processes=min(cli.workers, len(cli.episode)),
        initializer=_init_worker, initargs=(config, devices),
    ) as pool:
        for case in pool.imap_unordered(_run, cli.episode):
            cases.append(case)
            print(
                f"  [{len(cases):3d}/{len(cli.episode)}] ep{case['episode']:5d} "
                f"{case['posture']:8s} success={case.get('success')} "
                f"class={case.get('completion_class')} "
                f"动作={case.get('executed_actions')}/{budget} "
                f"查询={case.get('policy_queries')} {case.get('error', '')}",
                flush=True,
            )
    cases.sort(key=lambda case: case["episode"])
    summary = {
        "source": cli.source,
        "policy_backend": config["policy_backend"],
        "model": config["model"],
        "policy_contract_source": (
            "checkpoint" if contract else
            "openpi training manifest" if cli.policy_backend == "openpi-pi05" else
            "command line"
        ),
        "dataset_mismatch": dataset_mismatch,
        "dataset": dataset.name,
        "dataset_digest": dataset.digest(),
        "expert_action_budget": expert_budget,
        "max_actions": budget,
        "execution_horizon": cli.execution_horizon,
        "temporal_ensemble": cli.temporal_ensemble,
        "total": len(cases),
        "success": sum(1 for c in cases if c.get("success")),
        "on_time_success": sum(1 for c in cases if c.get("on_time_success")),
        "delayed_success": sum(1 for c in cases if c.get("delayed_success")),
        "diagnostic_success": sum(1 for c in cases if c.get("diagnostic_success")),
        "gate_success": sum(1 for c in cases if c.get("gate_success")),
        "by_posture": {
            posture: {
                "n": sum(1 for c in cases if c["posture"] == posture),
                "success": sum(1 for c in cases if c["posture"] == posture and c.get("success")),
            }
            for posture in sorted({c["posture"] for c in cases})
        },
        "cases": cases,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"  {cli.source:8s} {summary['success']}/{summary['total']}")
    for posture, value in summary["by_posture"].items():
        print(f"  {posture:8s} {value['success']}/{value['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
