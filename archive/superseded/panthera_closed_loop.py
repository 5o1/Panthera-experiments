"""One closed-loop rollout implementation, shared by training and auditing.

Training needs closed-loop success as its stopping signal because validation
loss does not predict it here -- the historical model that grasped reliably
scored a worse chunk L1 than the model that scored zero.  Auditing needs the
same rollout to explain a failure.  Two copies of this loop would drift, and the
whole point is that the number the trainer stops on is the number the audit
reports, so the loop lives here and both import it.

The policy enters as a ``predict`` callable rather than a model, which keeps
this module free of OpenVLA and testable against a scripted stand-in.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np

SAMPLE_PERIOD_STEPS = 5
PHYSICS_DT = 1.0 / 250.0


class Predict(Protocol):
    """Return a [chunk, 7] array of absolute joint targets plus gripper."""

    def __call__(
        self, image: np.ndarray, state: np.ndarray, instruction: str
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class RolloutConfig:
    robotwin_root: Path
    dataset_root: Path
    scene_registry: Path
    task_config: str
    task_name: str
    execution_horizon: int = 20
    max_actions: int = 3200

    def __post_init__(self) -> None:
        if self.execution_horizon < 1:
            raise ValueError("execution_horizon must be positive")
        if self.max_actions < 1:
            raise ValueError("max_actions must be positive")


@contextlib.contextmanager
def _inside(root: Path):
    """Run with RoboTwin's cwd, then put the caller's back.

    The training process has already chdir'd into the OpenVLA package root, and
    leaving it somewhere else would break checkpointing after validation.
    """
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


class ClosedLoopRunner:
    """Roll a policy through the executor that reproduces the expert."""

    def __init__(self, config: RolloutConfig) -> None:
        self.config = config
        root = Path(config.robotwin_root).resolve()
        os.environ["ASSETS_PATH"] = str(root)
        os.environ.setdefault("MUJOCO_GL", "egl")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        with _inside(root):
            module = importlib.import_module(f"envs.{config.task_name}")
            self._task_class = getattr(module, config.task_name)
            from audit_dense_execution import _task_args

            self._base_args = _task_args(root, config.task_config, config.task_name)
        self._scene = json.loads(
            (Path(config.dataset_root) / "scene_info.json").read_text(encoding="utf-8")
        )

    def _instruction(self, episode_id: int) -> str:
        path = (
            Path(self.config.dataset_root)
            / "instructions"
            / f"episode{episode_id}.json"
        )
        values = json.loads(path.read_text(encoding="utf-8"))["seen"]
        if not values:
            raise ValueError(f"episode {episode_id} has no seen instruction")
        return values[episode_id % len(values)]

    def observe(self, task) -> tuple[np.ndarray, np.ndarray]:
        """Build the observation the evaluation harness would build."""
        from PIL import Image
        from rlinf.envs.utils import center_crop_image

        observation = task.get_obs()
        frame = np.asarray(observation["observation"]["head_camera"]["rgb"])
        image = np.asarray(
            center_crop_image(Image.fromarray(frame).convert("RGB")), dtype=np.uint8
        )
        state = np.asarray(
            observation["observation"]["robot_state"]["vector"], dtype=np.float32
        )
        return image, state

    @staticmethod
    def _apply(task, target: np.ndarray, previous: np.ndarray) -> np.ndarray:
        arm = np.asarray(target[:6], dtype=np.float64)
        velocity = (arm - previous) / (SAMPLE_PERIOD_STEPS * PHYSICS_DT)
        gripper = float(np.clip(target[6], 0.0, 1.0))
        for _ in range(SAMPLE_PERIOD_STEPS):
            task.robot.set_arm_joints(arm, velocity, "left")
            task.robot.set_gripper(gripper, "left")
            task._step_scene()
        return arm

    def run(self, episode_id: int, predict: Predict, trace: bool = False) -> dict:
        """Roll one episode and report whether the task was solved."""
        from audit_dense_execution import _use_scene_registry, _verify_scene

        metadata = self._scene[f"episode_{episode_id}"]["panthera_episode"]
        seed = int(metadata["episode_seed"])
        case = {
            "episode": episode_id,
            "seed": seed,
            "posture": metadata["realized_geometry"]["cylinder_posture"],
        }
        rows: list[dict] = []
        root = Path(self.config.robotwin_root).resolve()
        with _inside(root):
            task = self._task_class()
            try:
                args = copy.deepcopy(self._base_args)
                args["step_lim"] = int(self.config.max_actions)
                _use_scene_registry(args, str(self.config.scene_registry))
                task.setup_demo(now_ep_num=seed, seed=seed, **args)
                case["scene"] = _verify_scene(task, metadata)
                if not case["scene"]["matches_recording"]:
                    raise ValueError("rebuilt scene does not match the recording")

                instruction = self._instruction(episode_id)
                previous = np.asarray(
                    task.robot.get_left_arm_real_jointState()[:6], dtype=np.float64
                )
                executed = 0
                queries = 0
                while executed < self.config.max_actions and not task.check_success():
                    image, state = self.observe(task)
                    chunk = np.asarray(predict(image, state, instruction), dtype=np.float64)
                    if chunk.ndim != 2 or chunk.shape[1] != 7:
                        raise ValueError(f"predict must return [N, 7], got {chunk.shape}")
                    queries += 1
                    for row in chunk[: self.config.execution_horizon]:
                        previous = self._apply(task, row, previous)
                        executed += 1
                        if trace:
                            pose = task.cylinder.get_pose()
                            rows.append(
                                {
                                    "action": executed,
                                    "commanded": [float(v) for v in row],
                                    "cylinder_xyz": [float(v) for v in pose.p],
                                }
                            )
                        if executed >= self.config.max_actions or task.check_success():
                            break
                if not task.check_success():
                    task._configure_release_contact_solver()
                    task._advance_until_inserted_stable()
                case.update(
                    {
                        "success": bool(task.check_success()),
                        "executed_actions": executed,
                        "policy_queries": queries,
                        "metrics": {
                            key: (float(v) if not isinstance(v, bool) else v)
                            for key, v in task.success_metrics().items()
                        },
                    }
                )
                if trace:
                    case["trace"] = rows
            except Exception as error:
                case["error"] = f"{type(error).__name__}: {error}"
                case["success"] = False
            finally:
                with contextlib.suppress(Exception):
                    task.close_env()
        return case

    def evaluate(self, episodes: Iterable[int], predict: Predict) -> dict:
        """Roll every episode and report the pass rate the trainer stops on."""
        cases = [self.run(episode, predict) for episode in episodes]
        successes = sum(1 for case in cases if case.get("success"))
        return {
            "successes": successes,
            "trials": len(cases),
            "pass_rate": successes / len(cases) if cases else 0.0,
            "cases": cases,
        }
