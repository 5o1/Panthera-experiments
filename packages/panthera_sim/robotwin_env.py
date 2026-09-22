"""Build a RoboTwin task from a dataset, and check it is the recorded one.

RoboTwin resolves task modules, assets and configs relative to its own tree, so
constructing a task means entering that tree.  The caller's working directory is
restored afterwards, because a training process that runs a closed-loop
validation has already chdir'd somewhere it needs to stay.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

ORIENTATION_TOLERANCE_DEG = 1.0
POSITION_TOLERANCE_M = 1e-3


class SceneMismatch(RuntimeError):
    """The rebuilt scene is not the one the episode was recorded in."""


@contextlib.contextmanager
def inside(root: Path):
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def prepare(robotwin_root: Path) -> None:
    """Make RoboTwin importable without moving our own code into its tree."""
    root = Path(robotwin_root).resolve()
    os.environ["ASSETS_PATH"] = str(root)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def task_class(robotwin_root: Path, task_name: str):
    prepare(robotwin_root)
    with inside(Path(robotwin_root).resolve()):
        module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)


def _task_config_root(root: Path) -> Path:
    """Where this tree keeps its task configs.

    RoboTwin moved them under ``env_cfg/`` between 0008ae6 and 6dde571, and the
    pre-migration checkout is still on disk for the entry points that have not
    been moved to the assembled runtime, so both layouts have to resolve.
    """
    for relative in ("env_cfg/task_config", "task_config"):
        candidate = root / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"no task_config directory under {root}")


def task_args(robotwin_root: Path, task_config: str, task_name: str) -> dict:
    """Resolve RoboTwin's task configuration into the kwargs setup_demo takes."""
    root = Path(robotwin_root).resolve()
    if Path(task_config).name != task_config:
        raise ValueError("task config must be a filename")
    configs = _task_config_root(root)
    with (configs / task_config).open("r", encoding="utf-8") as stream:
        args = yaml.safe_load(stream)
    with (configs / "_embodiment_config.yml").open("r", encoding="utf-8") as stream:
        registry = yaml.safe_load(stream)
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
        robot_root = (root / registry[name]["file_path"]).resolve()
        args[f"{side}_robot_file"] = str(robot_root)
        with (robot_root / "config.yml").open("r", encoding="utf-8") as stream:
            args[f"{side}_embodiment_config"] = yaml.safe_load(stream)
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


def bind_scenes(args: dict, scenes_path: Path, camera: Mapping[str, Any]) -> dict:
    """Point the task at the dataset's scene table and camera, not at a seed.

    A seed does not identify a scene: collection overrode the seed's posture and
    lying angle per shard and recorded only the outcome, so rebuilding from a
    seed alone lands on a different posture for about half the episodes.  The
    collection-side forcing flags are removed here so they cannot re-sample
    inside the forced sector instead of using the recorded angle.

    The camera is part of the scene for the same reason.  The task config named
    on the command line decides how the head camera is placed, and the pilot
    config samples it (``target_xy_jitter_m: 0.06``, no ``pose_mode``) while the
    dataset was collected with a pinned pose.  Evaluating a dataset with another
    config's camera put the head camera 47.6 cm away from where the episode was
    recorded, looking from the other side of the table.  Measured on the
    single-trajectory overfit model, episode 2: the action chunk was off by a
    median 187.2 mrad from the sampled camera and 3.9 mrad from the recorded
    one, against 2.7 mrad for the recorded frame itself.  Every policy number in
    ``docs/18`` was taken through the sampled camera.
    """
    randomization = args.setdefault("task_randomization", {})
    randomization["scene_registry"] = str(scenes_path)
    randomization["scene_registry_required"] = True
    randomization["camera_randomization"] = dict(camera)
    randomization.pop("forced_posture", None)
    randomization.pop("forced_lying_angle_bin", None)
    return args


def verify_scene(task, scene: Mapping[str, Any]) -> dict:
    """Confirm the placed actors match the recorded geometry."""
    expected_quaternion = np.asarray(scene["cylinder_quaternion_wxyz"], dtype=np.float64)
    pose = task.cylinder.get_pose()
    actual = np.asarray(pose.q, dtype=np.float64)
    alignment = abs(float(np.dot(expected_quaternion, actual)))
    orientation_error_deg = float(
        np.degrees(2.0 * np.arccos(np.clip(alignment, -1.0, 1.0)))
    )
    position_error_m = float(
        np.linalg.norm(
            np.asarray(pose.p, dtype=np.float64)[:2]
            - np.asarray(scene["cylinder_initial_xy_m"], dtype=np.float64)
        )
    )
    report = {
        "orientation_error_deg": orientation_error_deg,
        "position_error_m": position_error_m,
        "matches_recording": bool(
            orientation_error_deg < ORIENTATION_TOLERANCE_DEG
            and position_error_m < POSITION_TOLERANCE_M
        ),
    }
    return report


# The recorded pose is restored exactly, so anything above rounding means the
# camera was placed by a config rather than by the recording.
CAMERA_TOLERANCE_M = 1.0e-4


def _camera_error(task, camera_pose) -> float:
    """How far the head camera sits from where the episode was recorded."""
    cameras = task.scene.get_cameras()
    if not cameras:
        raise SceneMismatch("scene has no cameras to check against the recording")
    placed = np.asarray(cameras[0].get_entity_pose().p, dtype=float)
    return float(np.linalg.norm(placed - np.asarray(camera_pose, dtype=float)))


def build_task(robotwin_root: Path, task_config: str, task_name: str,
               episode, scenes_path: Path, step_limit: int, base_args=None,
               *, camera: Mapping[str, Any], camera_pose: Sequence[float]):
    """Construct one task in the episode's recorded scene and camera, or raise.

    ``camera`` and ``camera_pose`` are keyword-only and have no defaults on
    purpose: a default would let a caller silently fall back to whatever camera
    the task config happens to name, which is the defect this signature exists
    to close.
    """
    root = Path(robotwin_root).resolve()
    cls = task_class(root, task_name)
    args = copy.deepcopy(
        base_args if base_args is not None else task_args(root, task_config, task_name)
    )
    args["step_lim"] = int(step_limit)
    bind_scenes(args, scenes_path, camera)
    task = cls()
    task.setup_demo(now_ep_num=episode.seed, seed=episode.seed, **args)
    report = verify_scene(task, episode.scene)
    report["camera_error_m"] = _camera_error(task, camera_pose)
    if report["camera_error_m"] > CAMERA_TOLERANCE_M:
        placed = task.scene.get_cameras()[0].get_entity_pose().p
        task.close_env()
        raise SceneMismatch(
            f"episode {episode.episode_id}: head camera is "
            f"{report['camera_error_m'] * 100:.1f} cm from where the episode was "
            f"recorded ({np.round(placed, 4).tolist()} vs "
            f"{np.round(np.asarray(camera_pose, dtype=float), 4).tolist()})"
        )
    if not report["matches_recording"]:
        task.close_env()
        raise SceneMismatch(
            f"episode {episode.episode_id}: orientation off by "
            f"{report['orientation_error_deg']:.3f} deg, position off by "
            f"{report['position_error_m'] * 1000:.3f} mm"
        )
    return task, report
