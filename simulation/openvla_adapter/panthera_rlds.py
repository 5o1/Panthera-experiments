#!/usr/bin/env python3
"""Build and validate an RLDS view of Panthera RoboTwin episodes.

The RoboTwin recorder samples *after* applying the drive target.  This adapter
therefore pairs each image/measured state with the next 50 Hz drive target.
Only samples on the global five-physics-step grid are retained, and duplicate
captures at stage boundaries are collapsed by keeping the last capture.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import h5py
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds


DATASET_NAME = os.environ.get(
    "PANTHERA_RLDS_DATASET_NAME", "panthera_single_cylinder"
)
EXPECTED_SCHEMA_VERSION = int(
    os.environ.get("PANTHERA_RLDS_SCHEMA_VERSION", "3")
)
EXPECTED_SCENE_PROFILE = os.environ.get("PANTHERA_RLDS_SCENE_PROFILE", "")
SAMPLE_PERIOD_STEPS = 5
ACTION_DIMENSION = 7
ACTION_CHUNK = int(os.environ.get("PANTHERA_ACTION_CHUNK", "5"))
if ACTION_CHUNK <= 0:
    raise ValueError("PANTHERA_ACTION_CHUNK must be a positive integer")


@dataclass(frozen=True)
class Episode:
    episode_id: int
    hdf5_path: Path
    instruction_path: Path
    metadata: dict


@dataclass(frozen=True)
class _BuildContext:
    episodes_by_split: dict[str, tuple[Episode, ...]]


_BUILD_CONTEXT: _BuildContext | None = None


def _episode_id(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"unexpected episode filename: {path.name}")
    return int(match.group(1))


def discover_episodes(source_root: Path) -> list[Episode]:
    """Find HDF5 episodes and attach their instruction and scene metadata."""
    source_root = source_root.resolve()
    scene_path = source_root / "scene_info.json"
    if not scene_path.is_file():
        raise FileNotFoundError(f"missing scene metadata: {scene_path}")
    scene_info = json.loads(scene_path.read_text(encoding="utf-8"))
    episodes: list[Episode] = []
    for hdf5_path in sorted((source_root / "data").glob("episode*.hdf5"), key=_episode_id):
        episode_id = _episode_id(hdf5_path)
        instruction_path = source_root / "instructions" / f"episode{episode_id}.json"
        if not instruction_path.is_file():
            raise FileNotFoundError(f"missing instruction file: {instruction_path}")
        key = f"episode_{episode_id}"
        try:
            metadata = scene_info[key]["panthera_episode"]
        except KeyError as error:
            raise KeyError(f"missing {key}.panthera_episode in {scene_path}") from error
        if metadata.get("schema_version") != EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"episode {episode_id} is not schema v{EXPECTED_SCHEMA_VERSION}"
            )
        if (
            EXPECTED_SCENE_PROFILE
            and metadata.get("scene_profile") != EXPECTED_SCENE_PROFILE
        ):
            raise ValueError(f"episode {episode_id} scene profile mismatch")
        if metadata.get("robot_count") != 1 or metadata.get("action_dimension") != 7:
            raise ValueError(f"episode {episode_id} is not a one-Panthera 7-D episode")
        if metadata.get("attach_on_grasp"):
            raise ValueError(f"episode {episode_id} used a forbidden grasp attachment")
        episodes.append(Episode(episode_id, hdf5_path, instruction_path, metadata))
    if len(episodes) < 2:
        raise ValueError("at least two episodes are required for train/val splits")
    return episodes


def split_episodes(
    episodes: Sequence[Episode], validation_episode_ids: Sequence[int]
) -> dict[str, tuple[Episode, ...]]:
    """Create explicit, disjoint train/val splits."""
    available = {episode.episode_id for episode in episodes}
    validation = set(validation_episode_ids)
    missing = validation - available
    if missing:
        raise ValueError(f"validation episode IDs do not exist: {sorted(missing)}")
    if not validation:
        validation = {max(available)}
    train = tuple(episode for episode in episodes if episode.episode_id not in validation)
    val = tuple(episode for episode in episodes if episode.episode_id in validation)
    if not train or not val:
        raise ValueError("both train and val splits must contain at least one episode")
    return {"train": train, "val": val}


def _grid_indices(simulation_steps: np.ndarray) -> np.ndarray:
    """Return one monotonically increasing capture per global 50 Hz grid step."""
    if simulation_steps.ndim != 1 or simulation_steps.size < 2:
        raise ValueError("simulation_step_index must be a nontrivial vector")
    if np.any(np.diff(simulation_steps) < 0):
        raise ValueError("simulation step clock moved backwards")
    selected: dict[int, int] = {}
    for index, step in enumerate(simulation_steps.tolist()):
        if step % SAMPLE_PERIOD_STEPS == 0:
            selected[int(step)] = index
    ordered_steps = np.asarray(sorted(selected), dtype=np.int64)
    if ordered_steps.size < ACTION_CHUNK + 1:
        raise ValueError(f"episode is too short for a {ACTION_CHUNK}-action chunk")
    gaps = np.diff(ordered_steps)
    if not np.all(gaps == SAMPLE_PERIOD_STEPS):
        bad = sorted(set(gaps[gaps != SAMPLE_PERIOD_STEPS].tolist()))
        raise ValueError(f"global sample grid has gaps other than 5 steps: {bad}")
    return np.asarray([selected[int(step)] for step in ordered_steps], dtype=np.int64)


def _instruction(episode: Episode) -> str:
    values = json.loads(episode.instruction_path.read_text(encoding="utf-8"))
    seen = values.get("seen", [])
    if not seen or not isinstance(seen[0], str):
        raise ValueError(f"episode {episode.episode_id} has no seen instruction")
    return seen[episode.episode_id % len(seen)]


def _decode_rgb(encoded: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _episode_steps(episode: Episode) -> Iterator[dict]:
    """Yield fixed-rate (observation, next-action) transitions for one episode."""
    instruction = _instruction(episode)
    with h5py.File(episode.hdf5_path, "r") as data:
        simulation_steps = np.asarray(data["timing/simulation_step_index"], dtype=np.int64)
        indices = _grid_indices(simulation_steps)
        actions = np.asarray(data["joint_action/vector"], dtype=np.float32)
        states = np.asarray(data["observation/robot_state/vector"], dtype=np.float32)
        times = np.asarray(data["timing/simulation_time_s"], dtype=np.float64)
        images = data["observation/head_camera/rgb"]
        if actions.shape[1:] != (ACTION_DIMENSION,) or states.shape != actions.shape:
            raise ValueError(f"episode {episode.episode_id} does not use the 7-D contract")
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(states)):
            raise ValueError(f"episode {episode.episode_id} contains nonfinite state/action")
        for output_index, source_index in enumerate(indices.tolist()):
            # The image was captured after source_index's target was applied.  The
            # following grid target is the command conditioned on this image.
            next_source_index = int(indices[min(output_index + 1, len(indices) - 1)])
            last = output_index == len(indices) - 1
            yield {
                "observation": {
                    "image": _decode_rgb(bytes(images[source_index])),
                    "state": states[source_index],
                    "simulation_step_index": simulation_steps[source_index],
                    "simulation_time_s": times[source_index],
                },
                "action": actions[next_source_index],
                "language_instruction": instruction,
                "is_first": output_index == 0,
                "is_last": last,
                "is_terminal": last,
                "reward": np.float32(1.0 if last else 0.0),
                "discount": np.float32(0.0 if last else 1.0),
            }


class PantheraSingleCylinder(tfds.core.GeneratorBasedBuilder):
    """TFDS/RLDS builder for synchronized single-Panthera cylinder episodes."""

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {
        "2.0.0": "Single-Panthera 7-D measured-state, next-action contract."
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Synchronized single-Panthera RoboTwin demonstrations for "
                "cylinder insertion into a groove."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(240, 320, 3), encoding_format="jpeg"
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(ACTION_DIMENSION,), dtype=np.float32
                                    ),
                                    "simulation_step_index": np.int64,
                                    "simulation_time_s": np.float64,
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(ACTION_DIMENSION,), dtype=np.float32
                            ),
                            "language_instruction": tfds.features.Text(),
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                            "reward": np.float32,
                            "discount": np.float32,
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "episode_id": np.int64,
                            "episode_seed": np.int64,
                            "robot_source_commit": tfds.features.Text(),
                            "robotwin_source_commit": tfds.features.Text(),
                            "attach_on_grasp": np.bool_,
                            "physics_timestep_s": np.float64,
                            "cylinder_initial_xy_m": tfds.features.Tensor(
                                shape=(2,), dtype=np.float64
                            ),
                            "groove_target_xy_m": tfds.features.Tensor(
                                shape=(2,), dtype=np.float64
                            ),
                            "sample_period_steps": np.int64,
                            "action_lead_samples": np.int64,
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/5o1/panthera",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        if _BUILD_CONTEXT is None:
            raise RuntimeError("build context is unset; use build_dataset()")
        return {
            split: self._generate_examples(episodes)
            for split, episodes in _BUILD_CONTEXT.episodes_by_split.items()
        }

    def _generate_examples(self, episodes: Iterable[Episode]):
        for episode in episodes:
            geometry = episode.metadata["realized_geometry"]
            target_xy = geometry.get(
                "groove_target_xy_m", geometry.get("socket_target_xy_m")
            )
            if target_xy is None:
                raise ValueError(
                    f"episode {episode.episode_id} has no groove/socket target"
                )
            yield str(episode.episode_id), {
                "steps": _episode_steps(episode),
                "episode_metadata": {
                    "episode_id": episode.episode_id,
                    "episode_seed": int(episode.metadata["episode_seed"]),
                    "robot_source_commit": episode.metadata["robot_source_commit"],
                    "robotwin_source_commit": episode.metadata["robotwin_source_commit"],
                    "attach_on_grasp": bool(episode.metadata["attach_on_grasp"]),
                    "physics_timestep_s": float(episode.metadata["physics_timestep_s"]),
                    "cylinder_initial_xy_m": geometry["cylinder_initial_xy_m"],
                    "groove_target_xy_m": target_xy,
                    "sample_period_steps": SAMPLE_PERIOD_STEPS,
                    "action_lead_samples": 1,
                },
            }


class PantheraPhoneVerticalCylinder(PantheraSingleCylinder):
    """TFDS builder for the provisional phone-SRT-aligned vertical task."""

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "3.0.0": (
            "Single-Panthera 7-D upright-cylinder insertion with the "
            "provisional phone-SRT camera profile."
        )
    }


class PantheraPhoneCylinderSocketV2(PantheraSingleCylinder):
    """TFDS builder for schema-10 upright/lying cylinder insertion."""

    VERSION = tfds.core.Version("4.0.0")
    RELEASE_NOTES = {
        "4.0.0": (
            "Single-Panthera schema-10 demonstrations with broadly randomized "
            "cylinder/socket positions and upright or lying cylinder poses."
        )
    }


def panthera_dataset_transform(trajectory: dict) -> dict:
    """The generated RLDS episode already uses OpenVLA's expected schema."""
    return trajectory


def _install_single_joint_materialize_support() -> None:
    """Teach the fixed OpenVLA-OFT checkout one local 7-D absolute-joint schema."""
    from prismatic.vla.datasets.rlds.oxe import materialize

    if getattr(materialize, "_panthera_single_joint_support", False):
        return
    original = materialize.make_oxe_dataset_kwargs

    def make_dataset_kwargs(
        dataset_name,
        data_root_dir,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=True,
        load_language=True,
        action_proprio_normalization_type=None,
    ):
        if dataset_name != DATASET_NAME:
            call_kwargs = {
                "load_camera_views": load_camera_views,
                "load_depth": load_depth,
                "load_proprio": load_proprio,
                "load_language": load_language,
            }
            if action_proprio_normalization_type is not None:
                call_kwargs["action_proprio_normalization_type"] = (
                    action_proprio_normalization_type
                )
            return original(dataset_name, data_root_dir, **call_kwargs)

        from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
        from prismatic.vla.datasets.rlds.oxe.configs import (
            ActionEncoding,
            OXE_DATASET_CONFIGS,
        )
        from prismatic.vla.datasets.rlds.oxe.transforms import (
            OXE_STANDARDIZATION_TRANSFORMS,
        )

        dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
        if dataset_kwargs["action_encoding"] is not ActionEncoding.JOINT_POS:
            raise ValueError("single Panthera must use the absolute JOINT_POS marker")
        dataset_kwargs["absolute_action_mask"] = [True] * ACTION_DIMENSION
        dataset_kwargs["action_normalization_mask"] = [True] * ACTION_DIMENSION
        dataset_kwargs["action_proprio_normalization_type"] = (
            ACTION_PROPRIO_NORMALIZATION_TYPE
            if action_proprio_normalization_type is None
            else action_proprio_normalization_type
        )
        missing = set(load_camera_views) - set(dataset_kwargs["image_obs_keys"])
        if missing:
            raise ValueError(f"missing camera views: {sorted(missing)}")
        dataset_kwargs["image_obs_keys"] = {
            key: value
            for key, value in dataset_kwargs["image_obs_keys"].items()
            if key in load_camera_views
        }
        dataset_kwargs["depth_obs_keys"] = {
            key: value
            for key, value in dataset_kwargs["depth_obs_keys"].items()
            if key in load_camera_views
        }
        dataset_kwargs.pop("state_encoding")
        dataset_kwargs.pop("action_encoding")
        if not load_depth:
            dataset_kwargs.pop("depth_obs_keys")
        if not load_proprio:
            dataset_kwargs.pop("state_obs_keys")
        if load_language:
            dataset_kwargs["language_key"] = "language_instruction"
        dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]
        if "aux_kwargs" in dataset_kwargs:
            dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))
        return {"name": dataset_name, "data_dir": str(data_root_dir), **dataset_kwargs}

    materialize.make_oxe_dataset_kwargs = make_dataset_kwargs
    materialize._panthera_single_joint_support = True


def register_openvla_dataset() -> None:
    """Register Panthera's one-camera 7-D absolute-joint contract in OpenVLA-OFT."""
    from tensorflow_datasets.core.utils import gcs_utils

    # Training is fully local; do not let a new finetune process probe GCS.
    gcs_utils._is_gcs_disabled = True
    from prismatic.vla.datasets.rlds.oxe.configs import (
        ActionEncoding,
        OXE_DATASET_CONFIGS,
        StateEncoding,
    )
    from prismatic.vla.datasets.rlds.oxe.transforms import (
        OXE_STANDARDIZATION_TRANSFORMS,
    )

    OXE_DATASET_CONFIGS[DATASET_NAME] = {
        "image_obs_keys": {"primary": "image", "wrist": None},
        "depth_obs_keys": {"primary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
    }
    OXE_STANDARDIZATION_TRANSFORMS[DATASET_NAME] = panthera_dataset_transform
    _install_single_joint_materialize_support()


def build_dataset(source_root: Path, data_root: Path, validation_ids: Sequence[int]):
    """Build TFDS files from a validated RoboTwin contract dataset."""
    global _BUILD_CONTEXT
    # This is a local custom dataset.  Avoid TFDS probing the public Google
    # bucket during both DatasetInfo initialization and download preparation.
    from tensorflow_datasets.core.utils import gcs_utils

    gcs_utils._is_gcs_disabled = True
    episodes = discover_episodes(source_root)
    splits = split_episodes(episodes, validation_ids)
    _BUILD_CONTEXT = _BuildContext(splits)
    builders = {
        "panthera_single_cylinder": PantheraSingleCylinder,
        "panthera_phone_vertical_cylinder": PantheraPhoneVerticalCylinder,
        "panthera_phone_cylinder_socket_v2": PantheraPhoneCylinderSocketV2,
    }
    try:
        builder_class = builders[DATASET_NAME]
    except KeyError as error:
        raise ValueError(f"unsupported Panthera RLDS dataset: {DATASET_NAME}") from error
    builder = builder_class(data_dir=str(data_root.resolve()))
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(try_download_gcs=False)
    )
    return builder, splits


def validate_openvla_pipeline(data_root: Path) -> dict:
    """Read one batch through OpenVLA's real normalize/chunk/decode pipeline."""
    register_openvla_dataset()
    from prismatic.vla.datasets.rlds.dataset import make_single_dataset
    from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs

    kwargs = make_oxe_dataset_kwargs(
        DATASET_NAME,
        data_root.resolve(),
        load_camera_views=("primary", "wrist"),
        load_proprio=True,
        load_language=True,
    )
    dataset, episode_count, statistics = make_single_dataset(
        kwargs,
        train=True,
        traj_transform_kwargs={
            "window_size": 1,
            "future_action_window_size": ACTION_CHUNK - 1,
            "skip_unlabeled": True,
            "goal_relabeling_strategy": "uniform",
        },
        frame_transform_kwargs={
            "resize_size": (224, 224),
            "num_parallel_calls": 1,
        },
    )
    episode_batch = next(dataset.as_numpy_iterator())
    # make_single_dataset intentionally preserves the leading trajectory axis;
    # RLDSDataset flattens it later when interleaving datasets for training.
    action = np.asarray(episode_batch["action"])[0]
    proprio = np.asarray(episode_batch["observation"]["proprio"])[0]
    primary = np.asarray(episode_batch["observation"]["image_primary"])[0]
    if action.shape != (ACTION_CHUNK, ACTION_DIMENSION):
        raise ValueError(f"unexpected OpenVLA action chunk: {action.shape}")
    if proprio.shape != (1, ACTION_DIMENSION):
        raise ValueError(f"unexpected OpenVLA proprio window: {proprio.shape}")
    if primary.shape != (1, 224, 224, 3):
        raise ValueError(f"unexpected OpenVLA image window: {primary.shape}")
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(proprio)):
        raise ValueError("OpenVLA pipeline emitted nonfinite values")
    stats = statistics
    if tuple(np.asarray(stats["action"]["mean"]).shape) != (ACTION_DIMENSION,):
        raise ValueError("OpenVLA action statistics are not 7-D")
    if tuple(np.asarray(stats["proprio"]["mean"]).shape) != (ACTION_DIMENSION,):
        raise ValueError("OpenVLA proprio statistics are not 7-D")
    return {
        "dataset_name": DATASET_NAME,
        "statistics_episode_count": int(episode_count),
        "action_chunk_shape": list(action.shape),
        "proprio_window_shape": list(proprio.shape),
        "primary_image_shape": list(primary.shape),
        "action_min": np.asarray(stats["action"]["min"]).tolist(),
        "action_max": np.asarray(stats["action"]["max"]).tolist(),
        "proprio_min": np.asarray(stats["proprio"]["min"]).tolist(),
        "proprio_max": np.asarray(stats["proprio"]["max"]).tolist(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--validation-episode", type=int, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    builder, splits = build_dataset(
        args.source_root, args.data_root, args.validation_episode
    )
    summary = validate_openvla_pipeline(args.data_root)
    summary.update(
        {
            "status": "passed",
            "tfds_version": str(builder.version),
            "splits": {
                name: [episode.episode_id for episode in episodes]
                for name, episodes in splits.items()
            },
            "sample_period_physics_steps": SAMPLE_PERIOD_STEPS,
            "nominal_control_hz": 50,
            "action_alignment": "next_global_grid_sample",
            "task_schema_version": EXPECTED_SCHEMA_VERSION,
            "scene_profile": EXPECTED_SCENE_PROFILE or None,
        }
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
