"""Read a dataset through its own snapshot, never through scattered config.

A dataset used to describe itself only in ``scene_info.json``, and only by
accident: 149 of its fields were identical across all 1280 episodes -- the task
configuration, duplicated once per episode -- while the physics it was collected
under, the assets used, and the action budget it needs were either absent or
spread across files that lived somewhere else entirely.  Consumers filled the
gap by guessing.  Rebuilding a scene from its seed reproduced a different
posture for 57 of 112 training episodes, and an evaluation ran against an action
budget under half what the expert needs, capping it at 7% before a policy was
involved.

So a dataset now carries ``dataset.json`` (what it was made with) and
``scenes.json`` (what each seed actually is), and every consumer goes through
:func:`open_dataset`.  Nothing in this repository opens those files directly;
a test enforces it, which is what lets their layout change later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from config import SimConfig, from_snapshot, resolve

DATASET_FILE = "dataset.json"
SCENES_FILE = "scenes.json"
SCENE_INFO_FILE = "scene_info.json"
SUPPORTED_SCHEMA = 1

# Every field the environment reads off a scene before it places anything.
SCENE_KEYS = (
    "cylinder_initial_xy_m",
    "socket_target_xy_m",
    "groove_target_xy_m",
    "cylinder_center_z_m",
    "cylinder_quaternion_wxyz",
    "cylinder_axis_world",
    "cylinder_posture",
    "cylinder_angle_rad",
    "workspace",
)


class DatasetError(ValueError):
    """The dataset is missing a snapshot, or the snapshot disagrees with itself."""


@dataclass(frozen=True)
class Contract:
    """The action and timing contract the data was recorded under."""

    action_dim: int
    action_chunk: int
    sample_period_physics_steps: int
    physics_timestep_s: float
    required_action_budget: int

    @property
    def control_hz(self) -> float:
        return 1.0 / (self.physics_timestep_s * self.sample_period_physics_steps)

    def as_dict(self) -> dict:
        return {
            "action_dim": self.action_dim,
            "action_chunk": self.action_chunk,
            "sample_period_physics_steps": self.sample_period_physics_steps,
            "physics_timestep_s": self.physics_timestep_s,
            "control_hz": self.control_hz,
            # Measured from the recorded trajectories, so an evaluation reads the
            # budget the data needs instead of inheriting a default from a file
            # that is not under version control.
            "required_action_budget": self.required_action_budget,
        }


@dataclass(frozen=True)
class Episode:
    """One recorded episode, addressed by its id rather than by its seed."""

    episode_id: int
    seed: int
    posture: str
    scene: Mapping[str, Any]
    hdf5_path: Path
    instruction_path: Path

    def instruction(self) -> str:
        values = json.loads(self.instruction_path.read_text(encoding="utf-8"))["seen"]
        if not values:
            raise DatasetError(f"episode {self.episode_id} has no seen instruction")
        return values[self.episode_id % len(values)]


class Dataset:
    """A recorded dataset, read through its snapshot."""

    def __init__(self, root: Path, manifest: Mapping[str, Any], scenes: Mapping[int, dict],
                 episodes: Mapping[int, Episode]) -> None:
        self.root = Path(root)
        self._manifest = manifest
        self._scenes = scenes
        self._episodes = episodes
        self._records: Optional[Mapping[str, Any]] = None

    # --- identity -------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._manifest.get("name", self.root.name)

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest

    def digest(self) -> str:
        """Hash of the snapshot, for a checkpoint to record what it trained on."""
        return hashlib.sha256(
            json.dumps(self._manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def backfilled(self) -> bool:
        """True when the snapshot was reconstructed rather than written at collection."""
        return bool(self._manifest.get("provenance", {}).get("backfilled"))

    # --- contract and configuration -------------------------------------------

    @property
    def contract(self) -> Contract:
        values = self._manifest["contract"]
        return Contract(
            action_dim=int(values["action_dim"]),
            action_chunk=int(values["action_chunk"]),
            sample_period_physics_steps=int(values["sample_period_physics_steps"]),
            physics_timestep_s=float(values["physics_timestep_s"]),
            required_action_budget=int(values["required_action_budget"]),
        )

    @property
    def sim_config(self) -> SimConfig:
        return from_snapshot(self._manifest["sim"])

    @property
    def assets(self) -> Mapping[str, Any]:
        return self._manifest.get("assets", {})

    @property
    def upstream(self) -> Mapping[str, Any]:
        return self._manifest.get("upstream", {})

    def check_live_config(self, live: Optional[SimConfig] = None) -> dict:
        """Report how a live configuration differs from the recorded one."""
        return (live or resolve()).difference(self._manifest["sim"])

    # --- scenes and episodes ---------------------------------------------------

    def camera(self, episode_id: int) -> Mapping[str, Any]:
        """The camera configuration the episode was recorded with.

        The head camera's placement is decided by ``camera_randomization`` in
        the task config, and the configs differ: the pilot config samples the
        pose while the collection config pinned it.  Evaluating through the
        wrong one put the camera 47.6 cm from the recorded viewpoint, so this
        comes from the episode rather than from whichever config a command line
        happens to name.
        """
        record = self.episode_record(episode_id)
        try:
            return record["task_randomization"]["camera_randomization"]
        except KeyError as error:
            raise DatasetError(
                f"episode {episode_id} records no camera configuration, so it "
                "cannot be evaluated in the scene it was recorded in"
            ) from error

    def camera_pose(self, episode_id: int) -> list:
        """Where the head camera actually ended up when the episode was recorded."""
        record = self.episode_record(episode_id)
        try:
            return list(record["camera_randomization"]["position_xyz_m"])
        except KeyError as error:
            raise DatasetError(
                f"episode {episode_id} records no realised camera pose"
            ) from error

    @property
    def scenes_path(self) -> Path:
        return self.root / SCENES_FILE

    def scene_for_seed(self, seed: int) -> dict:
        try:
            return self._scenes[int(seed)]
        except KeyError as error:
            raise DatasetError(f"seed {seed} is not in {self.scenes_path}") from error

    def episode(self, episode_id: int) -> Episode:
        try:
            return self._episodes[int(episode_id)]
        except KeyError as error:
            raise DatasetError(f"episode {episode_id} is not in {self.root}") from error

    def episode_record(self, episode_id: int) -> Mapping[str, Any]:
        """The full per-episode record, for diagnostics that need the audits.

        The snapshot carries what a run needs to rebuild a scene; the collector
        also wrote per-episode audit trails -- retiming, settling, grasp route
        selection -- which only offline analysis reads.  Those stay in
        ``scene_info.json``, and are reached through here so that file still has
        no direct readers and its layout stays free to change.
        """
        if self._records is None:
            self._records = _load_scene_info(self.root)
        key = f"episode_{int(episode_id)}"
        try:
            return self._records[key]["panthera_episode"]
        except KeyError as error:
            raise DatasetError(
                f"{key} has no record in {self.root / SCENE_INFO_FILE}"
            ) from error

    @property
    def task_config(self) -> Mapping[str, Any]:
        """The collection configuration, which is task-level, not per episode."""
        return self._manifest.get("task", {}).get("config", {})

    def episode_ids(self) -> list[int]:
        return sorted(self._episodes)

    def episodes(self, postures: Optional[Sequence[str]] = None) -> Iterator[Episode]:
        wanted = None if postures is None else set(postures)
        for episode_id in self.episode_ids():
            episode = self._episodes[episode_id]
            if wanted is None or episode.posture in wanted:
                yield episode

    def posture_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for episode in self.episodes():
            counts[episode.posture] = counts.get(episode.posture, 0) + 1
        return counts

    def stratified_sample(self, size: int) -> list[int]:
        """Pick ids spread evenly across postures.

        A sample that happens to contain only the easy posture would let a gate
        pass while the hard half is broken, which is how a 20/20 upright result
        once sat beside a 1/20 lying one.
        """
        if size <= 0:
            return self.episode_ids()
        by_posture: dict[str, list[int]] = {}
        for episode in self.episodes():
            by_posture.setdefault(episode.posture, []).append(episode.episode_id)
        share = max(size // max(len(by_posture), 1), 1)
        chosen: list[int] = []
        for ids in by_posture.values():
            step = max(len(ids) // share, 1)
            chosen.extend(ids[::step][:share])
        return sorted(chosen)


def _load_scene_info(root: Path) -> dict:
    path = root / SCENE_INFO_FILE
    if not path.is_file():
        raise DatasetError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_scene(scene: Mapping[str, Any], where: str) -> None:
    missing = [key for key in SCENE_KEYS if key not in scene]
    if missing:
        raise DatasetError(f"{where} is missing {', '.join(missing)}")
    if scene["cylinder_posture"] not in {"upright", "lying"}:
        raise DatasetError(f"{where} has posture {scene['cylinder_posture']!r}")
    if scene["cylinder_posture"] == "upright" and float(scene["cylinder_angle_rad"]) != 0.0:
        raise DatasetError(f"{where} is upright but records a non-zero angle")


def _episode_file(root: Path, episode_id: int) -> Path:
    """Where this dataset keeps one episode's recording.

    RoboTwin renamed these between 0008ae6 and 6dde571, from ``episode3.hdf5``
    to ``episode_0000003.hdf5``.  Every dataset we hold predates the rename and
    anything collected from here on will not, so both have to resolve, and a
    dataset that has neither has to say which id it was looking for rather than
    failing later on a missing file.
    """
    data = Path(root) / "data"
    for name in (f"episode{episode_id}.hdf5", f"episode_{episode_id:07d}.hdf5"):
        candidate = data / name
        if candidate.exists():
            return candidate
    # Nothing on disk yet: name it the way this dataset's siblings are named, so
    # a half-written dataset keeps one convention.
    existing = sorted(data.glob("episode_*.hdf5")) if data.is_dir() else []
    if existing:
        return data / f"episode_{episode_id:07d}.hdf5"
    return data / f"episode{episode_id}.hdf5"


def open_dataset(root: Path) -> Dataset:
    """Open a dataset through its snapshot; refuse one that has none."""
    root = Path(root)
    manifest_path = root / DATASET_FILE
    if not manifest_path.is_file():
        raise DatasetError(
            f"{root} has no {DATASET_FILE}; run panthera_sim.dataset backfill first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SUPPORTED_SCHEMA:
        raise DatasetError(
            f"{manifest_path} declares schema_version "
            f"{manifest.get('schema_version')!r}, expected {SUPPORTED_SCHEMA}"
        )

    scenes_path = root / SCENES_FILE
    if not scenes_path.is_file():
        raise DatasetError(f"missing {scenes_path}")
    raw_scenes = json.loads(scenes_path.read_text(encoding="utf-8")).get("scenes", {})
    scenes: dict[int, dict] = {}
    for seed, scene in raw_scenes.items():
        _validate_scene(scene, f"{scenes_path} seed {seed}")
        scenes[int(seed)] = dict(scene)
    if not scenes:
        raise DatasetError(f"{scenes_path} contains no scenes")

    episodes: dict[int, Episode] = {}
    for entry in manifest["episodes"]:
        episode_id = int(entry["episode_id"])
        seed = int(entry["seed"])
        episodes[episode_id] = Episode(
            episode_id=episode_id,
            seed=seed,
            posture=str(entry["posture"]),
            scene=scenes[seed],
            hdf5_path=_episode_file(root, episode_id),
            instruction_path=root / "instructions" / f"episode{episode_id}.json",
        )
    return Dataset(root, manifest, scenes, episodes)
