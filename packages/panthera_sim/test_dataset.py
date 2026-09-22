"""Checks for the dataset access layer, including that it is actually used.

The layer exists so that a dataset's layout can change without hunting down
readers.  That only holds if nothing reads the files directly, so the last test
here scans the repository and fails on any direct open, with an explicit list of
files not yet migrated -- the debt is enumerated rather than hidden.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from dataset import DATASET_FILE, SCENES_FILE, DatasetError, open_dataset

REPO = Path(__file__).resolve().parents[2]


def _scene(posture: str = "upright", angle: float = 0.0) -> dict:
    if posture == "upright":
        quaternion = [math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]
        axis = [0.0, 0.0, 1.0]
        centre_z = 0.802
    else:
        quaternion = [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
        axis = [math.cos(angle), math.sin(angle), 0.0]
        centre_z = 0.7695
    return {
        "cylinder_initial_xy_m": [0.12, -0.07],
        "socket_target_xy_m": [-0.02, -0.10],
        "groove_target_xy_m": [-0.02, -0.10],
        "cylinder_center_z_m": centre_z,
        "cylinder_quaternion_wxyz": quaternion,
        "cylinder_axis_world": axis,
        "cylinder_posture": posture,
        "cylinder_angle_rad": angle,
        "workspace": {"robot_base_x_m": 0.0, "robot_base_y_m": -0.35},
    }


def _build(tmp_path: Path, episodes=((0, 1000, "upright", 0.0), (1, 1001, "lying", 0.3))) -> Path:
    from config import resolve

    root = tmp_path / "dataset"
    (root / "data").mkdir(parents=True)
    (root / "instructions").mkdir()
    scenes = {}
    entries = []
    for episode_id, seed, posture, angle in episodes:
        scenes[str(seed)] = _scene(posture, angle)
        entries.append({"episode_id": episode_id, "seed": seed, "posture": posture})
        (root / "data" / f"episode{episode_id}.hdf5").write_bytes(b"")
        (root / "instructions" / f"episode{episode_id}.json").write_text(
            json.dumps({"seen": ["pick it up", "place it"]}), encoding="utf-8"
        )
    manifest = {
        "schema_version": 1,
        "name": "demo",
        "contract": {
            "action_dim": 7, "action_chunk": 25,
            "sample_period_physics_steps": 5, "physics_timestep_s": 0.004,
            "required_action_budget": 5101,
        },
        "task": {"name": "demo_task"},
        "sim": resolve({}).snapshot(),
        "episodes": entries,
        "provenance": {"backfilled": False},
    }
    (root / DATASET_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    (root / SCENES_FILE).write_text(
        json.dumps({"schema_version": 1, "scenes": scenes}), encoding="utf-8"
    )
    return root


def test_contract_is_read_from_the_snapshot(tmp_path):
    dataset = open_dataset(_build(tmp_path))
    contract = dataset.contract
    assert contract.action_dim == 7 and contract.action_chunk == 25
    assert contract.control_hz == pytest.approx(50.0)
    # The budget an evaluation must honour comes from the data, not a default.
    assert contract.required_action_budget == 5101


def test_episode_is_addressed_by_id_and_carries_its_scene(tmp_path):
    dataset = open_dataset(_build(tmp_path))
    episode = dataset.episode(1)
    assert episode.seed == 1001 and episode.posture == "lying"
    assert episode.scene["cylinder_angle_rad"] == 0.3
    assert episode.instruction() == "place it"   # 1 % 2


def test_scene_is_found_by_seed(tmp_path):
    dataset = open_dataset(_build(tmp_path))
    assert dataset.scene_for_seed(1000)["cylinder_posture"] == "upright"


def test_unknown_seed_and_episode_are_named(tmp_path):
    dataset = open_dataset(_build(tmp_path))
    with pytest.raises(DatasetError, match="seed 4242"):
        dataset.scene_for_seed(4242)
    with pytest.raises(DatasetError, match="episode 99"):
        dataset.episode(99)


def test_missing_snapshot_tells_the_caller_what_to_run(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    with pytest.raises(DatasetError, match="backfill"):
        open_dataset(root)


def test_wrong_schema_version_rejected(tmp_path):
    root = _build(tmp_path)
    manifest = json.loads((root / DATASET_FILE).read_text())
    manifest["schema_version"] = 99
    (root / DATASET_FILE).write_text(json.dumps(manifest))
    with pytest.raises(DatasetError, match="schema_version"):
        open_dataset(root)


def test_scene_missing_a_required_field_rejected(tmp_path):
    root = _build(tmp_path)
    payload = json.loads((root / SCENES_FILE).read_text())
    del payload["scenes"]["1000"]["cylinder_axis_world"]
    (root / SCENES_FILE).write_text(json.dumps(payload))
    with pytest.raises(DatasetError, match="cylinder_axis_world"):
        open_dataset(root)


def test_upright_scene_with_an_angle_rejected(tmp_path):
    """The exact corruption a seed-rebuilt scene produces."""
    root = _build(tmp_path)
    payload = json.loads((root / SCENES_FILE).read_text())
    payload["scenes"]["1000"]["cylinder_angle_rad"] = 1.2478
    (root / SCENES_FILE).write_text(json.dumps(payload))
    with pytest.raises(DatasetError, match="non-zero angle"):
        open_dataset(root)


def test_stratified_sample_covers_both_postures(tmp_path):
    episodes = [(i, 1000 + i, "upright", 0.0) for i in range(10)]
    episodes += [(10 + i, 2000 + i, "lying", 0.3) for i in range(10)]
    dataset = open_dataset(_build(tmp_path, episodes))
    chosen = set(dataset.stratified_sample(4))
    postures = {e.posture for e in dataset.episodes() if e.episode_id in chosen}
    assert postures == {"upright", "lying"}


def test_digest_changes_with_the_snapshot(tmp_path):
    root = _build(tmp_path)
    before = open_dataset(root).digest()
    manifest = json.loads((root / DATASET_FILE).read_text())
    manifest["contract"]["required_action_budget"] = 1000
    (root / DATASET_FILE).write_text(json.dumps(manifest))
    assert open_dataset(root).digest() != before


def test_live_config_difference_is_reported(tmp_path):
    from config import resolve

    dataset = open_dataset(_build(tmp_path))
    assert dataset.check_live_config(resolve({})) == {}
    live = resolve({"PANTHERA_V2_SOLVER_POSITION_ITERATIONS": "64"})
    assert dataset.check_live_config(live) == {"solver_position_iterations": (20, 64)}


# --- the layer is actually used ----------------------------------------------

# Files that still open the dataset directly.  This list only shrinks; a new
# entry means a new direct reader, which is what the layer exists to prevent.
# Empty: every reader goes through the access layer.  An entry here would mean
# a new direct reader, which is what the layer exists to prevent.
NOT_YET_MIGRATED: set[str] = set()
DIRECT_READ = re.compile(r'["\'](?:scene_info|scenes|dataset)\.json["\']')


def test_nothing_opens_the_dataset_files_directly():
    offenders = []
    source_roots = tuple(
        REPO / name for name in ("packages", "pipelines", "tools", "overlays")
    )
    for path in (
        path for source_root in source_roots for path in source_root.rglob("*.py")
    ):
        relative = path.relative_to(REPO).as_posix()
        if any(part in relative for part in ("__pycache__", "archive/", "ros2_ws/")):
            continue
        if relative in NOT_YET_MIGRATED:
            continue
        # The layer and the one-shot migration are allowed to name the files.
        if relative.startswith("packages/panthera_sim/") and Path(relative).name in {
            "dataset.py", "backfill.py", "test_dataset.py"
        }:
            continue
        if DIRECT_READ.search(path.read_text(encoding="utf-8", errors="replace")):
            offenders.append(relative)
    assert offenders == [], (
        "these open a dataset file directly; go through panthera_sim.dataset: "
        + ", ".join(offenders)
    )


def test_per_episode_records_are_reachable_without_opening_the_file(tmp_path):
    """Diagnostics need the collector's audit trails, which the snapshot omits."""
    import json as _json

    root = _build(tmp_path)
    (root / "scene_info.json").write_text(
        _json.dumps(
            {
                "episode_0": {"panthera_episode": {
                    "episode_seed": 1000,
                    "continuous_motion_audit": [{"stage": "grasped"}],
                }},
            }
        ),
        encoding="utf-8",
    )
    dataset = open_dataset(root)
    record = dataset.episode_record(0)
    assert record["continuous_motion_audit"] == [{"stage": "grasped"}]


def test_missing_per_episode_record_is_named(tmp_path):
    import json as _json

    root = _build(tmp_path)
    (root / "scene_info.json").write_text(_json.dumps({}), encoding="utf-8")
    with pytest.raises(DatasetError, match="episode_0"):
        open_dataset(root).episode_record(0)


def test_task_config_is_task_level(tmp_path):
    root = _build(tmp_path)
    manifest = json.loads((root / DATASET_FILE).read_text())
    manifest["task"]["config"] = {"schema_version": 10, "robot_count": 1}
    (root / DATASET_FILE).write_text(json.dumps(manifest))
    assert open_dataset(root).task_config["schema_version"] == 10


def test_both_episode_file_namings_resolve(tmp_path):
    """RoboTwin renamed the recordings; a dataset may carry either name.

    Upstream went from ``episode3.hdf5`` to ``episode_0000003.hdf5`` in the 223
    commits between 0008ae6 and 6dde571.  Every dataset we hold uses the old
    name and anything collected from the migrated runtime will use the new one,
    so the access layer resolves both rather than each reader guessing.
    """
    from dataset import _episode_file

    data = tmp_path / "data"
    data.mkdir()
    (data / "episode7.hdf5").write_bytes(b"")
    assert _episode_file(tmp_path, 7).name == "episode7.hdf5"

    (data / "episode_0000008.hdf5").write_bytes(b"")
    assert _episode_file(tmp_path, 8).name == "episode_0000008.hdf5"


def test_a_missing_episode_is_named_like_its_siblings(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "episode_0000001.hdf5").write_bytes(b"")
    from dataset import _episode_file

    assert _episode_file(tmp_path, 9).name == "episode_0000009.hdf5"
