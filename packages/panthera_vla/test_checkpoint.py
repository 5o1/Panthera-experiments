"""Checks for the checkpoint contract.

Each of these corresponds to a way an evaluation could previously be wrong
without any error: the chunk length, the platform constant, the unnormalisation
key, the image preprocessing, or the dataset all came from outside the
checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkpoint import (
    CheckpointError,
    ObservationContract,
    PolicyContract,
    open_checkpoint,
    write_training_record,
)

CONTRACT = PolicyContract(
    action_dim=7, action_chunk=25, proprio_dim=7, use_proprio=True,
    use_l1_regression=True, num_images_in_input=1,
    unnorm_key="panthera_phone_cylinder_socket_v2", robot_platform="PANTHERA",
)
OBSERVATION = ObservationContract(image_size=(224, 224), center_crop=True)


def _write(tmp_path: Path, **overrides) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    (root / "dataset_statistics.json").write_text(
        json.dumps({"panthera_phone_cylinder_socket_v2": {"action": {}}}), encoding="utf-8"
    )
    arguments = dict(
        root=root, policy=CONTRACT, observation=OBSERVATION,
        dataset_name="fixedcam1280", dataset_digest="a" * 64,
        dataset_root="/data/fixedcam1280", base_model="/models/base",
        optimisation={"batch_size": 6, "learning_rate": 5e-4, "steps": 51000},
    )
    arguments.update(overrides)
    write_training_record(**arguments)
    return root


def test_contract_round_trips(tmp_path):
    checkpoint = open_checkpoint(_write(tmp_path))
    assert checkpoint.policy == CONTRACT
    assert checkpoint.observation == OBSERVATION
    assert checkpoint.dataset["name"] == "fixedcam1280"


def test_subset_lineage_is_recorded(tmp_path):
    root = _write(
        tmp_path,
        dataset_lineage={"source_root": "/data/full", "source_digest": "f" * 64},
    )
    dataset = open_checkpoint(root).dataset
    assert dataset["lineage"]["source_root"] == "/data/full"
    assert dataset["lineage"]["source_digest"] == "f" * 64


def test_base_model_records_the_random_action_head(tmp_path):
    record = json.loads((_write(tmp_path) / "training.json").read_text())
    assert record["base_model"]["action_head"] == "randomly_initialised"


def test_missing_record_says_what_is_wrong(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(CheckpointError, match="training.json"):
        open_checkpoint(bare)


def test_unknown_schema_rejected(tmp_path):
    root = _write(tmp_path)
    record = json.loads((root / "training.json").read_text())
    record["schema_version"] = 7
    (root / "training.json").write_text(json.dumps(record))
    with pytest.raises(CheckpointError, match="schema_version"):
        open_checkpoint(root)


@pytest.mark.parametrize("key", ["action_chunk", "unnorm_key", "robot_platform"])
def test_missing_policy_field_is_named(tmp_path, key):
    root = _write(tmp_path)
    record = json.loads((root / "training.json").read_text())
    del record["policy_contract"][key]
    (root / "training.json").write_text(json.dumps(record))
    with pytest.raises(CheckpointError, match=key):
        open_checkpoint(root)


def test_unnorm_key_absent_from_statistics_rejected(tmp_path):
    """The statistics are what the weights were normalised against."""
    root = _write(tmp_path)
    record = json.loads((root / "training.json").read_text())
    record["policy_contract"]["unnorm_key"] = "some_other_dataset"
    (root / "training.json").write_text(json.dumps(record))
    with pytest.raises(CheckpointError, match="dataset_statistics"):
        open_checkpoint(root)


def test_stopping_and_merge_records_are_absorbed(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "early-stopping.json").write_text(json.dumps({"best_loss": 0.008}))
    (root / "manual-merge.json").write_text(json.dumps({"merged_by": "out of band"}))
    write_training_record(
        root=root, policy=CONTRACT, observation=OBSERVATION,
        dataset_name="d", dataset_digest="b" * 64, dataset_root="/d",
        base_model="/m", optimisation={},
    )
    record = json.loads((root / "training.json").read_text())
    assert record["early_stopping"]["best_loss"] == 0.008
    assert record["manual_merge"]["merged_by"] == "out of band"


class _Dataset:
    def __init__(self, name, digest):
        self.name = name
        self._digest = digest

    def digest(self):
        return self._digest


def test_matching_dataset_passes_silently(tmp_path):
    checkpoint = open_checkpoint(_write(tmp_path))
    assert checkpoint.check_dataset(_Dataset("fixedcam1280", "a" * 64)) is None


def test_mismatched_dataset_is_reported_not_raised(tmp_path):
    checkpoint = open_checkpoint(_write(tmp_path))
    message = checkpoint.check_dataset(_Dataset("other", "c" * 64))
    assert message is not None and "mismatch" in message


def test_checkpoint_without_a_recorded_dataset_says_so(tmp_path):
    root = _write(tmp_path, dataset_digest="")
    message = open_checkpoint(root).check_dataset(_Dataset("x", "d" * 64))
    assert "does not record" in message


def test_backfilled_flag_is_visible(tmp_path):
    root = _write(tmp_path, backfilled=True, note="reconstructed from the run log")
    assert open_checkpoint(root).backfilled is True
