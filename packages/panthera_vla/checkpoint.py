"""A checkpoint carries what is needed to run it, instead of the caller guessing.

A merged checkpoint already stores weights, the backbone ``config.json`` and the
normalisation statistics.  It does not store what an evaluation actually has to
supply: the action chunk length, the unnormalisation key, whether proprioception
is used, how many images the model expects, the image preprocessing, or which
dataset it was fitted to.  Those came from environment variables read by the
evaluation entry point -- ``PANTHERA_ACTION_CHUNK``, ``ROBOT_PLATFORM``,
``PANTHERA_EVAL_UNNORM_KEY`` -- so a mismatch produced no error at all, only a
model that silently decoded the wrong thing.

``training.json`` closes that: the run writes what it trained under, and an
evaluation reads it rather than being told.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

TRAINING_FILE = "training.json"
SUPPORTED_SCHEMA = 1

# ROBOT_PLATFORM selects module-level constants inside prismatic, including the
# action chunk length, so it is part of the contract rather than an environment
# detail.
POLICY_KEYS = (
    "action_dim",
    "action_chunk",
    "proprio_dim",
    "use_proprio",
    "use_l1_regression",
    "num_images_in_input",
    "unnorm_key",
    "robot_platform",
)
OBSERVATION_KEYS = ("image_size", "center_crop")


class CheckpointError(ValueError):
    """The checkpoint has no contract, or one that disagrees with its weights."""


@dataclass(frozen=True)
class PolicyContract:
    """Everything the model loader must be told, taken from the run itself."""

    action_dim: int
    action_chunk: int
    proprio_dim: int
    use_proprio: bool
    use_l1_regression: bool
    num_images_in_input: int
    unnorm_key: str
    robot_platform: str

    def as_dict(self) -> dict:
        return {key: getattr(self, key) for key in POLICY_KEYS}


@dataclass(frozen=True)
class ObservationContract:
    """How an image must be prepared, so evaluation matches training."""

    image_size: tuple[int, int]
    center_crop: bool

    def as_dict(self) -> dict:
        return {"image_size": list(self.image_size), "center_crop": self.center_crop}


@dataclass(frozen=True)
class Checkpoint:
    """A merged checkpoint together with the contract it was produced under."""

    root: Path
    record: Mapping[str, Any]

    @property
    def policy(self) -> PolicyContract:
        values = self.record["policy_contract"]
        return PolicyContract(
            action_dim=int(values["action_dim"]),
            action_chunk=int(values["action_chunk"]),
            proprio_dim=int(values["proprio_dim"]),
            use_proprio=bool(values["use_proprio"]),
            use_l1_regression=bool(values["use_l1_regression"]),
            num_images_in_input=int(values["num_images_in_input"]),
            unnorm_key=str(values["unnorm_key"]),
            robot_platform=str(values["robot_platform"]),
        )

    @property
    def observation(self) -> ObservationContract:
        values = self.record["observation_contract"]
        return ObservationContract(
            image_size=tuple(values["image_size"]), center_crop=bool(values["center_crop"])
        )

    @property
    def dataset(self) -> Mapping[str, Any]:
        return self.record.get("dataset", {})

    @property
    def backfilled(self) -> bool:
        return bool(self.record.get("provenance", {}).get("backfilled"))

    def check_dataset(self, dataset) -> Optional[str]:
        """Say whether an evaluation dataset is the one this was trained on.

        Returning a message rather than raising lets a deliberate cross-dataset
        evaluation proceed and be recorded as such; what must not happen is it
        proceeding unnoticed.
        """
        recorded = self.dataset.get("digest")
        if not recorded:
            return "checkpoint does not record which dataset it trained on"
        if recorded != dataset.digest():
            return (
                f"dataset mismatch: trained on {self.dataset.get('name')} "
                f"({recorded[:12]}...), evaluating {dataset.name} "
                f"({dataset.digest()[:12]}...)"
            )
        return None


def open_checkpoint(root: Path) -> Checkpoint:
    """Open a checkpoint through its contract; refuse one that has none."""
    root = Path(root)
    path = root / TRAINING_FILE
    if not path.is_file():
        raise CheckpointError(
            f"{root} has no {TRAINING_FILE}; the run predates the contract, "
            "or was written by an entry point that does not record one"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != SUPPORTED_SCHEMA:
        raise CheckpointError(
            f"{path} declares schema_version {record.get('schema_version')!r}, "
            f"expected {SUPPORTED_SCHEMA}"
        )
    for section, keys in (("policy_contract", POLICY_KEYS),
                          ("observation_contract", OBSERVATION_KEYS)):
        values = record.get(section)
        if not isinstance(values, Mapping):
            raise CheckpointError(f"{path} has no {section}")
        missing = [key for key in keys if key not in values]
        if missing:
            raise CheckpointError(f"{path} {section} is missing {', '.join(missing)}")

    statistics = root / "dataset_statistics.json"
    if statistics.is_file():
        keys = set(json.loads(statistics.read_text(encoding="utf-8")))
        unnorm_key = record["policy_contract"]["unnorm_key"]
        if unnorm_key not in keys:
            # The stored statistics are what the weights were normalised against,
            # so a key absent from them cannot be the right one.
            raise CheckpointError(
                f"{path} names unnorm_key {unnorm_key!r}, which is not in "
                f"dataset_statistics.json ({sorted(keys)})"
            )
    return Checkpoint(root=root, record=record)


def write_training_record(
    root: Path,
    policy: PolicyContract,
    observation: ObservationContract,
    dataset_name: str,
    dataset_digest: str,
    dataset_root: str,
    base_model: str,
    optimisation: Mapping[str, Any],
    splits: Optional[Mapping[str, Any]] = None,
    dataset_lineage: Optional[Mapping[str, Any]] = None,
    backfilled: bool = False,
    note: Optional[str] = None,
) -> Path:
    """Write the contract beside the weights."""
    record = {
        "schema_version": SUPPORTED_SCHEMA,
        "policy_contract": policy.as_dict(),
        "observation_contract": observation.as_dict(),
        "dataset": {
            "name": dataset_name,
            "digest": dataset_digest,
            "root": dataset_root,
            "splits": dict(splits or {}),
            "lineage": dict(dataset_lineage or {}),
        },
        "base_model": {
            "path": base_model,
            # The base checkpoint ships no action head, so the 151M-parameter
            # head is randomly initialised on every run that starts from it.
            "action_head": "randomly_initialised",
        },
        "optimisation": dict(optimisation),
        "provenance": {"backfilled": backfilled, "note": note},
    }
    for name in ("early-stopping.json", "manual-merge.json"):
        candidate = Path(root) / name
        if candidate.is_file():
            record[name.removesuffix(".json").replace("-", "_")] = json.loads(
                candidate.read_text(encoding="utf-8")
            )
    path = Path(root) / TRAINING_FILE
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
