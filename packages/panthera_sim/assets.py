"""Load simulation assets as profiles, and let a dataset pin what it used.

An embodiment's kinematics live in RoboTwin's own config format, its contact
physics lived in the task YAML, its geometry lived in Python constants, and the
one behaviour it has -- the gripper ratchet -- lived inside an audit script.
Nothing tied them together, so a dataset could not say which end effector it was
collected with, and a downstream run could not check.

A profile is that missing unit.  It carries a digest over everything it declares
plus the upstream files it references, which is what a dataset snapshots and a
later run is compared against.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

PROFILE_FILE = "profile.yml"
SUPPORTED_SCHEMA = 1
BEHAVIOUR_CONTEXTS = ("collection", "evaluation", "deployment")


class AssetError(ValueError):
    """A profile is missing, malformed, or references something absent."""


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise AssetError(f"missing asset file: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise AssetError(f"{path} does not contain a mapping")
    schema = loaded.get("schema_version")
    if schema != SUPPORTED_SCHEMA:
        raise AssetError(f"{path} declares schema_version {schema!r}, expected {SUPPORTED_SCHEMA}")
    return loaded


@dataclass(frozen=True)
class ObjectProfile:
    """A scene object's intrinsic properties."""

    name: str
    geometry: Mapping[str, float]
    physics: Mapping[str, float]
    appearance: Mapping[str, Any]
    source: Path

    def digest(self) -> str:
        return hashlib.sha256(
            _canonical({
                "geometry": dict(self.geometry),
                "physics": dict(self.physics),
                "appearance": dict(self.appearance),
            })
        ).hexdigest()

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "geometry": dict(self.geometry),
            "physics": dict(self.physics),
            "appearance": dict(self.appearance),
            "digest": self.digest(),
        }


@dataclass(frozen=True)
class AssetProfile:
    """An embodiment together with its end effector and that end effector's behaviours."""

    name: str
    root: Path
    document: Mapping[str, Any]
    referenced: Mapping[str, str]

    @property
    def end_effector(self) -> Mapping[str, Any]:
        return self.document["end_effector"]

    @property
    def cameras(self) -> Mapping[str, str]:
        return self.document.get("cameras", {})

    def contact(self) -> Mapping[str, float]:
        """Friction the task must apply for this end effector to hold anything."""
        return self.end_effector["contact"]

    def behaviour(self, name: str, context: str) -> Optional[Any]:
        """Instantiate a declared behaviour, when the profile enables it here.

        Returning ``None`` rather than raising is deliberate: a caller asks for
        the ratchet in every context and the profile decides, so collection does
        not silently acquire a behaviour the recorded data was not made with.
        """
        if context not in BEHAVIOUR_CONTEXTS:
            raise AssetError(f"unknown behaviour context {context!r}")
        declared = self.end_effector.get("behaviours", {}).get(name)
        if declared is None:
            return None
        enabled = declared.get("enabled_for", [])
        if context not in enabled:
            return None
        module_name = declared["module"]
        entry = declared["entry"]
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise AssetError(f"behaviour {name!r} cannot import {module_name}") from error
        try:
            factory = getattr(module, entry)
        except AttributeError as error:
            raise AssetError(f"{module_name} has no {entry!r}") from error
        arguments = {
            key: value
            for key, value in declared.items()
            if key not in {"module", "entry", "enabled_for"}
        }
        return factory(**arguments)

    def digest(self) -> str:
        """Hash the profile and every upstream file it points at."""
        return hashlib.sha256(
            _canonical({"document": self.document, "referenced": dict(self.referenced)})
        ).hexdigest()

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "digest": self.digest(),
            "end_effector": dict(self.end_effector),
            "cameras": dict(self.cameras),
            "referenced": dict(self.referenced),
        }


def load_profile(profiles_root: Path, name: str, assets_root: Optional[Path] = None) -> AssetProfile:
    """Read one embodiment profile and hash the upstream files it references."""
    profiles_root = Path(profiles_root)
    path = profiles_root / name / PROFILE_FILE
    document = _read_yaml(path)
    if document.get("name") != name:
        raise AssetError(f"{path} declares name {document.get('name')!r}, expected {name!r}")
    for required in ("embodiment", "end_effector"):
        if required not in document:
            raise AssetError(f"{path} is missing {required!r}")
    end_effector = document["end_effector"]
    for required in ("name", "opening_range", "contact"):
        if required not in end_effector:
            raise AssetError(f"{path} end_effector is missing {required!r}")
    low, high = end_effector["opening_range"]
    if not low < high:
        raise AssetError(f"{path} opening_range must be increasing")

    # Embodiment paths are written relative to the overlay root, where RoboTwin's
    # loader resolves them, not relative to the profile directory.
    base = Path(assets_root) if assets_root is not None else profiles_root.parents[1]
    referenced: dict[str, str] = {}
    for key, relative in document["embodiment"].items():
        target = base / relative
        if not target.is_file():
            raise AssetError(f"{path} references a missing file: {target}")
        referenced[key] = hashlib.sha256(target.read_bytes()).hexdigest()
    return AssetProfile(
        name=name, root=profiles_root / name, document=document, referenced=referenced
    )


def load_object(profiles_root: Path, name: str) -> ObjectProfile:
    """Read one scene-object profile."""
    path = Path(profiles_root) / "objects" / f"{name}.yml"
    document = _read_yaml(path)
    if document.get("name") != name:
        raise AssetError(f"{path} declares name {document.get('name')!r}, expected {name!r}")
    return ObjectProfile(
        name=name,
        geometry=document.get("geometry", {}),
        physics=document.get("physics", {}),
        appearance=document.get("appearance", {}),
        source=path,
    )
