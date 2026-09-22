"""Checks for asset profiles and the behaviours they declare.

Two things are being guarded.  First, that a profile can be pinned: a dataset
records a digest, and a later run that silently swapped an end effector or its
friction must not match.  Second, that a behaviour belongs to the end effector
and is enabled per context -- collection must not pick up a ratchet the recorded
data was not made with.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from assets import AssetError, load_object, load_profile
from behaviours.ratchet import GripperRatchet

REPO_PROFILES = Path(__file__).resolve().parents[2] / "overlays/robotwin/assets/profiles"


def _write_profile(tmp_path: Path, **overrides) -> Path:
    profiles = tmp_path / "profiles"
    (profiles / "demo").mkdir(parents=True)
    (tmp_path / "embodiments").mkdir()
    (tmp_path / "embodiments" / "config.yml").write_text("joint_stiffness: 1000\n")
    document = {
        "schema_version": 1,
        "name": "demo",
        "embodiment": {"robotwin_config": "embodiments/config.yml"},
        "end_effector": {
            "name": "demo_gripper",
            "opening_range": [0.0, 1.0],
            "contact": {"static_friction": 8.0, "dynamic_friction": 6.0},
            "behaviours": {
                "ratchet": {
                    "module": "behaviours.ratchet",
                    "entry": "GripperRatchet",
                    "margin": 0.15,
                    "hold_steps": 3,
                    "contact_threshold": 2,
                    "enabled_for": ["evaluation", "deployment"],
                }
            },
        },
    }
    document.update(overrides)
    (profiles / "demo" / "profile.yml").write_text(yaml.safe_dump(document))
    return profiles


# --- the repository's own profiles --------------------------------------------

def test_repository_profile_loads_and_matches_recorded_physics():
    profile = load_profile(REPO_PROFILES, "panthera_phone_symmetric")
    contact = profile.contact()
    # These are the values the 1280-episode dataset recorded in
    # physics_parameters; a profile that disagrees would silently change grasps.
    assert contact["static_friction"] == 8.0
    assert contact["dynamic_friction"] == 6.0


def test_repository_object_profile_matches_recorded_physics():
    cylinder = load_object(REPO_PROFILES, "panthera_cylinder")
    assert cylinder.physics["mass_kg"] == 0.03
    assert cylinder.physics["static_friction"] == 1.2
    assert cylinder.physics["dynamic_friction"] == 1.0
    assert cylinder.physics["restitution"] == 0.0
    assert cylinder.geometry["radius_m"] == 0.0275
    assert cylinder.geometry["half_height_m"] == 0.06


def test_repository_profile_digest_is_stable():
    first = load_profile(REPO_PROFILES, "panthera_phone_symmetric").digest()
    second = load_profile(REPO_PROFILES, "panthera_phone_symmetric").digest()
    assert first == second and len(first) == 64


# --- pinning ------------------------------------------------------------------

def test_digest_changes_when_friction_changes(tmp_path):
    profiles = _write_profile(tmp_path)
    before = load_profile(profiles, "demo", assets_root=tmp_path).digest()
    document = yaml.safe_load((profiles / "demo" / "profile.yml").read_text())
    document["end_effector"]["contact"]["static_friction"] = 4.0
    (profiles / "demo" / "profile.yml").write_text(yaml.safe_dump(document))
    assert load_profile(profiles, "demo", assets_root=tmp_path).digest() != before


def test_digest_changes_when_the_referenced_embodiment_changes(tmp_path):
    profiles = _write_profile(tmp_path)
    before = load_profile(profiles, "demo", assets_root=tmp_path).digest()
    (tmp_path / "embodiments" / "config.yml").write_text("joint_stiffness: 500\n")
    assert load_profile(profiles, "demo", assets_root=tmp_path).digest() != before


def test_missing_referenced_file_is_named(tmp_path):
    profiles = _write_profile(tmp_path)
    (tmp_path / "embodiments" / "config.yml").unlink()
    with pytest.raises(AssetError, match="missing file"):
        load_profile(profiles, "demo", assets_root=tmp_path)


def test_wrong_schema_version_rejected(tmp_path):
    profiles = _write_profile(tmp_path, schema_version=2)
    with pytest.raises(AssetError, match="schema_version"):
        load_profile(profiles, "demo", assets_root=tmp_path)


def test_name_mismatch_rejected(tmp_path):
    profiles = _write_profile(tmp_path, name="other")
    with pytest.raises(AssetError, match="declares name"):
        load_profile(profiles, "demo", assets_root=tmp_path)


def test_inverted_opening_range_rejected(tmp_path):
    profiles = _write_profile(tmp_path)
    document = yaml.safe_load((profiles / "demo" / "profile.yml").read_text())
    document["end_effector"]["opening_range"] = [1.0, 0.0]
    (profiles / "demo" / "profile.yml").write_text(yaml.safe_dump(document))
    with pytest.raises(AssetError, match="increasing"):
        load_profile(profiles, "demo", assets_root=tmp_path)


# --- behaviours belong to the end effector ------------------------------------

def test_behaviour_is_enabled_per_context(tmp_path):
    profile = load_profile(_write_profile(tmp_path), "demo", assets_root=tmp_path)
    assert profile.behaviour("ratchet", "evaluation") is not None
    assert profile.behaviour("ratchet", "deployment") is not None
    # Collection must not acquire a behaviour the recorded data was not made with.
    assert profile.behaviour("ratchet", "collection") is None


def test_behaviour_is_built_with_the_profile_constants(tmp_path):
    profile = load_profile(_write_profile(tmp_path), "demo", assets_root=tmp_path)
    ratchet = profile.behaviour("ratchet", "evaluation")
    assert (ratchet.margin, ratchet.hold_steps, ratchet.contact_threshold) == (0.15, 3, 2)


def test_undeclared_behaviour_is_none(tmp_path):
    profile = load_profile(_write_profile(tmp_path), "demo", assets_root=tmp_path)
    assert profile.behaviour("suction", "evaluation") is None


def test_unknown_context_is_an_error(tmp_path):
    profile = load_profile(_write_profile(tmp_path), "demo", assets_root=tmp_path)
    with pytest.raises(AssetError, match="behaviour context"):
        profile.behaviour("ratchet", "training")


# --- the ratchet itself -------------------------------------------------------

def test_ratchet_passes_commands_through_before_contact():
    ratchet = GripperRatchet()
    assert ratchet(0.9, contacts=0, step=0) == 0.9
    assert ratchet(0.5, contacts=1, step=1) == 0.5
    assert not ratchet.holding


def test_ratchet_engages_on_contact_and_holds_against_jitter():
    ratchet = GripperRatchet(margin=0.15, hold_steps=3)
    ratchet(0.6, contacts=2, step=10)
    assert ratchet.holding and ratchet.engaged_at == 10
    assert ratchet(0.65, contacts=2, step=11) == 0.6
    assert ratchet(0.7, contacts=2, step=12) == 0.6
    assert ratchet.suppressed == 2


def test_ratchet_only_ever_tightens():
    ratchet = GripperRatchet()
    ratchet(0.6, contacts=2, step=0)
    assert ratchet(0.4, contacts=2, step=1) == 0.4
    assert ratchet(0.5, contacts=2, step=2) == 0.4


def test_sustained_wide_opening_releases():
    ratchet = GripperRatchet(margin=0.15, hold_steps=3)
    ratchet(0.3, contacts=2, step=0)
    for step in range(1, 3):
        assert ratchet(1.0, contacts=2, step=step) == 0.3
    assert ratchet(1.0, contacts=2, step=3) == 1.0
    assert not ratchet.holding


def test_transient_wide_opening_does_not_release():
    ratchet = GripperRatchet(margin=0.15, hold_steps=3)
    ratchet(0.3, contacts=2, step=0)
    ratchet(1.0, contacts=2, step=1)
    ratchet(0.3, contacts=2, step=2)   # streak broken
    assert ratchet(1.0, contacts=2, step=3) == 0.3
    assert ratchet.holding


def test_reset_clears_state():
    ratchet = GripperRatchet()
    ratchet(0.5, contacts=2, step=0)
    ratchet.reset()
    assert not ratchet.holding and ratchet.held_value is None
    assert ratchet.telemetry() == {
        "engaged_at": None, "suppressed_openings": 0, "still_holding": False
    }


@pytest.mark.parametrize(
    "kwargs", [{"margin": 0.0}, {"margin": 1.5}, {"hold_steps": 0}, {"contact_threshold": 0}]
)
def test_invalid_ratchet_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        GripperRatchet(**kwargs)


def test_the_live_objects_carry_the_values_the_baseline_was_measured_with():
    """The profile is the source, and it must still say what the data says.

    These numbers produced the 1249/1280 expert-replay baseline and the
    1280-episode dataset.  Moving them out of the environment modules and into
    the profiles is only safe while the profiles agree with them, so the values
    are pinned here rather than trusted.
    """
    profiles = REPO_PROFILES
    cylinder = load_object(profiles, "panthera_cylinder")
    socket = load_object(profiles, "panthera_socket")

    assert cylinder.geometry["radius_m"] == 0.0275
    assert cylinder.geometry["half_height_m"] == 0.06
    assert cylinder.physics["mass_kg"] == 0.03
    assert socket.geometry["inner_half_width_m"] == 0.034
    # The colour the renderer draws, and so what the recorded frames show.
    assert list(cylinder.appearance["base_color"]) == [0.98, 0.78, 0.02, 1.0]
    assert list(socket.appearance["base_color"]) == [0.98, 0.78, 0.02, 1.0]


def test_the_live_environment_reads_the_profile_instead_of_repeating_it():
    """No literal in the live task may restate a profile value.

    The profiles were written during the restructure and never read:
    ``load_object`` had zero callers, and the same numbers stayed in the
    environment modules.  The colour was the one value with no other home, so
    nothing recorded it and a change to it reached the training data unseen.
    """
    source = (
        REPO_PROFILES.parents[1] / "envs" / "place_vertical_cylinder_in_groove.py"
    ).read_text(encoding="utf-8")
    restated = (
        "CYLINDER_RADIUS_M = 0.0275",
        "CYLINDER_HALF_HEIGHT_M = 0.060",
        "CYLINDER_MASS_KG = 0.030",
        "SOCKET_INNER_HALF_WIDTH_M = 0.034",
        "base_color=[0.98, 0.78, 0.02, 1.0]",
        "socket_color = (0.98, 0.78, 0.02)",
    )
    for literal in restated:
        assert literal not in source, (
            f"{literal!r} is still written out in the live task; it belongs to "
            "the object profile"
        )
    assert "load_object" in source, "the live task no longer reads its profile"
