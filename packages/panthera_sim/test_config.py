"""Checks for the single simulation-parameter declaration.

The failures worth guarding against are the ones that already happened: a value
outside its range accepted silently, a snapshot that omits a parameter and so
cannot describe the physics a dataset was collected under, and a live
configuration that differs from a recorded one without anybody noticing.
"""

from __future__ import annotations

import pytest

from config import (
    PARAMETERS,
    SimConfig,
    SimConfigError,
    from_snapshot,
    resolve,
)


def test_every_parameter_resolves_without_an_environment():
    config = resolve({})
    assert set(config.values) == {parameter.name for parameter in PARAMETERS}
    assert all(source == "default" for source in config.sources.values())


def test_declared_names_and_variables_are_unique():
    names = [parameter.name for parameter in PARAMETERS]
    variables = [parameter.env for parameter in PARAMETERS if parameter.env]
    assert len(names) == len(set(names))
    assert len(variables) == len(set(variables))


def test_environment_overrides_are_typed_and_attributed():
    config = resolve({"PANTHERA_V2_SOLVER_POSITION_ITERATIONS": "64"})
    assert config["solver_position_iterations"] == 64
    assert config.sources["solver_position_iterations"] == "environment"
    assert config.sources["solver_velocity_iterations"] == "default"


def test_unparsable_value_names_the_variable():
    with pytest.raises(SimConfigError, match="PANTHERA_V2_SOLVER_POSITION_ITERATIONS"):
        resolve({"PANTHERA_V2_SOLVER_POSITION_ITERATIONS": "many"})


@pytest.mark.parametrize(
    "variable,value",
    [
        ("PANTHERA_V2_SOLVER_POSITION_ITERATIONS", "0"),
        ("PANTHERA_V2_SOLVER_POSITION_ITERATIONS", "256"),
        ("PANTHERA_V2_CYLINDER_LINEAR_DAMPING", "-1"),
        ("PANTHERA_V2_CYLINDER_ANGULAR_DAMPING", "11"),
        ("PANTHERA_V2_FORCED_LYING_ANGLE_BIN", "8"),
        ("PANTHERA_V2_DIRECT_RELEASE_BOTTOM_CLEARANCE_M", "0.2"),
        ("PANTHERA_V2_REORIENTATION_RELEASE_Z_M", "0.5"),
    ],
)
def test_out_of_range_values_are_rejected(variable, value):
    with pytest.raises(SimConfigError):
        resolve({variable: value})


def test_zero_depenetration_velocity_rejected_as_exclusive_bound():
    with pytest.raises(SimConfigError, match="> 0.0"):
        resolve({"PANTHERA_V2_MAX_DEPENETRATION_VELOCITY_MPS": "0"})
    assert resolve(
        {"PANTHERA_V2_MAX_DEPENETRATION_VELOCITY_MPS": "0.05"}
    )["max_depenetration_velocity_mps"] == 0.05


@pytest.mark.parametrize("value", ["base", "side"])
def test_choice_parameters_accept_their_choices(value):
    assert resolve({"PANTHERA_V2_REORIENTATION_RADIAL_MODE": value})[
        "reorientation_radial_mode"
    ] == value


def test_choice_parameter_rejects_anything_else():
    with pytest.raises(SimConfigError, match="must be one of"):
        resolve({"PANTHERA_V2_REORIENTATION_RADIAL_MODE": "middle"})


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("on", True), ("0", False), ("no", False)],
)
def test_assist_flag_parses_as_boolean(raw, expected):
    assert resolve({"PANTHERA_TERMINAL_TARGET_ASSIST": raw})[
        "terminal_target_assist"
    ] is expected


def test_assist_flag_rejects_nonsense():
    with pytest.raises(SimConfigError):
        resolve({"PANTHERA_TERMINAL_TARGET_ASSIST": "maybe"})


def test_snapshot_round_trips():
    config = resolve({"PANTHERA_V2_SOLVER_POSITION_ITERATIONS": "64"})
    restored = from_snapshot(config.snapshot())
    assert restored.values == config.values


def test_snapshot_carries_geometry_the_dataset_needs():
    geometry = resolve({}).snapshot()["geometry"]
    assert geometry["cylinder_radius_m"] == 0.0275
    assert geometry["table_height_m"] == 0.74


def test_snapshot_missing_a_parameter_is_rejected():
    snapshot = resolve({}).snapshot()
    del snapshot["values"]["solver_position_iterations"]
    with pytest.raises(SimConfigError, match="missing parameters"):
        from_snapshot(snapshot)


def test_snapshot_with_an_unknown_parameter_is_rejected():
    snapshot = resolve({}).snapshot()
    snapshot["values"]["cylinder_bounciness"] = 0.5
    with pytest.raises(SimConfigError, match="unknown parameters"):
        from_snapshot(snapshot)


def test_snapshot_with_an_out_of_range_value_is_rejected():
    snapshot = resolve({}).snapshot()
    snapshot["values"]["solver_position_iterations"] = 999
    with pytest.raises(SimConfigError):
        from_snapshot(snapshot)


def test_difference_reports_stored_then_live():
    """A downstream run compares its live configuration against the dataset's."""
    stored = resolve({}).snapshot()
    live = resolve({"PANTHERA_V2_SOLVER_POSITION_ITERATIONS": "64"})
    assert live.difference(stored) == {"solver_position_iterations": (20, 64)}
    assert resolve({}).difference(stored) == {}


def test_unknown_parameter_lookup_is_an_error():
    with pytest.raises(KeyError, match="cylinder_bounciness"):
        resolve({})["cylinder_bounciness"]


def test_every_parameter_states_why_it_exists():
    assert all(parameter.why.strip() for parameter in PARAMETERS)
