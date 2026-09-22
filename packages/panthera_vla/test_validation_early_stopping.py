import json

import pytest

from validation_early_stopping import ValidationEarlyStopping


def test_stops_after_three_small_or_rising_validation_results() -> None:
    stopper = ValidationEarlyStopping(min_delta=1.0e-3, patience=3)

    assert stopper.update(step=1000, loss=0.0500).improved
    assert not stopper.update(step=2000, loss=0.0495).should_stop
    assert not stopper.update(step=3000, loss=0.0497).should_stop
    decision = stopper.update(step=4000, loss=0.0501)

    assert decision.should_stop
    assert decision.bad_validation_count == 3
    assert stopper.best_loss == pytest.approx(0.0500)
    assert stopper.best_step == 1000


def test_significant_improvement_resets_patience() -> None:
    stopper = ValidationEarlyStopping(min_delta=1.0e-3, patience=3)

    stopper.update(step=1000, loss=0.0500)
    stopper.update(step=2000, loss=0.0495)
    decision = stopper.update(step=3000, loss=0.0489)

    assert decision.improved
    assert decision.bad_validation_count == 0
    assert stopper.best_step == 3000
    assert stopper.best_loss == pytest.approx(0.0489)


def test_writes_machine_readable_status_atomically(tmp_path) -> None:
    stopper = ValidationEarlyStopping(min_delta=1.0e-3, patience=3)
    stopper.update(step=1000, loss=0.05)
    output = tmp_path / "early-stopping.json"

    stopper.write_json(output, stop_reason="early_stopping")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["stop_reason"] == "early_stopping"
    assert payload["best_step"] == 1000
    assert payload["best_loss"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("min_delta", "patience"),
    [(0.0, 3), (-1.0, 3), (float("nan"), 3), (1.0e-3, 0)],
)
def test_rejects_invalid_configuration(min_delta: float, patience: int) -> None:
    with pytest.raises(ValueError):
        ValidationEarlyStopping(min_delta=min_delta, patience=patience)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_validation_loss(loss: float) -> None:
    stopper = ValidationEarlyStopping(min_delta=1.0e-3, patience=3)
    with pytest.raises(ValueError):
        stopper.update(step=1000, loss=loss)


# --- closed-loop pass-rate stopping ------------------------------------------
#
# The point of the single-trajectory gate is that no generalisation is asked
# for, so the run should end the moment the policy reproduces its own
# trajectory, and not before.

from validation_early_stopping import PassRateEarlyStopping


def test_stops_when_the_single_trajectory_succeeds():
    stopper = PassRateEarlyStopping()
    assert not stopper.update(step=250, successes=0, trials=1).should_stop
    decision = stopper.update(step=500, successes=1, trials=1)
    assert decision.pass_rate == 1.0
    assert decision.reached_target
    assert decision.should_stop


def test_failure_resets_the_confirmation_streak():
    stopper = PassRateEarlyStopping(confirmations=2)
    assert not stopper.update(step=250, successes=1, trials=1).should_stop
    assert not stopper.update(step=500, successes=0, trials=1).should_stop
    assert not stopper.update(step=750, successes=1, trials=1).should_stop
    assert stopper.update(step=1000, successes=1, trials=1).should_stop


def test_minimum_step_prevents_an_early_fluke_from_ending_the_run():
    stopper = PassRateEarlyStopping(minimum_step=1000)
    assert not stopper.update(step=250, successes=1, trials=1).should_stop
    assert stopper.update(step=1000, successes=1, trials=1).should_stop


def test_partial_pass_rate_against_a_lower_target():
    stopper = PassRateEarlyStopping(target=0.5)
    assert not stopper.update(step=250, successes=1, trials=4).should_stop
    assert stopper.update(step=500, successes=2, trials=4).should_stop


def test_best_pass_rate_tracks_the_maximum():
    stopper = PassRateEarlyStopping(target=1.0)
    stopper.update(step=250, successes=1, trials=4)
    stopper.update(step=500, successes=3, trials=4)
    stopper.update(step=750, successes=2, trials=4)
    assert stopper.best_pass_rate == 0.75
    assert stopper.best_step == 500


@pytest.mark.parametrize(
    "kwargs", [{"target": 0.0}, {"target": 1.5}, {"confirmations": 0}, {"minimum_step": -1}]
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        PassRateEarlyStopping(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"step": -1, "successes": 0, "trials": 1},
               {"step": 0, "successes": 0, "trials": 0},
               {"step": 0, "successes": 2, "trials": 1},
               {"step": 0, "successes": -1, "trials": 1}]
)
def test_invalid_observation_rejected(kwargs):
    with pytest.raises(ValueError):
        PassRateEarlyStopping().update(**kwargs)


def test_state_snapshot_names_the_criterion(tmp_path):
    stopper = PassRateEarlyStopping()
    stopper.update(step=500, successes=1, trials=1)
    path = tmp_path / "early-stopping.json"
    stopper.write_json(path, stop_reason="closed_loop_target_reached")
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["criterion"] == "closed_loop_pass_rate"
    assert state["best_pass_rate"] == 1.0
    assert state["stop_reason"] == "closed_loop_target_reached"
