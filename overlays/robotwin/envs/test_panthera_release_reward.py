from dataclasses import replace

import pytest

from envs.panthera_release_reward import (
    ReleaseEffectState,
    evaluate_transition,
    remaining_cost,
)


def state(**changes):
    baseline = ReleaseEffectState(
        tcp_to_object_m=0.20,
        target_xy_error_m=0.18,
        target_height_error_m=0.15,
        target_axis_error_deg=45.0,
        object_linear_speed_mps=0.0,
        object_angular_speed_radps=0.0,
        grasped=False,
        released_in_valid_volume=False,
        success=False,
        hard_failure=False,
    )
    return replace(baseline, **changes)


def trajectory_return(states, *, variant="r2"):
    return sum(
        evaluate_transition(previous, current, variant=variant).total
        for previous, current in zip(states, states[1:])
    )


def test_expert_grasp_transport_release_and_settle_ranks_highest():
    trace = [
        state(),
        state(tcp_to_object_m=0.01),
        state(tcp_to_object_m=0.01, grasped=True),
        state(
            tcp_to_object_m=0.01,
            grasped=True,
            target_xy_error_m=0.006,
            target_height_error_m=0.010,
            target_axis_error_deg=2.0,
        ),
        state(
            tcp_to_object_m=0.05,
            target_xy_error_m=0.005,
            target_height_error_m=0.008,
            target_axis_error_deg=2.0,
            released_in_valid_volume=True,
            object_linear_speed_mps=0.02,
        ),
        state(
            tcp_to_object_m=0.08,
            target_xy_error_m=0.002,
            target_height_error_m=0.002,
            target_axis_error_deg=1.0,
            released_in_valid_volume=True,
            success=True,
        ),
    ]
    assert trajectory_return(trace) > 0.95


def test_premature_release_outside_target_is_penalized_not_rewarded():
    before = state(tcp_to_object_m=0.01, grasped=True)
    premature = state(tcp_to_object_m=0.03, hard_failure=True)
    reward = evaluate_transition(before, premature, variant="r2")
    assert reward.task == -1.0
    assert reward.total < -1.0


def test_arrival_while_still_grasped_is_not_success():
    far = state(tcp_to_object_m=0.01, grasped=True)
    arrived = replace(
        far,
        target_xy_error_m=0.003,
        target_height_error_m=0.006,
        target_axis_error_deg=1.0,
    )
    reward = evaluate_transition(far, arrived, variant="r2")
    assert reward.task == 0.0
    assert reward.total < 1.0
    assert remaining_cost(arrived) > 0.0


def test_object_lost_outside_recoverable_workspace_is_hard_failure():
    grasped = state(tcp_to_object_m=0.01, grasped=True)
    lost = state(tcp_to_object_m=0.40, hard_failure=True)
    assert evaluate_transition(grasped, lost, variant="r1").total == -1.0


def test_repeated_open_close_cannot_outscore_successful_settlement():
    held = state(tcp_to_object_m=0.01, grasped=True, target_xy_error_m=0.008)
    released = replace(held, grasped=False, released_in_valid_volume=True)
    # Regrasping after a nominal release increases remaining cost again.  The
    # potential telescopes, so cycling cannot manufacture terminal reward.
    cycle = [held, released, held, released, held]
    success = replace(
        released,
        target_xy_error_m=0.002,
        target_height_error_m=0.002,
        target_axis_error_deg=1.0,
        success=True,
    )
    assert trajectory_return(cycle) < trajectory_return([held, released, success])


def test_success_is_paid_once_only_after_settled_state_latches():
    released_moving = state(
        released_in_valid_volume=True,
        target_xy_error_m=0.003,
        target_height_error_m=0.003,
        target_axis_error_deg=1.0,
        object_linear_speed_mps=0.08,
    )
    settled_success = replace(
        released_moving,
        object_linear_speed_mps=0.0,
        success=True,
    )
    first = evaluate_transition(released_moving, settled_success, variant="r1")
    absorbing = evaluate_transition(settled_success, settled_success, variant="r1")
    assert first.total == 1.0
    assert absorbing.total == 0.0


def test_smoothness_penalty_excludes_gripper_dimension():
    previous = state()
    current = replace(previous, tcp_to_object_m=0.19)
    arm_same_gripper_changed = evaluate_transition(
        previous,
        current,
        variant="r2",
        previous_action=[0.0] * 7,
        current_action=[0.0] * 6 + [1.0],
    )
    assert arm_same_gripper_changed.arm_action_change == pytest.approx(0.0)
