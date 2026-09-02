import numpy as np
import pytest

from panthera_vision_teleop.joint_command_guard import (
    ConsecutiveFailureLatch,
    JointCommandGuard,
    JointMotionLimiter,
)


NAMES = [f"joint{index}" for index in range(1, 7)]


def guard():
    return JointCommandGuard(
        NAMES,
        [-2.4, -0.1, -0.1, -1.6, -1.7, -2.5],
        [2.4, 3.2, 4.0, 1.6, 1.7, 2.5],
        [0.5] * 6,
        [2.0] * 6,
        stale_timeout_sec=0.25,
    )


def test_guard_accepts_slow_in_limit_sequence():
    state = guard()
    state.seed_feedback(NAMES, [0.0] * 6, 0.0)
    assert np.allclose(state.accept(NAMES, [0.01] * 6, 0.1), [0.01] * 6)
    assert np.allclose(state.accept(NAMES, [0.03] * 6, 0.2), [0.03] * 6)


@pytest.mark.parametrize("positions", [
    [2.5, 0, 0, 0, 0, 0],
    [0, -0.2, 0, 0, 0, 0],
    [0, 0, float("nan"), 0, 0, 0],
])
def test_guard_rejects_limit_or_nonfinite_targets(positions):
    with pytest.raises(ValueError):
        guard().accept(NAMES, positions, 0.0)


def test_guard_rejects_wrong_joint_order_and_rate_jumps():
    state = guard()
    state.seed_feedback(NAMES, [0.0] * 6, 0.0)
    with pytest.raises(ValueError, match="joint order"):
        state.accept(list(reversed(NAMES)), [0.0] * 6, 0.1)
    with pytest.raises(ValueError, match="velocity"):
        state.accept(NAMES, [0.2] * 6, 0.1)


def test_guard_emits_exactly_one_stale_hold():
    state = guard()
    state.seed_feedback(NAMES, [0.1] * 6, 1.0)
    assert state.stale_hold(1.25) is None
    assert np.allclose(state.stale_hold(1.251), [0.1] * 6)
    assert state.stale_hold(2.0) is None


def test_failure_latch_requires_consecutive_failures_and_explicit_reset():
    latch = ConsecutiveFailureLatch(3)
    assert not latch.failure("IK_FAILED")
    latch.success()
    assert latch.count == 0
    assert not latch.failure("GOAL_ERROR")
    assert not latch.failure("GOAL_ERROR")
    assert latch.failure("GOAL_ERROR")
    assert latch.locked and latch.reason == "GOAL_ERROR"
    latch.success()
    assert latch.locked
    latch.reset()
    assert not latch.locked and latch.count == 0 and latch.reason == ""


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_failure_latch_rejects_invalid_limit(limit):
    with pytest.raises(ValueError):
        ConsecutiveFailureLatch(limit)


def test_joint_motion_limiter_uses_host_velocity_and_acceleration_bounds():
    limiter = JointMotionLimiter(
        [-2.4, -0.1, -0.1, -1.6, -1.7, -2.5],
        [2.4, 3.2, 4.0, 1.6, 1.7, 2.5],
        [0.6] * 6,
        [2.0] * 6,
    )
    limiter.reset([0.0] * 6, 0.0)
    positions = []
    velocities = []
    target = [2.0, 2.0, 2.0, 1.5, 1.5, 2.0]
    for index in range(1, 101):
        position, velocity = limiter.update(target, index * 0.02)
        positions.append(position.copy())
        velocities.append(velocity.copy())
    velocities = np.asarray(velocities)
    assert np.max(np.abs(velocities)) <= 0.6 + 1e-12
    acceleration = np.diff(np.vstack((np.zeros(6), velocities)), axis=0) / 0.02
    assert np.max(np.abs(acceleration)) <= 2.0 + 1e-10
    assert np.all(np.diff(np.asarray(positions), axis=0) >= -1e-12)


def test_joint_motion_limiter_brakes_without_overshooting():
    limiter = JointMotionLimiter([-2.0], [2.0], [0.6], [2.0])
    limiter.reset([0.0], 0.0)
    samples = [limiter.update([0.2], index * 0.02)[0][0] for index in range(1, 101)]
    assert max(samples) <= 0.2 + 1e-12
    assert samples[-1] == pytest.approx(0.2)
    assert limiter.velocity[0] == pytest.approx(0.0)
