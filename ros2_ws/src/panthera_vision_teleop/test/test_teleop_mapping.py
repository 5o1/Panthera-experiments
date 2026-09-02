import numpy as np

from panthera_vision_teleop.teleop_logic import (
    CartesianMotionLimiter,
    DepthClutch,
    GestureDebouncer,
    MappingConfig,
    PerAxisCommandGate,
    PinchHysteresis,
    RotationMotionLimiter,
    TeleopMapping,
    TeleopState,
    TeleopStateMachine,
    limit_vector_step,
)
from panthera_vision_teleop.rotation_utils import rotation_angle, rpy_to_matrix


def mapping(**overrides):
    overrides.setdefault("max_position_velocity_mps", np.full(3, 1000.0))
    overrides.setdefault("max_position_acceleration_mps2", np.full(3, 1000.0))
    config = MappingConfig(position_alpha=1.0, **overrides)
    result = TeleopMapping(config)
    result.update_robot([0.3, 0.0, 0.4], [0.0, 0.1, 0.0])
    result.update_human([0.35, 0.12, -0.1], [0, 0, 0, 1], 1.0)
    return result


def test_disabled_and_uncalibrated_produce_no_target():
    state = mapping()
    assert state.target(1.0) is None
    state.set_enabled(True)
    assert state.target(1.0) is not None
    state.set_enabled(False)
    assert state.target(1.0) is None


def test_calibration_same_input_has_zero_offset_and_fixed_orientation():
    state = mapping()
    assert state.set_enabled(True)
    target = state.target(1.0)
    assert target is not None
    assert np.allclose(target.position, [0.3, 0.0, 0.4])
    assert np.allclose(target.rpy, [0.0, 0.1, 0.0])


def test_camera_axes_gain_and_limits():
    state = mapping(
        position_gain=np.array([0.25, 0.25, 0.12]),
        max_position_offset=np.array([0.02, 0.03, 0.04]),
    )
    state.set_enabled(True)
    # camera delta [right=.2, up=-.5, toward=-.4] maps robot [+x,+y,+z]
    state.update_human([0.55, -0.38, -0.5], [0, 0, 0, 1], 1.1)
    target = state.target(1.1)
    assert target is not None
    assert np.allclose(target.position, [0.32, 0.03, 0.44])


def test_safe_default_locks_noisy_monocular_depth_axis():
    state = mapping()
    state.set_enabled(True)
    # Change only camera depth.  The general mapper supports depth when an X
    # gain is supplied, but the real-demo default keeps robot X fixed.
    state.update_human([0.35, 0.12, -0.5], [0, 0, 0, 1], 1.1)
    target = state.target(1.1)
    assert target is not None
    assert np.allclose(target.position, [0.3, 0.0, 0.4])


def test_per_axis_deadband_rejects_small_jitter_continuously():
    state = mapping(position_deadband_m=np.array([0.0, 0.002, 0.002]))
    state.set_enabled(True)
    # Camera +x maps to robot +y. gain .25 makes this a 1 mm raw offset,
    # which is below the 2 mm robot-axis deadband.
    state.update_human([0.354, 0.12, -0.1], [0, 0, 0, 1], 1.1)
    target = state.target(1.1)
    assert target is not None
    assert np.allclose(target.position, [0.3, 0.0, 0.4])
    diagnostics = state.latest_diagnostics
    assert diagnostics is not None
    assert np.isclose(diagnostics.raw_robot_offset[1], 0.001)
    assert np.isclose(diagnostics.deadbanded_robot_offset[1], 0.0)


def test_vector_step_limit_preserves_direction_and_caps_distance():
    current = np.array([0.3, 0.0, 0.4])
    desired = np.array([0.3, 0.03, 0.04])
    limited = limit_vector_step(current, desired, 0.01)
    assert np.isclose(np.linalg.norm(limited - current), 0.01)
    assert np.allclose(
        (limited - current) / np.linalg.norm(limited - current),
        (desired - current) / np.linalg.norm(desired - current),
    )


def test_vector_step_limit_returns_nearby_target_unchanged():
    desired = np.array([0.3, 0.003, 0.404])
    assert np.allclose(limit_vector_step([0.3, 0.0, 0.4], desired, 0.01), desired)


def test_cartesian_motion_limiter_bounds_velocity_and_acceleration():
    limiter = CartesianMotionLimiter([0.2, 0.2, 0.2], [0.5, 0.5, 0.5])
    limiter.reset([0.0, 0.0, 0.0], 0.0)
    first = limiter.update([1.0, 0.0, 0.0], 0.1)
    second = limiter.update([1.0, 0.0, 0.0], 0.2)
    # Acceleration 0.5 m/s² permits 0.05 then 0.10 m/s.
    assert np.allclose(first, [0.005, 0.0, 0.0])
    assert np.allclose(second, [0.015, 0.0, 0.0])
    assert np.all(np.abs(limiter.velocity) <= 0.2)


def test_cartesian_motion_limiter_reaches_target_without_overshoot():
    limiter = CartesianMotionLimiter([1.0, 1.0, 1.0], [10.0, 10.0, 10.0])
    limiter.reset([0.0, 0.0, 0.0], 0.0)
    assert np.allclose(limiter.update([0.01, -0.01, 0.0], 0.1), [0.01, -0.01, 0.0])
    assert np.allclose(limiter.velocity, np.zeros(3))


def test_rotation_motion_limiter_bounds_angular_acceleration():
    limiter = RotationMotionLimiter(max_velocity=1.0, max_acceleration=2.0)
    limiter.reset(np.eye(3), 0.0)
    desired = rpy_to_matrix(0.0, 0.0, 1.0)
    first = limiter.update(desired, 0.1)
    second = limiter.update(desired, 0.2)
    assert np.isclose(rotation_angle(first), 0.02, atol=1e-8)
    assert np.isclose(rotation_angle(second), 0.06, atol=1e-8)
    assert np.linalg.norm(limiter.angular_velocity) <= 1.0


def test_per_axis_command_gate_has_enter_exit_hysteresis():
    gate = PerAxisCommandGate([0.006, 0.006, 0.006], [0.003, 0.003, 0.003])
    assert not gate.update([0.005, 0.0, 0.0])
    assert gate.update([0.006, 0.0, 0.0])
    assert gate.update([0.004, 0.0, 0.0])
    assert not gate.update([0.003, 0.0, 0.0])
    gate.update([0.0, 0.0, 0.01])
    gate.reset()
    assert not gate.update([0.0, 0.0, 0.004])


def test_stale_input_and_invalidation_stop_targets():
    state = mapping(stale_timeout_sec=0.5)
    state.set_enabled(True)
    assert state.target(1.49) is not None
    assert state.target(1.51) is None
    state.invalidate_input()
    assert state.target(1.1) is None


def test_gesture_needs_stable_confident_transition_only_once():
    gesture = GestureDebouncer(["Open_Palm"], ["Closed_Fist"], 0.7, 3)
    assert gesture.update("Open_Palm", 0.9, True) is None
    assert gesture.update("Open_Palm", 0.9, True) is None
    assert gesture.update("Open_Palm", 0.9, True) is True
    assert gesture.update("Open_Palm", 0.9, True) is None
    assert gesture.update("Closed_Fist", 0.6, True) is None
    assert gesture.update("Unknown", 0.99, True) is None
    assert gesture.update("Closed_Fist", 0.9, True) is None
    assert gesture.update("Closed_Fist", 0.9, True) is None
    assert gesture.update("Closed_Fist", 0.9, True) is False


def test_gesture_disabled_never_emits():
    gesture = GestureDebouncer(["Open_Palm"], ["Closed_Fist"], 0.7, 1)
    assert gesture.update("Open_Palm", 1.0, False) is None


def test_pinch_hysteresis_uses_stable_time_neutral_band_and_cooldown():
    pinch = PinchHysteresis(0.35, 0.65, stable_sec=0.2, cooldown_sec=0.5)
    assert pinch.update(0.2, 1.0, True) is None
    assert pinch.update(0.2, 1.19, True) is None
    assert pinch.update(0.2, 1.21, True) is False
    # Neutral-band samples do not request the opposite transition.
    assert pinch.update(0.5, 1.3, True) is None
    assert pinch.update(0.8, 1.4, True) is None
    assert pinch.update(0.8, 1.61, True) is None
    assert pinch.update(0.8, 1.72, True) is True


def test_pinch_disabled_or_invalid_resets_candidate():
    pinch = PinchHysteresis(0.35, 0.65, stable_sec=0.1, cooldown_sec=0.0)
    assert pinch.update(0.2, 1.0, True) is None
    assert pinch.update(0.2, 1.2, False) is None
    assert pinch.update(float("nan"), 1.3, True) is None
    assert pinch.update(0.2, 1.4, True) is None


def test_teleop_state_machine_requires_rearm_after_stale():
    state = TeleopStateMachine()
    assert state.source_enable(True, False) == TeleopState.CALIBRATING
    assert state.inputs_ready() == TeleopState.TRACKING
    assert state.stale() == TeleopState.STALE_LOCK
    assert state.source_enable(True, True) == TeleopState.STALE_LOCK
    assert state.source_enable(False, True) == TeleopState.DISABLED
    assert state.source_enable(True, True) == TeleopState.TRACKING


def test_teleop_state_machine_fault_uses_same_explicit_rearm_rule():
    state = TeleopStateMachine()
    state.source_enable(True, True)
    assert state.fault() == TeleopState.FAULT_LOCK
    assert state.source_enable(True, True) == TeleopState.FAULT_LOCK
    state.source_enable(False, True)
    assert state.source_enable(True, False) == TeleopState.CALIBRATING


def test_depth_clutch_holds_release_value_until_reengaged():
    clutch = DepthClutch()
    assert clutch.update(-0.1, False) == -0.1
    assert clutch.update(-0.2, False) == -0.1
    assert clutch.update(-0.2, True) == -0.2
    assert clutch.update(-0.3, False) == -0.2
    clutch.reset()
    assert clutch.update(-0.3, False) == -0.3
