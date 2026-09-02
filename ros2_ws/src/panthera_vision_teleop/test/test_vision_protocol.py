import json

import pytest

from panthera_vision_teleop.vision_protocol import EnableRearmLatch, ProtocolError, TransportLagGuard, latch_enable_on_pose_loss, parse_packet


def packet(**updates):
    value = {
        "v": 1, "seq": 1, "source_time_ms": 100,
        "enabled": True, "recalibrate": False,
        "pose_valid": True, "hand_valid": True,
        "shoulder": [0, 0, 0], "elbow": [0.1, 0.2, 0], "wrist": [0.3, 0.1, -0.1],
        "pose_score": 0.9, "gesture": "Open_Palm", "gesture_score": 0.8,
    }
    value.update(updates)
    return json.dumps(value)


def test_valid_packet_is_parsed():
    result = parse_packet(packet())
    assert result.seq == 1
    assert result.wrist == (0.3, 0.1, -0.1)
    # Protocol-v1 senders from before raw-feature support remain valid.
    assert result.pinch_ratio == -1.0
    assert not result.depth_clutch
    assert result.experiment_marker == ""


@pytest.mark.parametrize("raw", ["not json", "[]", packet(v=2), packet(pose_score=2), packet(shoulder=[0, 0]), packet(enabled=1), packet(pinch_ratio=float("nan")), packet(depth_clutch=1)])
def test_bad_packets_are_rejected(raw):
    with pytest.raises(ProtocolError):
        parse_packet(raw)


def test_sequence_must_increase_per_connection():
    with pytest.raises(ProtocolError):
        parse_packet(packet(seq=4), previous_seq=4)


def test_source_timestamp_must_increase_per_connection():
    with pytest.raises(ProtocolError):
        parse_packet(packet(seq=5, source_time_ms=100), previous_seq=4, previous_source_time_ms=100)


def test_transport_lag_guard_uses_best_clock_offset_as_baseline():
    guard = TransportLagGuard(max_extra_delay_ms=250.0)
    assert guard.accept(source_time_ms=1000, arrival_time_ms=5100.0)
    assert guard.accept(source_time_ms=1100, arrival_time_ms=5180.0)
    assert not guard.accept(source_time_ms=1200, arrival_time_ms=5551.0)
    assert guard.latest_extra_delay_ms == 271.0
    # A faster packet improves the baseline; clock epochs need not match.
    assert guard.accept(source_time_ms=1300, arrival_time_ms=5370.0)


def test_bridge_rearm_latch_requires_disable_then_enable_after_loss():
    latch = EnableRearmLatch()
    assert latch.update(True, True)
    assert not latch.update(True, False)
    assert not latch.update(True, True)
    assert not latch.update(False, True)
    assert latch.update(True, True)


def test_bridge_timeout_lock_also_requires_rearm_transition():
    latch = EnableRearmLatch()
    assert latch.update(True, True)
    latch.force_lock()
    assert not latch.update(True, True)
    assert not latch.update(False, True)
    assert latch.update(True, True)


def test_pose_loss_latches_enable_off_until_operator_reenables():
    enabled, latched = latch_enable_on_pose_loss(True, False, 0.0, 0.6)
    assert not enabled
    assert latched

    # Good tracking by itself cannot restore the previous enabled state.
    enabled, latched = latch_enable_on_pose_loss(enabled, True, 0.95, 0.6)
    assert not enabled
    assert not latched

    # A later explicit enable action is accepted only while tracking is good.
    enabled, latched = latch_enable_on_pose_loss(True, True, 0.95, 0.6)
    assert enabled
    assert not latched


def test_low_pose_score_is_also_a_tracking_loss():
    assert latch_enable_on_pose_loss(True, True, 0.59, 0.6) == (False, True)


def test_optional_raw_features_are_parsed():
    result = parse_packet(packet(
        depth_world_m=-0.21,
        image_reach_ratio=2.4,
        image_arm_scale=3.1,
        pinch_ratio=0.35,
        depth_clutch=True,
        orientation_valid=False,
    ))
    assert result.depth_world_m == -0.21
    assert result.image_reach_ratio == 2.4
    assert result.image_arm_scale == 3.1
    assert result.pinch_ratio == 0.35
    assert result.depth_clutch
    assert not result.orientation_valid
