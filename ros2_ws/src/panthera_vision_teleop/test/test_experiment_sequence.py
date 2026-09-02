from panthera_vision_teleop.experiment_sequence import ExperimentPhase, ExperimentSequence


def test_sequence_advances_only_on_valid_frames():
    sequence = ExperimentSequence([
        ExperimentPhase("static", "hold", 2, 3),
        ExperimentPhase("move", "move", 0, 2),
    ])
    assert sequence.arm_current() == "WARMUP_STARTED"
    assert sequence.update(False) is None
    assert sequence.remaining == 2
    assert sequence.update(True) is None
    assert sequence.update(True) == "RECORDING_STARTED"
    assert sequence.state == "RECORDING" and sequence.remaining == 3
    assert sequence.update(True) is None
    assert sequence.update(True) is None
    assert sequence.update(True) == "PHASE_COMPLETE:static"
    assert sequence.state == "WAITING_NEXT"
    assert sequence.arm_current() == "RECORDING_STARTED"
    sequence.update(True)
    assert sequence.update(True) == "PHASE_COMPLETE:move"
    assert sequence.state == "COMPLETE"


def test_arm_does_not_restart_running_phase():
    sequence = ExperimentSequence([ExperimentPhase("one", "one", 1, 1)])
    assert not sequence.phase_is_armed
    sequence.arm_current()
    assert sequence.phase_is_armed
    assert sequence.arm_current() == "ALREADY_RUNNING"


def test_real_source_single_key_can_arm_each_phase_before_enable():
    sequence = ExperimentSequence([
        ExperimentPhase("one", "one", 0, 1),
        ExperimentPhase("two", "two", 0, 1),
    ])
    assert not sequence.phase_is_armed
    sequence.arm_current()
    assert sequence.phase_is_armed
    assert sequence.update(True) == "PHASE_COMPLETE:one"
    assert sequence.state == "WAITING_NEXT"
    assert not sequence.phase_is_armed
    sequence.arm_current()
    assert sequence.phase_is_armed


def test_marker_before_update_labels_exact_recording_window():
    sequence = ExperimentSequence([ExperimentPhase("motion", "move", 2, 3)])
    sequence.arm_current()
    markers = []
    for _ in range(5):
        markers.append(sequence.marker())
        sequence.update(True)
    assert [marker.split(":")[1] for marker in markers] == [
        "WARMUP", "WARMUP", "RECORDING", "RECORDING", "RECORDING"
    ]
    assert sequence.state == "COMPLETE"


def test_auto_advance_uses_wall_clock_countdown_without_manual_arm():
    sequence = ExperimentSequence(
        [
            ExperimentPhase("first", "first", 0, 1),
            ExperimentPhase("second", "second", 1, 1),
        ],
        auto_advance_seconds=2.0,
    )
    sequence.arm_current()
    assert sequence.update(True, now_s=10.0) == "PHASE_COMPLETE:first"
    assert sequence.state == "INTERMISSION" and sequence.remaining == 2
    assert sequence.update(False, now_s=10.4) is None and sequence.remaining == 2
    assert sequence.update(False, now_s=11.1) is None and sequence.remaining == 1
    assert sequence.update(False, now_s=12.0) == "NEXT_PHASE_STARTED:second"
    assert sequence.state == "WARMUP"
    assert sequence.update(True) == "RECORDING_STARTED"
    assert sequence.update(True) == "PHASE_COMPLETE:second"
    assert sequence.state == "COMPLETE"
