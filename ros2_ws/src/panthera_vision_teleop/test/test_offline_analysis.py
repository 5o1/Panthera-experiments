import numpy as np

from panthera_vision_teleop.offline_analysis import recording_phase_intervals, scalar_summary, summarize_recording, timing_summary, vector_summary


def test_vector_summary_reports_range_noise_and_path():
    result = vector_summary([[0, 0, 0], [1, -1, 0], [2, 1, 0]], ["x", "y", "z"])
    assert result["count"] == 3
    assert result["axes"]["x"]["peak_to_peak"] == 2.0
    assert result["axes"]["y"]["min"] == -1.0
    assert np.isclose(result["path_length"], np.sqrt(2) + np.sqrt(5))


def test_recording_summary_detects_workspace_and_motion_limiting():
    summary = summarize_recording({
        "/teleop/diagnostics/deadbanded_robot_offset": [[0, 0, 0], [0, 0.2, 0]],
        "/teleop/diagnostics/limited_robot_offset": [[0, 0, 0], [0, 0.05, 0]],
        "/teleop/diagnostics/filtered_robot_offset": [[0, 0, 0], [0, 0.04, 0]],
        "/teleop/diagnostics/motion_limited_robot_offset": [[0, 0, 0], [0, 0.01, 0]],
    })
    assert summary["clipped_fraction"] == 0.5
    assert summary["motion_limited_fraction"] == 0.5


def test_empty_vector_series_is_supported():
    assert vector_summary([], ["x", "y", "z"]) == {"count": 0, "axes": {}}


def test_scalar_summary_includes_latency_percentile():
    result = scalar_summary([1.0, 2.0, 3.0, 100.0])
    assert result["count"] == 4
    assert result["max"] == 100.0
    assert 3.0 < result["p95"] < 100.0


def test_recording_phase_intervals_ignore_warmup_and_waiting():
    result = recording_phase_intervals([
        (1.0, "static:WARMUP:2"),
        (2.0, "static:RECORDING:3"),
        (3.0, "static:RECORDING:2"),
        (4.0, "move:WAITING_NEXT:0"),
        (5.0, "move:RECORDING:1"),
    ])
    assert result == {"static": (2.0, 3.0), "move": (5.0, 5.0)}


def test_timing_summary_reports_actual_rate_and_gaps():
    result = timing_summary([1.0, 1.1, 1.2, 1.5])
    assert result["count"] == 4
    assert np.isclose(result["mean_rate_hz"], 6.0)
    assert np.isclose(result["max_interval_ms"], 300.0)
