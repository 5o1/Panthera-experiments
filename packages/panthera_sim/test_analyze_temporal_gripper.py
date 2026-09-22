import numpy as np
import pytest

from analyze_temporal_gripper import reconstruct


def _chunk(gripper):
    chunk = np.zeros((3, 7), dtype=float)
    chunk[:, 6] = gripper
    return chunk.tolist()


def test_reconstruct_detects_open_suppressed_by_equal_average():
    report = reconstruct(
        [
            {"row": 0, "actions": _chunk([0.1, 0.1, 0.1])},
            {"row": 1, "actions": _chunk([0.8, 0.8, 0.8])},
        ],
        4,
        coefficient=0.0,
    )
    assert report["first_newest_open_row"] == 1
    assert report["first_ensemble_open_row"] == 3
    assert report["first_suppressed_open_row"] == 1
    assert report["raw_open_was_suppressed"] is True


def test_reconstruct_reports_model_never_opened():
    report = reconstruct(
        [{"row": 0, "actions": _chunk([0.2, 0.3, 0.4])}],
        3,
    )
    assert report["first_newest_open_row"] is None
    assert report["first_ensemble_open_row"] is None
    assert report["raw_open_was_suppressed"] is False


def test_bad_trace_shape_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        reconstruct([{"row": 0, "actions": [[0.0] * 6]}], 1)
