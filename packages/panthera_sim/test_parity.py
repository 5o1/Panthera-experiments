"""Unit tests for checkpoint-to-recording validation helpers."""

import json

import numpy as np

from mosaic_videos import _checkpoint_label
from parity import _action_chunks, _gripper_transitions, _normalize_actions


def test_bounds_q99_normalization_matches_training_contract():
    statistics = {
        "q01": [0.0, -2.0],
        "q99": [10.0, 2.0],
        "min": [0.0, -2.0],
        "max": [10.0, 2.0],
        "mask": [True, True],
    }
    value = np.asarray([[0.0, 0.0], [10.0, 4.0]])
    assert np.allclose(
        _normalize_actions(value, statistics, clip=False),
        [[-1.0, 0.0], [1.0, 2.0]],
    )
    assert np.allclose(
        _normalize_actions(value, statistics, clip=True),
        [[-1.0, 0.0], [1.0, 1.0]],
    )


def test_action_chunks_pad_tail_like_rlds():
    actions = np.arange(21, dtype=np.float64).reshape(3, 7)
    chunks = _action_chunks(actions, [0, 2], action_chunk=3)
    assert chunks.shape == (2, 3, 7)
    assert np.array_equal(chunks[0], actions)
    assert np.array_equal(chunks[1], np.repeat(actions[2][None, :], 3, axis=0))


def test_gripper_transitions_are_reported_on_the_new_state_row():
    actions = np.zeros((6, 7), dtype=np.float64)
    actions[:, 6] = [0.9, 0.8, 0.4, 0.3, 0.6, 0.7]
    assert _gripper_transitions(actions) == [
        {"row": 2, "kind": "close", "before": 0.8, "after": 0.4},
        {"row": 4, "kind": "open", "before": 0.3, "after": 0.6},
    ]


def test_mosaic_label_reads_offline_validation_loss(tmp_path):
    video = tmp_path / "step-5000.mp4"
    video.touch()
    video.with_suffix(".json").write_text(
        json.dumps(
            {"cases": [{"offline_validation": {"normalized_l1": 0.01234567}}]}
        ),
        encoding="utf-8",
    )
    assert _checkpoint_label(video) == "checkpoint 5000 | offline val L1 0.012346"


def test_mosaic_label_marks_historical_loss_as_unavailable(tmp_path):
    video = tmp_path / "step-10000.mp4"
    video.touch()
    assert _checkpoint_label(video).endswith("offline val L1 unavailable")
