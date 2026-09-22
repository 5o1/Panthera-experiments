"""Averaging overlapping chunks is what lets the policy hold still.

Measured on the single-trajectory overfit model, episode 2: the expert holds
still for 26.9% of its trajectory -- four settling blocks, two of them around
the gripper closing at step 231 and opening at 801 -- and moves 5.23 mrad per
step in between.  The policy held still for 0.0% of its rollout at 13.71 mrad
per step, so it never reached the state the release is predicted from.
"""

import numpy as np
import pytest

from rollout import TemporalEnsemble


def test_a_single_chunk_is_returned_unchanged():
    e = TemporalEnsemble()
    chunk = np.arange(21, dtype=float).reshape(3, 7)
    e.add(0, chunk)
    assert np.allclose(e.action(1), chunk[1])


def test_a_step_no_chunk_covers_is_an_error():
    e = TemporalEnsemble()
    e.add(0, np.zeros((3, 7)))
    with pytest.raises(ValueError, match="no chunk covers"):
        e.action(3)


def test_noise_around_a_held_pose_averages_out():
    # The failure this exists for: the underlying prediction is "stay here" and
    # the per-step output noise turns it into dithering.
    rng = np.random.default_rng(0)
    held = np.full(7, 0.5)
    e = TemporalEnsemble(coefficient=0.0)
    for start in range(16):
        steps = np.arange(start, start + 25)
        e.add(start, held + rng.normal(0, 0.01, (25, 7)))
    single = held + rng.normal(0, 0.01, 7)
    assert np.abs(e.action(20) - held).max() < np.abs(single - held).max()


def test_depth_counts_the_chunks_that_cover_a_step():
    e = TemporalEnsemble()
    for start in (0, 5, 10):
        e.add(start, np.zeros((25, 7)))
    assert e.depth(12) == 3
    assert e.depth(2) == 1


def test_a_fresher_chunk_outweighs_an_older_one():
    e = TemporalEnsemble(coefficient=1.0)
    e.add(0, np.zeros((25, 7)))
    e.add(0, np.ones((25, 7)))
    # Newest is age 0 with weight 1, oldest age 1 with weight e**-1.
    assert e.action(0)[0] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))


def test_zero_coefficient_weights_every_chunk_equally():
    e = TemporalEnsemble(coefficient=0.0)
    e.add(0, np.zeros((25, 7)))
    e.add(0, np.full((25, 7), 4.0))
    assert e.action(0)[0] == pytest.approx(2.0)


def test_old_chunks_are_dropped_beyond_the_horizon():
    e = TemporalEnsemble(horizon=4)
    for start in range(10):
        e.add(start, np.zeros((25, 7)))
    assert len(e._chunks) == 4


def test_a_negative_coefficient_is_refused():
    with pytest.raises(ValueError, match="nonnegative"):
        TemporalEnsemble(coefficient=-0.1)
