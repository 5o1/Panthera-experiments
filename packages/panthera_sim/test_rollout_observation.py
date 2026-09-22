"""Evaluation must prepare images the way training did.

Measured on the single-trajectory overfit run: fed its own recorded training
frame the model reproduced the expert chunk to a median 2.6 mrad, and fed the
frame this harness built it was off by 183 mrad.  The crop was only part of
that, but it was the part this harness controls, and it was applied
unconditionally to a model trained with ``--image_aug false``.
"""

import sys
import types

import numpy as np
import pytest

import rollout


class _Task:
    """The slice of the environment ``observe_task`` reads."""

    def __init__(self, frame):
        self._frame = frame

    def get_obs(self):
        return {
            "observation": {
                "head_camera": {"rgb": self._frame},
                "robot_state": {
                    "vector": np.zeros(7, dtype=np.float32),
                    "dynamics_vector": np.arange(28, dtype=np.float32),
                },
            }
        }


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)


@pytest.fixture
def cropper(monkeypatch):
    """Stand in for RLinf's cropper, which is not importable off the Lab."""
    calls = []

    def center_crop_image(picture):
        calls.append(picture)
        return picture.crop((0, 0, 224, 224))

    module = types.ModuleType("rlinf.envs.utils")
    module.center_crop_image = center_crop_image
    package = types.ModuleType("rlinf.envs")
    package.utils = module
    root = types.ModuleType("rlinf")
    root.envs = package
    monkeypatch.setitem(sys.modules, "rlinf", root)
    monkeypatch.setitem(sys.modules, "rlinf.envs", package)
    monkeypatch.setitem(sys.modules, "rlinf.envs.utils", module)
    return calls


def test_default_does_not_crop(frame, cropper):
    image, _ = rollout.observe_task(_Task(frame))
    assert image.shape == frame.shape
    assert cropper == [], "a model with no recorded crop must not be cropped for"


def test_the_renderer_is_handed_over_untouched(frame, cropper):
    # The channel order the collector leaves behind is repaired where it is
    # introduced, in the RLDS conversion, so nothing is compensated for here.
    # Evaluation showing the simulator's own output is what makes a dataset
    # built by any other route still line up.
    image, _ = rollout.observe_task(_Task(frame))
    assert np.array_equal(image, frame)


def test_crop_is_applied_when_the_contract_asks_for_it(frame, cropper):
    image, _ = rollout.observe_task(_Task(frame), center_crop=True)
    assert image.shape == (224, 224, 3)
    assert len(cropper) == 1


def test_state_is_untouched_by_the_crop(frame, cropper):
    _, plain = rollout.observe_task(_Task(frame))
    _, cropped = rollout.observe_task(_Task(frame), center_crop=True)
    assert np.array_equal(plain, cropped)
    assert plain.dtype == np.float32


def test_dynamics_checkpoint_selects_28d_state(frame, cropper):
    _, state = rollout.observe_task(_Task(frame), proprio_dim=28)
    assert state.shape == (28,)
    assert np.array_equal(state, np.arange(28, dtype=np.float32))


def test_unknown_proprioception_width_is_rejected(frame, cropper):
    with pytest.raises(ValueError, match="unsupported Panthera proprioception"):
        rollout.observe_task(_Task(frame), proprio_dim=14)
