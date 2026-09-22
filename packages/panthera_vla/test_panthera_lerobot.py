import io

import numpy as np
from PIL import Image
import pytest

from panthera_lerobot import FEATURES, lerobot_frame


def _jpeg() -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.full((240, 320, 3), 80, dtype=np.uint8)).save(stream, format="JPEG")
    return stream.getvalue()


def test_lerobot_features_keep_the_public_7d_contract():
    assert FEATURES["state"]["shape"] == (7,)
    assert FEATURES["actions"]["shape"] == (7,)
    assert FEATURES["image"]["shape"] == (240, 320, 3)


def test_step_mapping_keeps_aligned_action_and_instruction():
    state = np.arange(7, dtype=np.float32)
    action = state + 1
    frame = lerobot_frame(
        {
            "observation": {"image": _jpeg(), "state": state},
            "action": action,
            "language_instruction": "insert the yellow cylinder",
        }
    )
    assert frame["image"].shape == (240, 320, 3)
    assert np.array_equal(frame["state"], state)
    assert np.array_equal(frame["actions"], action)
    assert frame["task"] == "insert the yellow cylinder"


def test_nonfinite_action_is_rejected():
    with pytest.raises(ValueError, match="NaN"):
        lerobot_frame(
            {
                "observation": {"image": _jpeg(), "state": np.zeros(7)},
                "action": np.array([0, 0, 0, 0, 0, 0, np.nan]),
                "language_instruction": "task",
            }
        )
