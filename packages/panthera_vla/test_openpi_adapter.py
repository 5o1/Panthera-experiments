import numpy as np
import pytest

from openpi_adapter import ACTION_DIM, IMAGE_KEYS, PantheraInputs, PantheraOutputs, inference_observation


def test_single_camera_uses_one_real_slot_and_masks_padding():
    image = np.full((12, 16, 3), 127, dtype=np.uint8)
    transformed = PantheraInputs()(
        inference_observation(image, np.arange(ACTION_DIM), "insert the cylinder")
    )
    assert tuple(transformed["image"]) == IMAGE_KEYS
    assert np.array_equal(transformed["image"]["base_0_rgb"], image)
    assert not transformed["image"]["left_wrist_0_rgb"].any()
    assert transformed["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.False_,
    }


def test_lerobot_float_chw_image_is_converted_to_uint8_hwc():
    transformed = PantheraInputs()(
        {
            "observation/image": np.ones((3, 4, 5), dtype=np.float32) * 0.5,
            "observation/state": np.zeros(7),
        }
    )
    image = transformed["image"]["base_0_rgb"]
    assert image.shape == (4, 5, 3)
    assert image.dtype == np.uint8
    assert np.all(image == 127)


def test_training_actions_remain_physical_7d_before_openpi_padding():
    actions = np.arange(35, dtype=np.float32).reshape(5, 7)
    transformed = PantheraInputs()(
        {
            "observation/image": np.zeros((4, 5, 3), dtype=np.uint8),
            "observation/state": np.zeros(7),
            "actions": actions,
        }
    )
    assert np.array_equal(transformed["actions"], actions)


def test_outputs_drop_only_openpi_padding_dimensions():
    chunk = np.arange(3 * 32, dtype=np.float32).reshape(3, 32)
    output = PantheraOutputs()({"actions": chunk})["actions"]
    assert output.shape == (3, 7)
    assert np.array_equal(output, chunk[:, :7])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("state", np.zeros(6), "shape"),
        ("state", np.array([0, 0, 0, 0, 0, 0, np.nan]), "NaN"),
        ("image", np.zeros((3, 3)), "shape"),
    ],
)
def test_invalid_observation_is_rejected(field, value, match):
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    state = np.zeros(7)
    if field == "state":
        state = value
    else:
        image = value
    with pytest.raises(ValueError, match=match):
        inference_observation(image, state, "task")
