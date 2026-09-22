#!/usr/bin/env python3
"""Pure Panthera observation/action transforms for OpenPI.

The simulator and the real robot expose the same public contract: one RGB
image, six absolute joint positions and one absolute gripper command.  OpenPI
has three fixed image slots and pads state/actions to its model dimension later
in the transform pipeline.  Keeping this adapter independent from JAX/OpenPI
makes the contract testable on the laptop and prevents an OpenPI install from
becoming a dependency of the rest of Panthera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import numpy as np


ACTION_DIM = 7
MODEL_ACTION_DIM = 32
DEFAULT_ACTION_HORIZON = 25
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _vector(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (ACTION_DIM,):
        raise ValueError(f"{name} must have shape ({ACTION_DIM},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _actions(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape [N, {ACTION_DIM}], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("actions contain NaN or infinity")
    return array


def _image(value) -> np.ndarray:
    array = np.asarray(value)
    # LeRobot returns float CHW, while live inference supplies uint8 HWC.
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"image must have shape [H, W, 3], got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError("image contains NaN or infinity")
        # LeRobot image tensors are normalized to [0, 1].  Reject values that
        # would silently wrap when converted to uint8.
        if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
            raise ValueError("floating image must be normalized to [0, 1]")
        # Match OpenPI's own Libero/UR5 policy adapters exactly: multiplication
        # followed by uint8 conversion truncates instead of rounding.
        array = array * 255.0
    elif not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"unsupported image dtype: {array.dtype}")
    if array.size and (int(array.min()) < 0 or int(array.max()) > 255):
        raise ValueError("integer image values must be in [0, 255]")
    return np.ascontiguousarray(array, dtype=np.uint8)


@dataclass(frozen=True)
class PantheraInputs:
    """Map the single-camera 7-D Panthera contract to OpenPI model keys.

    This transform deliberately does not pad state/actions to 32 dimensions.
    OpenPI normalizes the physical seven dimensions first and then its
    ``PadStatesAndActions`` model transform pads them.  Padding here would mix
    synthetic zeros into the normalization statistics.
    """

    def __call__(self, data: Mapping[str, object]) -> dict:
        base = _image(data["observation/image"])
        result: MutableMapping[str, object] = {
            "state": _vector(data["observation/state"], name="state"),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": np.zeros_like(base),
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            result["actions"] = _actions(data["actions"])
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return dict(result)


@dataclass(frozen=True)
class PantheraOutputs:
    """Return only the seven physical dimensions from an OpenPI action chunk."""

    def __call__(self, data: Mapping[str, object]) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
            raise ValueError(
                f"OpenPI actions must have shape [N, >= {ACTION_DIM}], got {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("OpenPI actions contain NaN or infinity")
        return {"actions": np.ascontiguousarray(actions[:, :ACTION_DIM])}


def inference_observation(image, state, prompt: str) -> dict:
    """Build the raw dictionary accepted by a Panthera-configured OpenPI policy."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    return {
        "observation/image": _image(image),
        "observation/state": _vector(state, name="state"),
        "prompt": prompt,
    }
