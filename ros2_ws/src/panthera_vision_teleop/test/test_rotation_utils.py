import math

import numpy as np
import pytest

from panthera_vision_teleop.rotation_utils import (
    arm_frame, clamp_rotation, interpolate_rotation, matrix_to_quaternion,
    matrix_to_rpy, normalize, quaternion_to_matrix, rotation_angle,
    rotation_from_vector, rotation_vector, rpy_to_matrix,
)


def test_normalize_and_reject_bad_vectors():
    assert np.allclose(normalize([3, 0, 4]), [0.6, 0.0, 0.8])
    for vector in ([0, 0, 0], [math.nan, 0, 0], [math.inf, 0, 0]):
        with pytest.raises(ValueError):
            normalize(vector)


def test_arm_frame_is_proper_rotation():
    rotation = arm_frame([0, 0, 0], [0, 1, 0], [1, 1, 0])
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_arm_frame_rejects_straight_arm():
    with pytest.raises(ValueError):
        arm_frame([0, 0, 0], [1, 0, 0], [2, 0, 0])


@pytest.mark.parametrize("rpy", [[0, 0, 0], [0.2, -0.3, 0.4], [-1.0, 0.5, 2.0]])
def test_rpy_and_quaternion_roundtrips(rpy):
    rotation = rpy_to_matrix(*rpy)
    assert np.allclose(rpy_to_matrix(*matrix_to_rpy(rotation)), rotation, atol=1e-8)
    assert np.allclose(quaternion_to_matrix(matrix_to_quaternion(rotation)), rotation, atol=1e-8)


def test_slerp_endpoints_and_half_angle():
    start, end = np.eye(3), rpy_to_matrix(0, 0, math.radians(20))
    assert np.allclose(interpolate_rotation(start, end, 0), start)
    assert np.allclose(interpolate_rotation(start, end, 1), end)
    assert rotation_angle(interpolate_rotation(start, end, 0.5)) == pytest.approx(math.radians(10))


def test_rotation_clamp_preserves_axis_and_limits_angle():
    original = rpy_to_matrix(math.radians(40), 0, 0)
    limited = clamp_rotation(original, math.radians(12))
    assert rotation_angle(limited) == pytest.approx(math.radians(12))
    assert matrix_to_rpy(limited)[0] == pytest.approx(math.radians(12))


@pytest.mark.parametrize("vector", [[0, 0, 0], [0.2, -0.1, 0.3], [math.pi, 0, 0]])
def test_rotation_vector_roundtrip(vector):
    matrix = rotation_from_vector(vector)
    assert np.allclose(rotation_from_vector(rotation_vector(matrix)), matrix, atol=1e-7)
