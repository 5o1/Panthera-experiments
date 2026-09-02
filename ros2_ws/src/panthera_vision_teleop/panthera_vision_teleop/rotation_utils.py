"""Small, testable rotation helpers used by the teleoperation nodes.

Vectors are column vectors. A 3x3 matrix is an active rotation whose columns
are the local x/y/z axes in the parent frame. Quaternions use ROS ``x,y,z,w``
order. RPY means ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _finite_array(values: Sequence[float], shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    return array


def normalize(vector: Sequence[float], epsilon: float = 1.0e-8) -> np.ndarray:
    """Return a unit 3-vector; reject zero, NaN and infinite inputs."""
    value = _finite_array(vector, (3,))
    norm = float(np.linalg.norm(value))
    if norm <= epsilon:
        raise ValueError("cannot normalize a zero or near-zero vector")
    return value / norm


def arm_frame(shoulder: Sequence[float], elbow: Sequence[float], wrist: Sequence[float], epsilon: float = 1.0e-5) -> np.ndarray:
    """Build the right-handed shoulder/elbow/wrist frame.

    A nearly straight arm is rejected because its plane normal is undefined;
    callers should retain the previous valid orientation.
    """
    shoulder_v = _finite_array(shoulder, (3,))
    elbow_v = _finite_array(elbow, (3,))
    wrist_v = _finite_array(wrist, (3,))
    z_axis = normalize(wrist_v - elbow_v, epsilon)
    upper_arm = normalize(elbow_v - shoulder_v, epsilon)
    x_axis = normalize(np.cross(upper_arm, z_axis), epsilon)
    y_axis = normalize(np.cross(z_axis, x_axis), epsilon)
    result = np.column_stack((x_axis, y_axis, z_axis))
    if float(np.linalg.det(result)) < 0.999:
        raise ValueError("arm frame is not a proper right-handed rotation")
    return result


def quaternion_to_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Convert a ROS-order quaternion to a 3x3 active rotation matrix."""
    q = _finite_array(quaternion_xyzw, (4,))
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-10:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=float)


def matrix_to_quaternion(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a proper rotation matrix to normalized ROS-order quaternion."""
    rotation = _finite_array(matrix, (3, 3))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError("matrix is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-5):
        raise ValueError("matrix determinant must be +1")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25*s, (rotation[2,1]-rotation[1,2])/s, (rotation[0,2]-rotation[2,0])/s, (rotation[1,0]-rotation[0,1])/s
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0,0] - rotation[1,1] - rotation[2,2]) * 2.0
            x, y, z, w = 0.25*s, (rotation[0,1]+rotation[1,0])/s, (rotation[0,2]+rotation[2,0])/s, (rotation[2,1]-rotation[1,2])/s
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1,1] - rotation[0,0] - rotation[2,2]) * 2.0
            x, y, z, w = (rotation[0,1]+rotation[1,0])/s, 0.25*s, (rotation[1,2]+rotation[2,1])/s, (rotation[0,2]-rotation[2,0])/s
        else:
            s = math.sqrt(1.0 + rotation[2,2] - rotation[0,0] - rotation[1,1]) * 2.0
            x, y, z, w = (rotation[0,2]+rotation[2,0])/s, (rotation[1,2]+rotation[2,1])/s, 0.25*s, (rotation[1,0]-rotation[0,1])/s
    quaternion = np.array([x, y, z, w], dtype=float)
    return quaternion / np.linalg.norm(quaternion)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ROS roll/pitch/yaw radians to an active rotation matrix."""
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        raise ValueError("RPY values must be finite")
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr],
    ])


def matrix_to_rpy(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a rotation matrix to ROS roll/pitch/yaw radians."""
    rotation = _finite_array(matrix, (3, 3))
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-7:
        roll, yaw = math.atan2(rotation[2,1], rotation[2,2]), math.atan2(rotation[1,0], rotation[0,0])
    else:
        roll, yaw = math.atan2(-rotation[1,2], rotation[1,1]), 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def rotation_angle(matrix: Sequence[Sequence[float]]) -> float:
    """Return the unsigned principal rotation angle in radians."""
    rotation = _finite_array(matrix, (3, 3))
    return math.acos(float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))


def rotation_vector(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the principal axis-angle vector whose norm is the angle."""
    rotation = _finite_array(matrix, (3, 3))
    angle = rotation_angle(rotation)
    if angle <= 1.0e-10:
        return np.zeros(3)
    if abs(math.pi - angle) < 1.0e-5:
        # The usual skew formula becomes singular at pi. Recover an axis from
        # the diagonal, then choose signs from the symmetric off-diagonals.
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        largest = int(np.argmax(axis))
        if axis[largest] <= 1.0e-8:
            raise ValueError("cannot recover pi-rotation axis")
        if largest == 0:
            axis[1] = math.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        elif largest == 1:
            axis[0] = math.copysign(axis[0], rotation[0, 1] + rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
        else:
            axis[0] = math.copysign(axis[0], rotation[0, 2] + rotation[2, 0])
            axis[1] = math.copysign(axis[1], rotation[1, 2] + rotation[2, 1])
        return normalize(axis) * angle
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / (2.0 * math.sin(angle))
    return axis * angle


def rotation_from_vector(vector: Sequence[float]) -> np.ndarray:
    """Convert an axis-angle vector into an active rotation matrix."""
    value = _finite_array(vector, (3,))
    angle = float(np.linalg.norm(value))
    if angle <= 1.0e-10:
        return np.eye(3)
    x, y, z = value / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def interpolate_rotation(start: Sequence[Sequence[float]], end: Sequence[Sequence[float]], fraction: float) -> np.ndarray:
    """Spherical interpolation without averaging Euler angles."""
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    q0, q1 = matrix_to_quaternion(start), matrix_to_quaternion(end)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + fraction * (q1 - q0)
        return quaternion_to_matrix(q / np.linalg.norm(q))
    theta = math.acos(dot)
    q = (math.sin((1.0-fraction)*theta)/math.sin(theta)*q0 + math.sin(fraction*theta)/math.sin(theta)*q1)
    return quaternion_to_matrix(q)


def clamp_rotation(matrix: Sequence[Sequence[float]], max_angle: float) -> np.ndarray:
    """Keep the rotation axis but clamp its principal angle."""
    if not math.isfinite(max_angle) or max_angle < 0.0:
        raise ValueError("max_angle must be finite and non-negative")
    rotation = _finite_array(matrix, (3, 3))
    angle = rotation_angle(rotation)
    if angle <= max_angle or angle <= 1.0e-10:
        return rotation.copy()
    return interpolate_rotation(np.eye(3), rotation, max_angle / angle)
