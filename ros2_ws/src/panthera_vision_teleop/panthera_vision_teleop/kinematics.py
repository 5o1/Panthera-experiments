"""Small URDF serial-chain FK/Jacobian/IK implementation for backend comparison."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

import numpy as np

from .rotation_utils import rotation_from_vector, rotation_vector, rpy_to_matrix


@dataclass(frozen=True)
class ChainJoint:
    name: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class SerialChain:
    """A single URDF path with numerical damped-least-squares IK."""

    def __init__(self, joints: Sequence[ChainJoint]) -> None:
        self.segments = tuple(joints)
        self.moving = tuple(joint for joint in joints if joint.joint_type in ("revolute", "continuous"))
        if not self.moving:
            raise ValueError("chain has no movable joints")
        self.joint_names = tuple(joint.name for joint in self.moving)
        self.lower = np.array([joint.lower for joint in self.moving])
        self.upper = np.array([joint.upper for joint in self.moving])

    @classmethod
    def from_urdf(cls, path: Path | str, base_link: str, tip_link: str) -> "SerialChain":
        root = ET.parse(path).getroot()
        by_child = {}
        for element in root.findall("joint"):
            child = element.find("child")
            if child is not None:
                by_child[child.attrib["link"]] = element
        elements = []
        child_link = tip_link
        while child_link != base_link:
            if child_link not in by_child:
                raise ValueError(f"no URDF path from {base_link} to {tip_link}")
            joint = by_child[child_link]
            elements.append(joint)
            parent = joint.find("parent")
            if parent is None:
                raise ValueError("joint has no parent")
            child_link = parent.attrib["link"]
        elements.reverse()
        return cls([cls._parse_joint(element) for element in elements])

    @staticmethod
    def _parse_joint(element) -> ChainJoint:
        origin_element = element.find("origin")
        xyz = [0.0, 0.0, 0.0] if origin_element is None else [float(v) for v in origin_element.attrib.get("xyz", "0 0 0").split()]
        rpy = [0.0, 0.0, 0.0] if origin_element is None else [float(v) for v in origin_element.attrib.get("rpy", "0 0 0").split()]
        origin = np.eye(4)
        origin[:3, :3] = rpy_to_matrix(*rpy)
        origin[:3, 3] = xyz
        axis_element = element.find("axis")
        axis = np.array([1.0, 0.0, 0.0] if axis_element is None else [float(v) for v in axis_element.attrib.get("xyz", "1 0 0").split()])
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-10:
            raise ValueError("joint axis is zero")
        axis /= norm
        joint_type = element.attrib["type"]
        limit = element.find("limit")
        if joint_type == "continuous":
            lower, upper = -math.pi, math.pi
        elif joint_type == "revolute" and limit is not None:
            lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
        else:
            lower, upper = 0.0, 0.0
        return ChainJoint(element.attrib["name"], joint_type, origin, axis, lower, upper)

    def forward(self, positions: Sequence[float]) -> np.ndarray:
        transform, _, _ = self.forward_with_jacobian(positions)
        return transform

    def forward_with_jacobian(self, positions: Sequence[float]):
        q = np.asarray(positions, dtype=float)
        if q.shape != (len(self.moving),) or not np.all(np.isfinite(q)):
            raise ValueError("positions have wrong shape or non-finite values")
        transform = np.eye(4)
        axes_world = []
        origins_world = []
        moving_index = 0
        for joint in self.segments:
            transform = transform @ joint.origin
            if joint.joint_type in ("revolute", "continuous"):
                origins_world.append(transform[:3, 3].copy())
                axes_world.append(transform[:3, :3] @ joint.axis)
                rotation = np.eye(4)
                rotation[:3, :3] = rotation_from_vector(joint.axis * q[moving_index])
                transform = transform @ rotation
                moving_index += 1
        endpoint = transform[:3, 3]
        jacobian = np.zeros((6, len(self.moving)))
        for index, (axis, origin) in enumerate(zip(axes_world, origins_world)):
            jacobian[:3, index] = np.cross(axis, endpoint - origin)
            jacobian[3:, index] = axis
        return transform, jacobian, q

    def inverse(
        self,
        target: Sequence[Sequence[float]],
        seed: Sequence[float],
        max_iterations: int = 120,
        damping: float = 0.03,
    ) -> np.ndarray:
        target_v = np.asarray(target, dtype=float)
        q = np.asarray(seed, dtype=float).copy()
        if target_v.shape != (4, 4) or not np.all(np.isfinite(target_v)):
            raise ValueError("target must be a finite 4x4 transform")
        if q.shape != (len(self.moving),) or not np.all(np.isfinite(q)):
            raise ValueError("seed has wrong shape or non-finite values")
        q = np.clip(q, self.lower, self.upper)
        for _ in range(max_iterations):
            current, jacobian, _ = self.forward_with_jacobian(q)
            error = np.concatenate((
                target_v[:3, 3] - current[:3, 3],
                rotation_vector(target_v[:3, :3] @ current[:3, :3].T),
            ))
            if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 1e-3:
                return q
            system = jacobian @ jacobian.T + (damping * damping) * np.eye(6)
            step = jacobian.T @ np.linalg.solve(system, error)
            step_norm = float(np.linalg.norm(step))
            if step_norm > 0.1:
                step *= 0.1 / step_norm
            q = np.clip(q + step, self.lower, self.upper)
        raise ValueError("IK did not converge")
