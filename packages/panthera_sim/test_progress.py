"""Checks that the staged metric names where an episode failed, not just how far.

Success is one bit, and the nine criteria saturate once the object is untouched,
so two very different failures read identically.  These tests pin the cases
actually observed: an episode that never reached the object, one that grasped
and lost it, one that carried it and released too high, and one that inserted it
but never came to rest.
"""

from __future__ import annotations

import numpy as np
import pytest

from executor import _Progress


class _Pose:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)


class _Cylinder:
    def __init__(self, scripted):
        self.scripted = scripted
        self.index = 0

    def get_pose(self):
        return _Pose(self.scripted[min(self.index, len(self.scripted) - 1)])


class _Robot:
    def __init__(self, task):
        self.task = task

    def get_left_ee_pose(self):
        return list(self.task.ee[min(self.task.cylinder.index,
                                     len(self.task.ee) - 1)]) + [1, 0, 0, 0]


class _Task:
    """Scripted object and gripper motion; nothing simulated."""

    def __init__(self, cylinder_path, ee_path, contacts):
        self.cylinder = _Cylinder(cylinder_path)
        self.ee = ee_path
        self.contacts = contacts
        self.robot = _Robot(self)
        self.realized_geometry = {"socket_target_xy_m": [1.0, 0.0]}

    def get_gripper_actor_contact_position(self, _name):
        return [0] * self.contacts[min(self.cylinder.index, len(self.contacts) - 1)]


def _run(task, steps):
    progress = _Progress(task)
    for step in range(steps):
        task.cylinder.index = step
        progress.observe(step)
    return progress


def test_never_approached():
    task = _Task([[0, 0, 0.8]] * 5, [[0.5, 0.5, 1.0]] * 5, [0] * 5)
    report = _run(task, 5).failure()
    assert report["reached_stage"] == 0
    assert report["failed_at"] == "接近"
    assert not report["regressed"]


def test_approached_but_never_grasped():
    task = _Task([[0, 0, 0.8]] * 5, [[0.02, 0, 0.8]] * 5, [1] * 5)
    report = _run(task, 5).failure()
    assert report["reached_stage"] == 1
    assert report["failed_at"] == "接触夹持"


def test_grasped_then_lost_is_reported_as_a_regression():
    """Grasped and lifted, then the object slips and falls back."""
    path = [[0, 0, 0.8], [0, 0, 0.85], [0, 0, 0.9], [0, 0, 0.8], [0, 0, 0.8]]
    task = _Task(path, [[0, 0, 0.85]] * 5, [2, 2, 2, 0, 0])
    report = _run(task, 5).failure()
    assert report["reached_stage"] == 3
    assert report["final_stage"] < 3
    assert report["regressed"] is True


def test_carried_but_never_inserted_names_the_missing_stage():
    """The ep896 shape: carried to the socket, released too high, toppled beside it.

    The object stays near the socket, so nothing regresses -- the defect is that
    insertion never happened, and that is what has to be named.
    """
    path = [[0, 0, 0.8], [0.5, 0, 0.9], [0.94, 0, 0.9], [0.9, 0, 0.8], [0.88, 0, 0.8]]
    task = _Task(path, [[0.5, 0, 0.9]] * 5, [2, 2, 2, 0, 0])
    report = _run(task, 5).failure()
    assert report["reached_stage"] == 4
    assert report["failed_at"] == "插入"
    assert report["regressed"] is False
    assert report["first_reached_step"][4] == 1


def test_object_knocked_back_past_its_start_is_a_regression():
    """Carried partway, then dropped and rolled back beyond where it began."""
    path = [[0, 0, 0.8], [0.5, 0, 0.9], [0.94, 0, 0.9], [-0.3, 0, 0.8], [-0.4, 0, 0.8]]
    task = _Task(path, [[0.5, 0, 0.9]] * 5, [2, 2, 2, 0, 0])
    report = _run(task, 5).failure()
    assert report["reached_stage"] == 4
    assert report["final_stage"] < 4
    assert report["regressed"] is True


def test_inserted_and_still_there():
    path = [[0, 0, 0.8], [0.5, 0, 0.9], [1.0, 0, 0.86], [1.0, 0, 0.86]]
    task = _Task(path, [[1.0, 0, 0.9]] * 4, [2, 2, 2, 0])
    report = _run(task, 4).failure()
    assert report["reached_stage"] == 5
    assert report["final_stage"] == 5
    assert not report["regressed"]
    assert report["failed_at"] is None


def test_continuous_quantities_move_while_the_object_is_untouched():
    """The nine success criteria saturate here; these must not."""
    far = _Task([[0, 0, 0.8]] * 4, [[0.9, 0, 0.8]] * 4, [0] * 4)
    near = _Task([[0, 0, 0.8]] * 4, [[0.12, 0, 0.8]] * 4, [0] * 4)
    a = _run(far, 4).report()
    b = _run(near, 4).report()
    assert a["stage"] == b["stage"] == 0
    assert b["nearest_gripper_to_object_m"] < a["nearest_gripper_to_object_m"]


def test_expert_deviation_separates_two_zero_scores():
    reference = np.zeros((6, 7))
    close = _Task([[0, 0, 0.8]] * 6, [[0.9, 0, 0.8]] * 6, [0] * 6)
    close._actual_robot_state = lambda: {"arm_qpos": np.full(6, 0.01)}
    drift = _Task([[0, 0, 0.8]] * 6, [[0.9, 0, 0.8]] * 6, [0] * 6)
    drift._actual_robot_state = lambda: {"arm_qpos": np.full(6, 0.9)}

    a = _Progress(close, reference)
    b = _Progress(drift, reference)
    for step in range(6):
        a.observe(step)
        b.observe(step)
    assert a.deviation_report()["median_mrad"] < b.deviation_report()["median_mrad"]
    assert a.deviation_report()["first_step_over_100mrad"] is None
    assert b.deviation_report()["first_step_over_100mrad"] == 0
