#!/usr/bin/env python3
"""Headless SAPIEN acceptance for a generated Panthera RoboTwin embodiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import sapien.core as sapien
import yaml


ARM_JOINTS = [f"joint{index}" for index in range(1, 7)]
GRIPPER_JOINTS = ["L_finger_joint", "R_finger_joint"]
EXPECTED_LIMITS = {
    "joint1": [-2.4, 2.4],
    "joint2": [0.0, 3.2],
    "joint3": [0.0, 4.0],
    "joint4": [-1.6, 1.6],
    "joint5": [-1.7, 1.7],
    "joint6": [-2.5, 2.5],
    "L_finger_joint": [0.0, 0.04],
    "R_finger_joint": [-0.04, 0.0],
}


def parse_args() -> argparse.Namespace:
    """Parse the generated embodiment directory and report location."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiment-root", required=True, type=Path)
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def finite_pose(link) -> list[float]:
    """Return and validate one link's world pose in SAPIEN wxyz convention."""
    pose = link.get_entity_pose()
    values = np.concatenate((np.asarray(pose.p), np.asarray(pose.q)))
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite pose for {link.get_name()}: {values}")
    return values.tolist()


def plan_summary(result: dict) -> dict:
    """Return a compact serializable summary for one MPLib attempt."""
    status = str(result.get("status", "missing"))
    position = np.asarray(result.get("position", []), dtype=float)
    velocity = np.asarray(result.get("velocity", []), dtype=float)
    if status != "Success":
        return {"status": status, "trajectory_steps": 0, "degrees_of_freedom": 0}
    if position.ndim != 2 or position.shape[1] != 6:
        raise ValueError(f"unexpected planned position shape: {position.shape}")
    if velocity.shape != position.shape:
        raise ValueError(
            f"planned velocity shape {velocity.shape} differs from position {position.shape}"
        )
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("MPLib plan contains NaN or infinity")
    return {
        "status": status,
        "trajectory_steps": int(position.shape[0]),
        "degrees_of_freedom": int(position.shape[1]),
    }


def validate_robotwin_contract(robotwin_root: Path, embodiment_root: Path) -> dict:
    """Load two Pantheras and probe current plus six 1 cm Cartesian targets."""
    sys.path.insert(0, str(robotwin_root))
    from envs.robot.robot import Robot

    config = yaml.safe_load((embodiment_root / "config.yml").read_text(encoding="utf-8"))
    scene = sapien.Scene()
    scene.set_timestep(1.0 / 250.0)
    robot = Robot(
        scene,
        need_topp=False,
        left_embodiment_config=config,
        right_embodiment_config=config,
        left_robot_file=str(embodiment_root),
        right_robot_file=str(embodiment_root),
        dual_arm_embodied=False,
        embodiment_dis=0.6,
        planner_backend="mplib",
    )
    robot.init_joints()
    robot.set_planner(scene=scene)
    robot.move_to_homestate()
    # RoboTwin opens both grippers before any task-space plan.  Mirror that
    # lifecycle here so the two fingertip collision meshes do not start in
    # their physically closed/contacting state during IK validation.
    robot.set_gripper(1.0, "left", gripper_eps=0.0)
    robot.set_gripper(1.0, "right", gripper_eps=0.0)
    for _ in range(250):
        robot.left_entity.set_qf(
            robot.left_entity.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
        )
        robot.right_entity.set_qf(
            robot.right_entity.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
        )
        scene.step()

    contact_pairs: list[dict] = []
    for contact in scene.get_contacts():
        impulse = sum(
            (np.asarray(point.impulse, dtype=float) for point in contact.points),
            start=np.zeros(3),
        )
        magnitude = float(np.linalg.norm(impulse))
        if magnitude > 1.0e-8:
            contact_pairs.append(
                {
                    "body0": contact.bodies[0].get_name(),
                    "body1": contact.bodies[1].get_name(),
                    "impulse_norm": magnitude,
                }
            )

    deltas = {
        "current": [0.0, 0.0, 0.0],
        "x_plus_1cm": [0.01, 0.0, 0.0],
        "x_minus_1cm": [-0.01, 0.0, 0.0],
        "y_plus_1cm": [0.0, 0.01, 0.0],
        "y_minus_1cm": [0.0, -0.01, 0.0],
        "z_plus_1cm": [0.0, 0.0, 0.01],
        "z_minus_1cm": [0.0, 0.0, -0.01],
    }
    plans: dict[str, dict] = {}
    settled_qpos: dict[str, list[float]] = {}
    for side in ("left", "right"):
        entity = getattr(robot, f"{side}_entity")
        planner = getattr(robot, f"{side}_planner")
        link = entity.find_link_by_name("link6")
        current_pose = link.get_entity_pose()
        settled_qpos[side] = np.asarray(entity.get_qpos(), dtype=float).tolist()
        plans[side] = {}
        for name, delta in deltas.items():
            target_pose = sapien.Pose(
                np.asarray(current_pose.p) + np.asarray(delta), current_pose.q
            )
            result = planner.plan_path(
                entity.get_qpos(),
                target_pose,
                constraint_pose=None,
                arms_tag=side,
            )
            plans[side][name] = plan_summary(result)

    for side in ("left", "right"):
        if plans[side]["current"]["status"] != "Success":
            diagnostic = {
                "settled_qpos": settled_qpos,
                "contact_pairs": contact_pairs,
                "plans": plans,
            }
            raise ValueError(
                "MPLib cannot solve the current Panthera pose: "
                + json.dumps(diagnostic, sort_keys=True)
            )
        nonzero_successes = [
            name
            for name, summary in plans[side].items()
            if name != "current"
            and summary["status"] == "Success"
            and summary["trajectory_steps"] > 0
        ]
        if not nonzero_successes:
            raise ValueError(
                f"MPLib found no nonzero 1 cm plan for {side}: "
                + json.dumps(plans[side], sort_keys=True)
            )

    action_dimension = (
        len(config["arm_joints_name"][0])
        + 1
        + len(config["arm_joints_name"][1])
        + 1
    )
    if action_dimension != 14:
        raise ValueError(f"unexpected dual-arm action dimension: {action_dimension}")
    return {
        "dual_arm_action_dimension": action_dimension,
        "action_order": [
            "left_joint1..joint6",
            "left_gripper",
            "right_joint1..joint6",
            "right_gripper",
        ],
        "base_separation_m": 0.6,
        "settled_qpos": settled_qpos,
        "nonzero_contact_pairs": contact_pairs,
        "plans": plans,
    }


def main() -> int:
    """Load, inspect, open/close the gripper, and step without wall-clock sleeps."""
    args = parse_args()
    root = args.embodiment_root.resolve()
    robotwin_root = args.robotwin_root.resolve()
    config = yaml.safe_load((root / "config.yml").read_text(encoding="utf-8"))
    urdf = (root / config["urdf_path"]).resolve()
    scene = sapien.Scene()
    scene.set_timestep(1.0 / 250.0)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(str(urdf))
    if robot is None:
        raise RuntimeError("SAPIEN returned no articulation")

    links = {link.get_name(): link for link in robot.get_links()}
    joints = {joint.get_name(): joint for joint in robot.get_joints()}
    active = [joint.get_name() for joint in robot.get_active_joints()]
    missing = sorted(set(ARM_JOINTS + GRIPPER_JOINTS) - joints.keys())
    if missing:
        raise ValueError(f"missing joints after SAPIEN load: {missing}")
    if "link6" not in links or "gripper_center" not in links:
        raise ValueError("link6 or gripper_center missing after SAPIEN load")

    observed_limits: dict[str, list[float]] = {}
    for name, expected in EXPECTED_LIMITS.items():
        limits = np.asarray(joints[name].get_limits(), dtype=float)
        if limits.shape != (1, 2):
            raise ValueError(f"unexpected limit shape for {name}: {limits.shape}")
        pair = limits[0].tolist()
        if not np.allclose(pair, expected, atol=1.0e-7):
            raise ValueError(f"unexpected SAPIEN limits for {name}: {pair}")
        observed_limits[name] = pair

    for joint in robot.get_active_joints():
        joint.set_drive_property(stiffness=1000.0, damping=200.0)

    robot.set_root_pose(sapien.Pose([0.0, 0.0, 0.75]))
    for joint in robot.get_active_joints():
        joint.set_drive_target(0.0)
    for _ in range(250):
        robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        scene.step()
    closed_pose = finite_pose(links["gripper_center"])

    joints["L_finger_joint"].set_drive_target(0.04)
    joints["R_finger_joint"].set_drive_target(-0.04)
    for _ in range(250):
        robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        scene.step()
    open_qpos = {
        name: float(joints[name].get_drive_target()[0]) for name in GRIPPER_JOINTS
    }
    if not math.isclose(open_qpos["L_finger_joint"], 0.04, abs_tol=1.0e-8):
        raise ValueError(f"left finger did not accept open target: {open_qpos}")
    if not math.isclose(open_qpos["R_finger_joint"], -0.04, abs_tol=1.0e-8):
        raise ValueError(f"right finger did not accept open target: {open_qpos}")

    report = {
        "status": "passed",
        "urdf": str(urdf),
        "links": sorted(links),
        "all_joints": sorted(joints),
        "active_joints": active,
        "limits": observed_limits,
        "closed_gripper_center_pose_wxyz": closed_pose,
        "open_gripper_drive_targets": open_qpos,
        "simulation_steps": 500,
        "wall_clock_sleep_calls": 0,
        "robotwin_contract": validate_robotwin_contract(robotwin_root, root),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
