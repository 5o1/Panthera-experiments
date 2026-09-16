#!/usr/bin/env python3
"""Build a RoboTwin Panthera embodiment from the pinned official ROS 2 model.

The generated files are simulation assets only.  The source ROS 2 checkout is
left untouched, and no robot device or ROS driver is accessed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


EXPECTED_JOINTS = {
    "joint1": ("revolute", -2.4, 2.4),
    "joint2": ("revolute", 0.0, 3.2),
    "joint3": ("revolute", 0.0, 4.0),
    "joint4": ("revolute", -1.6, 1.6),
    "joint5": ("revolute", -1.7, 1.7),
    "joint6": ("revolute", -2.5, 2.5),
    "L_finger_joint": ("prismatic", 0.0, 0.04),
    "R_finger_joint": ("prismatic", -0.04, 0.0),
}


CONFIG_YAML = """\
urdf_path: "./panthera.urdf"
srdf_path: "./panthera.srdf"
joint_stiffness: 1000
joint_damping: 200
gripper_stiffness: 1000
gripper_damping: 200
move_group: ["link6", "link6"]
ee_joints: ["joint6", "joint6"]
ee_pose_from_child_link: true
arm_joints_name:
  - ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
  - ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
gripper_name:
  - base: "L_finger_joint"
    mimic: [["R_finger_joint", -1.0, 0.0]]
  - base: "L_finger_joint"
    mimic: [["R_finger_joint", -1.0, 0.0]]
gripper_bias: 0.165
gripper_scale: [0.0, 0.04]
homestate:
  - [0.0, 1.6, 1.6, 0.0, 0.0, 0.0]
  - [0.0, 1.6, 1.6, 0.0, 0.0, 0.0]
delta_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
global_trans_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
robot_pose: [[0, -0.45, 0.75, 0.70710678, 0, 0, 0.70710678]]
planner: "mplib_RRT"
dual_arm: false
rotate_lim: [0.2, 1.0]
grasp_perfect_direction: ["front_right", "front_left"]
static_camera_list:
  - name: head_camera
    position: [-0.032, -0.45, 1.35]
    forward: [0, 0.6, -0.8]
    left: [-1, 0, 0]
"""


SRDF = """\
<?xml version="1.0"?>
<robot name="panthera_ht_ros_description">
  <group name="arm">
    <joint name="joint1"/>
    <joint name="joint2"/>
    <joint name="joint3"/>
    <joint name="joint4"/>
    <joint name="joint5"/>
    <joint name="joint6"/>
    <chain base_link="base_link" tip_link="link6"/>
  </group>
  <disable_collisions link1="base_link" link2="link1" reason="Adjacent"/>
  <disable_collisions link1="base_link" link2="link2" reason="Never"/>
  <disable_collisions link1="link1" link2="link2" reason="Adjacent"/>
  <disable_collisions link1="link2" link2="link3" reason="Adjacent"/>
  <disable_collisions link1="link2" link2="link4" reason="Never"/>
  <disable_collisions link1="link2" link2="link5" reason="Never"/>
  <disable_collisions link1="link2" link2="link6" reason="Never"/>
  <disable_collisions link1="link3" link2="link4" reason="Adjacent"/>
  <disable_collisions link1="link3" link2="link5" reason="Never"/>
  <disable_collisions link1="link3" link2="link6" reason="Never"/>
  <disable_collisions link1="link4" link2="link5" reason="Adjacent"/>
  <disable_collisions link1="link4" link2="link6" reason="Never"/>
  <disable_collisions link1="link5" link2="link6" reason="Adjacent"/>
  <disable_collisions link1="link6" link2="L_finger" reason="Adjacent"/>
  <disable_collisions link1="link6" link2="R_finger" reason="Adjacent"/>
  <disable_collisions link1="link6" link2="gripper_center" reason="Adjacent"/>
  <disable_collisions link1="L_finger" link2="R_finger" reason="Never"/>
  <disable_collisions link1="gripper_center" link2="L_finger" reason="Default"/>
  <disable_collisions link1="gripper_center" link2="R_finger" reason="Default"/>
  <disable_collisions link1="L_finger" link2="link3" reason="Never"/>
  <disable_collisions link1="L_finger" link2="link4" reason="Never"/>
  <disable_collisions link1="L_finger" link2="link5" reason="Never"/>
  <disable_collisions link1="R_finger" link2="link3" reason="Never"/>
  <disable_collisions link1="R_finger" link2="link4" reason="Never"/>
  <disable_collisions link1="R_finger" link2="link5" reason="Never"/>
  <disable_collisions link1="gripper_center" link2="link3" reason="Never"/>
  <disable_collisions link1="gripper_center" link2="link4" reason="Never"/>
  <disable_collisions link1="gripper_center" link2="link5" reason="Never"/>
</robot>
"""


def parse_args() -> argparse.Namespace:
    """Parse explicit source and output locations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="root of the official Panthera_HT_ROS2 checkout",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a stable content hash for provenance and later verification."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_and_rewrite_urdf(source: Path, destination: Path) -> list[str]:
    """Validate official joints and rewrite ROS package mesh URIs for SAPIEN."""
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("name", "panthera_ht_ros_description")

    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    for name, (expected_type, expected_lower, expected_upper) in EXPECTED_JOINTS.items():
        joint = joints.get(name)
        if joint is None or joint.get("type") != expected_type:
            raise ValueError(f"missing or unexpected joint: {name}")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"joint has no limit: {name}")
        bounds = (float(limit.get("lower", "nan")), float(limit.get("upper", "nan")))
        if bounds != (expected_lower, expected_upper):
            raise ValueError(f"unexpected limits for {name}: {bounds}")

    mimic = joints["R_finger_joint"].find("mimic")
    if mimic is None or mimic.get("joint") != "L_finger_joint":
        raise ValueError("official right-finger mimic definition is missing")
    if float(mimic.get("multiplier", "nan")) != -1.0:
        raise ValueError("official right-finger mimic multiplier is not -1")

    meshes: list[str] = []
    prefix = "package://panthera_ht_ros_description/meshes/"
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if not filename.startswith(prefix):
            raise ValueError(f"unexpected mesh URI: {filename}")
        basename = filename.removeprefix(prefix)
        if "/" in basename or not basename:
            raise ValueError(f"unsafe mesh basename: {basename}")
        mesh.set("filename", f"meshes/{basename}")
        meshes.append(basename)

    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return sorted(set(meshes))


def main() -> int:
    """Generate a deterministic Panthera embodiment and its provenance record."""
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    description_root = source_root / "src" / "panthera_ht_ros_description"
    source_urdf = description_root / "urdf" / "panthera_ht_ros_description_gripper.xacro"
    source_mesh_root = description_root / "meshes"
    source_license = source_root / "LICENSE"
    for required in (source_urdf, source_mesh_root, source_license):
        if not required.exists():
            raise FileNotFoundError(required)

    output_root.mkdir(parents=True, exist_ok=True)
    mesh_output = output_root / "meshes"
    mesh_output.mkdir(exist_ok=True)
    mesh_names = validate_and_rewrite_urdf(source_urdf, output_root / "panthera.urdf")

    hashes: dict[str, str] = {}
    for mesh_name in mesh_names:
        source_mesh = source_mesh_root / mesh_name
        if not source_mesh.is_file():
            raise FileNotFoundError(source_mesh)
        destination = mesh_output / mesh_name
        shutil.copy2(source_mesh, destination)
        hashes[f"meshes/{mesh_name}"] = sha256(destination)

    (output_root / "panthera.srdf").write_text(SRDF, encoding="utf-8")
    (output_root / "config.yml").write_text(CONFIG_YAML, encoding="utf-8")
    shutil.copy2(source_license, output_root / "LICENSE.HighTorque-Robotics")
    for name in ("panthera.urdf", "panthera.srdf", "config.yml", "LICENSE.HighTorque-Robotics"):
        hashes[name] = sha256(output_root / name)

    provenance = {
        "source": "https://github.com/HighTorque-Robotics/Panthera-HT_ROS2.git",
        "source_commit": args.source_commit,
        "source_urdf": str(source_urdf.relative_to(source_root)),
        "simulation_changes": [
            "rewrote package:// mesh URIs to embodiment-relative paths",
            "added a minimal SRDF arm group derived from the official MoveIt SRDF",
            "added RoboTwin joint, gripper, TCP and placement metadata",
            "used the official MoveIt pose1 group state as the non-singular simulation home",
        ],
        "files_sha256": dict(sorted(hashes.items())),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"built Panthera embodiment at {output_root}")
    print(f"meshes: {', '.join(mesh_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
