#!/usr/bin/env python3
"""Derive a phone-SRT camera embodiment from the verified Panthera asset."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import yaml


PHONE_CAMERA = {
    "name": "head_camera",
    # The archived 16:9 phone frame centers the task while the Panthera base is
    # left of it.  The task itself is shifted to x=-0.25 m, so centering the
    # camera on the same x preserves that composition without moving the robot
    # model.  These values remain non-metric until the mount is restored.
    "position": [-0.25, 0.72, 1.05],
    "forward": [0.0, -1.0, -0.28],
    "left": [1.0, 0.0, 0.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera-x", type=float, default=-0.25)
    parser.add_argument(
        "--profile", default="phone_srt_provisional_wide_v3"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    for required in ("panthera.urdf", "panthera.srdf", "config.yml"):
        if not (source / required).is_file():
            raise FileNotFoundError(source / required)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite phone embodiment without archival: {output}"
        )

    shutil.copytree(source, output)
    config_path = output / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    robot_pose = config.get("robot_pose")
    if not isinstance(robot_pose, list) or len(robot_pose) != 1 or len(robot_pose[0]) != 7:
        raise ValueError("expected one 7-value Panthera robot_pose")
    robot_pose[0][1] = -0.35
    phone_camera = deepcopy(PHONE_CAMERA)
    phone_camera["position"][0] = args.camera_x
    config["static_camera_list"] = [phone_camera]
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    provenance_path = output / "provenance.phone-srt.json"
    provenance_path.write_text(
        json.dumps(
            {
                "derived_from": str(source),
                "profile": args.profile,
                "policy_crop": {
                    "source_aspect": "16:9",
                    "crop": "center_4_3_then_downscale",
                    "simulation_resolution": [320, 240],
                    "vertical_fov_deg": 75,
                },
                "static_camera": phone_camera,
                "simulation_robot_pose": robot_pose[0],
                "calibration_status": "inferred_from_archived_view_not_metric",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"built provisional phone-SRT Panthera embodiment at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
