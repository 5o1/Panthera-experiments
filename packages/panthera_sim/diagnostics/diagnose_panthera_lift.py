#!/usr/bin/env python3
"""Probe Panthera post-grasp lift IK candidates in one simulator instance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from run_oracle_smoke import _load_task_args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.robotwin_root.resolve()
    os.environ["ASSETS_PATH"] = str(root)
    sys.path.insert(0, str(root))
    os.chdir(root)
    from envs.place_vertical_cylinder_in_groove import (
        place_vertical_cylinder_in_groove,
    )
    from envs.utils import ArmTag

    task_args = _load_task_args(
        root,
        "panthera_phone_vertical_oracle.yml",
        "place_vertical_cylinder_in_groove",
    )
    task = place_vertical_cylinder_in_groove()
    results = []
    try:
        task.setup_demo(now_ep_num=0, seed=0, **task_args)
        arm = ArmTag("left")
        task._move_arm(task.close_gripper(arm, pos=0.9), "preclose")
        task._move_arm(task._top_down_grasp(arm, 0), "grasped")
        task._advance_physics(25)
        origin = np.asarray(task.robot.get_left_ee_pose(), dtype=float)
        orientations = {
            "measured": origin[3:].tolist(),
            "commanded": task.grasp_quaternion_wxyz,
        }
        for name, quaternion in orientations.items():
            for dz in (0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10):
                target = origin.copy()
                target[2] += dz
                target[3:] = quaternion
                task.plan_success = True
                result = task.left_move_to_pose(target)
                status = "missing" if result is None else str(result.get("status"))
                position = None if result is None else result.get("position")
                results.append({
                    "orientation": name,
                    "dz_m": dz,
                    "status": status,
                    "trajectory_steps": 0 if position is None else int(position.shape[0]),
                })
    finally:
        task.close_env(clear_cache=True)
    payload = {"origin_ee_pose": origin.tolist(), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
