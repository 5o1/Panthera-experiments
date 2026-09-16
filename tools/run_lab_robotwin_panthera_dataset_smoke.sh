#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
robotwin_root="${workspace}/RoboTwin"
overlay_root="${workspace}/panthera-robotwin-overlay"
state_root="${workspace}/.panthera-dataset-smoke-state"
dataset_root="${workspace}/data/place_cylinder_in_groove/panthera_cylinder_dataset_smoke"
activation_script="${workspace}/activate_lab_vla.sh"
task_config="panthera_cylinder_dataset_smoke"
expected_robotwin_commit="0008ae6800df9f75fc8de7098bacb01735fd8fd2"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock git timeout; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  fi
done
if [[ ! -f "${workspace}/.panthera-cylinder-oracle-state/oracle.ok" ]]; then
  echo "错误：双 Panthera oracle 尚未通过四种子验收。" >&2
  exit 1
fi
if [[ ! -f "$activation_script" ]]; then
  echo "错误：缺少环境激活脚本：${activation_script}" >&2
  exit 1
fi
if [[ $(git -C "$robotwin_root" rev-parse HEAD) != "$expected_robotwin_commit" ]]; then
  echo "错误：RoboTwin checkout 不是已固定提交。" >&2
  exit 1
fi

mkdir -p "$state_root"
exec 9>"${state_root}/dataset.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera dataset 流程正在运行。" >&2
  exit 1
fi

for relative_path in \
  envs/place_cylinder_in_groove.py \
  description/task_instruction/place_cylinder_in_groove.json \
  "task_config/${task_config}.yml"; do
  if [[ ! -s "${overlay_root}/${relative_path}" ]]; then
    echo "错误：overlay 文件缺失或为空：${relative_path}" >&2
    exit 1
  fi
done

install -m 0644 \
  "${overlay_root}/envs/place_cylinder_in_groove.py" \
  "${robotwin_root}/envs/place_cylinder_in_groove.py"
install -m 0644 \
  "${overlay_root}/description/task_instruction/place_cylinder_in_groove.json" \
  "${robotwin_root}/description/task_instruction/place_cylinder_in_groove.json"
install -m 0644 \
  "${overlay_root}/task_config/${task_config}.yml" \
  "${robotwin_root}/task_config/${task_config}.yml"

# shellcheck disable=SC1090
source "$activation_script"
python -m py_compile "${robotwin_root}/envs/place_cylinder_in_groove.py"

run_stamp=$(date +%Y%m%d-%H%M%S)
run_log="${state_root}/dataset-${run_stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"

set +e
(
  cd "$robotwin_root"
  timeout --signal=INT --kill-after=60s \
    "${PANTHERA_DATASET_TIMEOUT:-45m}" \
    python script/collect_data.py place_cylinder_in_groove "$task_config"
) 2>&1 | tee "$run_log"
dataset_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$dataset_status" >"${state_root}/exit-code.txt"

if (( dataset_status != 0 )); then
  echo "错误：双 Panthera episode 采集未通过，退出码 ${dataset_status}。" >&2
  exit "$dataset_status"
fi
if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|AssertionError:|Collect Error' "$run_log"; then
  echo "错误：采集返回 0 但日志含致命异常，拒绝写入成功标记。" >&2
  exit 1
fi

python - "$dataset_root" "${state_root}/dataset-summary.json" <<'PY'
import json
from pathlib import Path
import sys

import h5py
import numpy as np

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
limits = np.array(
    [
        [-2.4, 2.4],
        [0.0, 3.2],
        [0.0, 4.0],
        [-1.6, 1.6],
        [-1.7, 1.7],
        [-2.5, 2.5],
    ],
    dtype=float,
)
episodes = []
for index in range(4):
    hdf5_path = root / "data" / f"episode{index}.hdf5"
    video_path = root / "video" / f"episode{index}.mp4"
    if not hdf5_path.is_file() or hdf5_path.stat().st_size == 0:
        raise SystemExit(f"missing nonempty HDF5: {hdf5_path}")
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise SystemExit(f"missing nonempty video: {video_path}")
    with h5py.File(hdf5_path, "r") as episode:
        qpos = np.asarray(episode["joint_action/vector"], dtype=float)
        rgb = episode["observation/head_camera/rgb"]
        left = np.asarray(episode["joint_action/left_arm"], dtype=float)
        right = np.asarray(episode["joint_action/right_arm"], dtype=float)
        left_gripper = np.asarray(episode["joint_action/left_gripper"], dtype=float)
        right_gripper = np.asarray(episode["joint_action/right_gripper"], dtype=float)
        if qpos.ndim != 2 or qpos.shape[1] != 14 or qpos.shape[0] < 50:
            raise SystemExit(f"unexpected 14-D trajectory shape: {qpos.shape}")
        frame_count = int(qpos.shape[0])
        expected_shapes = {
            "rgb": int(rgb.shape[0]),
            "left": tuple(left.shape),
            "right": tuple(right.shape),
            "left_gripper": tuple(left_gripper.shape),
            "right_gripper": tuple(right_gripper.shape),
        }
        if int(rgb.shape[0]) != frame_count:
            raise SystemExit(f"RGB/qpos frame count mismatch: {expected_shapes}")
        if left.shape != (frame_count, 6) or right.shape != (frame_count, 6):
            raise SystemExit(f"arm trajectory shape mismatch: {expected_shapes}")
        if left_gripper.shape != (frame_count,) or right_gripper.shape != (frame_count,):
            raise SystemExit(f"gripper trajectory shape mismatch: {expected_shapes}")
        if not np.all(np.isfinite(qpos)):
            raise SystemExit(f"episode {index} contains NaN or infinity")
        if np.any(left < limits[:, 0] - 1.0e-5) or np.any(left > limits[:, 1] + 1.0e-5):
            raise SystemExit(f"episode {index} left arm exceeds official limits")
        if np.any(right < limits[:, 0] - 1.0e-5) or np.any(right > limits[:, 1] + 1.0e-5):
            raise SystemExit(f"episode {index} right arm exceeds official limits")
        if np.any(left_gripper < 0.0) or np.any(left_gripper > 1.0):
            raise SystemExit(f"episode {index} left gripper is not normalized")
        if np.any(right_gripper < 0.0) or np.any(right_gripper > 1.0):
            raise SystemExit(f"episode {index} right gripper is not normalized")
        max_step = float(np.max(np.abs(np.diff(qpos, axis=0))))
    episodes.append(
        {
            "index": index,
            "frames": frame_count,
            "hdf5_bytes": hdf5_path.stat().st_size,
            "video_bytes": video_path.stat().st_size,
            "max_sample_delta": max_step,
        }
    )

summary = {
    "status": "passed",
    "episodes": episodes,
    "action_dimension": 14,
    "action_order": [
        "left_joint1..joint6",
        "left_gripper",
        "right_joint1..joint6",
        "right_gripper",
    ],
    "physics_hz": 250,
    "capture_every_sim_steps": 5,
    "nominal_capture_hz": 50,
    "wall_clock_sleep_in_success_path": False,
    "timestamps_present_in_upstream_hdf5": False,
}
temporary = summary_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(summary_path)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if find "$dataset_root/.cache" -type f -print -quit 2>/dev/null | grep -q .; then
  echo "错误：episode 合并后仍有未清理的采集缓存。" >&2
  exit 1
fi
if pgrep -af 'ray::|raylet|gcs_server' | grep -v pgrep >/dev/null; then
  echo "错误：采集结束后仍有 Ray 进程。" >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "错误：采集结束后仍有 GPU 计算进程。" >&2
  exit 1
fi

date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/dataset.ok"
echo "双 Panthera 50 Hz episode smoke 通过：${dataset_root}"
