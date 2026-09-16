#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
robotwin_root="${workspace}/RoboTwin"
overlay_root="${workspace}/panthera-robotwin-overlay"
state_root="${workspace}/.panthera-contract-smoke-state"
task_name="place_cylinder_in_groove"
task_config="panthera_cylinder_contract_smoke"
dataset_root="${workspace}/data/${task_name}/${task_config}"
activation_script="${workspace}/activate_lab_vla.sh"
expected_robotwin_commit="0008ae6800df9f75fc8de7098bacb01735fd8fd2"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock git timeout; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
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
if ! grep -q 'def _step_scene' "${robotwin_root}/envs/_base_task.py"; then
  echo "错误：确定性仿真步计数补丁尚未安装。" >&2
  exit 1
fi

mkdir -p "$state_root"
exec 9>"${state_root}/contract.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera contract 流程正在运行。" >&2
  exit 1
fi

for relative_path in \
  "envs/${task_name}.py" \
  "description/task_instruction/${task_name}.json" \
  "task_config/${task_config}.yml"; do
  if [[ ! -s "${overlay_root}/${relative_path}" ]]; then
    echo "错误：overlay 文件缺失或为空：${relative_path}" >&2
    exit 1
  fi
done
install -m 0644 "${overlay_root}/envs/${task_name}.py" "${robotwin_root}/envs/${task_name}.py"
install -m 0644 \
  "${overlay_root}/description/task_instruction/${task_name}.json" \
  "${robotwin_root}/description/task_instruction/${task_name}.json"
install -m 0644 \
  "${overlay_root}/task_config/${task_config}.yml" \
  "${robotwin_root}/task_config/${task_config}.yml"

# shellcheck disable=SC1090
source "$activation_script"
python -m py_compile "${robotwin_root}/envs/${task_name}.py"

run_stamp=$(date +%Y%m%d-%H%M%S)
run_log="${state_root}/contract-${run_stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"
set +e
(
  cd "$robotwin_root"
  timeout --signal=INT --kill-after=60s \
    "${PANTHERA_CONTRACT_TIMEOUT:-45m}" \
    python script/collect_data.py "$task_name" "$task_config"
) 2>&1 | tee "$run_log"
contract_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$contract_status" >"${state_root}/exit-code.txt"
if (( contract_status != 0 )); then
  echo "错误：Panthera 正式契约 smoke 失败，退出码 ${contract_status}。" >&2
  exit "$contract_status"
fi
if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|AssertionError:|Collect Error|ERROR: Scene info' "$run_log"; then
  echo "错误：采集日志含致命异常。" >&2
  exit 1
fi

python - "$dataset_root" "${state_root}/contract-summary.json" <<'PY'
import json
from pathlib import Path
import sys

import h5py
import numpy as np

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
limits = np.array(
    [[-2.4, 2.4], [0.0, 3.2], [0.0, 4.0], [-1.6, 1.6], [-1.7, 1.7], [-2.5, 2.5]],
    dtype=float,
)
scene_info = json.loads((root / "scene_info.json").read_text(encoding="utf-8"))
episodes = []
geometries = []
for index in range(4):
    hdf5_path = root / "data" / f"episode{index}.hdf5"
    video_path = root / "video" / f"episode{index}.mp4"
    instruction_path = root / "instructions" / f"episode{index}.json"
    for path in (hdf5_path, video_path, instruction_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing nonempty artifact: {path}")
    instruction = json.loads(instruction_path.read_text(encoding="utf-8"))
    if not instruction.get("seen"):
        raise SystemExit(f"episode {index} has no generated instruction")
    metadata = scene_info[f"episode_{index}"]["panthera_episode"]
    if metadata["episode_seed"] != index or metadata["schema_version"] != 1:
        raise SystemExit(f"episode {index} metadata is inconsistent")
    geometry = metadata["realized_geometry"]
    physics_timestep_s = float(metadata["physics_timestep_s"])
    geometries.append(json.dumps(geometry, sort_keys=True))
    with h5py.File(hdf5_path, "r") as episode:
        action = np.asarray(episode["joint_action/vector"], dtype=float)
        state = np.asarray(episode["observation/robot_state/vector"], dtype=float)
        left_state = np.asarray(episode["observation/robot_state/left_arm_qpos"], dtype=float)
        right_state = np.asarray(episode["observation/robot_state/right_arm_qpos"], dtype=float)
        left_gripper = np.asarray(episode["observation/robot_state/left_gripper_qpos"], dtype=float)
        right_gripper = np.asarray(episode["observation/robot_state/right_gripper_qpos"], dtype=float)
        steps = np.asarray(episode["timing/simulation_step_index"], dtype=np.int64)
        times = np.asarray(episode["timing/simulation_time_s"], dtype=float)
        frame_count = int(action.shape[0])
        if action.shape != (frame_count, 14) or state.shape != (frame_count, 14) or frame_count < 50:
            raise SystemExit(f"episode {index} has invalid action/state shapes")
        if left_state.shape != (frame_count, 6) or right_state.shape != (frame_count, 6):
            raise SystemExit(f"episode {index} has invalid arm state shapes")
        if left_gripper.shape != (frame_count,) or right_gripper.shape != (frame_count,):
            raise SystemExit(f"episode {index} has invalid gripper state shapes")
        if steps.shape != (frame_count,) or times.shape != (frame_count,):
            raise SystemExit(f"episode {index} has invalid timing shapes")
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(state)) or not np.all(np.isfinite(times)):
            raise SystemExit(f"episode {index} contains nonfinite values")
        if np.any(np.diff(steps) < 0) or steps[-1] <= steps[0]:
            raise SystemExit(f"episode {index} simulation clock is not monotonic")
        if not np.allclose(times, steps * physics_timestep_s, atol=1.0e-9, rtol=0.0):
            raise SystemExit(f"episode {index} time does not match the 250 Hz step clock")
        for arm in (left_state, right_state):
            if np.any(arm < limits[:, 0] - 1.0e-5) or np.any(arm > limits[:, 1] + 1.0e-5):
                raise SystemExit(f"episode {index} measured arm state exceeds official limits")
        if np.any(left_gripper < 0.0) or np.any(left_gripper > 1.0):
            raise SystemExit(f"episode {index} measured left gripper is invalid")
        if np.any(right_gripper < 0.0) or np.any(right_gripper > 1.0):
            raise SystemExit(f"episode {index} measured right gripper is invalid")
        tracking_error = float(np.max(np.abs(action[:, [0,1,2,3,4,5,7,8,9,10,11,12]] - state[:, [0,1,2,3,4,5,7,8,9,10,11,12]])))
        positive_step_deltas = np.diff(steps)[np.diff(steps) > 0]
        if positive_step_deltas.max() > 5:
            raise SystemExit(f"episode {index} has a discontinuous sample gap")
    episodes.append({
        "index": index,
        "frames": frame_count,
        "first_simulation_step": int(steps[0]),
        "last_simulation_step": int(steps[-1]),
        "physics_timestep_s": physics_timestep_s,
        "minimum_positive_step_delta": int(positive_step_deltas.min()),
        "maximum_step_delta": int(positive_step_deltas.max()),
        "max_arm_tracking_error_rad": tracking_error,
        "instruction_count": len(instruction["seen"]),
        "realized_geometry": geometry,
        "hdf5_bytes": hdf5_path.stat().st_size,
        "video_bytes": video_path.stat().st_size,
    })
if len(set(geometries)) != 4:
    raise SystemExit("task randomization did not produce four distinct geometries")

summary = {
    "status": "passed",
    "schema_version": 1,
    "episodes": episodes,
    "action_dimension": 14,
    "measured_state_dimension": 14,
    "physics_hz": 250,
    "nominal_capture_every_steps": 5,
    "exact_per_frame_step_and_time": True,
    "target_and_measured_state_are_separate": True,
    "wall_clock_sleep_in_success_path": False,
}
temporary = summary_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(summary_path)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if find "$dataset_root/.cache" -type f -print -quit 2>/dev/null | grep -q .; then
  echo "错误：合并后仍有未清理的数据缓存。" >&2
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
touch "${state_root}/contract.ok"
echo "双 Panthera 正式 episode 契约 smoke 通过：${dataset_root}"
