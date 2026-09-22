#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
# Upstream is read-only under externals/; this runs against a runtime
# assembled from it plus the overlay plus the patches.
upstream_root="${workspace}/externals/RoboTwin"
robotwin_root="${workspace}/runtime/robotwin"
python3 "${workspace}/pipelines/assemble_runtime.py" \
  --upstream robotwin --source "$upstream_root" --runtime "$robotwin_root" >/dev/null
overlay_root="${workspace}/overlays/robotwin"
replay_script="${workspace}/archive/superseded/replay_panthera_dataset.py"
oracle_runner="${workspace}/packages/panthera_sim/diagnostics/run_oracle_smoke.py"
state_root="${workspace}/state/panthera-single-sft-dataset-state"
task_name="place_cylinder_in_groove"
task_config="panthera_single_cylinder_sft_v1"
dataset_root="${workspace}/data/${task_name}/${task_config}"
activation_script="${workspace}/tools/activate_lab_vla.sh"
expected_episodes=128

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock timeout; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
for required in \
  "$activation_script" \
  "${overlay_root}/envs/${task_name}.py" \
  "$replay_script" \
  "$oracle_runner" \
  "${overlay_root}/description/task_instruction/${task_name}.json" \
  "${overlay_root}/task_config/${task_config}.yml"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少前置文件：${required}" >&2
    exit 1
  fi
done
for marker in \
  "${workspace}/state/panthera-embodiment-state/verified.ok" \
  "${workspace}/state/panthera-physical-replay-probe-state/probe.ok"; do
  if [[ ! -f "$marker" ]]; then
    echo "错误：缺少前置验收标记：${marker}" >&2
    exit 1
  fi
done

mkdir -p "$state_root"
exec 9>"${state_root}/dataset.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera SFT 数据流程正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/dataset.ok" ]]; then
  python3 -m json.tool "${state_root}/dataset-summary.json"
  echo "单 Panthera SFT v1 数据集已通过，无需重复采集。"
  exit 0
fi
reuse_existing=false
if [[ -e "$dataset_root" ]]; then
  existing_count=$(find "${dataset_root}/data" -maxdepth 1 -type f -name 'episode*.hdf5' 2>/dev/null | wc -l)
  if [[ -s "${dataset_root}/scene_info.json" && "$existing_count" -eq "$expected_episodes" ]]; then
    reuse_existing=true
    echo "复用已完整采集的 128 个 HDF5，重新执行确定性验收。"
  else
    echo "错误：发现不完整数据目录，请先人工归档：${dataset_root}" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "$activation_script"
python -m py_compile "${robotwin_root}/envs/${task_name}.py"

# 在昂贵的 128 集采集前，用同一随机化配置证明多个固定 seed 均可规划成功。
# 该检查按仿真事件推进，不用墙钟 sleep；失败时停止，不进入上游无界 seed 重试。
preflight_root="${state_root}/planning-preflight"
if [[ ! -f "${state_root}/planning-preflight.ok" ]]; then
  PYTHONUNBUFFERED=1 timeout --signal=INT --kill-after=60s \
    "${PANTHERA_SFT_PREFLIGHT_TIMEOUT:-30m}" \
    python "$oracle_runner" \
      --robotwin-root "$robotwin_root" \
      --output-root "$preflight_root" \
      --task-config "${task_config}.yml" \
      --seeds 0 1 2 3
  touch "${state_root}/planning-preflight.ok"
fi

if [[ "$reuse_existing" == false ]]; then
  run_stamp=$(date +%Y%m%d-%H%M%S)
  run_log="${state_root}/dataset-${run_stamp}.log"
  printf '%s\n' "$run_log" >"${state_root}/run-log.txt"
  set +e
  (
    cd "$robotwin_root"
    PYTHONUNBUFFERED=1 timeout --signal=INT --kill-after=90s \
      "${PANTHERA_SFT_DATASET_TIMEOUT:-3h}" \
      python scripts/collect_data.py "$task_name" "$task_config"
  ) 2>&1 | tee "$run_log"
  dataset_status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$dataset_status" >"${state_root}/exit-code.txt"
  if (( dataset_status != 0 )); then
    echo "错误：单 Panthera SFT v1 采集失败，退出码 ${dataset_status}。" >&2
    exit "$dataset_status"
  fi
  if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|AssertionError:|Collect Error' "$run_log"; then
    echo "错误：采集日志含致命异常。" >&2
    exit 1
  fi
else
  run_log=$(cat "${state_root}/run-log.txt")
fi

python - "$dataset_root" "${state_root}/dataset-summary.json" "$run_log" "$expected_episodes" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
run_log = Path(sys.argv[3])
expected = int(sys.argv[4])
scene_info = json.loads((root / "scene_info.json").read_text(encoding="utf-8"))
if len(scene_info) != expected:
    raise SystemExit(f"expected {expected} scene records, got {len(scene_info)}")

geometry_keys = set()
episode_seeds = set()
camera_keys = set()
image_keys = set()
video_sizes = []
frame_counts = []
grid_counts = []
action_mins = []
action_maxs = []
state_mins = []
state_maxs = []
for index in range(expected):
    hdf5_path = root / "data" / f"episode{index}.hdf5"
    instruction_path = root / "instructions" / f"episode{index}.json"
    video_path = root / "video" / f"episode{index}.mp4"
    if not hdf5_path.is_file() or hdf5_path.stat().st_size == 0:
        raise SystemExit(f"missing episode data: {hdf5_path}")
    if not instruction_path.is_file() or instruction_path.stat().st_size == 0:
        raise SystemExit(f"missing episode instruction: {instruction_path}")
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise SystemExit(f"missing episode video: {video_path}")
    video_sizes.append(video_path.stat().st_size)
    instructions = json.loads(instruction_path.read_text(encoding="utf-8"))
    if not instructions.get("seen"):
        raise SystemExit(f"episode {index} has no language instruction")
    metadata = scene_info[f"episode_{index}"]["panthera_episode"]
    if metadata.get("schema_version") != 3:
        raise SystemExit(f"episode {index} did not use single-arm schema v3")
    if metadata.get("robot_count") != 1 or metadata.get("action_dimension") != 7:
        raise SystemExit(f"episode {index} did not use one-Panthera 7-D contract")
    if metadata.get("attach_on_grasp"):
        raise SystemExit(f"episode {index} used forbidden grasp attachment")
    episode_seed = int(metadata["episode_seed"])
    if episode_seed in episode_seeds:
        raise SystemExit(f"episode {index} reuses seed {episode_seed}")
    episode_seeds.add(episode_seed)
    if int(metadata.get("sample_period_physics_steps", -1)) != 5:
        raise SystemExit(f"episode {index} does not record the 5-step sample period")
    randomization = metadata.get("task_randomization", {})
    if randomization.get("cylinder_x_m") != 0.02 or randomization.get("groove_x_m") != 0.02:
        raise SystemExit(f"episode {index} task randomization metadata mismatch")
    domain = metadata.get("domain_randomization", {})
    if not domain.get("random_background") or not domain.get("random_light"):
        raise SystemExit(f"episode {index} visual randomization metadata mismatch")
    geometry = metadata["realized_geometry"]
    cylinder_xy = np.asarray(geometry["cylinder_initial_xy_m"], dtype=float)
    groove_xy = np.asarray(geometry["groove_target_xy_m"], dtype=float)
    if abs(cylinder_xy[0]) > 0.020001 or abs(cylinder_xy[1] + 0.02) > 0.015001:
        raise SystemExit(f"episode {index} cylinder outside configured bounds")
    if abs(groove_xy[0]) > 0.020001 or abs(groove_xy[1] + 0.16) > 0.015001:
        raise SystemExit(f"episode {index} groove outside configured bounds")
    geometry_keys.add(tuple(np.round(np.concatenate([cylinder_xy, groove_xy]), 8)))

    with h5py.File(hdf5_path, "r") as episode:
        action = np.asarray(episode["joint_action/vector"], dtype=np.float32)
        state = np.asarray(episode["observation/robot_state/vector"], dtype=np.float32)
        steps = np.asarray(episode["timing/simulation_step_index"], dtype=np.int64)
        times = np.asarray(episode["timing/simulation_time_s"], dtype=np.float64)
        images = episode["observation/head_camera/rgb"]
        camera = np.asarray(episode["observation/head_camera/cam2world_gl"][0], dtype=float)
        if action.ndim != 2 or action.shape[1] != 7 or state.shape != action.shape:
            raise SystemExit(f"episode {index} state/action contract mismatch")
        if steps.shape != (len(action),) or times.shape != (len(action),):
            raise SystemExit(f"episode {index} timing contract mismatch")
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(state)):
            raise SystemExit(f"episode {index} contains nonfinite state/action")
        if np.any(np.diff(steps) < 0):
            raise SystemExit(f"episode {index} simulation clock moved backwards")
        grid_steps = np.unique(steps[steps % 5 == 0])
        if len(grid_steps) < 26 or not np.all(np.diff(grid_steps) == 5):
            raise SystemExit(f"episode {index} does not contain a contiguous 50 Hz grid")
        frame_counts.append(len(action))
        grid_counts.append(len(grid_steps))
        action_mins.append(action.min(axis=0))
        action_maxs.append(action.max(axis=0))
        state_mins.append(state.min(axis=0))
        state_maxs.append(state.max(axis=0))
        camera_keys.add(tuple(np.round(camera.reshape(-1), 6)))
        image_keys.add(hashlib.sha256(bytes(images[0])).hexdigest())

if len(geometry_keys) != expected:
    raise SystemExit("realized object/groove geometry is not unique per episode")
if len(episode_seeds) != expected:
    raise SystemExit("successful episode seeds are not unique")
if len(camera_keys) < expected // 2:
    raise SystemExit("head camera randomization did not produce enough distinct extrinsics")
if len(image_keys) < expected // 2:
    raise SystemExit("visual randomization did not produce enough distinct first frames")

log_text = run_log.read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"failed\s+\x1b\[[0-9;]*m(\d+)\x1b\[[0-9;]*m times / (\d+) tries", log_text)
summary = {
    "status": "contract_passed",
    "dataset_name": "panthera_single_cylinder_sft_v1",
    "task_schema_version": 3,
    "robot_count": 1,
    "action_dimension": 7,
    "episodes": expected,
    "episode_seed_min": min(episode_seeds),
    "episode_seed_max": max(episode_seeds),
    "train_episode_ids": [0, 111],
    "validation_episode_ids": [112, 127],
    "raw_frame_count_min": min(frame_counts),
    "raw_frame_count_max": max(frame_counts),
    "fixed_grid_frame_count_min": min(grid_counts),
    "fixed_grid_frame_count_max": max(grid_counts),
    "unique_geometries": len(geometry_keys),
    "unique_camera_extrinsics": len(camera_keys),
    "unique_first_images": len(image_keys),
    "video_count": len(video_sizes),
    "video_bytes_min": min(video_sizes),
    "action_min": np.min(np.stack(action_mins), axis=0).tolist(),
    "action_max": np.max(np.stack(action_maxs), axis=0).tolist(),
    "proprio_min": np.min(np.stack(state_mins), axis=0).tolist(),
    "proprio_max": np.max(np.stack(state_maxs), axis=0).tolist(),
    "collector_failure_summaries": [list(map(int, match)) for match in matches],
    "nominal_control_hz": 50,
    "wall_clock_sleep_in_success_path": False,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "回放训练/验证样本，确认真实评测动作路径不依赖 oracle D6 约束……"
python "$replay_script" \
  --robotwin-root "$robotwin_root" \
  --dataset-root "$dataset_root" \
  --task-config "${task_config}.yml" \
  --summary "${state_root}/replay-summary.json" \
  --episode 0 \
  --episode 112
python - "${state_root}/dataset-summary.json" "${state_root}/replay-summary.json" <<'PY'
import json
from pathlib import Path
import sys

dataset_path = Path(sys.argv[1])
replay_path = Path(sys.argv[2])
dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
replay = json.loads(replay_path.read_text(encoding="utf-8"))
if dataset.get("status") != "contract_passed" or replay.get("status") != "passed":
    raise SystemExit("dataset contract or no-attachment replay did not pass")
dataset["status"] = "passed"
dataset["no_attachment_replay"] = replay
dataset_path.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
PY

if pgrep -af 'ray::|raylet|collect_data.py|eval_policy.py' | grep -v grep; then
  echo "错误：SFT 数据验收后仍有 RoboTwin/Ray 进程。" >&2
  exit 1
fi
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/dataset.ok"
echo "单 Panthera SFT v1 的 128-episode 数据集通过：${dataset_root}"
