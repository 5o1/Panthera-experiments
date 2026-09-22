#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
upstream_root="${workspace}/externals/RoboTwin"
robotwin_root="${workspace}/runtime/robotwin"
overlay_root="${workspace}/overlays/robotwin"
oracle_runner="${workspace}/packages/panthera_sim/diagnostics/run_oracle_smoke.py"
replay_script="${workspace}/archive/superseded/replay_panthera_dataset.py"
state_root="${workspace}/state/panthera-phone-sft-dataset-state"
shard_root="${workspace}/datasets/shards/data_phone_shards/place_vertical_cylinder_in_groove"
dataset_root="${workspace}/data/place_vertical_cylinder_in_groove/panthera_phone_vertical_sft_v1"
activation_script="${workspace}/tools/activate_lab_vla.sh"
oracle_runner="${workspace}/bin/run_lab_panthera_phone_oracle.sh"
template="${overlay_root}/task_config/panthera_phone_vertical_sft_v1.yml"
task_name="place_vertical_cylinder_in_groove"
task_config="panthera_phone_vertical_sft_v1"
shard_count=4
episodes_per_shard=32
expected_episodes=128
read -r -a collector_gpus <<<"${PANTHERA_PHONE_DATASET_GPUS:-0 1 2 3}"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock git timeout nvidia-smi; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
if (( ${#collector_gpus[@]} == 0 )); then
  echo "错误：PANTHERA_PHONE_DATASET_GPUS 不能为空。" >&2
  exit 1
fi
for gpu in "${collector_gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "错误：非法 GPU 编号：${gpu}" >&2
    exit 1
  fi
done
for required in \
  "$activation_script" "$oracle_runner" "$collector_patch" "$template" \
  "${overlay_root}/envs/${task_name}.py" \
  "${overlay_root}/description/task_instruction/${task_name}.json" \
  "$oracle_runner" \
  "$replay_script"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少 phone-SRT 数据集文件：${required}" >&2
    exit 1
  fi
done

mkdir -p "$state_root"
exec 9>"${state_root}/dataset.lock"
if ! flock -n 9; then
  echo "错误：另一个 phone-SRT 数据集流程正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/dataset.ok" ]]; then
  python3 -m json.tool "${state_root}/dataset-summary.json"
  exit 0
fi

bash "$oracle_runner"
# shellcheck disable=SC1090
source "$activation_script"

# Assembled from the pinned upstream instead of patched into it.
python3 "${workspace}/pipelines/assemble_runtime.py" \
  --upstream robotwin \
  --source "$upstream_root" \
  --runtime "$robotwin_root"
grep -q 'seed_start = int(' "$robotwin_root/script/collect_data.py" \
  || { echo "错误：装配出的运行树缺少所需改动。" >&2; exit 1; }

install -m 0644 "${overlay_root}/envs/${task_name}.py" \
  "${robotwin_root}/envs/${task_name}.py"
install -m 0644 \
  "${overlay_root}/description/task_instruction/${task_name}.json" \
  "${robotwin_root}/description/task_instruction/${task_name}.json"
install -m 0644 "$template" \
  "${robotwin_root}/env_cfg/task_config/${task_config}.yml"

python - "$template" "${robotwin_root}/env_cfg/task_config" "$shard_count" \
  "$episodes_per_shard" <<'PY'
from pathlib import Path
import sys
import yaml

template = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
output = Path(sys.argv[2])
shards = int(sys.argv[3])
episodes = int(sys.argv[4])
for shard in range(shards):
    config = dict(template)
    config["episode_num"] = episodes
    config["seed_start"] = shard * 10000
    config["retry_delay_s"] = 0
    config["save_path"] = "../data_phone_shards"
    name = f"panthera_phone_vertical_sft_shard{shard}.yml"
    (output / name).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
PY

# Exercise all four disjoint seed ranges before starting the expensive shards.
if [[ ! -f "${state_root}/planning-preflight.ok" ]]; then
  CUDA_VISIBLE_DEVICES="${collector_gpus[0]}" PYTHONUNBUFFERED=1 timeout --signal=INT --kill-after=60s \
    "${PANTHERA_PHONE_PREFLIGHT_TIMEOUT:-45m}" \
    python "$oracle_runner" \
      --robotwin-root "$robotwin_root" \
      --output-root "${state_root}/planning-preflight" \
      --task-name "$task_name" \
      --task-config "${task_config}.yml" \
      --seeds 0 10000 20000 30000
  touch "${state_root}/planning-preflight.ok"
fi

for gpu in "${collector_gpus[@]}"; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "错误：GPU ${gpu} 当前有其他计算进程，拒绝启动采集。" >&2
    exit 1
  fi
done

declare -a shard_pids=()
cleanup_children() {
  local pid
  for pid in "${shard_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${shard_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup_children INT TERM EXIT

shard_artifacts_complete() {
  local shard="$1"
  local root="${shard_root}/panthera_phone_vertical_sft_shard${shard}"
  [[ -s "${root}/scene_info.json" && -s "${root}/seed.txt" ]] || return 1
  [[ $(find "${root}/data" -maxdepth 1 -name 'episode*.hdf5' 2>/dev/null | wc -l) -eq "$episodes_per_shard" ]] || return 1
  [[ $(find "${root}/video" -maxdepth 1 -name 'episode*.mp4' 2>/dev/null | wc -l) -eq "$episodes_per_shard" ]] || return 1
  [[ $(find "${root}/instructions" -maxdepth 1 -name 'episode*.json' 2>/dev/null | wc -l) -eq "$episodes_per_shard" ]] || return 1
}

collection_status=0
gpu_count=${#collector_gpus[@]}
echo "使用 GPU ${collector_gpus[*]} 分批运行 4 个独立的 32-episode 分片……"
for ((batch_start=0; batch_start<shard_count; batch_start+=gpu_count)); do
  shard_pids=()
  batch_end=$((batch_start + gpu_count))
  (( batch_end > shard_count )) && batch_end=$shard_count
  for ((shard=batch_start; shard<batch_end; shard++)); do
    gpu_index=$((shard - batch_start))
    gpu="${collector_gpus[$gpu_index]}"
    shard_state="${state_root}/shard-${shard}"
    mkdir -p "$shard_state"
    if shard_artifacts_complete "$shard"; then
      printf '%s\n' 0 >"${shard_state}/exit-code.txt"
      echo "复用已完成的分片 ${shard}，不重复采集。"
      continue
    fi
    run_log="${shard_state}/collector.log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      cd "$robotwin_root"
      set +e
      PYTHONUNBUFFERED=1 timeout --signal=INT --kill-after=90s \
        "${PANTHERA_PHONE_SHARD_TIMEOUT:-2h}" \
        python scripts/collect_data.py "$task_name" \
          "panthera_phone_vertical_sft_shard${shard}" \
        >"$run_log" 2>&1
      code=$?
      set -e
      printf '%s\n' "$code" >"${shard_state}/exit-code.txt"
      exit "$code"
    ) &
    child_pid=$!
    shard_pids+=("$child_pid")
    printf '%s\n' "$child_pid" >"${shard_state}/pid.txt"
  done
  for child_pid in "${shard_pids[@]}"; do
    if ! wait "$child_pid"; then
      collection_status=1
    fi
  done
done
trap - INT TERM EXIT
if (( collection_status != 0 )); then
  echo "错误：至少一个 phone-SRT 仿真分片失败。" >&2
  exit 1
fi

python - "$shard_root" "$dataset_root" "$shard_count" \
  "$episodes_per_shard" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

source_root = Path(sys.argv[1])
destination = Path(sys.argv[2])
shard_count = int(sys.argv[3])
episodes_per_shard = int(sys.argv[4])
expected = shard_count * episodes_per_shard

def complete(root: Path) -> bool:
    return (
        (root / "scene_info.json").is_file()
        and len(list((root / "data").glob("episode*.hdf5"))) == expected
        and len(list((root / "video").glob("episode*.mp4"))) == expected
        and len(list((root / "instructions").glob("episode*.json"))) == expected
    )

if destination.exists():
    if not complete(destination):
        raise SystemExit(f"incomplete merged dataset requires archival: {destination}")
    print(f"reusing complete merged dataset: {destination}")
    raise SystemExit(0)

staging = destination.with_name(destination.name + ".building")
if staging.exists():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    staging.rename(staging.with_name(staging.name + f".failed-{stamp}"))
for name in ("data", "video", "instructions"):
    (staging / name).mkdir(parents=True, exist_ok=True)

merged_scene = {}
all_seeds = []
for shard in range(shard_count):
    config_name = f"panthera_phone_vertical_sft_shard{shard}"
    root = source_root / config_name
    scene = json.loads((root / "scene_info.json").read_text(encoding="utf-8"))
    seeds = [int(value) for value in (root / "seed.txt").read_text().split()]
    if len(scene) != episodes_per_shard or len(seeds) != episodes_per_shard:
        raise SystemExit(f"shard {shard} is incomplete")
    all_seeds.extend(seeds)
    for local_id in range(episodes_per_shard):
        global_id = shard * episodes_per_shard + local_id
        metadata = scene[f"episode_{local_id}"]
        metadata["panthera_episode"]["collection_shard"] = shard
        metadata["panthera_episode"]["shard_episode_id"] = local_id
        merged_scene[f"episode_{global_id}"] = metadata
        for folder, suffix in (
            ("data", "hdf5"),
            ("video", "mp4"),
            ("instructions", "json"),
        ):
            source = root / folder / f"episode{local_id}.{suffix}"
            target = staging / folder / f"episode{global_id}.{suffix}"
            if not source.is_file() or source.stat().st_size == 0:
                raise SystemExit(f"missing shard artifact: {source}")
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)

if len(set(all_seeds)) != expected:
    raise SystemExit("shard seeds are not globally unique")
(staging / "scene_info.json").write_text(
    json.dumps(merged_scene, indent=2) + "\n", encoding="utf-8"
)
(staging / "seed.txt").write_text(
    " ".join(map(str, all_seeds)) + "\n", encoding="utf-8"
)
destination.parent.mkdir(parents=True, exist_ok=True)
staging.rename(destination)
print(f"atomically merged {expected} episodes into {destination}")
PY

python - "$dataset_root" "${state_root}/dataset-summary.json" \
  "${#collector_gpus[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

root = Path(sys.argv[1])
output = Path(sys.argv[2])
parallel_gpu_count = int(sys.argv[3])
scene = json.loads((root / "scene_info.json").read_text(encoding="utf-8"))
expected = 128
seeds = set()
geometries = set()
camera_extrinsics = set()
first_images = set()
frame_counts = []
video_sizes = []
shards = set()
for episode_id in range(expected):
    metadata = scene[f"episode_{episode_id}"]["panthera_episode"]
    if metadata.get("schema_version") != 4:
        raise SystemExit(f"episode {episode_id} is not schema v4")
    if metadata.get("scene_profile") != "phone_srt_vertical_socket":
        raise SystemExit(f"episode {episode_id} scene profile mismatch")
    if metadata.get("robot_count") != 1 or metadata.get("action_dimension") != 7:
        raise SystemExit(f"episode {episode_id} is not one-Panthera 7-D")
    if metadata.get("attach_on_grasp"):
        raise SystemExit(f"episode {episode_id} used forbidden attachment")
    seeds.add(int(metadata["episode_seed"]))
    shards.add(int(metadata["collection_shard"]))
    geometry = metadata["realized_geometry"]
    geometries.add(tuple(np.round(
        geometry["cylinder_initial_xy_m"] + geometry["groove_target_xy_m"], 8
    )))
    hdf5_path = root / "data" / f"episode{episode_id}.hdf5"
    video_path = root / "video" / f"episode{episode_id}.mp4"
    instruction_path = root / "instructions" / f"episode{episode_id}.json"
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise SystemExit(f"episode {episode_id} video is missing")
    if not instruction_path.is_file() or instruction_path.stat().st_size == 0:
        raise SystemExit(f"episode {episode_id} instruction is missing")
    with h5py.File(hdf5_path, "r") as episode:
        action = np.asarray(episode["joint_action/vector"])
        state = np.asarray(episode["observation/robot_state/vector"])
        if action.ndim != 2 or action.shape[1] != 7 or state.shape != action.shape:
            raise SystemExit(f"episode {episode_id} state/action shape mismatch")
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(state)):
            raise SystemExit(f"episode {episode_id} contains nonfinite values")
        camera = np.asarray(
            episode["observation/head_camera/cam2world_gl"][0], dtype=float
        )
        image = bytes(episode["observation/head_camera/rgb"][0])
        camera_extrinsics.add(tuple(np.round(camera.reshape(-1), 6)))
        first_images.add(hashlib.sha256(image).hexdigest())
        frame_counts.append(len(action))
    video_sizes.append(video_path.stat().st_size)
if len(scene) != expected or len(seeds) != expected:
    raise SystemExit("merged episode/seed count mismatch")
if shards != {0, 1, 2, 3}:
    raise SystemExit(f"expected four collection shards, got {shards}")
if len(geometries) != expected:
    raise SystemExit("realized geometry is not unique")
if len(camera_extrinsics) < expected // 2 or len(first_images) < expected // 2:
    raise SystemExit("camera or appearance randomization is insufficient")
summary = {
    "status": "contract_passed",
    "dataset": "panthera_phone_vertical_sft_v1",
    "task": "place_vertical_cylinder_in_groove",
    "scene_profile": "phone_srt_vertical_socket",
    "task_schema_version": 4,
    "robot_count": 1,
    "action_dimension": 7,
    "episodes": expected,
    "collection_shards": 4,
    "episodes_per_shard": 32,
    "parallel_gpu_count": parallel_gpu_count,
    "unique_seeds": len(seeds),
    "unique_geometries": len(geometries),
    "unique_camera_extrinsics": len(camera_extrinsics),
    "unique_first_images": len(first_images),
    "raw_frame_count_min": min(frame_counts),
    "raw_frame_count_max": max(frame_counts),
    "video_count": len(video_sizes),
    "video_bytes_min": min(video_sizes),
    "wall_clock_sleep_in_success_path": False,
}
output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

CUDA_VISIBLE_DEVICES="${collector_gpus[0]}" python \
  "$replay_script" \
  --robotwin-root "$robotwin_root" \
  --dataset-root "$dataset_root" \
  --task-name "$task_name" \
  --task-config "${task_config}.yml" \
  --summary "${state_root}/replay-summary.json" \
  --episode 0 \
  --episode 112
python - "${state_root}/dataset-summary.json" \
  "${state_root}/replay-summary.json" <<'PY'
import json
from pathlib import Path
import sys

dataset_path = Path(sys.argv[1])
replay = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
if replay.get("status") != "passed" or replay.get("passed") != 2:
    raise SystemExit("phone-SRT no-attachment replay failed")
dataset["status"] = "passed"
dataset["no_attachment_replay"] = replay
dataset_path.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
PY

if pgrep -af 'ray::|raylet|collect_data.py|eval_policy.py' | grep -v grep; then
  echo "错误：phone-SRT 数据验收后仍有 RoboTwin/Ray 进程。" >&2
  exit 1
fi
for gpu in "${collector_gpus[@]}"; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "错误：数据验收后 GPU ${gpu} 仍有计算进程。" >&2
    exit 1
  fi
done
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/dataset.ok"
echo "phone-SRT 对齐版 128 集数据集通过（并行 GPU 数：${#collector_gpus[@]}）：${dataset_root}"
