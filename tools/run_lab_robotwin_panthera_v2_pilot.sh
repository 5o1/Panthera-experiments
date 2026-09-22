#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
# The collector used to patch and install straight into the upstream checkout,
# which left it with 424 untracked files and 9 modified ones and no way to say
# which change was ours.  Upstream is now read-only under externals/ and this
# runs against a runtime assembled from it plus the overlay plus the patches.
upstream_root="${workspace}/externals/RoboTwin"
robotwin_root="${workspace}/runtime/robotwin"
overlay_root="${workspace}/overlays/robotwin"
dataset_tier="${PANTHERA_V2_DATASET_TIER:-pilot}"
case "$dataset_tier" in
  pilot)
    dataset_name="panthera_phone_cylinder_socket_v2_single_grasp_pilot"
    upright_shards=4
    upright_episodes_per_shard=8
    lying_episodes_per_shard=4
    expected_episodes=64
    minimum_side_count=24
    lying_shards_per_bin=1
    human_review_tier=false
    collector_gpu_default="1 2 3"
    ;;
  formal)
    dataset_name="panthera_phone_cylinder_socket_v2_single_grasp_sft_v1"
    upright_shards=8
    upright_episodes_per_shard=8
    lying_episodes_per_shard=8
    expected_episodes=128
    minimum_side_count=48
    lying_shards_per_bin=1
    human_review_tier=false
    collector_gpu_default="1 2 3"
    ;;
  expanded)
    dataset_name="panthera_phone_cylinder_socket_v2_single_grasp_sft_v2"
    upright_shards=80
    upright_episodes_per_shard=8
    lying_shards_per_bin=10
    lying_episodes_per_shard=8
    expected_episodes=1280
    minimum_side_count=480
    human_review_tier=true
    collector_gpu_default="0 1 2 3"
    ;;
  # The smallest run the shard structure allows: one upright shard plus the
  # eight lying angle bins that the collector always splits by.  It exists for
  # gate 2, which needs one trajectory whose recorded frames were drawn by the
  # appearance pinning in docs/20 -- the 1280-episode dataset predates that fix
  # and its frames cannot be reproduced by the renderer.  Not a training set.
  gate)
    dataset_name="panthera_phone_cylinder_socket_v2_gate_appearance_pinned"
    upright_shards=1
    upright_episodes_per_shard=2
    lying_shards_per_bin=1
    lying_episodes_per_shard=1
    expected_episodes=10
    minimum_side_count=1
    human_review_tier=false
    collector_gpu_default="0 1 2 3"
    ;;
  # Same 1280-episode contract as expanded, but every episode reuses one fixed
  # front-on camera pose instead of a per-episode sampled pose.
  expanded_fixed)
    dataset_name="panthera_phone_cylinder_socket_v2_single_grasp_sft_v2_fixedcam"
    upright_shards=80
    upright_episodes_per_shard=8
    lying_shards_per_bin=10
    lying_episodes_per_shard=8
    expected_episodes=1280
    minimum_side_count=480
    human_review_tier=true
    collector_gpu_default="0 1 2 3"
    ;;
  *)
    echo "错误：PANTHERA_V2_DATASET_TIER 必须是 gate、pilot、formal、expanded 或 expanded_fixed。" >&2
    exit 1
    ;;
esac
shard_count=$((upright_shards + 8 * lying_shards_per_bin))
state_root="${workspace}/state/panthera-v2-single-grasp-${dataset_tier}-state"
shard_parent="${workspace}/datasets/shards/data_phone_v2_single_grasp_${dataset_tier}_shards/place_randomized_cylinder_in_socket"
dataset_root="${workspace}/data/place_randomized_cylinder_in_socket/${dataset_name}"
review_root="${workspace}/reports/dataset-review/${dataset_name}"
review_video="${review_root}/balanced-stratified-10min-2x-1024x768.mp4"
template="${overlay_root}/task_config/panthera_phone_cylinder_socket_v2_pilot.yml"
task_name="place_randomized_cylinder_in_socket"
default_collector_gpus="$collector_gpu_default"
read -r -a collector_gpus <<<"${PANTHERA_V2_COLLECTOR_GPUS:-${PANTHERA_V2_PILOT_GPUS:-$default_collector_gpus}}"
if [[ "$dataset_tier" == "pilot" ]]; then
  collectors_per_gpu="${PANTHERA_V2_COLLECTORS_PER_GPU:-1}"
else
  collectors_per_gpu="${PANTHERA_V2_COLLECTORS_PER_GPU:-3}"
fi
shard_max_attempts="${PANTHERA_V2_SHARD_MAX_ATTEMPTS:-3}"
completion_marker="${state_root}/dataset.ok"
[[ "$human_review_tier" == "true" ]] && completion_marker="${state_root}/automated-audit.ok"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：禁止使用 root 运行 v2 pilot。" >&2
  exit 1
fi
for command_name in ffmpeg ffprobe flock git nvidia-smi timeout; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令 ${command_name}。" >&2
    exit 1
  }
done
if (( ${#collector_gpus[@]} == 0 )); then
  echo "错误：GPU 列表不能为空。" >&2
  exit 1
fi
if [[ ! "$collectors_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：PANTHERA_V2_COLLECTORS_PER_GPU 必须是正整数。" >&2
  exit 1
fi
if [[ ! "$shard_max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：PANTHERA_V2_SHARD_MAX_ATTEMPTS 必须是正整数。" >&2
  exit 1
fi
for gpu in "${collector_gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "错误：GPU 编号必须是非负整数：${gpu}。" >&2
    exit 1
  fi
done
for required in \
  "${workspace}/tools/activate_lab_vla.sh" \
  "$template" \
  "${workspace}/pipelines/assemble_runtime.py" \
  "${overlay_root}/envs/panthera_camera_randomization.py" \
  "${overlay_root}/envs/utils/async_episode_writer.py" \
  "${overlay_root}/envs/panthera_v2_sampling.py" \
  "${overlay_root}/envs/place_randomized_cylinder_in_socket.py" \
  "${overlay_root}/envs/place_vertical_cylinder_in_groove.py" \
  "${workspace}/packages/panthera_sim/build_panthera_phone_embodiment.py" \
  "${overlay_root}/description/task_instruction/${task_name}.json"; do
  [[ -s "$required" ]] || { echo "错误：缺少 ${required}" >&2; exit 1; }
done

mkdir -p "$state_root" "$review_root"
exec 9>"${state_root}/pilot.lock"
flock -n 9 || { echo "错误：另一个 v2 pilot 正在运行。" >&2; exit 1; }
if [[ -f "$completion_marker" && -s "${state_root}/dataset-summary.json" && -s "$review_video" ]]; then
  python3 -m json.tool "${state_root}/dataset-summary.json"
  echo "审查视频：${review_video}"
  exit 0
fi
# A changed camera contract must never inherit partially generated expanded
# shards from a previous definition.  The state path is versioned by tier and
# dataset name; completed shard reuse below additionally verifies metadata.

# shellcheck disable=SC1090
source "${workspace}/tools/activate_lab_vla.sh"
python3 "${workspace}/pipelines/assemble_runtime.py" \
  --upstream robotwin \
  --source "$upstream_root" \
  --runtime "$robotwin_root"

task_config_root="${robotwin_root}/env_cfg/task_config"
collector_source="${robotwin_root}/scripts/collect_data.py"
for marker in \
  'seed_start = int(args.get("seed_start", 0))' \
  'from envs.utils.async_episode_writer import' \
  'max_seed_attempts = int('; do
  if ! grep -q "$marker" "$collector_source"; then
    echo "错误：装配出的采集器缺少改动：${marker}" >&2
    exit 1
  fi
done

symmetric_embodiment="${robotwin_root}/assets/embodiments/panthera_phone_symmetric"
if [[ ! -s "${symmetric_embodiment}/config.yml" ]]; then
  python "${workspace}/packages/panthera_sim/build_panthera_phone_embodiment.py" \
    --source-root "${robotwin_root}/assets/embodiments/panthera_phone" \
    --output-root "$symmetric_embodiment" \
    --camera-x 0.0 \
    --profile phone_srt_symmetric_front_semicircle_v2
fi
python - "${task_config_root}/_embodiment_config.yml" "$symmetric_embodiment" <<'PY'
from pathlib import Path
import sys
import yaml

registry_path = Path(sys.argv[1])
embodiment_path = Path(sys.argv[2])
registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
expected = "./" + str(embodiment_path.relative_to(embodiment_path.parents[2]))
entry = registry.get("panthera_phone_symmetric")
if entry != {"file_path": expected}:
    registry["panthera_phone_symmetric"] = {"file_path": expected}
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
PY

python - "$template" "$task_config_root" "$dataset_tier" \
  "$upright_shards" "$upright_episodes_per_shard" \
  "$lying_shards_per_bin" "$lying_episodes_per_shard" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys
import yaml

template = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
output = Path(sys.argv[2])
dataset_tier = sys.argv[3]
upright_shards = int(sys.argv[4])
upright_episodes = int(sys.argv[5])
lying_shards_per_bin = int(sys.argv[6])
lying_episodes = int(sys.argv[7])
specs = []
seed_base = 10000000 if dataset_tier == "expanded_fixed" else 0
for shard in range(upright_shards):
    specs.append((shard, "upright", None, upright_episodes, seed_base + shard * 10000))
for angle_bin in range(8):
    for bin_shard in range(lying_shards_per_bin):
        shard = upright_shards + angle_bin * lying_shards_per_bin + bin_shard
        specs.append(
            (shard, "lying", angle_bin, lying_episodes, seed_base + shard * 10000)
        )
for shard, posture, angle_bin, episodes, seed_start in specs:
    config = deepcopy(template)
    config["episode_num"] = episodes
    config["seed_start"] = seed_start
    config["retry_delay_s"] = 0
    config["save_path"] = f"../data_phone_v2_single_grasp_{dataset_tier}_shards"
    config["task_randomization"]["forced_posture"] = posture
    if angle_bin is not None:
        config["task_randomization"]["forced_lying_angle_bin"] = angle_bin
    if dataset_tier == "expanded_fixed":
        # One front-on pose shared by every episode; see
        # docs/17_panthera_v2_expanded_dataset_pipeline_2026-09-17.md section 3.1.
        config["task_randomization"]["camera_randomization"].update(
            {
                "pose_mode": "fixed",
                "fixed_azimuth_deg": 90.0,
                "fixed_height_above_table_m": 0.60,
                "fixed_horizontal_distance_m": 0.55,
                "fixed_look_at_xyz_m": [0.0013, -0.0929, 0.8851],
                "minimum_horizontal_distance_m": 0.55,
                "target_xy_jitter_m": 0.0,
            }
        )
    name = f"panthera_v2_single_grasp_{dataset_tier}_shard{shard}.yml"
    (output / name).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
PY

for gpu in "${collector_gpus[@]}"; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "错误：GPU ${gpu} 正被占用。" >&2
    exit 1
  fi
done

declare -a child_pids=()
declare -a worker_gpus=()
for gpu in "${collector_gpus[@]}"; do
  for ((slot=0; slot<collectors_per_gpu; slot++)); do
    worker_gpus+=("$gpu")
  done
done
cleanup_children() {
  local pid
  for pid in "${child_pids[@]:-}"; do
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${child_pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup_children INT TERM EXIT

shard_complete() {
  local shard="$1"
  local expected="$lying_episodes_per_shard"
  (( shard < upright_shards )) && expected="$upright_episodes_per_shard"
  local root="${shard_parent}/panthera_v2_single_grasp_${dataset_tier}_shard${shard}"
  [[ -s "${root}/scene_info.json" && -s "${root}/seed.txt" ]] || return 1
  [[ $(find "${root}/data" -maxdepth 1 -name 'episode*.hdf5' 2>/dev/null | wc -l) -eq "$expected" ]] || return 1
  [[ $(find "${root}/video" -maxdepth 1 -name 'episode*.mp4' 2>/dev/null | wc -l) -eq "$expected" ]] || return 1
  if [[ "$human_review_tier" == "true" ]]; then
    [[ $(find "${root}/.episode_commits" -maxdepth 1 -name 'episode*.json' 2>/dev/null | wc -l) -eq "$expected" ]] || return 1
    python - "$root" "$expected" <<'PY' >/dev/null || return 1
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = int(sys.argv[2])
scene = json.loads((root / "scene_info.json").read_text(encoding="utf-8"))
if len(scene) != expected:
    raise SystemExit(1)
for episode_id in range(expected):
    camera = scene[f"episode_{episode_id}"]["panthera_episode"].get("camera_randomization", {})
    if (
        camera.get("enabled") is not True
        or camera.get("schema_version") != 1
        or camera.get("rendered_visibility", {}).get("passed") is not True
    ):
        raise SystemExit(1)
PY
  fi
}

gpu_count=${#worker_gpus[@]}
for ((batch=0; batch<shard_count; batch+=gpu_count)); do
  batch_complete=0
  for ((attempt=1; attempt<=shard_max_attempts; attempt++)); do
    child_pids=()
    for ((offset=0; offset<gpu_count && batch+offset<shard_count; offset++)); do
      shard=$((batch + offset))
      gpu="${worker_gpus[$offset]}"
      if shard_complete "$shard"; then
        if (( attempt == 1 )); then
          echo "复用已完成分片 ${shard}。"
        fi
        continue
      fi
      mkdir -p "${state_root}/shard-${shard}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
        cd "$robotwin_root"
        timeout --signal=INT --kill-after=90s \
          "${PANTHERA_V2_SHARD_TIMEOUT:-4h}" \
          python scripts/collect_data.py "$task_name" \
            "panthera_v2_single_grasp_${dataset_tier}_shard${shard}" \
          >"${state_root}/shard-${shard}/collector.log" 2>&1
      ) &
      child_pids+=("$!")
      echo "启动分片 ${shard}：GPU ${gpu}，尝试 ${attempt}。"
    done
    for pid in "${child_pids[@]}"; do
      wait "$pid" || true
    done
    batch_complete=1
    for ((offset=0; offset<gpu_count && batch+offset<shard_count; offset++)); do
      shard=$((batch + offset))
      shard_complete "$shard" || batch_complete=0
    done
    (( batch_complete == 1 )) && break
  done
  (( batch_complete == 1 )) || {
    echo "错误：分片批次 ${batch} 在重试后仍不完整。" >&2
    exit 1
  }
done
trap - INT TERM EXIT

python - "$shard_parent" "$dataset_root" "$dataset_tier" \
  "$upright_shards" "$upright_episodes_per_shard" \
  "$lying_shards_per_bin" "$lying_episodes_per_shard" \
  "$expected_episodes" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

source_root = Path(sys.argv[1])
destination = Path(sys.argv[2])
dataset_tier = sys.argv[3]
upright_shards = int(sys.argv[4])
upright_episodes = int(sys.argv[5])
lying_shards_per_bin = int(sys.argv[6])
lying_episodes = int(sys.argv[7])
expected_total = int(sys.argv[8])
expected_per_shard = (
    [upright_episodes] * upright_shards
    + [lying_episodes] * (8 * lying_shards_per_bin)
)
if destination.exists():
    if len(list((destination / "data").glob("episode*.hdf5"))) == expected_total:
        print(f"复用已合并数据集：{destination}")
        raise SystemExit(0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    destination.rename(destination.with_name(destination.name + f".incomplete-{stamp}"))
staging = destination.with_name(destination.name + ".building")
if staging.exists():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    staging.rename(staging.with_name(staging.name + f".failed-{stamp}"))
for folder in ("data", "video", "instructions"):
    (staging / folder).mkdir(parents=True, exist_ok=True)

merged = {}
all_seeds = []
global_id = 0
for shard, expected in enumerate(expected_per_shard):
    root = source_root / f"panthera_v2_single_grasp_{dataset_tier}_shard{shard}"
    scene = json.loads((root / "scene_info.json").read_text(encoding="utf-8"))
    seeds = [int(value) for value in (root / "seed.txt").read_text().split()]
    if len(scene) != expected or len(seeds) != expected:
        raise SystemExit(f"分片 {shard} 不完整")
    all_seeds.extend(seeds)
    for local_id in range(expected):
        metadata = scene[f"episode_{local_id}"]
        metadata["panthera_episode"]["collection_shard"] = shard
        metadata["panthera_episode"]["shard_episode_id"] = local_id
        merged[f"episode_{global_id}"] = metadata
        for folder, suffix in (("data", "hdf5"), ("video", "mp4"), ("instructions", "json")):
            source = root / folder / f"episode{local_id}.{suffix}"
            target = staging / folder / f"episode{global_id}.{suffix}"
            if not source.is_file() or source.stat().st_size == 0:
                raise SystemExit(f"缺少分片产物：{source}")
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        global_id += 1
if global_id != expected_total or len(set(all_seeds)) != expected_total:
    raise SystemExit("合并后的 episode 或 seed 数量错误")
(staging / "scene_info.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
(staging / "seed.txt").write_text(" ".join(map(str, all_seeds)) + "\n", encoding="utf-8")
destination.parent.mkdir(parents=True, exist_ok=True)
staging.rename(destination)
PY

python - "$dataset_root" "${state_root}/dataset-summary.json" \
  "$dataset_name" "$expected_episodes" "$minimum_side_count" \
  "$state_root" "${#worker_gpus[@]}" "${#collector_gpus[@]}" \
  "$dataset_tier" "$human_review_tier" <<'PY'
import json
import math
from pathlib import Path
import re
import sys

import h5py
import numpy as np

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
dataset_name = sys.argv[3]
expected = int(sys.argv[4])
minimum_side_count = int(sys.argv[5])
state_root = Path(sys.argv[6])
parallel_collectors = int(sys.argv[7])
physical_gpu_count = int(sys.argv[8])
dataset_tier = sys.argv[9]
human_review_tier = sys.argv[10] == "true"
scene = json.loads((root / "scene_info.json").read_text(encoding="utf-8"))
postures = {"upright": 0, "lying": 0}
angle_bins = {str(index): 0 for index in range(8)}
sides = {
    "cylinder_left": 0,
    "cylinder_right": 0,
    "socket_left": 0,
    "socket_right": 0,
}
cylinder_cells = set()
socket_cells = set()
frame_counts = []
planned_near_zero_fractions = []
actual_near_zero_fractions = []
actual_longest_near_zero_runs_s = []
actual_near_zero_runs_over_80ms = []
retime_peak_velocities = []
retime_peak_accelerations = []
actual_jump_ratios = []
actual_jump_violations = []
actual_jump_pairs = []
# Empirical: the rejected schema-9 pilot peaks at 17.4x this budget while both
# schema-10 1280-episode sets sit at 1.04x, which is controller tracking rather
# than a planning defect. 1.5x separates those by an order of magnitude.
actual_velocity_jump_gate_ratio = 1.5
camera_azimuths_deg = []
camera_heights_above_table_m = []
camera_downward_angles_deg = []
camera_minimum_central_margins = []
camera_positions = set()
camera_visible_pixels = {"robot": [], "cylinder": [], "socket": []}
seeds = set()
for episode_id in range(expected):
    metadata = scene[f"episode_{episode_id}"]["panthera_episode"]
    if metadata["schema_version"] != 10 or metadata["robot_count"] != 1:
        raise SystemExit(f"episode {episode_id} contract mismatch")
    if metadata["action_dimension"] != 7 or metadata["attach_on_grasp"]:
        raise SystemExit(f"episode {episode_id} action/attachment mismatch")
    if metadata.get("motion_profile") != "chaikin_arc_length_quintic_single_grasp_v3":
        raise SystemExit(f"episode {episode_id} motion profile mismatch")
    if metadata.get("single_grasp") is not True:
        raise SystemExit(f"episode {episode_id} is not a single-grasp trajectory")
    release_clearance = float(metadata["direct_release_bottom_clearance_m"])
    lowest_clearance = float(metadata["direct_release_lowest_validated_clearance_m"])
    clearance_margin = float(metadata["direct_release_clearance_margin_m"])
    if not math.isclose(
        release_clearance - lowest_clearance,
        clearance_margin,
        abs_tol=1.0e-9,
    ):
        raise SystemExit(f"episode {episode_id} release clearance mismatch")
    motion_audit = metadata.get("continuous_motion_audit", [])
    if len(motion_audit) != 1:
        raise SystemExit(
            f"episode {episode_id} must contain one episode-local motion audit"
        )
    fractions = [
        float(item["interior_near_zero_speed_fraction"])
        for item in motion_audit
    ]
    if max(fractions) > 0.02:
        raise SystemExit(f"episode {episode_id} contains internal stop points")
    planned_near_zero_fractions.extend(fractions)
    retime_audit = metadata.get("joint_retime_audit", [])
    if len(retime_audit) != 1:
        raise SystemExit(f"episode {episode_id} lacks one retiming audit")
    retime = retime_audit[0]
    retime_peak_velocities.append(float(retime["peak_joint_velocity_radps"]))
    retime_peak_accelerations.append(
        float(retime["peak_joint_acceleration_radps2"])
    )
    if (
        float(retime["peak_joint_velocity_radps"]) > 0.601
        or float(retime["peak_joint_acceleration_radps2"]) > 2.002
        or float(retime["longest_internal_near_zero_run_s"]) >= 0.08
    ):
        raise SystemExit(f"episode {episode_id} retiming contract mismatch")
    settle_audit = metadata.get("motion_settle_audit", [])
    if not settle_audit or not all(bool(item["settled"]) for item in settle_audit):
        raise SystemExit(f"episode {episode_id} contains an unsettled motion stage")
    geometry = metadata["realized_geometry"]
    camera = metadata.get("camera_randomization", {})
    if camera.get("enabled") is not True or camera.get("schema_version") != 1:
        raise SystemExit(f"episode {episode_id} lacks constrained camera metadata")
    rendered = camera.get("rendered_visibility", {})
    if rendered.get("passed") is not True or rendered.get("schema_version") != 1:
        raise SystemExit(f"episode {episode_id} failed rendered visibility gate")
    for group_name in camera_visible_pixels:
        group = rendered.get("groups", {}).get(group_name, {})
        if (
            group.get("passed") is not True
            or group.get("all_visible_pixels_in_central_region") is not True
        ):
            raise SystemExit(
                f"episode {episode_id} {group_name} is not visibly centered"
            )
        camera_visible_pixels[group_name].append(int(group["visible_pixels"]))
    constraints = camera["constraints"]
    if float(camera["height_above_table_m"]) + 1.0e-9 < float(
        constraints["minimum_height_above_table_m"]
    ):
        raise SystemExit(f"episode {episode_id} camera is too low")
    if float(camera["downward_angle_deg"]) + 1.0e-9 < float(
        constraints["minimum_downward_angle_deg"]
    ):
        raise SystemExit(f"episode {episode_id} camera is not overhead")
    names = list(camera["keypoint_names"])
    if not any(name.startswith("robot_") for name in names):
        raise SystemExit(f"episode {episode_id} camera lacks robot keypoints")
    if not any(name.startswith("cylinder_") for name in names):
        raise SystemExit(f"episode {episode_id} camera lacks cylinder keypoints")
    if not any(name.startswith("socket_") for name in names):
        raise SystemExit(f"episode {episode_id} camera lacks socket keypoints")
    points = np.asarray(camera["keypoints_world_xyz_m"], dtype=float)
    position = np.asarray(camera["position_xyz_m"], dtype=float)
    forward = np.asarray(camera["forward_xyz"], dtype=float)
    left = np.asarray(camera["left_xyz"], dtype=float)
    up = np.asarray(camera["up_xyz"], dtype=float)
    if (
        points.shape != (len(names), 3)
        or position.shape != (3,)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(position))
    ):
        raise SystemExit(f"episode {episode_id} camera values are invalid")
    rotation = np.column_stack((forward, left, up))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise SystemExit(f"episode {episode_id} camera frame is not orthonormal")
    relative = points - position
    depth = relative @ forward
    tan_y = math.tan(math.radians(float(camera["vertical_fov_deg"])) / 2.0)
    tan_x = tan_y * float(camera["image_width_px"]) / float(camera["image_height_px"])
    recomputed_uv = np.column_stack((
        (1.0 - (relative @ left) / (depth * tan_x)) / 2.0,
        (1.0 - (relative @ up) / (depth * tan_y)) / 2.0,
    ))
    recorded_uv = np.asarray(camera["keypoints_uv"], dtype=float)
    if np.any(depth <= 0.10) or not np.allclose(recomputed_uv, recorded_uv, atol=1.0e-8):
        raise SystemExit(f"episode {episode_id} camera projection mismatch")
    lower_u, lower_v, upper_u, upper_v = map(float, camera["central_region_uv"])
    if (
        np.any(recomputed_uv[:, 0] < lower_u - 1.0e-9)
        or np.any(recomputed_uv[:, 0] > upper_u + 1.0e-9)
        or np.any(recomputed_uv[:, 1] < lower_v - 1.0e-9)
        or np.any(recomputed_uv[:, 1] > upper_v + 1.0e-9)
    ):
        raise SystemExit(f"episode {episode_id} keypoint leaves central image region")
    camera_azimuths_deg.append(float(camera["azimuth_deg"]))
    camera_heights_above_table_m.append(float(camera["height_above_table_m"]))
    camera_downward_angles_deg.append(float(camera["downward_angle_deg"]))
    camera_minimum_central_margins.append(
        float(camera["minimum_central_margin_fraction"])
    )
    camera_positions.add(tuple(np.round(position, 4)))
    posture = metadata["cylinder_posture"]
    postures[posture] += 1
    if posture == "lying":
        angle_bins[str(geometry["lying_angle_bin"])] += 1
    cylinder_cells.add((geometry["cylinder_radial_bin"], geometry["cylinder_angular_bin"]))
    socket_cells.add((geometry["socket_radial_bin"], geometry["socket_angular_bin"]))
    workspace = geometry["workspace"]
    base = np.array([workspace["robot_base_x_m"], workspace["robot_base_y_m"]])
    sides["cylinder_left" if geometry["cylinder_initial_xy_m"][0] < base[0] else "cylinder_right"] += 1
    sides["socket_left" if geometry["socket_target_xy_m"][0] < base[0] else "socket_right"] += 1
    for key in ("cylinder_initial_xy_m", "socket_target_xy_m"):
        radius = float(np.linalg.norm(np.asarray(geometry[key]) - base))
        if radius > workspace["maximum_radius_m"] + 1e-9:
            raise SystemExit(f"episode {episode_id} exceeds 75% workspace")
    if not math.isclose(workspace["reach_fraction"], 0.75):
        raise SystemExit("workspace fraction mismatch")
    seeds.add(int(metadata["episode_seed"]))
    with h5py.File(root / "data" / f"episode{episode_id}.hdf5", "r") as episode:
        actions = np.asarray(episode["joint_action/vector"])
        states = np.asarray(episode["observation/robot_state/vector"])
        if actions.ndim != 2 or actions.shape[1] != 7 or states.shape != actions.shape:
            raise SystemExit(f"episode {episode_id} HDF5 shape mismatch")
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(states)):
            raise SystemExit(f"episode {episode_id} contains nonfinite values")
        frame_counts.append(len(actions))
        arm_qpos = np.asarray(episode["observation/robot_state/arm_qpos"])
        step_index = np.asarray(episode["timing/simulation_step_index"])
        simulation_time = np.asarray(episode["timing/simulation_time_s"])
    route_audit = motion_audit[0]
    route_mask = (
        (step_index >= int(route_audit["start_simulation_step"]))
        & (step_index <= int(route_audit["end_simulation_step"]))
    )
    route_qpos = arm_qpos[route_mask]
    route_time = simulation_time[route_mask]
    if len(route_qpos) < 3:
        raise SystemExit(f"episode {episode_id} route has too few recorded samples")
    sample_dt = np.diff(route_time)
    sample_dq = np.linalg.norm(np.diff(route_qpos, axis=0), axis=1)
    sample_mid_time = (route_time[:-1] + route_time[1:]) / 2.0
    route_phase = (
        (sample_mid_time - route_time[0])
        / (route_time[-1] - route_time[0])
    )
    geometric_progress = (
        10.0 * route_phase**3
        - 15.0 * route_phase**4
        + 6.0 * route_phase**5
    )
    interior_mask = (
        (sample_dt > 1.0e-9)
        & (geometric_progress >= 0.05)
        & (geometric_progress <= 0.95)
    )
    if np.count_nonzero(interior_mask) < 2:
        raise SystemExit(f"episode {episode_id} has too few interior samples")
    interior_speed = sample_dq[interior_mask] / sample_dt[interior_mask]
    interior_dt = sample_dt[interior_mask]
    low_speed = interior_speed < 0.02
    run_durations = []
    run_duration = 0.0
    for is_low, duration in zip(low_speed, interior_dt):
        if is_low:
            run_duration += float(duration)
        elif run_duration:
            run_durations.append(run_duration)
            run_duration = 0.0
    if run_duration:
        run_durations.append(run_duration)
    actual_fraction = float(np.mean(low_speed))
    longest_run = max(run_durations, default=0.0)
    runs_over_80ms = sum(duration >= 0.08 for duration in run_durations)
    if longest_run >= 0.08:
        raise SystemExit(
            f"episode {episode_id} actual qpos contains a {longest_run:.3f}s stop"
        )
    actual_near_zero_fractions.append(actual_fraction)
    actual_longest_near_zero_runs_s.append(longest_run)
    actual_near_zero_runs_over_80ms.append(runs_over_80ms)
    # The stall check above catches a route that stops; it cannot catch a route
    # whose velocity jumps. Bounded acceleration allows at most a_limit * dt of
    # velocity change per sample, so the ratio to that budget is the
    # discontinuity measured in units of the retiming contract. Only regularly
    # spaced pairs count: the recorder inserts diagnostic frames at stage
    # boundaries, and a difference taken across an irregular gap measures the
    # sampling rather than the trajectory.
    safe_dt = np.where(sample_dt > 1.0e-9, sample_dt, np.nan)
    sample_velocity = np.diff(route_qpos, axis=0) / safe_dt[:, None]
    nominal_dt = float(np.median(sample_dt[interior_mask]))
    regular_mask = np.abs(sample_dt - nominal_dt) <= 0.25 * nominal_dt
    jump_mask = (
        interior_mask[:-1]
        & interior_mask[1:]
        & regular_mask[:-1]
        & regular_mask[1:]
    )
    if not np.any(jump_mask):
        raise SystemExit(
            f"episode {episode_id} has no regularly spaced interior sample pair"
        )
    jump_ratio = np.abs(np.diff(sample_velocity, axis=0)) / np.maximum(
        2.002 * sample_dt[1:][:, None], 1.0e-12
    )
    episode_jump_ratio = float(jump_ratio[jump_mask].max())
    if episode_jump_ratio > actual_velocity_jump_gate_ratio:
        raise SystemExit(
            f"episode {episode_id} actual qpos velocity jump is "
            f"{episode_jump_ratio:.3f}x the acceleration budget"
        )
    actual_jump_ratios.append(episode_jump_ratio)
    actual_jump_violations.append(int(np.count_nonzero(jump_ratio[jump_mask] > 1.0)))
    actual_jump_pairs.append(int(np.count_nonzero(jump_mask)))
if postures != {"upright": expected // 2, "lying": expected // 2}:
    raise SystemExit(f"posture balance mismatch: {postures}")
if set(angle_bins.values()) != {expected // 16}:
    raise SystemExit(f"lying angle balance mismatch: {angle_bins}")
if len(seeds) != expected:
    raise SystemExit("seed uniqueness mismatch")
if min(sides.values()) < minimum_side_count:
    raise SystemExit(f"left/right balance mismatch: {sides}")
if len(cylinder_cells) < 24 or len(socket_cells) < 24:
    raise SystemExit(
        f"workspace coverage too low: cylinder={len(cylinder_cells)}, socket={len(socket_cells)}"
    )
if dataset_tier == "expanded_fixed":
    # 固定机位 tier 的正确契约与随机机位相反：必须恰好只有一个机位。
    if len(camera_positions) != 1:
        raise SystemExit(
            f"fixed camera tier must share exactly one pose: {len(camera_positions)}"
        )
elif expected >= 1280:
    if len(camera_positions) < 1000:
        raise SystemExit(f"camera pose diversity too low: {len(camera_positions)}")
    if max(camera_azimuths_deg) - min(camera_azimuths_deg) < 100.0:
        raise SystemExit("camera azimuth coverage is too narrow")
    if max(camera_heights_above_table_m) - min(camera_heights_above_table_m) < 0.30:
        raise SystemExit("camera height coverage is too narrow")
rejected_candidates = 0
collector_final_seed_values = []
ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
for log_path in sorted(state_root.glob("shard-*/collector.log")):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    text = ansi_escape.sub("", text)
    matches = re.findall(r"failed\s+(\d+)\s+times\s+/\s+(\d+)\s+tries", text)
    if len(matches) != 1:
        raise SystemExit(f"cannot audit collector candidate count: {log_path}")
    failed, tries = map(int, matches[0])
    rejected_candidates += failed
    collector_final_seed_values.append(tries)
summary = {
    "status": f"{dataset_tier}_contract_passed",
    "approval_status": (
        "awaiting_human_review" if human_review_tier else "automated_gate_passed"
    ),
    "schema_version": 10,
    "dataset_tier": dataset_tier,
    "dataset": dataset_name,
    "episodes": expected,
    "collector_candidates": expected + rejected_candidates,
    "collector_rejected_candidates": rejected_candidates,
    "collector_final_seed_values": collector_final_seed_values,
    "parallel_collectors": parallel_collectors,
    "physical_gpu_count": physical_gpu_count,
    "postures": postures,
    "lying_angle_bins": angle_bins,
    "left_right_counts": sides,
    "unique_cylinder_workspace_cells": len(cylinder_cells),
    "unique_socket_workspace_cells": len(socket_cells),
    "unique_camera_positions_rounded_1e4_m": len(camera_positions),
    "camera_azimuth_deg_min": min(camera_azimuths_deg),
    "camera_azimuth_deg_max": max(camera_azimuths_deg),
    "camera_height_above_table_m_min": min(camera_heights_above_table_m),
    "camera_height_above_table_m_max": max(camera_heights_above_table_m),
    "camera_downward_angle_deg_min": min(camera_downward_angles_deg),
    "camera_downward_angle_deg_max": max(camera_downward_angles_deg),
    "camera_minimum_central_margin_fraction": min(camera_minimum_central_margins),
    "camera_minimum_visible_pixels": {
        name: min(values) for name, values in camera_visible_pixels.items()
    },
    "camera_visibility_contract": "robot+cylinder+socket keypoints inside central 80% image region",
    "raw_frame_count_min": min(frame_counts),
    "raw_frame_count_max": max(frame_counts),
    "maximum_planned_near_zero_speed_fraction": max(planned_near_zero_fractions),
    "maximum_actual_near_zero_speed_fraction": max(actual_near_zero_fractions),
    "maximum_actual_near_zero_run_s": max(actual_longest_near_zero_runs_s),
    "actual_near_zero_runs_over_80ms": sum(actual_near_zero_runs_over_80ms),
    "maximum_actual_velocity_jump_over_budget_ratio": max(actual_jump_ratios),
    "actual_velocity_jump_gate_ratio": actual_velocity_jump_gate_ratio,
    "actual_velocity_jump_violations": sum(actual_jump_violations),
    "actual_velocity_jump_evaluated_pairs": sum(actual_jump_pairs),
    "actual_motion_interior_definition": (
        "qpos speed below 0.02 rad/s within quintic geometric progress 0.05..0.95"
    ),
    "motion_profile": "chaikin_arc_length_quintic_single_grasp_v3",
    "single_grasp": True,
    "direct_release_bottom_clearance_m": -0.014,
    "direct_release_clearance_margin_m": 0.001,
    "continuous_speed_scale": 0.6,
    "nominal_joint_acceleration_limit_radps2": 2.0,
    "joint_acceleration_audit_tolerance_radps2": 0.002,
    "maximum_retime_joint_velocity_radps": max(retime_peak_velocities),
    "maximum_retime_joint_acceleration_radps2": max(retime_peak_accelerations),
    "wall_clock_sleep_in_success_path": False,
    "async_episode_writer": {
        "enabled": True,
        "workers_per_collector": 1,
        "max_pending_episodes_per_collector": 2,
        "atomic_hdf5_video_commit": True,
    },
    "training_started": False,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

python - "$dataset_root" "${state_root}/review-selection-balanced-stratified.json" \
  "${state_root}/review-concat-balanced-stratified.txt" "$expected_episodes" <<'PY'
import json
from pathlib import Path
import random
import subprocess
import sys

root = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
concat_path = Path(sys.argv[3])
expected = int(sys.argv[4])
rng = random.Random(20260916)
scene = json.loads((root / "scene_info.json").read_text(encoding="utf-8"))
upright_by_side = {"left": [], "right": []}
lying_by_bin_side = {
    index: {"left": [], "right": []} for index in range(8)
}
durations = {}
episode_sides = {}
for episode_id in range(expected):
    path = root / "video" / f"episode{episode_id}.mp4"
    value = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True)
    durations[episode_id] = float(value.strip())
    metadata = scene[f"episode_{episode_id}"]["panthera_episode"]
    geometry = metadata["realized_geometry"]
    side = (
        "left"
        if geometry["cylinder_initial_xy_m"][0]
        < geometry["workspace"]["robot_base_x_m"]
        else "right"
    )
    episode_sides[episode_id] = side
    if metadata["cylinder_posture"] == "upright":
        upright_by_side[side].append(episode_id)
    else:
        angle_bin = int(geometry["lying_angle_bin"])
        lying_by_bin_side[angle_bin][side].append(episode_id)

selected_ids = []
selected_lying_sides = {"left": 0, "right": 0}
for index in range(8):
    available_sides = [
        side for side in ("left", "right") if lying_by_bin_side[index][side]
    ]
    if not available_sides:
        raise SystemExit(f"review lacks lying bin {index}")
    side = min(available_sides, key=lambda value: (selected_lying_sides[value], value))
    selected_ids.append(rng.choice(lying_by_bin_side[index][side]))
    selected_lying_sides[side] += 1
if min(selected_lying_sides.values()) < 3:
    raise SystemExit(f"review lying-side balance is too low: {selected_lying_sides}")
for side in ("left", "right"):
    if not upright_by_side[side]:
        raise SystemExit(f"review lacks upright sample on {side} side")
    selected_ids.append(rng.choice(upright_by_side[side]))
selected_side_counts = {
    side: sum(episode_sides[episode_id] == side for episode_id in selected_ids)
    for side in ("left", "right")
}
remaining_by_side = {
    side: [
        episode_id
        for episode_id in range(expected)
        if episode_id not in selected_ids and episode_sides[episode_id] == side
    ]
    for side in ("left", "right")
}
for candidates in remaining_by_side.values():
    rng.shuffle(candidates)
source_duration = sum(durations[episode_id] for episode_id in selected_ids)
while source_duration < 1200.0:
    available_sides = [
        side for side in ("left", "right") if remaining_by_side[side]
    ]
    if not available_sides:
        raise SystemExit("not enough video duration for review")
    side = min(available_sides, key=lambda value: (selected_side_counts[value], value))
    episode_id = remaining_by_side[side].pop()
    selected_ids.append(episode_id)
    selected_side_counts[side] += 1
    source_duration += durations[episode_id]
rng.shuffle(selected_ids)
selected = [
    {
        "episode": episode_id,
        "path": str(root / "video" / f"episode{episode_id}.mp4"),
        "duration_s": durations[episode_id],
        "posture": scene[f"episode_{episode_id}"]["panthera_episode"]["cylinder_posture"],
        "cylinder_side": (
            "left"
            if scene[f"episode_{episode_id}"]["panthera_episode"]["realized_geometry"]["cylinder_initial_xy_m"][0]
            < scene[f"episode_{episode_id}"]["panthera_episode"]["realized_geometry"]["workspace"]["robot_base_x_m"]
            else "right"
        ),
        "lying_angle_bin": scene[f"episode_{episode_id}"]["panthera_episode"]["realized_geometry"]["lying_angle_bin"],
    }
    for episode_id in selected_ids
]
selection_path.write_text(json.dumps({
    "selection_seed": 20260916,
    "selection": "one complete trajectory per lying-angle bin balanced across sides, one upright per side, then random fill",
    "playback_speed": 2.0,
    "target_output_duration_s": 600,
    "selected_source_duration_s": source_duration,
    "expected_output_duration_s": source_duration / 2.0,
    "episodes": selected,
}, indent=2) + "\n", encoding="utf-8")
concat_path.write_text("".join(f"file '{item['path']}'\n" for item in selected), encoding="utf-8")
PY

if [[ ! -s "$review_video" ]]; then
  ffmpeg -hide_banner -loglevel error -f concat -safe 0 \
    -i "${state_root}/review-concat-balanced-stratified.txt" \
    -vf "setpts=0.5*PTS,scale=1024:768:flags=lanczos,setsar=1" \
    -c:v libx264 -preset medium -crf 20 -an "$review_video"
fi
ffprobe -v error -show_entries stream=width,height:format=duration,size \
  -of json "$review_video" >"${state_root}/review-video-probe.json"
sha256sum "$review_video" >"${state_root}/review-video.sha256"

date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "$completion_marker"
if [[ "$human_review_tier" == "true" ]]; then
  cat >"${state_root}/awaiting-human-review.txt" <<EOF
自动生成、全量机器审计和 10 分钟分层审阅视频已完成。
当前数据集尚未获人工验收，不得自动转换 RLDS、训练或评测。
EOF
  echo "v2 expanded 数据集已通过自动审计，现停在人工验收门前；未启动训练。"
else
  echo "v2 ${dataset_tier} 数据集已通过；未启动训练。"
fi
echo "数据集：${dataset_root}"
echo "10 分钟随机审查视频：${review_video}"
