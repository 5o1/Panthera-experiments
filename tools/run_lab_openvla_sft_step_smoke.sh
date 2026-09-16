#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
activation_script="${workspace}/activate_lab_vla.sh"
adapter_root="${workspace}/panthera-openvla-adapter"
data_root="${PANTHERA_SFT_DATA_ROOT:-${workspace}/rlds_single_sft_v1}"
model_root="${PANTHERA_SFT_INITIAL_MODEL:-${workspace}/models/openvla-oft-place-empty-cup}"
openvla_constants_patch="${workspace}/patches/openvla_oft_panthera_constants.patch"
openvla_site_packages="${workspace}/RLinf/.venv/lib/python3.11/site-packages"
dataset_name="${PANTHERA_SFT_DATASET_NAME:-panthera_single_cylinder}"
dataset_version="${PANTHERA_SFT_DATASET_VERSION:-2.0.0}"
run_root="${PANTHERA_SFT_SMOKE_RUN_ROOT:-${workspace}/runs/panthera-single-openvla-sft-smoke}"
run_id="${PANTHERA_SFT_SMOKE_RUN_ID:-panthera-single-cylinder-sft-step-smoke}"
run_dir="${run_root}/${run_id}"
state_root="${PANTHERA_SFT_SMOKE_STATE_ROOT:-${workspace}/.panthera-single-openvla-sft-smoke-state}"
rlds_marker="${PANTHERA_SFT_RLDS_MARKER:-${workspace}/.panthera-single-sft-rlds-state/rlds.ok}"
smoke_gpu="${PANTHERA_SFT_GPU:-0}"
action_chunk="${PANTHERA_ACTION_CHUNK:-5}"
robot_platform="${PANTHERA_ROBOT_PLATFORM:-BRIDGE}"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
if [[ ! "$action_chunk" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：PANTHERA_ACTION_CHUNK 必须是正整数。" >&2
  exit 1
fi
for command_name in flock patch timeout nvidia-smi; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
for required in \
  "$activation_script" \
  "${adapter_root}/panthera_rlds.py" \
  "$openvla_constants_patch" \
  "${model_root}/config.json" \
  "${data_root}/${dataset_name}/${dataset_version}/dataset_info.json"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少前置文件：${required}" >&2
    exit 1
  fi
done
constants_source="${openvla_site_packages}/prismatic/vla/constants.py"
if grep -Fq 'PANTHERA_CONSTANTS = {' "$constants_source" \
  && grep -Fq "'PANTHERA': 'PANTHERA'" "$constants_source" \
  && grep -Fq 'elif ROBOT_PLATFORM == "PANTHERA":' "$constants_source"; then
  echo "OpenVLA-OFT Panthera 25x7 常量补丁已存在。"
elif patch --batch --silent --dry-run -p1 \
  -d "$openvla_site_packages" <"$openvla_constants_patch"; then
  patch --batch --silent -p1 -d "$openvla_site_packages" <"$openvla_constants_patch"
  echo "已应用 OpenVLA-OFT Panthera 25x7 常量补丁。"
else
  echo "错误：OpenVLA-OFT Panthera 25x7 常量补丁无法安全应用。" >&2
  exit 1
fi
if ! grep -Fq 'PANTHERA_CONSTANTS = {' "$constants_source" \
  || ! grep -Fq "'PANTHERA': 'PANTHERA'" "$constants_source" \
  || ! grep -Fq 'elif ROBOT_PLATFORM == "PANTHERA":' "$constants_source"; then
  echo "错误：Panthera 25x7 常量补丁分支结束后源码标记不完整。" >&2
  exit 1
fi
if [[ ! -f "$rlds_marker" ]]; then
  echo "错误：Panthera SFT v1 RLDS 尚未通过验收。" >&2
  exit 1
fi

mkdir -p "$state_root" "$run_root" "${workspace}/wandb"
exec 9>"${state_root}/train.lock"
if ! flock -n 9; then
  echo "错误：另一个 OpenVLA SFT smoke 正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/train.ok" ]]; then
  python3 -m json.tool "${state_root}/train-summary.json"
  echo "OpenVLA SFT optimizer smoke 已通过，无需重复运行。"
  exit 0
fi
if [[ -e "$run_dir" ]]; then
  failed_run="${run_dir}.failed-$(date +%Y%m%d-%H%M%S)"
  mv "$run_dir" "$failed_run"
  echo "已归档上次失败输出：${failed_run}"
fi
if nvidia-smi -i "$smoke_gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "错误：GPU ${smoke_gpu} 当前有其他计算进程，拒绝启动 smoke。" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$activation_script"
export PYTHONPATH="${adapter_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="$smoke_gpu"
export WANDB_MODE=offline
export WANDB_DIR="${workspace}/wandb"
export TF_CPP_MIN_LOG_LEVEL=2
export ROBOT_PLATFORM="$robot_platform"
export PANTHERA_ACTION_CHUNK="$action_chunk"
export PANTHERA_RLDS_DATASET_NAME="$dataset_name"
export PANTHERA_RLDS_SCHEMA_VERSION="${PANTHERA_SFT_SCHEMA_VERSION:-3}"
export PANTHERA_RLDS_SCENE_PROFILE="${PANTHERA_SFT_SCENE_PROFILE:-}"

run_stamp=$(date +%Y%m%d-%H%M%S)
run_log="${state_root}/train-${run_stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"
set +e
timeout --signal=INT --kill-after=90s "${PANTHERA_SFT_STEP_TIMEOUT:-45m}" \
  torchrun --standalone --nproc-per-node=1 "${adapter_root}/run_finetune.py" \
    --vla_path "$model_root" \
    --data_root_dir "$data_root" \
    --dataset_name "$dataset_name" \
    --run_root_dir "$run_root" \
    --run_id_override "$run_id" \
    --shuffle_buffer_size 1024 \
    --use_l1_regression true \
    --use_diffusion false \
    --num_images_in_input 1 \
    --use_proprio true \
    --batch_size 1 \
    --max_steps 1 \
    --save_freq 999999 \
    --image_aug false \
    --lora_rank 8 \
    --merge_lora_during_training false \
    --use_val_set false \
    --wandb_entity panthera-local \
    --wandb_project panthera-openvla-sft-smoke \
    --wandb_log_freq 1 \
  2>&1 | tee "$run_log"
train_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$train_status" >"${state_root}/exit-code.txt"
if (( train_status != 0 )); then
  echo "错误：OpenVLA SFT optimizer smoke 失败，退出码 ${train_status}。" >&2
  exit "$train_status"
fi
if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|CUDA out of memory|nan|NaN' "$run_log"; then
  echo "错误：训练日志含致命异常或非有限值。" >&2
  exit 1
fi
if ! grep -q 'Max step 1 reached' "$run_log"; then
  echo "错误：训练主循环未到达预定退出点。" >&2
  exit 1
fi
if [[ ! -s "${run_dir}/dataset_statistics.json" ]]; then
  echo "错误：训练入口没有保存 Panthera 数据统计。" >&2
  exit 1
fi
python - "$run_log" "${state_root}/train-summary.json" "$dataset_name" \
  "${PANTHERA_SFT_SCHEMA_VERSION:-3}" \
  "${PANTHERA_SFT_SCENE_PROFILE:-}" "$action_chunk" "$robot_platform" <<'PY'
import json
from pathlib import Path
import sys

summary = {
    "status": "passed",
    "log": str(Path(sys.argv[1])),
    "gpu_count": 1,
    "batch_size": 1,
    "configured_max_steps": 1,
    "dataset": sys.argv[3],
    "task_schema_version": int(sys.argv[4]),
    "scene_profile": sys.argv[5] or None,
    "action_dimension": 7,
    "action_chunk": int(sys.argv[6]),
    "robot_platform": sys.argv[7],
    "use_proprio": True,
    "objective": "l1_regression",
    "lora_rank": 8,
    "checkpoint_saved": False,
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
if nvidia-smi -i "$smoke_gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "错误：训练 smoke 结束后 GPU ${smoke_gpu} 仍有计算进程。" >&2
  exit 1
fi
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/train.ok"
echo "OpenVLA SFT optimizer smoke 通过：${run_dir}"
