#!/usr/bin/env bash

set -euo pipefail

# 已弃用（2026-09-17）：128 条 schema 10 线退役，正式训练改用固定机位 1280。
# 此入口只消费冻结的 4.0.0 产物，保留作历史证据，不要再运行。
workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
export PANTHERA_SFT_DATA_ROOT="${workspace}/datasets/rlds/rlds_phone_cylinder_socket_v2_sft_v1"
export PANTHERA_SFT_DATASET_NAME=panthera_phone_cylinder_socket_v2
export PANTHERA_SFT_DATASET_VERSION=4.0.0
export PANTHERA_SFT_SCHEMA_VERSION=10
export PANTHERA_SFT_SCENE_PROFILE=panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2
export PANTHERA_SFT_DATASET_MARKER="${workspace}/state/panthera-v2-single-grasp-formal-state/dataset.ok"
export PANTHERA_SFT_RLDS_MARKER="${workspace}/state/panthera-v2-schema10-rlds-state/rlds.ok"
export PANTHERA_SFT_SMOKE_MARKER="${workspace}/state/panthera-v2-schema10-openvla-smoke-state/train.ok"
export PANTHERA_SFT_INITIAL_MODEL="${workspace}/models/openvla-oft-place-empty-cup"
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_ROBOT_PLATFORM=PANTHERA
export PANTHERA_SFT_MAX_STEPS=10000
export PANTHERA_SFT_LEARNING_RATE=0.00005
export PANTHERA_SFT_LR_WARMUP_STEPS=500
export PANTHERA_SFT_NUM_STEPS_BEFORE_DECAY=8000
export PANTHERA_SFT_GPUS=1,2,3
export PANTHERA_SFT_STATE_ROOT="${workspace}/state/panthera-v2-schema10-openvla-10k-state"
export PANTHERA_SFT_RUN_ROOT="${workspace}/runs/panthera-v2-schema10-openvla"
export PANTHERA_SFT_RUN_ID=panthera-v2-schema10-25x7-sft-10000steps
export PANTHERA_SFT_TIMEOUT=24h

run_dir="${PANTHERA_SFT_RUN_ROOT}/${PANTHERA_SFT_RUN_ID}"
state_root="$PANTHERA_SFT_STATE_ROOT"
if [[ -e "$run_dir" && ! -f "${state_root}/train.ok" ]]; then
  for gpu in 1 2 3; do
    if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
      echo "错误：GPU ${gpu} 仍有计算进程，不归档可能仍在写入的训练目录。" >&2
      exit 1
    fi
  done
  interrupted="${run_dir}.interrupted-$(date +%Y%m%d-%H%M%S)"
  mv "$run_dir" "$interrupted"
  echo "已可恢复归档中断的完整训练任务：${interrupted}"
fi

exec bash "${workspace}/bin/run_lab_openvla_sft.sh"
