#!/usr/bin/env bash
# 阶段：验证集早停 SFT，固定使用 4 张卡。
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
tier="${PANTHERA_V2_CHAIN_TIER:-expanded}"
case "$tier" in
  expanded)
    data_root="${workspace}/datasets/rlds/rlds_phone_cylinder_socket_v2_sft_v2"
    dataset_state="${workspace}/state/panthera-v2-single-grasp-expanded-state"
    rlds_state="${workspace}/state/panthera-v2-expanded-rlds-state"
    smoke_state="${workspace}/state/panthera-v2-expanded-openvla-smoke-state"
    run_root="${workspace}/runs/panthera-v2-expanded-openvla"
    run_id=panthera-v2-expanded-25x7-sft-earlystop-md1e-3-p3
    state_root="${workspace}/state/panthera-v2-expanded-openvla-earlystop-state"
    ;;
  expanded_fixed)
    data_root="${workspace}/datasets/rlds/rlds_phone_cylinder_socket_v2_sft_v2_fixedcam"
    dataset_state="${workspace}/state/panthera-v2-single-grasp-expanded_fixed-state"
    rlds_state="${workspace}/state/panthera-v2-expanded-fixed-rlds-state"
    smoke_state="${workspace}/state/panthera-v2-expanded-fixed-openvla-smoke-state"
    run_root="${workspace}/runs/panthera-v2-expanded-fixed-openvla"
    run_id=panthera-v2-expanded-fixed-25x7-sft-earlystop-md1e-3-p3
    state_root="${workspace}/state/panthera-v2-expanded-fixed-openvla-earlystop-state"
    ;;
  *) echo "错误：未知 tier ${tier}。" >&2; exit 1 ;;
esac

export PANTHERA_SFT_DATA_ROOT="$data_root"
export PANTHERA_SFT_DATASET_NAME=panthera_phone_cylinder_socket_v2
export PANTHERA_SFT_DATASET_VERSION=4.1.0
export PANTHERA_SFT_SCHEMA_VERSION=10
export PANTHERA_SFT_SCENE_PROFILE=panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2
export PANTHERA_SFT_DATASET_MARKER="${dataset_state}/dataset.ok"
export PANTHERA_SFT_RLDS_MARKER="${rlds_state}/rlds.ok"
export PANTHERA_SFT_SMOKE_MARKER="${smoke_state}/train.ok"
export PANTHERA_SFT_INITIAL_MODEL="${workspace}/models/openvla-oft-place-empty-cup"
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_ROBOT_PLATFORM=PANTHERA
# 停止判据改为固定时间预算：验证集 L1 早停已被闭环实测否定（docs/10 第 9 节，
# 20k 的离线指标全面优于 15k 而闭环 2/16 对 11/16）。耐心值默认设到不会触发，
# 只保留周期性验证作为遥测；真正停下训练的是 PANTHERA_SFT_TIMEOUT。
export PANTHERA_SFT_EARLY_STOPPING_MIN_DELTA="${PANTHERA_SFT_EARLY_STOPPING_MIN_DELTA:-0.001}"
export PANTHERA_SFT_EARLY_STOPPING_PATIENCE="${PANTHERA_SFT_EARLY_STOPPING_PATIENCE:-100000}"
export PANTHERA_SFT_ACCEPT_TIME_BUDGET="${PANTHERA_SFT_ACCEPT_TIME_BUDGET:-1}"
export PANTHERA_SFT_VAL_FREQ="${PANTHERA_SFT_VAL_FREQ:-1000}"
export PANTHERA_SFT_LEARNING_RATE="${PANTHERA_SFT_LEARNING_RATE:-0.00005}"
export PANTHERA_SFT_LR_WARMUP_STEPS="${PANTHERA_SFT_LR_WARMUP_STEPS:-500}"
export PANTHERA_SFT_NUM_STEPS_BEFORE_DECAY="${PANTHERA_SFT_NUM_STEPS_BEFORE_DECAY:-8000}"
export PANTHERA_SFT_GPUS="${PANTHERA_SFT_GPUS:-0,1,2,3}"
export PANTHERA_SFT_STATE_ROOT="$state_root"
export PANTHERA_SFT_RUN_ROOT="$run_root"
export PANTHERA_SFT_RUN_ID="$run_id"
export PANTHERA_SFT_TIMEOUT="${PANTHERA_SFT_TIMEOUT:-3h}"

exec bash "${workspace}/bin/run_lab_openvla_panthera_v2_sft_earlystop.sh"
