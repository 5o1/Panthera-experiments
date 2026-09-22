#!/usr/bin/env bash
# 阶段：一步优化器 smoke。底层脚本是单卡单步检查，不支持 4 卡。
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
tier="${PANTHERA_V2_CHAIN_TIER:-expanded}"
case "$tier" in
  expanded)
    data_root="${workspace}/rlds_phone_cylinder_socket_v2_sft_v2"
    rlds_marker="${workspace}/.panthera-v2-expanded-rlds-state/rlds.ok"
    run_root="${workspace}/runs/panthera-v2-expanded-openvla-smoke"
    run_id=panthera-v2-expanded-25x7-step-smoke
    state_root="${workspace}/.panthera-v2-expanded-openvla-smoke-state"
    ;;
  expanded_fixed)
    data_root="${workspace}/rlds_phone_cylinder_socket_v2_sft_v2_fixedcam"
    rlds_marker="${workspace}/.panthera-v2-expanded-fixed-rlds-state/rlds.ok"
    run_root="${workspace}/runs/panthera-v2-expanded-fixed-openvla-smoke"
    run_id=panthera-v2-expanded-fixed-25x7-step-smoke
    state_root="${workspace}/.panthera-v2-expanded-fixed-openvla-smoke-state"
    ;;
  *) echo "错误：未知 tier ${tier}。" >&2; exit 1 ;;
esac

export PANTHERA_SFT_DATA_ROOT="$data_root"
export PANTHERA_SFT_DATASET_NAME=panthera_phone_cylinder_socket_v2
export PANTHERA_SFT_DATASET_VERSION=4.1.0
export PANTHERA_SFT_SCHEMA_VERSION=10
export PANTHERA_SFT_SCENE_PROFILE=panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2
export PANTHERA_SFT_RLDS_MARKER="$rlds_marker"
export PANTHERA_SFT_INITIAL_MODEL="${workspace}/models/openvla-oft-place-empty-cup"
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_ROBOT_PLATFORM=PANTHERA
export PANTHERA_SFT_GPU="${PANTHERA_SFT_GPU:-0}"
export PANTHERA_SFT_SMOKE_RUN_ROOT="$run_root"
export PANTHERA_SFT_SMOKE_RUN_ID="$run_id"
export PANTHERA_SFT_SMOKE_STATE_ROOT="$state_root"

exec bash "${workspace}/run_lab_openvla_sft_step_smoke.sh"
