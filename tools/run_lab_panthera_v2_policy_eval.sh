#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
split="${PANTHERA_V2_EVAL_SPLIT:-${1:-dev}}"
case "$split" in
  dev)
    seed_file=panthera_v2_schema10_dev_seeds.json
    state_name=.panthera-v2-schema10-policy-dev18-state
    log_name=panthera-v2-schema10-policy-dev18
    ;;
  final)
    seed_file=panthera_v2_schema10_final_seeds.json
    state_name=.panthera-v2-schema10-policy-final18-state
    log_name=panthera-v2-schema10-policy-final18
    [[ -f "${workspace}/state/panthera-v2-schema10-policy-dev18-state/eval.ok" ]] || {
      echo "错误：开发集未达到 75%，禁止运行 final。" >&2
      exit 1
    }
    ;;
  *)
    echo "错误：评测 split 必须为 dev 或 final。" >&2
    exit 1
    ;;
esac

export PANTHERA_EVAL_STATE_ROOT="${workspace}/${state_name}"
export PANTHERA_EVAL_LOG_ROOT="${workspace}/logs/${log_name}"
export PANTHERA_EVAL_TASK_NAME=place_randomized_cylinder_in_socket
export PANTHERA_EVAL_SCENE_PROFILE=panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2
export PANTHERA_EVAL_UNNORM_KEY=panthera_phone_cylinder_socket_v2
export PANTHERA_EVAL_CONFIG_NAME=robotwin_panthera_v2_openvlaoft_eval
export PANTHERA_EVAL_ENV_SOURCE="${workspace}/overlays/rlinf/config/env/robotwin_place_randomized_cylinder_in_socket.yaml"
export PANTHERA_EVAL_CONFIG_SOURCE="${workspace}/overlays/rlinf/evaluations/robotwin_panthera_v2_openvlaoft_eval.yaml"
export PANTHERA_EVAL_SEED_SOURCE="${workspace}/overlays/rlinf/seeds/${seed_file}"
export PANTHERA_EVAL_ENV_TARGET="${workspace}/runtime/rlinf/examples/embodiment/config/env/robotwin_place_randomized_cylinder_in_socket.yaml"
export PANTHERA_EVAL_CONFIG_TARGET="${workspace}/runtime/rlinf/evaluations/robotwin/robotwin_panthera_v2_openvlaoft_eval.yaml"
export PANTHERA_EVAL_TASK_SOURCE="${workspace}/overlays/robotwin/envs/place_randomized_cylinder_in_socket.py"
export PANTHERA_EVAL_INSTRUCTION_SOURCE="${workspace}/overlays/robotwin/description/task_instruction/place_randomized_cylinder_in_socket.json"
export PANTHERA_EVAL_CONFIG_MARKER="${workspace}/state/panthera-v2-schema10-eval-config-state/config.ok"
export PANTHERA_EVAL_TRAIN_MARKER="${workspace}/state/panthera-v2-schema10-openvla-10k-state/train.ok"
export PANTHERA_EVAL_TRAIN_STATE="${workspace}/state/panthera-v2-schema10-openvla-10k-state"
export PANTHERA_EVAL_GPUS=1,2,3
export PANTHERA_EVAL_TRAJECTORIES=18
export PANTHERA_EVAL_MAX_EPISODE_STEPS=1600
export PANTHERA_EVAL_MIN_SUCCESS=0.75
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_EVAL_EXECUTION_HORIZON=20
export PANTHERA_ROBOT_PLATFORM=PANTHERA
export PANTHERA_TERMINAL_TARGET_ASSIST=0
export PANTHERA_TERMINAL_INSERTION_ASSIST_M=0
export PANTHERA_POLICY_EVAL_TIMEOUT=12h

exec bash "${workspace}/bin/run_lab_panthera_policy_eval.sh"
