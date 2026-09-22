#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
train_state="${workspace}/.panthera-phone-openvla-sft-20k-state"

export PANTHERA_EVAL_STATE_ROOT="${workspace}/.panthera-phone-policy-eval-pregrip09-state"
export PANTHERA_EVAL_LOG_ROOT="${workspace}/logs/panthera-phone-policy-eval-pregrip09"
export PANTHERA_EVAL_TRACE_DIR="${PANTHERA_EVAL_STATE_ROOT}/trajectory-traces"
export PANTHERA_EVAL_SCENE_PROFILE=phone_srt_vertical_socket_pregrip09
export PANTHERA_EVAL_ENV_SOURCE="${workspace}/panthera-rlinf-overlay/config/env/robotwin_place_vertical_cylinder_in_groove_pregrip09.yaml"
export PANTHERA_EVAL_SEED_SOURCE="${workspace}/panthera-rlinf-overlay/seeds/panthera_phone_vertical_diagnostic_seeds.json"
export PANTHERA_EVAL_TRAIN_MARKER="${train_state}/train.ok"
export PANTHERA_EVAL_TRAIN_STATE="$train_state"
export PANTHERA_EVAL_TRAJECTORIES=2
export PANTHERA_EVAL_GPUS="${PANTHERA_EVAL_DIAGNOSTIC_GPUS:-1,2}"
export PANTHERA_EVAL_MIN_SUCCESS="${PANTHERA_EVAL_MIN_SUCCESS:-1.0}"
export PANTHERA_EVAL_REQUIRED_INITIAL_GRIPPER_OPENING=0.9

exec bash "${workspace}/run_lab_panthera_phone_policy_eval.sh"
