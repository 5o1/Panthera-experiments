#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
rlinf_root="${workspace}/runtime/rlinf"
rlinf_overlay="${workspace}/overlays/rlinf"
robotwin_overlay="${workspace}/overlays/robotwin"

export PANTHERA_EVAL_STATE_ROOT="${PANTHERA_EVAL_STATE_ROOT:-${workspace}/state/panthera-phone-policy-eval-state}"
export PANTHERA_EVAL_LOG_ROOT="${PANTHERA_EVAL_LOG_ROOT:-${workspace}/logs/panthera-phone-policy-eval}"
export PANTHERA_EVAL_TASK_NAME="${PANTHERA_EVAL_TASK_NAME:-place_vertical_cylinder_in_groove}"
export PANTHERA_EVAL_SCENE_PROFILE="${PANTHERA_EVAL_SCENE_PROFILE:-phone_srt_vertical_socket}"
export PANTHERA_EVAL_UNNORM_KEY="${PANTHERA_EVAL_UNNORM_KEY:-panthera_phone_vertical_cylinder}"
export PANTHERA_EVAL_CONFIG_NAME="${PANTHERA_EVAL_CONFIG_NAME:-robotwin_panthera_phone_vertical_openvlaoft_eval}"
export PANTHERA_EVAL_ENV_SOURCE="${PANTHERA_EVAL_ENV_SOURCE:-${rlinf_overlay}/config/env/robotwin_place_vertical_cylinder_in_groove.yaml}"
export PANTHERA_EVAL_CONFIG_SOURCE="${PANTHERA_EVAL_CONFIG_SOURCE:-${rlinf_overlay}/evaluations/robotwin_panthera_phone_vertical_openvlaoft_eval.yaml}"
export PANTHERA_EVAL_SEED_SOURCE="${PANTHERA_EVAL_SEED_SOURCE:-${rlinf_overlay}/seeds/panthera_phone_vertical_eval_seeds.json}"
export PANTHERA_EVAL_ENV_TARGET="${PANTHERA_EVAL_ENV_TARGET:-${rlinf_root}/examples/embodiment/config/env/robotwin_place_vertical_cylinder_in_groove.yaml}"
export PANTHERA_EVAL_CONFIG_TARGET="${PANTHERA_EVAL_CONFIG_TARGET:-${rlinf_root}/evaluations/robotwin/robotwin_panthera_phone_vertical_openvlaoft_eval.yaml}"
export PANTHERA_EVAL_TASK_SOURCE="${PANTHERA_EVAL_TASK_SOURCE:-${robotwin_overlay}/envs/place_vertical_cylinder_in_groove.py}"
export PANTHERA_EVAL_INSTRUCTION_SOURCE="${PANTHERA_EVAL_INSTRUCTION_SOURCE:-${robotwin_overlay}/description/task_instruction/place_vertical_cylinder_in_groove.json}"
export PANTHERA_EVAL_CONFIG_MARKER="${PANTHERA_EVAL_CONFIG_MARKER:-${workspace}/state/panthera-phone-eval-config-state/config.ok}"
export PANTHERA_EVAL_TRAIN_MARKER="${PANTHERA_EVAL_TRAIN_MARKER:-${workspace}/state/panthera-phone-openvla-sft-state/train.ok}"
export PANTHERA_EVAL_TRAIN_STATE="${PANTHERA_EVAL_TRAIN_STATE:-${workspace}/state/panthera-phone-openvla-sft-state}"

exec bash "${workspace}/bin/run_lab_panthera_policy_eval.sh"
