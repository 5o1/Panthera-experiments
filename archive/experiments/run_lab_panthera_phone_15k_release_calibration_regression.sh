#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
train_state="${workspace}/.panthera-phone-openvla-sft-25x7-low-lr-15k-state"

export PANTHERA_POLICY_MODEL="${workspace}/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-25x7-15000steps-lr5e5-nowarmup"
export PANTHERA_EVAL_STATE_ROOT="${workspace}/.panthera-phone-15k-release-calibration-regression-state"
export PANTHERA_EVAL_LOG_ROOT="${workspace}/logs/panthera-phone-15k-release-calibration-regression"
export PANTHERA_EVAL_TRACE_DIR="${PANTHERA_EVAL_STATE_ROOT}/trajectory-traces"
export PANTHERA_EVAL_SEED_SOURCE="${workspace}/panthera-rlinf-overlay/seeds/panthera_phone_vertical_release_calibration_seeds.json"
export PANTHERA_EVAL_TRAIN_MARKER="${train_state}/train.ok"
export PANTHERA_EVAL_TRAIN_STATE="$train_state"
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_EVAL_EXECUTION_HORIZON=18
export PANTHERA_EVAL_MAX_EPISODE_STEPS=1200
export PANTHERA_TERMINAL_INSERTION_ASSIST_M=0
export PANTHERA_TERMINAL_TARGET_ASSIST=1
export PANTHERA_TERMINAL_TARGET_ASSIST_TRIGGER=release
export PANTHERA_ROBOT_PLATFORM=PANTHERA
export PANTHERA_EVAL_TRAJECTORIES=4
export PANTHERA_EVAL_GPUS=1,2
export PANTHERA_EVAL_MIN_SUCCESS=0.75

exec bash "${workspace}/run_lab_panthera_phone_policy_eval.sh"
