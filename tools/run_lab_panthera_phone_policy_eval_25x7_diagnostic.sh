#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
train_state="${workspace}/state/panthera-phone-openvla-sft-25x7-10k-state"

export PANTHERA_EVAL_STATE_ROOT="${workspace}/state/panthera-phone-policy-eval-25x7-diagnostic-state"
export PANTHERA_EVAL_LOG_ROOT="${workspace}/logs/panthera-phone-policy-eval-25x7-diagnostic"
export PANTHERA_EVAL_TRACE_DIR="${PANTHERA_EVAL_STATE_ROOT}/trajectory-traces"
export PANTHERA_EVAL_SEED_SOURCE="${workspace}/overlays/rlinf/seeds/panthera_phone_vertical_diagnostic_seeds.json"
export PANTHERA_EVAL_TRAIN_MARKER="${train_state}/train.ok"
export PANTHERA_EVAL_TRAIN_STATE="$train_state"
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_ROBOT_PLATFORM=PANTHERA
export PANTHERA_EVAL_TRAJECTORIES=2
export PANTHERA_EVAL_GPUS="${PANTHERA_EVAL_DIAGNOSTIC_GPUS:-1,2}"
export PANTHERA_EVAL_MIN_SUCCESS="${PANTHERA_EVAL_MIN_SUCCESS:-1.0}"

exec bash "${workspace}/bin/run_lab_panthera_phone_policy_eval.sh"
