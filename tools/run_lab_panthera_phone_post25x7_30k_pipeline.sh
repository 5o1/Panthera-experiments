#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
train_state="${workspace}/.panthera-phone-openvla-sft-25x7-30k-state"
model="${workspace}/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-25x7-30000steps"

run_gate() {
  local name=$1
  local seeds=$2

  PANTHERA_POLICY_MODEL="$model" \
  PANTHERA_EVAL_STATE_ROOT="${workspace}/.panthera-phone-${name}-state" \
  PANTHERA_EVAL_LOG_ROOT="${workspace}/logs/panthera-phone-${name}" \
  PANTHERA_EVAL_TRACE_DIR="${workspace}/.panthera-phone-${name}-state/trajectory-traces" \
  PANTHERA_EVAL_SEED_SOURCE="$seeds" \
  PANTHERA_EVAL_TRAIN_MARKER="${train_state}/train.ok" \
  PANTHERA_EVAL_TRAIN_STATE="$train_state" \
  PANTHERA_ACTION_CHUNK=25 \
  PANTHERA_EVAL_EXECUTION_HORIZON=18 \
  PANTHERA_EVAL_MAX_EPISODE_STEPS=1200 \
  PANTHERA_TERMINAL_INSERTION_ASSIST_M=0 \
  PANTHERA_TERMINAL_TARGET_ASSIST=1 \
  PANTHERA_ROBOT_PLATFORM=PANTHERA \
  PANTHERA_EVAL_TRAJECTORIES=16 \
  PANTHERA_EVAL_GPUS=1,2 \
  PANTHERA_EVAL_MIN_SUCCESS=0.75 \
  bash "${workspace}/run_lab_panthera_phone_policy_eval.sh"
}

run_gate \
  policy-eval-25x7-30k-lift-height-handoff-h18-dev16 \
  "${workspace}/panthera-rlinf-overlay/seeds/panthera_phone_vertical_eval_seeds.json"

run_gate \
  policy-eval-25x7-30k-lift-height-handoff-h18-final16 \
  "${workspace}/panthera-rlinf-overlay/seeds/panthera_phone_vertical_final_seeds_v2.json"

date --iso-8601=seconds >"${workspace}/.panthera-phone-25x7-30k-gates.ok"
