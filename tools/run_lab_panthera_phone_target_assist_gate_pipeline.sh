#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
train_state="${workspace}/.panthera-phone-openvla-sft-25x7-10k-state"
model="${workspace}/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-25x7-10000steps"

run_gate() {
  local name=$1
  local seeds=$2
  local trajectories=$3
  local gpus=$4
  local minimum_success=$5

  PANTHERA_POLICY_MODEL="$model" \
  PANTHERA_EVAL_STATE_ROOT="${workspace}/.panthera-phone-${name}-state" \
  PANTHERA_EVAL_LOG_ROOT="${workspace}/logs/panthera-phone-${name}" \
  PANTHERA_EVAL_TRACE_DIR="${workspace}/.panthera-phone-${name}-state/trajectory-traces" \
  PANTHERA_EVAL_SEED_SOURCE="$seeds" \
  PANTHERA_EVAL_TRAIN_MARKER="${train_state}/train.ok" \
  PANTHERA_EVAL_TRAIN_STATE="$train_state" \
  PANTHERA_ACTION_CHUNK=25 \
  PANTHERA_EVAL_EXECUTION_HORIZON=20 \
  PANTHERA_EVAL_MAX_EPISODE_STEPS=1200 \
  PANTHERA_TERMINAL_INSERTION_ASSIST_M=0 \
  PANTHERA_TERMINAL_TARGET_ASSIST=1 \
  PANTHERA_ROBOT_PLATFORM=PANTHERA \
  PANTHERA_EVAL_TRAJECTORIES="$trajectories" \
  PANTHERA_EVAL_GPUS="$gpus" \
  PANTHERA_EVAL_MIN_SUCCESS="$minimum_success" \
  bash "${workspace}/run_lab_panthera_phone_policy_eval.sh"
}

diagnostic_seeds="${workspace}/panthera-rlinf-overlay/seeds/panthera_phone_vertical_diagnostic_seeds.json"
heldout_seeds="${workspace}/panthera-rlinf-overlay/seeds/panthera_phone_vertical_eval_seeds.json"

# The two-seed gate must be perfect before spending compute on the held-out set.
run_gate \
  policy-eval-25x7-target-assist-two-seed \
  "$diagnostic_seeds" 2 1,2 1.0

# Formal acceptance uses 16 unseen seeds and the project-wide 75% threshold.
run_gate \
  policy-eval-25x7-target-assist-heldout16 \
  "$heldout_seeds" 16 1,2 0.75

date --iso-8601=seconds >"${workspace}/.panthera-phone-target-assist-gates.ok"
