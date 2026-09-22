#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"

export PANTHERA_EVAL_STATE_ROOT="${workspace}/.panthera-phone-policy-eval-l1-probe-state"
export PANTHERA_EVAL_LOG_ROOT="${workspace}/logs/panthera-phone-policy-eval-l1-probe"
export PANTHERA_EVAL_TRAJECTORIES=2
export PANTHERA_EVAL_GPUS="${PANTHERA_EVAL_PROBE_GPUS:-1,2}"
export PANTHERA_EVAL_MIN_SUCCESS=0.5

exec bash "${workspace}/run_lab_panthera_phone_policy_eval.sh"
