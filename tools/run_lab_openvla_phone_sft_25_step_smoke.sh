#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
source_model="${workspace}/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-20000steps"

export PANTHERA_SFT_INITIAL_MODEL="$source_model"
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_ROBOT_PLATFORM=PANTHERA
export PANTHERA_SFT_GPU=1
export PANTHERA_SFT_SMOKE_RUN_ROOT="${workspace}/runs/panthera-phone-openvla-sft-25x7-smoke"
export PANTHERA_SFT_SMOKE_RUN_ID=panthera-phone-vertical-sft-25x7-step-smoke
export PANTHERA_SFT_SMOKE_STATE_ROOT="${workspace}/.panthera-phone-openvla-sft-25x7-smoke-state"

exec bash "${workspace}/run_lab_openvla_phone_sft_step_smoke.sh"
