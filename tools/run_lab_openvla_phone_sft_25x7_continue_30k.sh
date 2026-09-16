#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
source_model="${workspace}/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-25x7-10000steps"

export PANTHERA_SFT_INITIAL_MODEL="$source_model"
export PANTHERA_SFT_RESUME_STEP=10000
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_ROBOT_PLATFORM=PANTHERA
export PANTHERA_SFT_MAX_STEPS=30000
export PANTHERA_SFT_GPUS=1,2,3
export PANTHERA_SFT_SMOKE_MARKER="${workspace}/.panthera-phone-openvla-sft-25x7-smoke-state/train.ok"
export PANTHERA_SFT_STATE_ROOT="${workspace}/.panthera-phone-openvla-sft-25x7-30k-state"
export PANTHERA_SFT_RUN_ID=panthera-phone-wide-v3-vertical-sft-25x7-30000steps
export PANTHERA_SFT_TIMEOUT=8h

exec bash "${workspace}/run_lab_openvla_phone_sft.sh"
