#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
source_model="${workspace}/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-5000steps"

export PANTHERA_SFT_INITIAL_MODEL="$source_model"
export PANTHERA_SFT_RESUME_STEP=5000
export PANTHERA_SFT_MAX_STEPS=20000
export PANTHERA_SFT_GPUS=1,2,3
export PANTHERA_SFT_STATE_ROOT="${workspace}/.panthera-phone-openvla-sft-20k-state"
export PANTHERA_SFT_RUN_ID=panthera-phone-wide-v3-vertical-sft-20000steps
export PANTHERA_SFT_TIMEOUT=8h

exec bash "${workspace}/run_lab_openvla_phone_sft.sh"
