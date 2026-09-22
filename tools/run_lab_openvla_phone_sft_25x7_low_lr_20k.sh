#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
source_model="${workspace}/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-25x7-15000steps-lr5e5-nowarmup"

export PANTHERA_SFT_INITIAL_MODEL="$source_model"
export PANTHERA_SFT_RESUME_STEP=15000
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_ROBOT_PLATFORM=PANTHERA
export PANTHERA_SFT_MAX_STEPS=20000
export PANTHERA_SFT_LEARNING_RATE=0.00001
export PANTHERA_SFT_LR_WARMUP_STEPS=0
export PANTHERA_SFT_NUM_STEPS_BEFORE_DECAY=25000
export PANTHERA_SFT_GPUS=1,2,3
export PANTHERA_SFT_SMOKE_MARKER="${workspace}/state/panthera-phone-openvla-sft-25x7-smoke-state/train.ok"
export PANTHERA_SFT_STATE_ROOT="${workspace}/state/panthera-phone-openvla-sft-25x7-low-lr-20k-state"
export PANTHERA_SFT_RUN_ID=panthera-phone-wide-v3-vertical-sft-25x7-20000steps-lr1e5-nowarmup
export PANTHERA_SFT_TIMEOUT=8h

exec bash "${workspace}/bin/run_lab_openvla_phone_sft.sh"
