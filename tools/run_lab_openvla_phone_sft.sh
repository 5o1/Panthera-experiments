#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
max_steps="${PANTHERA_SFT_MAX_STEPS:-5000}"
export PANTHERA_SFT_DATA_ROOT="${workspace}/rlds_phone_vertical_sft_v1"
export PANTHERA_SFT_DATASET_NAME=panthera_phone_vertical_cylinder
export PANTHERA_SFT_DATASET_VERSION=3.0.0
export PANTHERA_SFT_SCHEMA_VERSION=4
export PANTHERA_SFT_SCENE_PROFILE=phone_srt_vertical_socket
export PANTHERA_SFT_DATASET_MARKER="${workspace}/.panthera-phone-sft-dataset-state/dataset.ok"
export PANTHERA_SFT_RLDS_MARKER="${workspace}/.panthera-phone-sft-rlds-state/rlds.ok"
export PANTHERA_SFT_SMOKE_MARKER="${workspace}/.panthera-phone-openvla-sft-smoke-state/train.ok"
export PANTHERA_SFT_RUN_ROOT="${workspace}/runs/panthera-phone-openvla-sft"
export PANTHERA_SFT_STATE_ROOT="${PANTHERA_SFT_STATE_ROOT:-${workspace}/.panthera-phone-openvla-sft-state}"
export PANTHERA_SFT_RUN_ID="${PANTHERA_SFT_RUN_ID:-panthera-phone-wide-v3-vertical-sft-${max_steps}steps}"

exec bash "${workspace}/run_lab_openvla_sft.sh"
