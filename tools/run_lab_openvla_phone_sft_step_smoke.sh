#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
export PANTHERA_SFT_DATA_ROOT="${workspace}/datasets/rlds/rlds_phone_vertical_sft_v1"
export PANTHERA_SFT_DATASET_NAME=panthera_phone_vertical_cylinder
export PANTHERA_SFT_DATASET_VERSION=3.0.0
export PANTHERA_SFT_SCHEMA_VERSION=4
export PANTHERA_SFT_SCENE_PROFILE=phone_srt_vertical_socket
export PANTHERA_SFT_RLDS_MARKER="${workspace}/state/panthera-phone-sft-rlds-state/rlds.ok"
export PANTHERA_SFT_SMOKE_RUN_ROOT="${PANTHERA_SFT_SMOKE_RUN_ROOT:-${workspace}/runs/panthera-phone-openvla-sft-smoke}"
export PANTHERA_SFT_SMOKE_RUN_ID="${PANTHERA_SFT_SMOKE_RUN_ID:-panthera-phone-vertical-sft-step-smoke}"
export PANTHERA_SFT_SMOKE_STATE_ROOT="${PANTHERA_SFT_SMOKE_STATE_ROOT:-${workspace}/state/panthera-phone-openvla-sft-smoke-state}"

exec bash "${workspace}/bin/run_lab_openvla_sft_step_smoke.sh"
