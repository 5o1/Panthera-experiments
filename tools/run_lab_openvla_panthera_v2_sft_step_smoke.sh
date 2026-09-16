#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
export PANTHERA_SFT_DATA_ROOT="${workspace}/rlds_phone_cylinder_socket_v2_sft_v1"
export PANTHERA_SFT_DATASET_NAME=panthera_phone_cylinder_socket_v2
export PANTHERA_SFT_DATASET_VERSION=4.0.0
export PANTHERA_SFT_SCHEMA_VERSION=10
export PANTHERA_SFT_SCENE_PROFILE=panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2
export PANTHERA_SFT_RLDS_MARKER="${workspace}/.panthera-v2-schema10-rlds-state/rlds.ok"
export PANTHERA_SFT_INITIAL_MODEL="${workspace}/models/openvla-oft-place-empty-cup"
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_ROBOT_PLATFORM=PANTHERA
export PANTHERA_SFT_GPU=1
export PANTHERA_SFT_SMOKE_RUN_ROOT="${workspace}/runs/panthera-v2-schema10-openvla-smoke"
export PANTHERA_SFT_SMOKE_RUN_ID=panthera-v2-schema10-25x7-step-smoke
export PANTHERA_SFT_SMOKE_STATE_ROOT="${workspace}/.panthera-v2-schema10-openvla-smoke-state"

exec bash "${workspace}/run_lab_openvla_sft_step_smoke.sh"
