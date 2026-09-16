#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"

bash "${workspace}/run_lab_openvla_phone_sft_25x7_low_lr_15k.sh"
bash "${workspace}/run_lab_panthera_phone_post25x7_low_lr_15k_pipeline.sh"
