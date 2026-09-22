#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"

bash "${workspace}/bin/run_lab_openvla_phone_sft_25x7_low_lr_20k.sh"
bash "${workspace}/bin/run_lab_panthera_phone_post25x7_low_lr_20k_pipeline.sh"
