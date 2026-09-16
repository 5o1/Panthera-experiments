#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PANTHERA_V2_DATASET_TIER=formal
exec "${script_dir}/run_lab_robotwin_panthera_v2_pilot.sh" "$@"
