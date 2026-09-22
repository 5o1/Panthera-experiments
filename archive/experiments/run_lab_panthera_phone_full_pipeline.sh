#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"

# The collector may already be running independently. Wait on its advisory
# lock instead of failing or polling before entering the resumable pipeline.
dataset_state="${workspace}/.panthera-phone-sft-dataset-state"
mkdir -p "$dataset_state"
exec 9>"${dataset_state}/dataset.lock"
echo "等待正在运行的 phone-SRT 数据采集完成……"
flock 9
flock -u 9
exec 9>&-

bash "${workspace}/run_lab_panthera_phone_sft_pipeline.sh"
bash "${workspace}/run_lab_panthera_phone_policy_pipeline.sh"
