#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
dataset_name=panthera_phone_cylinder_socket_v2_single_grasp_sft_v1
dataset_root="${workspace}/data/place_randomized_cylinder_in_socket/${dataset_name}"
dataset_state="${workspace}/state/panthera-v2-single-grasp-formal-state"
state_root="${workspace}/state/panthera-v2-schema10-media-audit-state"
activation_script="${workspace}/tools/activate_lab_vla.sh"
auditor="${workspace}/verify_lab_panthera_single_dataset.py"
summary="${state_root}/media-summary.json"
scene_profile=panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2
workers="${PANTHERA_V2_MEDIA_WORKERS:-8}"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：禁止使用 root 运行媒体审计。" >&2
  exit 1
fi
for command_name in ffprobe flock; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令 ${command_name}。" >&2
    exit 1
  }
done
for required in "$activation_script" "$auditor" \
  "${dataset_state}/human-review-approval.json"; do
  [[ -s "$required" ]] || { echo "错误：缺少前置文件 ${required}。" >&2; exit 1; }
done
[[ -f "${dataset_state}/dataset.ok" ]] || {
  echo "错误：正式数据集尚未通过。" >&2
  exit 1
}

mkdir -p "$state_root"
exec 9>"${state_root}/media.lock"
flock -n 9 || { echo "错误：媒体审计已在运行。" >&2; exit 1; }
if [[ -f "${state_root}/media.ok" && -s "$summary" ]]; then
  python3 -m json.tool "$summary"
  exit 0
fi

# shellcheck disable=SC1090
source "$activation_script"
python "$auditor" \
  --dataset-root "$dataset_root" \
  --summary "$summary" \
  --expected-episodes 128 \
  --expected-schema-version 10 \
  --expected-scene-profile "$scene_profile" \
  --dataset-name "$dataset_name" \
  --workers "$workers"
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/media.ok"
echo "schema 10 正式集逐集媒体审计通过。"
