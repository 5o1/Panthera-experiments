#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
dataset_root="${workspace}/data/place_vertical_cylinder_in_groove/panthera_phone_vertical_sft_v1"
state_root="${workspace}/.panthera-phone-media-audit-state"
activation_script="${workspace}/activate_lab_vla.sh"
auditor="${workspace}/verify_lab_panthera_single_dataset.py"
summary="${state_root}/media-summary.json"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in ffprobe flock; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
for required in "$activation_script" "$auditor"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少 phone-SRT 媒体验收文件：${required}" >&2
    exit 1
  fi
done
if [[ ! -f "${workspace}/.panthera-phone-sft-dataset-state/dataset.ok" ]]; then
  echo "错误：phone-SRT 对齐版数据尚未通过基础验收。" >&2
  exit 1
fi

mkdir -p "$state_root"
exec 9>"${state_root}/media.lock"
if ! flock -n 9; then
  echo "错误：另一个 phone-SRT 媒体验收正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/media.ok" ]]; then
  python3 -m json.tool "$summary"
  exit 0
fi

# shellcheck disable=SC1090
source "$activation_script"
python "$auditor" \
  --dataset-root "$dataset_root" \
  --summary "$summary" \
  --expected-episodes 128 \
  --expected-schema-version 4 \
  --expected-scene-profile phone_srt_vertical_socket \
  --dataset-name panthera_phone_vertical_sft_v1
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/media.ok"
echo "phone-SRT 对齐版数据与视频逐集验收通过。"
