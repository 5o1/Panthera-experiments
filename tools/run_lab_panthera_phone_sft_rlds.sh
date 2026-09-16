#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
source_root="${workspace}/data/place_vertical_cylinder_in_groove/panthera_phone_vertical_sft_v1"
data_root="${workspace}/rlds_phone_vertical_sft_v1"
adapter_root="${workspace}/panthera-openvla-adapter"
state_root="${workspace}/.panthera-phone-sft-rlds-state"
activation_script="${workspace}/activate_lab_vla.sh"
summary_path="${state_root}/rlds-summary.json"
dataset_name="panthera_phone_vertical_cylinder"
dataset_version_root="${data_root}/${dataset_name}/3.0.0"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock timeout; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
for required in "$activation_script" "${adapter_root}/panthera_rlds.py"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少 phone-SRT RLDS 文件：${required}" >&2
    exit 1
  fi
done
if [[ ! -f "${workspace}/.panthera-phone-sft-dataset-state/dataset.ok" ]]; then
  echo "错误：phone-SRT 对齐版 128 集数据尚未通过验收。" >&2
  exit 1
fi

mkdir -p "$state_root" "$data_root"
exec 9>"${state_root}/rlds.lock"
if ! flock -n 9; then
  echo "错误：另一个 phone-SRT RLDS 流程正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/rlds.ok" ]]; then
  python3 -m json.tool "$summary_path"
  exit 0
fi
if [[ -e "$dataset_version_root" && ! -s "${dataset_version_root}/dataset_info.json" ]]; then
  echo "错误：发现不完整 RLDS 输出，请先人工归档：${dataset_version_root}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$activation_script"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${adapter_root}${PYTHONPATH:+:${PYTHONPATH}}"
export ROBOT_PLATFORM=BRIDGE
export PANTHERA_RLDS_DATASET_NAME="$dataset_name"
export PANTHERA_RLDS_SCHEMA_VERSION=4
export PANTHERA_RLDS_SCENE_PROFILE=phone_srt_vertical_socket

validation_args=()
for ((episode=112; episode<128; episode++)); do
  validation_args+=(--validation-episode "$episode")
done
stamp=$(date +%Y%m%d-%H%M%S)
run_log="${state_root}/rlds-${stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"
set +e
timeout --signal=INT --kill-after=60s \
  "${PANTHERA_PHONE_RLDS_TIMEOUT:-60m}" \
  python "${adapter_root}/panthera_rlds.py" \
    --source-root "$source_root" \
    --data-root "$data_root" \
    --summary "$summary_path" \
    "${validation_args[@]}" \
  2>&1 | tee "$run_log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" >"${state_root}/exit-code.txt"
if (( status != 0 )); then
  echo "错误：phone-SRT RLDS 转换失败。" >&2
  exit "$status"
fi
python - "$summary_path" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "passed":
    raise SystemExit("phone RLDS did not pass")
if summary.get("dataset_name") != "panthera_phone_vertical_cylinder":
    raise SystemExit("phone RLDS dataset name mismatch")
if summary.get("tfds_version") != "3.0.0":
    raise SystemExit("phone RLDS version mismatch")
if summary.get("task_schema_version") != 4:
    raise SystemExit("phone RLDS schema mismatch")
if summary.get("scene_profile") != "phone_srt_vertical_socket":
    raise SystemExit("phone RLDS scene profile mismatch")
if summary.get("statistics_episode_count") != 128:
    raise SystemExit("phone RLDS statistics do not cover 128 episodes")
if summary.get("action_chunk_shape") != [5, 7]:
    raise SystemExit("phone RLDS action chunk is not 5x7")
if summary.get("proprio_window_shape") != [1, 7]:
    raise SystemExit("phone RLDS proprio window is not 1x7")
expected = {"train": list(range(112)), "val": list(range(112, 128))}
if summary.get("splits") != expected:
    raise SystemExit("phone RLDS split mismatch")
PY
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/rlds.ok"
echo "phone-SRT 对齐版 RLDS 通过：${dataset_version_root}"
