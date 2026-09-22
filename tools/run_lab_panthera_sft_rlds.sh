#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
source_root="${workspace}/data/place_cylinder_in_groove/panthera_single_cylinder_sft_v1"
data_root="${workspace}/datasets/rlds/rlds_single_sft_v1"
adapter_root="${workspace}/packages/panthera_vla"
state_root="${workspace}/state/panthera-single-sft-rlds-state"
activation_script="${workspace}/tools/activate_lab_vla.sh"
summary_path="${state_root}/rlds-summary.json"
dataset_version_root="${data_root}/panthera_single_cylinder/2.0.0"

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
    echo "错误：缺少前置文件：${required}" >&2
    exit 1
  fi
done
if [[ ! -f "${workspace}/state/panthera-single-sft-dataset-state/dataset.ok" ]]; then
  echo "错误：单 Panthera 128-episode SFT v1 数据集尚未通过验收。" >&2
  exit 1
fi

mkdir -p "$state_root" "$data_root"
exec 9>"${state_root}/rlds.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera SFT RLDS 流程正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/rlds.ok" ]]; then
  python3 -m json.tool "$summary_path"
  echo "Panthera SFT v1 RLDS 已通过，无需重复转换。"
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

validation_args=()
for ((episode=112; episode<128; episode++)); do
  validation_args+=(--validation-episode "$episode")
done
run_stamp=$(date +%Y%m%d-%H%M%S)
run_log="${state_root}/rlds-${run_stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"
set +e
timeout --signal=INT --kill-after=60s "${PANTHERA_SFT_RLDS_TIMEOUT:-60m}" \
  python "${adapter_root}/panthera_rlds.py" \
    --source-root "$source_root" \
    --data-root "$data_root" \
    --summary "$summary_path" \
    "${validation_args[@]}" \
  2>&1 | tee "$run_log"
rlds_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rlds_status" >"${state_root}/exit-code.txt"
if (( rlds_status != 0 )); then
  echo "错误：Panthera SFT RLDS 转换失败，退出码 ${rlds_status}。" >&2
  exit "$rlds_status"
fi
if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|AssertionError:' "$run_log"; then
  echo "错误：RLDS 日志含致命异常。" >&2
  exit 1
fi
python - "$summary_path" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_train = list(range(112))
expected_val = list(range(112, 128))
if summary.get("status") != "passed":
    raise SystemExit("SFT RLDS summary did not pass")
if summary.get("splits") != {"train": expected_train, "val": expected_val}:
    raise SystemExit("SFT RLDS split mismatch")
if summary.get("statistics_episode_count") != 128:
    raise SystemExit("SFT RLDS statistics do not cover 128 episodes")
if summary.get("action_chunk_shape") != [5, 7]:
    raise SystemExit("OpenVLA action chunk is not 5x7")
if summary.get("proprio_window_shape") != [1, 7]:
    raise SystemExit("OpenVLA proprio window is not 1x7")
PY

if pgrep -af 'ray::|raylet|collect_data.py|eval_policy.py' | grep -v grep; then
  echo "错误：SFT RLDS 验收后仍有 RoboTwin/Ray 进程。" >&2
  exit 1
fi
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/rlds.ok"
echo "Panthera SFT v1 RLDS 通过：${dataset_version_root}"
