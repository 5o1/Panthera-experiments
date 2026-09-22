#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
source_root="${workspace}/data/place_cylinder_in_groove/panthera_cylinder_contract_smoke"
data_root="${workspace}/datasets/rlds"
adapter_root="${workspace}/packages/panthera_vla"
state_root="${workspace}/state/panthera-rlds-smoke-state"
activation_script="${workspace}/tools/activate_lab_vla.sh"
summary_path="${state_root}/rlds-summary.json"
dataset_version_root="${data_root}/panthera_cylinder/1.0.0"

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
for required in \
  "$activation_script" \
  "${adapter_root}/panthera_rlds.py"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少前置文件：${required}" >&2
    exit 1
  fi
done
if [[ ! -f "${workspace}/state/panthera-contract-smoke-state/contract.ok" ]]; then
  echo "错误：统一时钟的 Panthera contract smoke 尚未通过。" >&2
  exit 1
fi

mkdir -p "$state_root" "$data_root"
exec 9>"${state_root}/rlds.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera RLDS 流程正在运行。" >&2
  exit 1
fi

if [[ -f "${state_root}/rlds.ok" ]]; then
  python3 -m json.tool "$summary_path"
  echo "Panthera RLDS smoke 已通过，无需重复转换。"
  exit 0
fi
if [[ -e "$dataset_version_root" ]]; then
  if [[ ! -s "${dataset_version_root}/dataset_info.json" ]]; then
    echo "错误：发现不完整的 RLDS 输出，请先人工归档：${dataset_version_root}" >&2
    exit 1
  fi
  echo "复用已完整生成的 RLDS shards，继续 OpenVLA 读取验收。"
fi

# shellcheck disable=SC1090
source "$activation_script"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${adapter_root}${PYTHONPATH:+:${PYTHONPATH}}"

run_stamp=$(date +%Y%m%d-%H%M%S)
run_log="${state_root}/rlds-${run_stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"
set +e
timeout --signal=INT --kill-after=60s "${PANTHERA_RLDS_TIMEOUT:-30m}" \
  python "${adapter_root}/panthera_rlds.py" \
    --source-root "$source_root" \
    --data-root "$data_root" \
    --summary "$summary_path" \
    --validation-episode 3 \
  2>&1 | tee "$run_log"
rlds_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rlds_status" >"${state_root}/exit-code.txt"
if (( rlds_status != 0 )); then
  echo "错误：Panthera RLDS 转换或 OpenVLA 读取验收失败，退出码 ${rlds_status}。" >&2
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
if summary.get("status") != "passed":
    raise SystemExit("RLDS summary did not pass")
if summary.get("splits") != {"train": [0, 1, 2], "val": [3]}:
    raise SystemExit(f"unexpected dataset split: {summary.get('splits')}")
if summary.get("action_chunk_shape") != [25, 14]:
    raise SystemExit("OpenVLA action chunk is not 25x14")
if summary.get("proprio_window_shape") != [1, 14]:
    raise SystemExit("OpenVLA proprio window is not 1x14")
if summary.get("primary_image_shape") != [1, 224, 224, 3]:
    raise SystemExit("OpenVLA primary image was not decoded and resized")
PY

if pgrep -af 'ray::|raylet|collect_data.py|eval_policy.py' | grep -v grep; then
  echo "错误：RLDS 验收后仍有 RoboTwin/Ray 进程。" >&2
  exit 1
fi
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/rlds.ok"
echo "Panthera RLDS + OpenVLA 单批次 smoke 通过：${dataset_version_root}"
