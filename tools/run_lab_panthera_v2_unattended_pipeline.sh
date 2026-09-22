#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state_root="${workspace}/state/panthera-v2-schema10-unattended-state"
mkdir -p "$state_root"
exec 9>"${state_root}/pipeline.lock"
flock -n 9 || { echo "错误：schema 10 无人值守流水线已在运行。" >&2; exit 1; }

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：禁止使用 root 运行训练流水线。" >&2
  exit 1
fi
if [[ -f "${state_root}/pipeline.ok" ]]; then
  python3 -m json.tool "${state_root}/pipeline-summary.json"
  exit 0
fi

current_stage=preflight
printf '%s\n' "$current_stage" >"${state_root}/current-stage.txt"
pipeline_status=failed
write_terminal_summary() {
  local exit_status=$?
  python3 - "$state_root" "$current_stage" "$pipeline_status" "$exit_status" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

root = Path(sys.argv[1])
payload = {
    "status": sys.argv[3],
    "last_stage": sys.argv[2],
    "exit_code": int(sys.argv[4]),
    "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "gpu_policy": {"training": [1, 2, 3], "evaluation": [1, 2, 3], "gpu0_used": False},
}
temporary = root / "pipeline-summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(root / "pipeline-summary.json")
PY
}
trap write_terminal_summary EXIT

run_stage() {
  current_stage="$1"
  shift
  printf '%s\n' "$current_stage" >"${state_root}/current-stage.txt"
  echo "[$(date --iso-8601=seconds)] 开始阶段：${current_stage}"
  "$@"
  date --iso-8601=seconds >"${state_root}/${current_stage}.ok"
}

wait_for_gpus() {
  local deadline=$((SECONDS + 86400))
  local gpu
  while :; do
    local busy=0
    for gpu in 1 2 3; do
      if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
        busy=1
      fi
    done
    (( busy == 0 )) && return 0
    (( SECONDS < deadline )) || {
      echo "错误：等待 GPU1–3 空闲超过 24 小时。" >&2
      return 1
    }
    sleep 60
  done
}

[[ -f "${workspace}/state/panthera-v2-single-grasp-formal-state/dataset.ok" ]] || {
  echo "错误：正式数据集门禁不存在。" >&2
  exit 1
}
[[ -s "${workspace}/state/panthera-v2-single-grasp-formal-state/human-review-approval.json" ]] || {
  echo "错误：缺少人工审阅批准记录。" >&2
  exit 1
}

run_stage media_audit bash "${workspace}/bin/run_lab_panthera_v2_media_audit.sh"
run_stage rlds_4_0_0 bash "${workspace}/bin/run_lab_panthera_v2_sft_rlds.sh"
run_stage eval_config bash "${workspace}/bin/run_lab_panthera_v2_eval_config_smoke.sh"
wait_for_gpus
run_stage optimizer_smoke bash "${workspace}/bin/run_lab_openvla_panthera_v2_sft_step_smoke.sh"
wait_for_gpus
run_stage sft_10k_gpu123 bash "${workspace}/bin/run_lab_openvla_panthera_v2_sft_10k.sh"

current_stage=dev18_gpu123
printf '%s\n' "$current_stage" >"${state_root}/current-stage.txt"
wait_for_gpus
set +e
bash "${workspace}/bin/run_lab_panthera_v2_policy_eval.sh" dev
dev_status=$?
set -e
if (( dev_status != 0 )); then
  if [[ -s "${workspace}/state/panthera-v2-schema10-policy-dev18-state/eval-summary.json" ]]; then
    pipeline_status=stopped_below_dev_threshold
  fi
  exit "$dev_status"
fi
date --iso-8601=seconds >"${state_root}/dev18_gpu123.ok"

current_stage=final18_gpu123
printf '%s\n' "$current_stage" >"${state_root}/current-stage.txt"
wait_for_gpus
set +e
bash "${workspace}/bin/run_lab_panthera_v2_policy_eval.sh" final
final_status=$?
set -e
if (( final_status != 0 )); then
  if [[ -s "${workspace}/state/panthera-v2-schema10-policy-final18-state/eval-summary.json" ]]; then
    pipeline_status=stopped_below_final_threshold
  fi
  exit "$final_status"
fi
date --iso-8601=seconds >"${state_root}/final18_gpu123.ok"

current_stage=complete
pipeline_status=passed
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/pipeline.ok"
echo "schema 10 数据转换、训练、开发集与最终闭环评测全部通过。"
