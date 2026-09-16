#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
train_state="${workspace}/.panthera-phone-openvla-sft-25x7-10k-state"
model_root="${workspace}/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-25x7-10000steps"
sweep_state="${workspace}/.panthera-phone-execution-horizon-sweep-state"
stamp=$(date +%Y%m%d-%H%M%S)
report_root="${workspace}/reports/panthera-phone-vertical-sft-v3/action-horizon-25/execution-horizon-sweep-${stamp}"
horizon_spec="${PANTHERA_EXECUTION_HORIZONS:-10 15 20}"
read -r -a horizons <<<"$horizon_spec"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行。" >&2
  exit 1
fi
if [[ ! -f "${train_state}/train.ok" ]]; then
  echo "错误：25x7 训练尚未通过产物门。" >&2
  exit 1
fi
if (( ${#horizons[@]} == 0 )); then
  echo "错误：执行视野列表不能为空。" >&2
  exit 1
fi
declare -A seen=()
for horizon in "${horizons[@]}"; do
  if [[ ! "$horizon" =~ ^[1-9][0-9]*$ ]] || (( horizon > 25 )); then
    echo "错误：执行视野必须是 1..25 的整数：${horizon}" >&2
    exit 1
  fi
  if [[ -n "${seen[$horizon]:-}" ]]; then
    echo "错误：执行视野重复：${horizon}" >&2
    exit 1
  fi
  seen[$horizon]=1
done

mkdir -p "$sweep_state" "$report_root"
exec 9>"${sweep_state}/sweep.lock"
if ! flock -n 9; then
  echo "错误：执行视野 sweep 已在运行。" >&2
  exit 1
fi
printf '%s\n' "$report_root" >"${sweep_state}/report-root.txt"

write_checksums() {
  local destination=$1
  (
    cd "$destination"
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 -r sha256sum >SHA256SUMS
  )
}

for horizon in "${horizons[@]}"; do
  state="${workspace}/.panthera-phone-policy-eval-25x7-horizon-${horizon}-seed0-state"
  logs="${workspace}/logs/panthera-phone-policy-eval-25x7-horizon-${horizon}-seed0"
  destination="${report_root}/horizon-${horizon}"
  if [[ -e "$state" || -e "$logs" ]]; then
    echo "错误：发现未归档的 horizon ${horizon} 状态或日志。" >&2
    exit 1
  fi

  echo "运行 seed 0，预测视野 25，执行视野 ${horizon}……"
  set +e
  PANTHERA_POLICY_MODEL="$model_root" \
  PANTHERA_EVAL_STATE_ROOT="$state" \
  PANTHERA_EVAL_LOG_ROOT="$logs" \
  PANTHERA_EVAL_TRACE_DIR="${state}/trajectory-traces" \
  PANTHERA_EVAL_SEED_SOURCE="${workspace}/panthera-rlinf-overlay/seeds/panthera_phone_vertical_diagnostic_seeds.json" \
  PANTHERA_EVAL_TRAIN_MARKER="${train_state}/train.ok" \
  PANTHERA_EVAL_TRAIN_STATE="$train_state" \
  PANTHERA_ACTION_CHUNK=25 \
  PANTHERA_EVAL_EXECUTION_HORIZON="$horizon" \
  PANTHERA_ROBOT_PLATFORM=PANTHERA \
  PANTHERA_EVAL_TRAJECTORIES=1 \
  PANTHERA_EVAL_GPUS=1 \
  PANTHERA_EVAL_MIN_SUCCESS=0 \
  bash "${workspace}/run_lab_panthera_phone_policy_eval.sh"
  status=$?
  set -e

  mkdir -p "$destination"
  if [[ -d "$state" ]]; then
    mv "$state" "$destination/state"
  fi
  if [[ -d "$logs" ]]; then
    mv "$logs" "$destination/logs"
  fi
  if [[ -s "${destination}/state/eval-summary.json" ]]; then
    output=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_root"])' \
      "${destination}/state/eval-summary.json")
    if [[ -d "${output}/video/eval" ]]; then
      cp -a "${output}/video/eval" "$destination/videos"
    fi
  fi
  printf '%s\n' "$status" >"${destination}/runner-exit-code.txt"
  write_checksums "$destination"
  if (( status != 0 )); then
    echo "错误：horizon ${horizon} 技术运行失败，证据位于 ${destination}。" >&2
    exit "$status"
  fi
done

python3 - "$report_root" "${sweep_state}/sweep-summary.json" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
cases = []
for case_dir in sorted(root.glob("horizon-*"), key=lambda p: int(p.name.split("-")[-1])):
    summary = json.loads((case_dir / "state/eval-summary.json").read_text())
    traces = sorted((case_dir / "state/trajectory-traces").glob("*.jsonl"))
    if len(traces) != 1:
        raise SystemExit(f"expected one trace for {case_dir}, got {len(traces)}")
    rows = [json.loads(line) for line in traces[0].read_text().splitlines() if line]
    final = rows[-1]["success_metrics"]
    cases.append(
        {
            "execution_horizon": summary["execution_horizon"],
            "success_once": summary["success_once"],
            "chunks": len(rows),
            "final_success_metrics": final,
            "video_count": summary["video_count"],
        }
    )
result = {
    "status": "completed",
    "prediction_horizon": 25,
    "seed": 0,
    "cases": cases,
}
encoded = json.dumps(result, indent=2) + "\n"
Path(sys.argv[2]).write_text(encoded)
(root / "sweep-summary.json").write_text(encoded)
print(encoded, end="")
PY

write_checksums "$report_root"
date --iso-8601=seconds >"${sweep_state}/completed-at.txt"
touch "${sweep_state}/sweep.ok"
echo "执行视野 sweep 完成：${report_root}"
