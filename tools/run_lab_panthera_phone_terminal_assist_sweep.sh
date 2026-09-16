#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
train_state="${workspace}/.panthera-phone-openvla-sft-25x7-10k-state"
model_root="${workspace}/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-25x7-10000steps"
sweep_state="${workspace}/.panthera-phone-terminal-assist-sweep-state"
stamp=$(date +%Y%m%d-%H%M%S)
report_root="${workspace}/reports/panthera-phone-vertical-sft-v3/action-horizon-25/terminal-assist-sweep-${stamp}"
distance_spec="${PANTHERA_TERMINAL_ASSIST_DISTANCES_M:-0.025 0.040 0.055}"
read -r -a distances <<<"$distance_spec"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行。" >&2
  exit 1
fi
if [[ ! -f "${train_state}/train.ok" ]]; then
  echo "错误：25x7 训练尚未通过产物门。" >&2
  exit 1
fi
if (( ${#distances[@]} == 0 )); then
  echo "错误：终端插入距离列表不能为空。" >&2
  exit 1
fi
for distance in "${distances[@]}"; do
  python3 - "$distance" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or not 0.0 < value <= 0.080:
    raise SystemExit(f"terminal assist distance must be in (0, 0.080], got {value}")
PY
done

mkdir -p "$sweep_state" "$report_root"
exec 9>"${sweep_state}/sweep.lock"
if ! flock -n 9; then
  echo "错误：终端插入技能 sweep 已在运行。" >&2
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

for distance in "${distances[@]}"; do
  label=${distance/./p}
  state="${workspace}/.panthera-phone-policy-eval-25x7-assist-${label}-seed0-state"
  logs="${workspace}/logs/panthera-phone-policy-eval-25x7-assist-${label}-seed0"
  destination="${report_root}/assist-${label}m"
  if [[ -e "$state" || -e "$logs" ]]; then
    echo "错误：发现未归档的 assist ${distance} 状态或日志。" >&2
    exit 1
  fi

  echo "运行 seed 0，预测视野 25，执行视野 20，终端直线插入 ${distance} m……"
  set +e
  PANTHERA_POLICY_MODEL="$model_root" \
  PANTHERA_EVAL_STATE_ROOT="$state" \
  PANTHERA_EVAL_LOG_ROOT="$logs" \
  PANTHERA_EVAL_TRACE_DIR="${state}/trajectory-traces" \
  PANTHERA_EVAL_SEED_SOURCE="${workspace}/panthera-rlinf-overlay/seeds/panthera_phone_vertical_diagnostic_seeds.json" \
  PANTHERA_EVAL_TRAIN_MARKER="${train_state}/train.ok" \
  PANTHERA_EVAL_TRAIN_STATE="$train_state" \
  PANTHERA_ACTION_CHUNK=25 \
  PANTHERA_EVAL_EXECUTION_HORIZON=20 \
  PANTHERA_TERMINAL_INSERTION_ASSIST_M="$distance" \
  PANTHERA_ROBOT_PLATFORM=PANTHERA \
  PANTHERA_EVAL_TRAJECTORIES=1 \
  PANTHERA_EVAL_GPUS=1 \
  PANTHERA_EVAL_MIN_SUCCESS=0 \
  bash "${workspace}/run_lab_panthera_phone_policy_eval.sh"
  status=$?
  set -e

  mkdir -p "$destination"
  [[ ! -d "$state" ]] || mv "$state" "$destination/state"
  [[ ! -d "$logs" ]] || mv "$logs" "$destination/logs"
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
    echo "错误：assist ${distance} 技术运行失败，证据位于 ${destination}。" >&2
    exit "$status"
  fi
done

python3 - "$report_root" "${sweep_state}/sweep-summary.json" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
cases = []
for case_dir in sorted(root.glob("assist-*m")):
    summary = json.loads((case_dir / "state/eval-summary.json").read_text())
    traces = sorted((case_dir / "state/trajectory-traces").glob("*.jsonl"))
    if len(traces) != 1:
        raise SystemExit(f"expected one trace for {case_dir}, got {len(traces)}")
    rows = [json.loads(line) for line in traces[0].read_text().splitlines() if line]
    assists = [
        row["terminal_insertion_assist"]
        for row in rows
        if row.get("terminal_insertion_assist", {}).get("attempted")
    ]
    cases.append(
        {
            "terminal_insertion_assist_m": summary["terminal_insertion_assist_m"],
            "success_once": summary["success_once"],
            "chunks": len(rows),
            "assist_attempts": assists,
            "final_success_metrics": rows[-1]["success_metrics"],
            "video_count": summary["video_count"],
        }
    )
result = {
    "status": "completed",
    "prediction_horizon": 25,
    "execution_horizon": 20,
    "seed": 0,
    "assist_uses_object_pose": False,
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
echo "终端插入技能 sweep 完成：${report_root}"
