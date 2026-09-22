#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state_root="${workspace}/state/panthera-single-sft-pipeline-state"
dataset_state="${workspace}/state/panthera-single-sft-dataset-state"
dataset_runner="${workspace}/bin/run_lab_robotwin_panthera_sft_dataset.sh"
rlds_runner="${workspace}/bin/run_lab_panthera_sft_rlds.sh"
train_runner="${workspace}/bin/run_lab_openvla_sft_step_smoke.sh"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock python3; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
for required in "$dataset_runner" "$rlds_runner" "$train_runner"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少流水线阶段脚本：${required}" >&2
    exit 1
  fi
done

mkdir -p "$state_root" "$dataset_state"
exec 9>"${state_root}/pipeline.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera SFT 流水线正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/pipeline.ok" ]]; then
  python3 -m json.tool "${state_root}/pipeline-summary.json"
  echo "Panthera SFT 无人值守流水线已通过，无需重复运行。"
  exit 0
fi

# 当前可能有旧版采集脚本持有此锁。阻塞等待锁由内核事件唤醒，不轮询、不使用 sleep。
echo "等待当前 128-episode 采集/验收流程释放数据集锁……"
exec 8>"${dataset_state}/dataset.lock"
flock 8
flock -u 8
exec 8>&-

echo "[1/3] 验收或复用 Panthera SFT v1 数据集"
bash "$dataset_runner"

echo "[2/3] 转换并验收 Panthera RLDS"
bash "$rlds_runner"

echo "[3/3] 运行 OpenVLA 最小优化器 smoke"
bash "$train_runner"

python3 - "$workspace" "${state_root}/pipeline-summary.json" <<'PY'
import json
from pathlib import Path
import sys

workspace = Path(sys.argv[1])
output = Path(sys.argv[2])
required = {
    "dataset": workspace / ".panthera-single-sft-dataset-state/dataset-summary.json",
    "rlds": workspace / ".panthera-single-sft-rlds-state/rlds-summary.json",
    "optimizer_smoke": workspace / ".panthera-single-openvla-sft-smoke-state/train-summary.json",
}
summary = {"status": "passed", "stages": {}}
for name, path in required.items():
    if not path.is_file():
        raise SystemExit(f"missing stage summary: {path}")
    stage = json.loads(path.read_text(encoding="utf-8"))
    if stage.get("status") != "passed":
        raise SystemExit(f"stage did not pass: {name}")
    summary["stages"][name] = stage
output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/pipeline.ok"
echo "单 Panthera SFT 无人值守流水线通过。"
