#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state_root="${workspace}/.panthera-phone-sft-pipeline-state"
dataset_runner="${workspace}/run_lab_robotwin_panthera_phone_sft_dataset.sh"
rlds_runner="${workspace}/run_lab_panthera_phone_sft_rlds.sh"
smoke_runner="${workspace}/run_lab_openvla_phone_sft_step_smoke.sh"
config_runner="${workspace}/run_lab_panthera_phone_eval_config_smoke.sh"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for required in "$dataset_runner" "$rlds_runner" "$smoke_runner" "$config_runner"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少 phone-SRT SFT 阶段脚本：${required}" >&2
    exit 1
  fi
done

mkdir -p "$state_root"
exec 9>"${state_root}/pipeline.lock"
if ! flock -n 9; then
  echo "错误：另一个 phone-SRT SFT 流水线正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/pipeline.ok" ]]; then
  python3 -m json.tool "${state_root}/pipeline-summary.json"
  exit 0
fi

echo "[1/4] 多 GPU 采集并验收 phone-SRT 对齐数据"
bash "$dataset_runner"
echo "[2/4] 转换并验收 phone-SRT RLDS"
bash "$rlds_runner"
echo "[3/4] 验收 phone-SRT 四环境闭环配置"
bash "$config_runner"
echo "[4/4] 运行 phone-SRT OpenVLA 一步优化器 smoke"
bash "$smoke_runner"

python3 - "$workspace" "${state_root}/pipeline-summary.json" <<'PY'
import json
from pathlib import Path
import sys

workspace = Path(sys.argv[1])
output = Path(sys.argv[2])
paths = {
    "dataset": workspace / ".panthera-phone-sft-dataset-state/dataset-summary.json",
    "rlds": workspace / ".panthera-phone-sft-rlds-state/rlds-summary.json",
    "eval_config": workspace / ".panthera-phone-eval-config-state/config-summary.json",
    "optimizer_smoke": workspace / ".panthera-phone-openvla-sft-smoke-state/train-summary.json",
}
summary = {"status": "passed", "scene_profile": "phone_srt_vertical_socket", "stages": {}}
for name, path in paths.items():
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed":
        raise SystemExit(f"phone-SRT stage did not pass: {name}")
    summary["stages"][name] = value
output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/pipeline.ok"
echo "phone-SRT 对齐版 SFT 上游流水线通过。"
