#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
upstream_state="${workspace}/.panthera-phone-sft-pipeline-state"
state_root="${workspace}/.panthera-phone-policy-pipeline-state"
media_runner="${workspace}/run_lab_panthera_phone_media_audit.sh"
train_runner="${workspace}/run_lab_openvla_phone_sft.sh"
eval_runner="${workspace}/run_lab_panthera_phone_policy_eval.sh"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for required in "$media_runner" "$train_runner" "$eval_runner"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少 phone-SRT 策略阶段脚本：${required}" >&2
    exit 1
  fi
done

mkdir -p "$state_root" "$upstream_state"
exec 9>"${state_root}/pipeline.lock"
if ! flock -n 9; then
  echo "错误：另一个 phone-SRT 策略流水线正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/pipeline.ok" ]]; then
  python3 -m json.tool "${state_root}/pipeline-summary.json"
  exit 0
fi

echo "等待 phone-SRT 数据/RLDS/配置/smoke 流水线释放锁……"
exec 8>"${upstream_state}/pipeline.lock"
flock 8
flock -u 8
exec 8>&-
if [[ ! -f "${upstream_state}/pipeline.ok" ]]; then
  echo "错误：phone-SRT 上游结束但没有通过标记。" >&2
  exit 1
fi

echo "[1/3] 逐集核对对齐版 HDF5、语言和连续视频"
bash "$media_runner"
echo "[2/3] 在选定 GPU 上训练 phone-SRT 对齐版 OpenVLA"
bash "$train_runner"
echo "[3/3] 四环境运行 16 条 held-out 闭环轨迹"
bash "$eval_runner"

python3 - "$workspace" "${state_root}/pipeline-summary.json" <<'PY'
import json
from pathlib import Path
import sys

workspace = Path(sys.argv[1])
output = Path(sys.argv[2])
paths = {
    "media": workspace / ".panthera-phone-media-audit-state/media-summary.json",
    "training": workspace / ".panthera-phone-openvla-sft-state/train-summary.json",
    "evaluation": workspace / ".panthera-phone-policy-eval-state/eval-summary.json",
}
summary = {"status": "passed", "scene_profile": "phone_srt_vertical_socket", "stages": {}}
for name, path in paths.items():
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed":
        raise SystemExit(f"phone-SRT policy stage did not pass: {name}")
    summary["stages"][name] = value
output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/pipeline.ok"
echo "phone-SRT 对齐仿真全部通过。"
