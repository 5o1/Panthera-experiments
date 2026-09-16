#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
robotwin_root="${workspace}/RoboTwin"
overlay_root="${workspace}/panthera-robotwin-overlay"
state_root="${workspace}/.panthera-phone-oracle-state"
result_root="${workspace}/results/panthera-phone-vertical-oracle"
activation_script="${workspace}/activate_lab_vla.sh"
scene_runner="${workspace}/bootstrap_lab_panthera_phone_scene.sh"
oracle_gpu="${PANTHERA_PHONE_ORACLE_GPU:-2}"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock timeout nvidia-smi; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
for required in \
  "$activation_script" "$scene_runner" \
  "${overlay_root}/run_oracle_smoke.py"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少 phone-SRT oracle 文件：${required}" >&2
    exit 1
  fi
done

mkdir -p "$state_root"
exec 9>"${state_root}/oracle.lock"
echo "等待现有 phone-SRT oracle（如有）释放锁……"
flock 9
if [[ -f "${state_root}/oracle.ok" ]]; then
  python3 -m json.tool "${result_root}/summary.json"
  exit 0
fi

bash "$scene_runner"
# shellcheck disable=SC1090
source "$activation_script"
if nvidia-smi -i "$oracle_gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "错误：GPU ${oracle_gpu} 被其他实验占用，拒绝启动对齐场景 oracle。" >&2
  exit 1
fi

stamp=$(date +%Y%m%d-%H%M%S)
run_log="${state_root}/oracle-${stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"
set +e
CUDA_VISIBLE_DEVICES="$oracle_gpu" PYTHONUNBUFFERED=1 timeout --signal=INT --kill-after=60s \
  "${PANTHERA_PHONE_ORACLE_TIMEOUT:-45m}" \
  python "${overlay_root}/run_oracle_smoke.py" \
    --robotwin-root "$robotwin_root" \
    --output-root "$result_root" \
    --task-name place_vertical_cylinder_in_groove \
    --task-config panthera_phone_vertical_oracle.yml \
    --seeds 0 1 2 3 \
  2>&1 | tee "$run_log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" >"${state_root}/exit-code.txt"
if (( status != 0 )); then
  echo "错误：phone-SRT 对齐场景 oracle 未通过。" >&2
  exit "$status"
fi
python3 - "${result_root}/summary.json" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("task") != "place_vertical_cylinder_in_groove":
    raise SystemExit("phone oracle task mismatch")
if summary.get("passed") != 4 or summary.get("total") != 4:
    raise SystemExit("phone oracle did not pass 4/4 seeds")
for case in summary["cases"]:
    metrics = case.get("metrics", {})
    if metrics.get("insertion_depth_m", 0.0) < 0.03:
        raise SystemExit("phone oracle insertion depth is insufficient")
PY
if pgrep -af 'ray::|raylet|collect_data.py|eval_policy.py' | grep -v grep; then
  echo "错误：phone oracle 后仍有 RoboTwin/Ray 进程。" >&2
  exit 1
fi
if nvidia-smi -i "$oracle_gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "错误：phone oracle 后 GPU ${oracle_gpu} 仍有计算进程。" >&2
  exit 1
fi
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/oracle.ok"
echo "phone-SRT 对齐场景 oracle 通过。"
