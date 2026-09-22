#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
# Upstream is read-only under externals/; this runs against a runtime
# assembled from it plus the overlay plus the patches.
upstream_root="${workspace}/externals/RoboTwin"
robotwin_root="${workspace}/runtime/robotwin"
python3 "${workspace}/pipelines/assemble_runtime.py" \
  --upstream robotwin --source "$upstream_root" --runtime "$robotwin_root" >/dev/null
overlay_root="${workspace}/overlays/robotwin"
oracle_runner="${workspace}/packages/panthera_sim/diagnostics/run_oracle_smoke.py"
state_root="${workspace}/state/panthera-cylinder-oracle-state"
result_root="${workspace}/results/panthera-cylinder-oracle-smoke"
activation_script="${workspace}/tools/activate_lab_vla.sh"
task_config="panthera_cylinder_oracle_panthera.yml"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock git timeout; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  fi
done
if [[ ! -f "${state_root%/*}/.panthera-embodiment-state/verified.ok" ]]; then
  echo "错误：Panthera embodiment 尚未通过静态与规划验收。" >&2
  exit 1
fi
if [[ ! -f "$activation_script" ]]; then
  echo "错误：缺少环境激活脚本：${activation_script}" >&2
  exit 1
fi

mkdir -p "$state_root" "$result_root"
exec 9>"${state_root}/oracle.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera cylinder oracle 正在运行。" >&2
  exit 1
fi

for relative_path in \
  envs/place_cylinder_in_groove.py \
  description/task_instruction/place_cylinder_in_groove.json \
  "task_config/${task_config}"; do
  if [[ ! -s "${overlay_root}/${relative_path}" ]]; then
    echo "错误：overlay 文件缺失或为空：${relative_path}" >&2
    exit 1
  fi
done

# shellcheck disable=SC1090
source "$activation_script"
python -m py_compile "${robotwin_root}/envs/place_cylinder_in_groove.py"

run_stamp=$(date +%Y%m%d-%H%M%S)
run_log="${state_root}/oracle-${run_stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"

set +e
timeout --signal=INT --kill-after=60s \
  "${PANTHERA_ORACLE_TIMEOUT:-30m}" \
  python "$oracle_runner" \
    --robotwin-root "$robotwin_root" \
    --output-root "$result_root" \
    --task-config "$task_config" \
    --seeds 0 1 2 3 2>&1 | tee "$run_log"
oracle_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$oracle_status" >"${state_root}/exit-code.txt"

if (( oracle_status != 0 )); then
  echo "错误：双 Panthera 圆柱入槽 oracle 未通过，退出码 ${oracle_status}。" >&2
  exit "$oracle_status"
fi
if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|AssertionError:' "$run_log"; then
  echo "错误：oracle 返回 0 但日志含致命异常，拒绝写入成功标记。" >&2
  exit 1
fi
python - "$result_root/summary.json" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("all_passed") is not True or summary.get("passed") != 4:
    raise SystemExit("summary does not contain four passing Panthera seeds")
if summary.get("embodiment") != "panthera+panthera":
    raise SystemExit("summary did not use the Panthera embodiment")
PY

find "$result_root" -type f -name 'contact-sheet.png' -size +0c \
  -printf '%p\n' | sort >"${state_root}/contact-sheets.txt"
if [[ $(wc -l <"${state_root}/contact-sheets.txt") -ne 4 ]]; then
  echo "错误：没有生成四张非空 Panthera contact sheet。" >&2
  exit 1
fi

date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/oracle.ok"
echo "双 Panthera 圆柱入槽 oracle 通过：${result_root}"
