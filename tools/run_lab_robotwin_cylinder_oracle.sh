#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
robotwin_root="${workspace}/RoboTwin"
overlay_root="${workspace}/panthera-robotwin-overlay"
state_root="${workspace}/.cylinder-oracle-state"
result_root="${workspace}/results/cylinder-oracle-smoke"
activation_script="${workspace}/activate_lab_vla.sh"
expected_robotwin_commit="0008ae6800df9f75fc8de7098bacb01735fd8fd2"

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
if [[ ! -f "${workspace}/.bootstrap-state/verified.ok" ]]; then
  echo "错误：RLinf/RoboTwin Conda 环境尚未通过验收。" >&2
  exit 1
fi
if [[ ! -f "$activation_script" ]]; then
  echo "错误：缺少环境激活脚本：${activation_script}" >&2
  exit 1
fi
if [[ $(git -C "$robotwin_root" rev-parse HEAD) != "$expected_robotwin_commit" ]]; then
  echo "错误：RoboTwin checkout 不是已固定提交。" >&2
  exit 1
fi

mkdir -p "$state_root" "$result_root"
exec 9>"${state_root}/oracle.lock"
if ! flock -n 9; then
  echo "错误：另一个 cylinder oracle 流程正在运行。" >&2
  exit 1
fi

for relative_path in \
  envs/place_cylinder_in_groove.py \
  description/task_instruction/place_cylinder_in_groove.json \
  task_config/panthera_cylinder_oracle.yml \
  run_oracle_smoke.py; do
  if [[ ! -s "${overlay_root}/${relative_path}" ]]; then
    echo "错误：overlay 文件缺失或为空：${relative_path}" >&2
    exit 1
  fi
done

install -m 0644 \
  "${overlay_root}/envs/place_cylinder_in_groove.py" \
  "${robotwin_root}/envs/place_cylinder_in_groove.py"
install -m 0644 \
  "${overlay_root}/description/task_instruction/place_cylinder_in_groove.json" \
  "${robotwin_root}/description/task_instruction/place_cylinder_in_groove.json"
install -m 0644 \
  "${overlay_root}/task_config/panthera_cylinder_oracle.yml" \
  "${robotwin_root}/task_config/panthera_cylinder_oracle.yml"

# shellcheck disable=SC1090
source "$activation_script"
python -m py_compile "${robotwin_root}/envs/place_cylinder_in_groove.py"
python -m json.tool \
  "${robotwin_root}/description/task_instruction/place_cylinder_in_groove.json" \
  >/dev/null

run_stamp=$(date +%Y%m%d-%H%M%S)
run_log="${state_root}/oracle-${run_stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"

set +e
timeout --signal=INT --kill-after=60s \
  "${PANTHERA_ORACLE_TIMEOUT:-30m}" \
  python "${overlay_root}/run_oracle_smoke.py" \
    --robotwin-root "$robotwin_root" \
    --output-root "$result_root" \
    --seeds 0 1 2 3 2>&1 | tee "$run_log"
oracle_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$oracle_status" >"${state_root}/exit-code.txt"

if (( oracle_status != 0 )); then
  echo "错误：圆柱入槽 scripted oracle 未通过，退出码 ${oracle_status}。" >&2
  exit "$oracle_status"
fi
if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|AssertionError:' "$run_log"; then
  echo "错误：oracle 返回 0 但日志含致命异常，拒绝写入成功标记。" >&2
  exit 1
fi
python - "$result_root/summary.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("all_passed") is not True or summary.get("passed") != 4:
    raise SystemExit("summary does not contain four passing seeds")
for case in summary["cases"]:
    if len(case.get("stages", [])) < 8:
        raise SystemExit(f"seed {case['seed']} has incomplete milestone frames")
PY

find "$result_root" -type f -name 'contact-sheet.png' -size +0c \
  -printf '%p\n' | sort >"${state_root}/contact-sheets.txt"
if [[ $(wc -l <"${state_root}/contact-sheets.txt") -ne 4 ]]; then
  echo "错误：没有生成四张非空 seed contact sheet。" >&2
  exit 1
fi

date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/oracle.ok"
echo "圆柱入槽 scripted oracle 通过：${result_root}"
