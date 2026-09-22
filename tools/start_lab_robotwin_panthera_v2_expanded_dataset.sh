#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state_root="${workspace}/state/panthera-v2-single-grasp-expanded-state"
runner="${workspace}/bin/run_lab_robotwin_panthera_v2_expanded_dataset.sh"
mkdir -p "$state_root"
[[ -x "$runner" ]] || { echo "错误：缺少可执行采集器 ${runner}。" >&2; exit 1; }

if [[ -f "${state_root}/automated-audit.ok" ]]; then
  echo "扩充数据已完成自动审计，当前等待人工验收；不会自动重复生成或启动训练。"
  exit 0
fi
if [[ -s "${state_root}/launcher.pid" ]]; then
  old_pid=$(<"${state_root}/launcher.pid")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "扩充数据任务已在运行，PID=${old_pid}。"
    exit 0
  fi
fi

stamp=$(date +%Y%m%d-%H%M%S)
log="${state_root}/expanded-dataset-${stamp}.log"
nohup setsid bash "$runner" >"$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >"${state_root}/launcher.pid"
printf '%s\n' "$log" >"${state_root}/current-log.txt"
if ! kill -0 "$pid" 2>/dev/null; then
  echo "错误：扩充数据任务启动后立即退出。" >&2
  tail -n 80 "$log" >&2
  exit 1
fi
echo "已启动 1280 条无人值守扩充数据任务：PID=${pid}"
echo "完成后将停在人工验收门前，不会自动转换 RLDS 或训练。"
echo "日志：${log}"
