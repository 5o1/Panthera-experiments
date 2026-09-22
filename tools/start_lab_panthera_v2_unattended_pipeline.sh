#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state_root="${workspace}/state/panthera-v2-schema10-unattended-state"
runner="${workspace}/bin/run_lab_panthera_v2_unattended_pipeline.sh"
mkdir -p "$state_root"
[[ -x "$runner" ]] || { echo "错误：缺少可执行流水线 ${runner}。" >&2; exit 1; }

if [[ -s "${state_root}/pipeline-summary.json" ]]; then
  terminal_status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' \
    "${state_root}/pipeline-summary.json")
  case "$terminal_status" in
    passed|stopped_below_dev_threshold|stopped_below_final_threshold)
      echo "流水线处于终态：${terminal_status}，不会自动重复实验。"
      exit 0
      ;;
  esac
fi

if [[ -s "${state_root}/launcher.pid" ]]; then
  old_pid=$(<"${state_root}/launcher.pid")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "流水线已在运行，PID=${old_pid}。"
    exit 0
  fi
fi

stamp=$(date +%Y%m%d-%H%M%S)
log="${state_root}/pipeline-${stamp}.log"
nohup setsid bash "$runner" >"$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >"${state_root}/launcher.pid"
printf '%s\n' "$log" >"${state_root}/current-log.txt"
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
  echo "错误：流水线启动后立即退出。" >&2
  tail -n 80 "$log" >&2
  exit 1
fi
echo "已启动 schema 10 无人值守流水线：PID=${pid}"
echo "日志：${log}"
