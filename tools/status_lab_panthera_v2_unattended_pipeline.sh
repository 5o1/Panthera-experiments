#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state_root="${workspace}/.panthera-v2-schema10-unattended-state"
pid=""
[[ -s "${state_root}/launcher.pid" ]] && pid=$(<"${state_root}/launcher.pid")
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  echo "进程：运行中（PID=${pid}）"
else
  echo "进程：未运行"
fi
if [[ -s "${state_root}/current-stage.txt" ]]; then
  echo "阶段：$(<"${state_root}/current-stage.txt")"
fi
if [[ -s "${state_root}/pipeline-summary.json" ]]; then
  python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print("状态："+p["status"]+"，最后阶段："+p["last_stage"]+"，退出码："+str(p["exit_code"]))' \
    "${state_root}/pipeline-summary.json"
fi
echo "GPU："
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
if [[ -s "${state_root}/current-log.txt" ]]; then
  log=$(<"${state_root}/current-log.txt")
  echo "日志：${log}"
  [[ -s "$log" ]] && tail -n 20 "$log"
fi
