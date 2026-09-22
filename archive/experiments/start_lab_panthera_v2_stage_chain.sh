#!/usr/bin/env bash
# 启动后台串流编排。可重复执行：已在运行则不重复拉起。
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
chain_state="${workspace}/.panthera-v2-stage-chain-state"
runner="${workspace}/run_lab_panthera_v2_stage_chain.sh"
mkdir -p "$chain_state"
[[ -x "$runner" ]] || { echo "错误：缺少编排脚本 ${runner}。" >&2; exit 1; }

if [[ -s "${chain_state}/chain.pid" ]]; then
  old_pid=$(<"${chain_state}/chain.pid")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "阶段串流已在运行，PID=${old_pid}。"
    exit 0
  fi
fi
if [[ -f "${chain_state}/chain.ok" ]]; then
  echo "阶段串流已全部完成；不会再重复执行。"
  exit 0
fi

stamp=$(date +%Y%m%d-%H%M%S)
log="${chain_state}/launcher-${stamp}.log"
nohup setsid bash "$runner" >"$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$log" >"${chain_state}/current-log.txt"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "错误：阶段串流启动后立即退出。" >&2
  tail -n 40 "$log" >&2
  exit 1
fi
echo "已启动阶段串流编排：PID=${pid}"
echo "阶段顺序：固定机位采集 → 等待批准 → 随机机位媒体审计/RLDS/smoke/SFT → 等待批准 → 固定机位同链路"
echo "日志：${log}"
