#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state="${workspace}/.panthera-phone-target-assist-stage-diagnostic-state"

if [[ -d "$state" ]] && find "$state" -mindepth 1 -print -quit | grep -q .; then
  echo "错误：末端技能阶段诊断状态目录非空：${state}" >&2
  exit 1
fi
if pgrep -u "$(id -u)" -x raylet >/dev/null 2>&1; then
  echo "错误：当前用户已有 Ray 集群。" >&2
  exit 1
fi
mkdir -p "$state"
nohup bash "${workspace}/run_lab_panthera_phone_target_assist_stage_diagnostic.sh" \
  >"${state}/launcher.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"${state}/launcher-pid.txt"
echo "$pid"
