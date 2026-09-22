#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state_root="${workspace}/.panthera-phone-policy-eval-l1-probe-state"
log_root="${workspace}/logs/panthera-phone-policy-eval-l1-probe"
launcher_log="${log_root}/launcher.log"

if pgrep -u "$(id -u)" -f 'eval_embodied_agent.py' >/dev/null 2>&1; then
  echo "错误：当前用户已有策略评测进程。" >&2
  exit 1
fi
if pgrep -u "$(id -u)" -x raylet >/dev/null 2>&1; then
  echo "错误：当前用户已有 Ray 集群。" >&2
  exit 1
fi
if [[ -d "$state_root" ]] && find "$state_root" -mindepth 1 -print -quit | grep -q .; then
  echo "错误：诊断状态目录非空，请先归档：${state_root}" >&2
  exit 1
fi

mkdir -p "$state_root" "$log_root"
nohup bash "${workspace}/run_lab_panthera_phone_policy_eval_probe.sh" \
  >"$launcher_log" 2>&1 </dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" >"${state_root}/launcher-pid.txt"
echo "$launcher_pid"
