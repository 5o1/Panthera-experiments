#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state_root="${workspace}/state/panthera-phone-post25x7-pipeline-state"
launcher_log="${state_root}/launcher.log"

mkdir -p "$state_root"
if [[ -s "${state_root}/launcher-pid.txt" ]]; then
  old_pid=$(<"${state_root}/launcher-pid.txt")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "错误：25x7 后续评测流水线已经在运行（PID ${old_pid}）。" >&2
    exit 1
  fi
fi
if [[ -f "${state_root}/pipeline.ok" ]]; then
  echo "25x7 后续评测流水线已经通过。"
  exit 0
fi

nohup bash "${workspace}/bin/run_lab_panthera_phone_post25x7_pipeline.sh" \
  >"$launcher_log" 2>&1 </dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" >"${state_root}/launcher-pid.txt"
echo "$launcher_pid"
