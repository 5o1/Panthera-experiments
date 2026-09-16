#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state_root="${workspace}/.panthera-phone-openvla-sft-25x7-10k-state"
launcher_log="${state_root}/launcher.log"

if pgrep -u "$(id -u)" -x raylet >/dev/null 2>&1; then
  echo "错误：当前用户已有 Ray 集群。" >&2
  exit 1
fi
if ps -u "$(id -u)" -o comm=,args= | awk \
  '$1 ~ /^(python|torchrun)/ && /run_finetune[.]py/ { found=1 } END { exit !found }'; then
  echo "错误：当前用户已有 OpenVLA 训练进程。" >&2
  exit 1
fi
if [[ -d "$state_root" ]] && find "$state_root" -mindepth 1 -print -quit | grep -q .; then
  echo "错误：25x7 训练状态目录非空：${state_root}" >&2
  exit 1
fi
for gpu in 1 2 3; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "错误：GPU ${gpu} 当前有计算进程。" >&2
    exit 1
  fi
done

mkdir -p "$state_root"
nohup bash "${workspace}/run_lab_openvla_phone_sft_25x7_10k.sh" \
  >"$launcher_log" 2>&1 </dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" >"${state_root}/launcher-pid.txt"
echo "$launcher_pid"
