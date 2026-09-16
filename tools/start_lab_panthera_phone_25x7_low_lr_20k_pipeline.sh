#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state="${workspace}/.panthera-phone-25x7-low-lr-20k-pipeline-state"

if [[ -d "$state" ]] && find "$state" -mindepth 1 -print -quit | grep -q .; then
  echo "错误：25x7 超低学习率 20k 流水线状态目录非空：${state}" >&2
  exit 1
fi
if pgrep -u "$(id -u)" -x raylet >/dev/null 2>&1; then
  echo "错误：当前用户已有 Ray 集群。" >&2
  exit 1
fi
if ps -u "$(id -u)" -o comm=,args= | awk \
  '$1 ~ /^(python|torchrun)/ && /run_finetune[.]py/ { found=1 } END { exit !found }'; then
  echo "错误：当前用户已有 OpenVLA 训练进程。" >&2
  exit 1
fi
for gpu in 1 2 3; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader \
    | grep -q '[0-9]'; then
    echo "错误：GPU ${gpu} 当前有计算进程。" >&2
    exit 1
  fi
done

mkdir -p "$state"
nohup bash "${workspace}/run_lab_panthera_phone_25x7_low_lr_20k_pipeline.sh" \
  >"${state}/launcher.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"${state}/launcher-pid.txt"
echo "$pid"
