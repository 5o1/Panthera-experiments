#!/usr/bin/env bash
# Wait for the active 7-D overfit lock, then start the 28-D successor unchanged.
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state_root="${CI_DYNAMICS_STATE_ROOT:-${workspace}/state/dynamics-overfit-gpu0}"
dataset_root="${CI_DYNAMICS_DATASET_ROOT:-${workspace}/ci/dynamics-ep2/dataset}"
queue_root="${CI_DYNAMICS_QUEUE_ROOT:-${workspace}/state/dynamics-overfit-queue}"
mkdir -p "$queue_root" "${workspace}/state"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行。" >&2
  exit 1
fi
if [[ ! -f "${state_root}/prepare.ok" ]]; then
  echo "错误：GPU0 dynamics smoke 尚未通过。" >&2
  exit 1
fi

exec 8>"${workspace}/state/single-trajectory-overfit.lock"
echo "等待当前单轨迹过拟合释放 GPU1–3……"
flock 8

latest_file="${workspace}/state/single-trajectory-overfit-24h.latest"
if [[ ! -s "$latest_file" ]]; then
  echo "错误：找不到当前过拟合运行记录。" >&2
  exit 1
fi
previous="$(cat "$latest_file")"
previous_root="${workspace}/ci/overfit-ep2-24h/${previous}"
if [[ ! -f "${previous_root}/TRAINING_COMPLETE" ]]; then
  echo "错误：前序任务没有正常收口，28-D 训练不自动启动。" >&2
  touch "${queue_root}/blocked"
  exit 1
fi

run_id="dynamics-28d-ep2-24h-$(date -u +%Y%m%dT%H%M%SZ)"
export CI_OVERFIT_DATASET_ROOT="$dataset_root"
export CI_OVERFIT_DATASET_NAME=panthera_phone_cylinder_socket_v2_dynamics
export CI_OVERFIT_PROPRIO_DIM=28
export CI_OVERFIT_TRAIN_GPUS=1,2,3
export CI_OVERFIT_BATCH_SIZE=6
export CI_OVERFIT_DURATION=24h
export CI_OVERFIT_SAVE_FREQ=5000
export CI_OVERFIT_WANDB_MODE=online
export CI_OVERFIT_WANDB_ENTITY=assanekowww
export CI_OVERFIT_WANDB_PROJECT=panthera-ci-overfit-dynamics-24h
export CI_OVERFIT_RUN_ID="$run_id"
export CI_OVERFIT_ROOT="${workspace}/ci/overfit-ep2-dynamics-24h/${run_id}"
export CI_OVERFIT_INHERITED_LOCK_FD=8

printf '%s\n' "$run_id" >"${queue_root}/started-run.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"${queue_root}/started-at.txt"
exec bash "${workspace}/pipelines/ci/run_single_trajectory_overfit_24h.sh"
