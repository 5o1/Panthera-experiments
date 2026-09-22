#!/usr/bin/env bash
# Evaluate immutable 28-D checkpoints on GPU0 while training continues on
# GPU1-3. Poll new checkpoints until TRAINING_COMPLETE closes the backlog.
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
queue_root="${CI_DYNAMICS_QUEUE_ROOT:-${workspace}/state/dynamics-overfit-queue}"
runner="${workspace}/pipelines/ci/evaluate_numbered_overfit_checkpoint_pairs.sh"
mkdir -p "$queue_root" "${workspace}/state"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行。" >&2
  exit 1
fi
[[ -x "$runner" ]] || { echo "错误：缺少 ${runner}" >&2; exit 1; }

run_id_file="${queue_root}/started-run.txt"
if [[ ! -s "$run_id_file" ]]; then
  echo "错误：找不到 28 维训练 run id。" >&2
  touch "${queue_root}/evaluation.blocked"
  exit 1
fi
run_id=$(cat "$run_id_file")
run_root="${workspace}/ci/overfit-ep2-dynamics-24h/${run_id}"
if [[ ! -s "${run_root}/run/overfit/training.json" ]]; then
  echo "错误：28 维训练尚未写出训练合同，不启动评测。" >&2
  touch "${queue_root}/evaluation.blocked"
  exit 1
fi

date -u +%Y-%m-%dT%H:%M:%SZ >"${queue_root}/evaluation-started-at.txt"
export CI_OVERFIT_RUN_ROOT="$run_root"
export CI_OVERFIT_EVAL_GPU="${CI_DYNAMICS_EVAL_GPU:-0}"
export CI_OVERFIT_EVAL_MIN_FREE_MIB="${CI_DYNAMICS_EVAL_MIN_FREE_MIB:-28000}"
export CI_OVERFIT_PAIR_MIN_STEP="${CI_DYNAMICS_EVAL_MIN_STEP:-5000}"
export CI_OVERFIT_TRAINING_COMPLETE_MARKER="${run_root}/TRAINING_COMPLETE"

set +e
bash "$runner"
evaluation_exit=$?
set -e
printf '%s\n' "$evaluation_exit" >"${queue_root}/evaluation-exit-code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"${queue_root}/evaluation-finished-at.txt"
if [[ "$evaluation_exit" != "0" ]]; then
  echo "错误：28 维 checkpoint 评测异常退出 ${evaluation_exit}。" >&2
  exit "$evaluation_exit"
fi

touch "${queue_root}/evaluation.complete"
echo "28 维 checkpoint 双模式评测全部完成：${run_root}"
