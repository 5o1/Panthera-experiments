#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
train_state="${workspace}/state/panthera-phone-openvla-sft-20k-state"
diagnostic_state="${workspace}/state/panthera-phone-policy-eval-diagnostic-state"
formal_state="${workspace}/state/panthera-phone-policy-eval-state"
archive_root="${workspace}/reports/panthera-phone-vertical-sft-v3/post20k"
pipeline_state="${workspace}/state/panthera-phone-post20k-pipeline-state"
poll_seconds="${PANTHERA_PIPELINE_POLL_SECONDS:-30}"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行。" >&2
  exit 1
fi
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：PANTHERA_PIPELINE_POLL_SECONDS 必须是正整数。" >&2
  exit 1
fi

mkdir -p "$pipeline_state" "$archive_root/archives"
exec 9>"${pipeline_state}/pipeline.lock"
if ! flock -n 9; then
  echo "错误：20k 后续评测流水线已经在运行。" >&2
  exit 1
fi

archive_if_nonempty() {
  local source=$1
  local label=$2
  if [[ -d "$source" ]] && find "$source" -mindepth 1 -print -quit | grep -q .; then
    local stamp
    stamp=$(date +%Y%m%d-%H%M%S)
    mv "$source" "${archive_root}/archives/${label}-${stamp}"
  fi
}

echo "等待 20k 续训通过产物门……"
while [[ ! -f "${train_state}/train.ok" ]]; do
  if [[ -f "${train_state}/exit-code.txt" ]]; then
    status=$(<"${train_state}/exit-code.txt")
    if [[ "$status" != 0 ]]; then
      echo "错误：20k 续训退出码为 ${status}，停止后续评测。" >&2
      exit "$status"
    fi
  fi
  if [[ -s "${train_state}/launcher-pid.txt" ]]; then
    train_pid=$(<"${train_state}/launcher-pid.txt")
    if [[ "$train_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$train_pid" 2>/dev/null; then
      echo "错误：训练 launcher 已终止，但 train.ok 未生成；停止等待并保留状态。" >&2
      exit 1
    fi
  fi
  sleep "$poll_seconds"
done

model_root=$(<"${train_state}/final-model.txt")
for required in \
  "${model_root}/config.json" \
  "${model_root}/dataset_statistics.json"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：20k 模型缺少 ${required}。" >&2
    exit 1
  fi
done
if ! find "$model_root" -maxdepth 2 -type f -name 'action_head--*_checkpoint.pt' -size +0c | grep -q .; then
  echo "错误：20k 模型缺少连续动作头。" >&2
  exit 1
fi

printf '%s\n' "$model_root" >"${pipeline_state}/model.txt"
date --iso-8601=seconds >"${pipeline_state}/training-gate-passed-at.txt"

archive_if_nonempty "$diagnostic_state" diagnostic-state
archive_if_nonempty "${workspace}/logs/panthera-phone-policy-eval-diagnostic" diagnostic-logs

echo "运行训练种子与留出种子的 2 条闭环诊断……"
set +e
PANTHERA_POLICY_MODEL="$model_root" \
PANTHERA_EVAL_TRAIN_MARKER="${train_state}/train.ok" \
PANTHERA_EVAL_TRAIN_STATE="$train_state" \
PANTHERA_EVAL_MIN_SUCCESS=1.0 \
bash "${workspace}/bin/run_lab_panthera_phone_policy_eval_diagnostic.sh"
diagnostic_status=$?
set -e

diagnostic_archive="${archive_root}/diagnostic-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$diagnostic_archive"
cp "${diagnostic_state}/eval-summary.json" "$diagnostic_archive/"
cp "${diagnostic_state}/resolved-config.yaml" "$diagnostic_archive/"
cp "$(<"${diagnostic_state}/run-log.txt")" "${diagnostic_archive}/eval.log"
if [[ -d "${diagnostic_state}/trajectory-traces" ]]; then
  cp -a "${diagnostic_state}/trajectory-traces" "$diagnostic_archive/"
fi
diagnostic_output=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_root"])' "${diagnostic_state}/eval-summary.json")
cp -a "${diagnostic_output}/video/eval" "$diagnostic_archive/videos"
if (( diagnostic_status != 0 )); then
  printf '%s\n' "$diagnostic_status" >"${diagnostic_archive}/exit-code.txt"
  echo "错误：诊断未通过；失败证据已归档至 ${diagnostic_archive}。" >&2
  exit "$diagnostic_status"
fi
date --iso-8601=seconds >"${pipeline_state}/diagnostic-passed-at.txt"

archive_if_nonempty "$formal_state" formal-state
archive_if_nonempty "${workspace}/logs/panthera-phone-policy-eval" formal-logs

echo "诊断 2/2 通过，运行 16 条留出种子正式评测……"
set +e
PANTHERA_POLICY_MODEL="$model_root" \
PANTHERA_EVAL_TRAIN_MARKER="${train_state}/train.ok" \
PANTHERA_EVAL_TRAIN_STATE="$train_state" \
PANTHERA_EVAL_GPUS=1,2 \
PANTHERA_EVAL_TRAJECTORIES=16 \
PANTHERA_EVAL_MIN_SUCCESS=0.75 \
PANTHERA_EVAL_TRACE_DIR="${formal_state}/trajectory-traces" \
bash "${workspace}/bin/run_lab_panthera_phone_policy_eval.sh"
formal_status=$?
set -e

formal_archive="${archive_root}/formal-16-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$formal_archive"
cp "${formal_state}/eval-summary.json" "$formal_archive/"
cp "${formal_state}/resolved-config.yaml" "$formal_archive/"
cp "$(<"${formal_state}/run-log.txt")" "${formal_archive}/eval.log"
if [[ -d "${formal_state}/trajectory-traces" ]]; then
  cp -a "${formal_state}/trajectory-traces" "$formal_archive/"
fi
formal_output=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_root"])' "${formal_state}/eval-summary.json")
cp -a "${formal_output}/video/eval" "$formal_archive/videos"
if (( formal_status != 0 )); then
  printf '%s\n' "$formal_status" >"${formal_archive}/exit-code.txt"
  echo "错误：正式评测未通过；失败证据已归档至 ${formal_archive}。" >&2
  exit "$formal_status"
fi
date --iso-8601=seconds >"${pipeline_state}/formal-passed-at.txt"
touch "${pipeline_state}/pipeline.ok"
echo "20k 模型正式仿真评测通过，证据已归档至 ${archive_root}。"
