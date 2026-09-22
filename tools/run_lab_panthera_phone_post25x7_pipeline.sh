#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
train_state="${workspace}/state/panthera-phone-openvla-sft-25x7-10k-state"
diagnostic_state="${workspace}/state/panthera-phone-policy-eval-25x7-diagnostic-state"
formal_state="${workspace}/state/panthera-phone-policy-eval-25x7-state"
archive_root="${workspace}/reports/panthera-phone-vertical-sft-v3/action-horizon-25"
pipeline_state="${workspace}/state/panthera-phone-post25x7-pipeline-state"
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
  echo "错误：25x7 后续评测流水线已经在运行。" >&2
  exit 1
fi

archive_if_nonempty() {
  local source=$1
  local label=$2
  if [[ -d "$source" ]] && find "$source" -mindepth 1 -print -quit | grep -q .; then
    mv "$source" "${archive_root}/archives/${label}-$(date +%Y%m%d-%H%M%S)"
  fi
}

write_checksums() {
  local destination=$1
  (
    cd "$destination"
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 -r sha256sum >SHA256SUMS
  )
}

archive_eval() {
  local state=$1
  local label=$2
  local destination="${archive_root}/${label}-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$destination"
  local artifact
  for artifact in eval-summary.json resolved-config.yaml run-log.txt exit-code.txt; do
    if [[ -f "${state}/${artifact}" ]]; then
      cp "${state}/${artifact}" "$destination/"
    fi
  done
  if [[ -s "${state}/run-log.txt" ]]; then
    local run_log
    run_log=$(<"${state}/run-log.txt")
    if [[ -f "$run_log" ]]; then
      cp "$run_log" "$destination/eval.log"
    fi
  fi
  if [[ -d "${state}/trajectory-traces" ]]; then
    cp -a "${state}/trajectory-traces" "$destination/"
  fi
  if [[ -s "${state}/eval-summary.json" ]]; then
    local output
    output=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("output_root", ""))' "${state}/eval-summary.json")
    if [[ -n "$output" && -d "${output}/video/eval" ]]; then
      cp -a "${output}/video/eval" "$destination/videos"
    fi
  fi
  write_checksums "$destination"
  printf '%s\n' "$destination"
}

echo "等待 25x7 训练通过产物门……"
while [[ ! -f "${train_state}/train.ok" ]]; do
  if [[ -f "${train_state}/exit-code.txt" ]]; then
    status=$(<"${train_state}/exit-code.txt")
    if [[ "$status" != 0 ]]; then
      echo "错误：25x7 训练退出码为 ${status}。" >&2
      exit "$status"
    fi
  fi
  if [[ -s "${train_state}/launcher-pid.txt" ]]; then
    train_pid=$(<"${train_state}/launcher-pid.txt")
    if [[ "$train_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$train_pid" 2>/dev/null; then
      echo "错误：25x7 训练 launcher 已终止，但 train.ok 未生成。" >&2
      exit 1
    fi
  fi
  sleep "$poll_seconds"
done

model_root=$(<"${train_state}/final-model.txt")
for required in \
  "${model_root}/config.json" \
  "${model_root}/dataset_statistics.json" \
  "${model_root}/action_head--latest_checkpoint.pt" \
  "${model_root}/proprio_projector--latest_checkpoint.pt"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：25x7 模型产物缺失或为空：${required}" >&2
    exit 1
  fi
done
printf '%s\n' "$model_root" >"${pipeline_state}/model.txt"
date --iso-8601=seconds >"${pipeline_state}/training-gate-passed-at.txt"
archive_if_nonempty "$diagnostic_state" diagnostic-state
archive_if_nonempty "${workspace}/logs/panthera-phone-policy-eval-25x7-diagnostic" diagnostic-logs

set +e
PANTHERA_POLICY_MODEL="$model_root" \
bash "${workspace}/bin/run_lab_panthera_phone_policy_eval_25x7_diagnostic.sh"
diagnostic_status=$?
set -e
diagnostic_archive=$(archive_eval "$diagnostic_state" diagnostic)
if [[ -f "${train_state}/train-summary.json" ]]; then
  cp "${train_state}/train-summary.json" "$diagnostic_archive/"
fi
if (( diagnostic_status != 0 )); then
  printf '%s\n' "$diagnostic_status" >"${diagnostic_archive}/exit-code.txt"
fi
write_checksums "$diagnostic_archive"
if (( diagnostic_status != 0 )); then
  echo "错误：25x7 双轨诊断未通过；证据位于 ${diagnostic_archive}。" >&2
  exit "$diagnostic_status"
fi
date --iso-8601=seconds >"${pipeline_state}/diagnostic-passed-at.txt"

archive_if_nonempty "$formal_state" formal-state
archive_if_nonempty "${workspace}/logs/panthera-phone-policy-eval-25x7" formal-logs
set +e
PANTHERA_POLICY_MODEL="$model_root" \
PANTHERA_EVAL_STATE_ROOT="$formal_state" \
PANTHERA_EVAL_LOG_ROOT="${workspace}/logs/panthera-phone-policy-eval-25x7" \
PANTHERA_EVAL_TRACE_DIR="${formal_state}/trajectory-traces" \
PANTHERA_EVAL_TRAIN_MARKER="${train_state}/train.ok" \
PANTHERA_EVAL_TRAIN_STATE="$train_state" \
PANTHERA_ACTION_CHUNK=25 \
PANTHERA_ROBOT_PLATFORM=PANTHERA \
PANTHERA_EVAL_GPUS=1,2 \
PANTHERA_EVAL_TRAJECTORIES=16 \
PANTHERA_EVAL_MIN_SUCCESS=0.75 \
bash "${workspace}/bin/run_lab_panthera_phone_policy_eval.sh"
formal_status=$?
set -e
formal_archive=$(archive_eval "$formal_state" formal-16)
if (( formal_status != 0 )); then
  printf '%s\n' "$formal_status" >"${formal_archive}/exit-code.txt"
fi
write_checksums "$formal_archive"
if (( formal_status != 0 )); then
  echo "错误：25x7 正式评测未通过；证据位于 ${formal_archive}。" >&2
  exit "$formal_status"
fi

date --iso-8601=seconds >"${pipeline_state}/formal-passed-at.txt"
date --iso-8601=seconds >"${pipeline_state}/completed-at.txt"
touch "${pipeline_state}/pipeline.ok"
echo "25x7 模型通过正式仿真评测：${formal_archive}"
