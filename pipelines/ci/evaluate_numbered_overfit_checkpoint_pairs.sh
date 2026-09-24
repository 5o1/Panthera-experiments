#!/usr/bin/env bash
# Poll immutable numbered checkpoints and make both execution-mode previews for
# every new checkpoint. Each checkpoint is handled as one unattended job.
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
run_root="${CI_OVERFIT_RUN_ROOT:?set CI_OVERFIT_RUN_ROOT}"
training_unit="${CI_OVERFIT_TRAINING_UNIT:-panthera-overfit-ep2-24h-b6-20260920T140123Z.service}"
training_complete_marker="${CI_OVERFIT_TRAINING_COMPLETE_MARKER:-}"

checkpoint_complete() {
  local checkpoint=$1
  local step=$2
  local required
  for required in \
    "$checkpoint/lora_adapter/adapter_model.safetensors" \
    "$checkpoint/lora_adapter/adapter_config.json" \
    "$checkpoint/action_head--${step}_checkpoint.pt" \
    "$checkpoint/proprio_projector--${step}_checkpoint.pt" \
    "$checkpoint/dataset_statistics.json"; do
    [[ -s "$required" ]] || return 1
  done
}
poll_seconds="${CI_OVERFIT_EVAL_POLL_SECONDS:-60}"
min_step="${CI_OVERFIT_PAIR_MIN_STEP:-0}"
pair_runner="${workspace}/pipelines/ci/evaluate_overfit_checkpoint_preview_pair.sh"
mosaic="${workspace}/packages/panthera_sim/mosaic_videos.py"

[[ -x "$pair_runner" ]] || { echo "错误：缺少 ${pair_runner}" >&2; exit 1; }
[[ -s "$mosaic" ]] || { echo "错误：缺少 ${mosaic}" >&2; exit 1; }

training_is_active() {
  if [[ -n "$training_complete_marker" ]]; then
    if [[ -f "$training_complete_marker" ]]; then
      return 1
    fi
    training_exit="$(dirname "$training_complete_marker")/exit-code.txt"
    if [[ -f "$training_exit" ]]; then
      echo "错误：训练已经退出但没有 TRAINING_COMPLETE：${training_exit}" >&2
      exit 1
    fi
    return 0
  fi
  systemctl --user is-active --quiet "$training_unit"
}

while true; do
  pending_checkpoint=0
  mapfile -t checkpoints < <(
    find "${run_root}/run" -maxdepth 1 -type d -name 'overfit--*_chkpt' -print | sort -V
  )

  for checkpoint in "${checkpoints[@]}"; do
    name=$(basename "$checkpoint")
    [[ "$name" =~ ^overfit--([0-9]+)_chkpt$ ]] || continue
    step="${BASH_REMATCH[1]}"
    (( step >= min_step )) || continue
    if ! checkpoint_complete "$checkpoint" "$step"; then
      echo "checkpoint ${step} 仍在写入；本轮不启动该 checkpoint 的双模式评测。"
      pending_checkpoint=1
      continue
    fi
    pair_root="${run_root}/paired-preview-step-${step}"
    comparison="${pair_root}/step-${step}-two-mode-comparison.mp4"
    summary="${pair_root}/pair-summary.json"
    if [[ -s "$comparison" && -s "$summary" ]]; then
      continue
    fi

    set +e
    CI_OVERFIT_PREVIEW_STEP="$step" \
      CI_OVERFIT_PREVIEW_RESULTS="$pair_root" \
      CI_OVERFIT_RESUME_UNIT= \
      bash "$pair_runner"
    pair_exit=$?
    set -e
    if [[ "$pair_exit" == "75" ]]; then
      pending_checkpoint=1
      continue
    fi
    if [[ "$pair_exit" != "0" ]]; then
      exit "$pair_exit"
    fi
  done

  if training_is_active; then
    echo "当前 checkpoint 双版本预览已完成；训练仍在运行，${poll_seconds}s 后重查。"
    sleep "$poll_seconds"
    continue
  fi
  if [[ "$pending_checkpoint" == "1" ]]; then
    echo "错误：训练已结束，但仍有不完整 checkpoint。" >&2
    exit 1
  fi
  break
done

mapfile -t comparisons < <(
  find "$run_root" -maxdepth 2 -type f \
    -path '*/paired-preview-step-*/step-*-two-mode-comparison.mp4' -print | sort -V
)
if [[ "${#comparisons[@]}" -gt 0 ]]; then
  python3 "$mosaic" \
    --input "${comparisons[@]}" \
    --fps "${CI_OVERFIT_EVAL_VIDEO_FPS:-25}" \
    --output "${run_root}/all-checkpoints-two-mode-comparison.mp4"
fi

touch "${run_root}/PAIRED_PREVIEW_BACKLOG_COMPLETE"
echo "双版本 checkpoint 预览队列完成：${run_root}"
