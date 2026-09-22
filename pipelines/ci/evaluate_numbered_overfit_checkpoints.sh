#!/usr/bin/env bash
# Evaluate immutable, numbered checkpoints while the three-GPU trainer keeps
# running. One merged scratch model is materialised at a time and removed after
# its result is committed, so checkpoint storage stays compact.
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
packages="${PANTHERA_PACKAGES:-${workspace}/packages}"
run_root="${CI_OVERFIT_RUN_ROOT:?set CI_OVERFIT_RUN_ROOT}"
checkpoint_prefix="${run_root}/run/overfit"
training_record="${checkpoint_prefix}/training.json"
dataset_root="${run_root}/subset"
robotwin_root="${workspace}/runtime/robotwin"
base_model="${CI_OVERFIT_BASE_MODEL:-${workspace}/models/openvla-oft-place-empty-cup}"
gpu="${CI_OVERFIT_EVAL_GPU:-0}"
min_free_mib="${CI_OVERFIT_EVAL_MIN_FREE_MIB:-28000}"
task_config="${CI_OVERFIT_TASK_CONFIG:-panthera_phone_cylinder_socket_v2_pilot.yml}"
task_name="${CI_OVERFIT_TASK_NAME:-place_randomized_cylinder_in_socket}"
episode="${CI_OVERFIT_EPISODE:-2}"
video_stride="${CI_OVERFIT_EVAL_VIDEO_STRIDE:-2}"
video_fps="${CI_OVERFIT_EVAL_VIDEO_FPS:-25}"
validation_batch_size="${CI_OVERFIT_EVAL_VALIDATION_BATCH_SIZE:-2}"
validation_reference="${CI_OVERFIT_EVAL_VALIDATION_REFERENCE:-}"
plot_temporal_loss="${CI_OVERFIT_EVAL_PLOT_TEMPORAL_LOSS:-1}"
execution_horizon="${CI_OVERFIT_EVAL_EXECUTION_HORIZON:-20}"
temporal_ensemble="${CI_OVERFIT_EVAL_TEMPORAL_ENSEMBLE:-}"
variant_label="${CI_OVERFIT_EVAL_VARIANT_LABEL:-}"
results="${CI_OVERFIT_EVAL_RESULTS:-${run_root}/numbered-eval}"
scratch="${results}/scratch"
training_unit="${CI_OVERFIT_TRAINING_UNIT:-panthera-overfit-ep2-24h-b6-20260920T140123Z.service}"
poll_seconds="${CI_OVERFIT_EVAL_POLL_SECONDS:-60}"
only_step="${CI_OVERFIT_EVAL_ONLY_STEP:-}"
refresh_mosaic_enabled="${CI_OVERFIT_EVAL_REFRESH_MOSAIC:-1}"

ensemble_args=()
if [[ -n "$temporal_ensemble" ]]; then
  ensemble_args=(--temporal-ensemble "$temporal_ensemble")
fi
variant_annotation_args=()
if [[ -n "$variant_label" ]]; then
  variant_annotation_args=(--annotation "$variant_label")
fi
offline_validation_args=()
if [[ -z "$validation_reference" ]]; then
  offline_validation_args=(
    --offline-validation-loss
    --offline-validation-trace
    --offline-validation-batch-size "$validation_batch_size"
  )
fi

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行。" >&2
  exit 1
fi
for required in \
  "$training_record" \
  "$dataset_root/dataset.json" \
  "$base_model/config.json" \
  "$packages/panthera_vla/merge_lora_checkpoint.py" \
  "$packages/panthera_sim/rollout.py"; do
  [[ -s "$required" ]] || { echo "错误：缺少 ${required}" >&2; exit 1; }
done

# shellcheck disable=SC1091
source "${workspace}/tools/activate_lab_vla.sh" >/dev/null 2>&1
export ROBOT_PLATFORM=PANTHERA
proprio_dim=$(python3 - "$training_record" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
value = int(record["policy_contract"]["proprio_dim"])
if value not in (7, 28):
    raise SystemExit(f"unsupported Panthera proprioception width: {value}")
print(value)
PY
)
export PANTHERA_PROPRIO_DIM="$proprio_dim"
mkdir -p "$results" "${workspace}/state"
exec 9>"${workspace}/state/single-trajectory-numbered-eval.lock"
if ! flock -n 9; then
  echo "错误：已有 numbered checkpoint 评测正在运行。" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${active_scratch:-}" && -d "$active_scratch" ]]; then
    rm -rf -- "$active_scratch"
  fi
}
trap cleanup EXIT INT TERM

refresh_mosaic() {
  [[ "$refresh_mosaic_enabled" == "1" ]] || return 0
  mapfile -t videos < <(find "$results" -maxdepth 1 -type f -name 'step-*.mp4' -print | sort -V)
  if [[ "${#videos[@]}" -gt 0 ]]; then
    python3 "$packages/panthera_sim/mosaic_videos.py" \
      --input "${videos[@]}" \
      --fps "$video_fps" \
      --output "${results}/all-checkpoints-comparison.mp4"
  fi
}

while true; do
  mapfile -t checkpoints < <(
    find "${run_root}/run" -maxdepth 1 -type d -name 'overfit--*_chkpt' -print | sort -V
  )
  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    if systemctl --user is-active --quiet "$training_unit"; then
      echo "尚无完整编号 checkpoint；${training_unit} 仍在训练，${poll_seconds}s 后重查。"
      sleep "$poll_seconds"
      continue
    fi
    echo "错误：训练已停止且没有发现编号 checkpoint。" >&2
    exit 1
  fi

  for checkpoint in "${checkpoints[@]}"; do
  name=$(basename "$checkpoint")
  if [[ ! "$name" =~ ^overfit--([0-9]+)_chkpt$ ]]; then
    echo "跳过无法解析的目录：${checkpoint}" >&2
    continue
  fi
  step="${BASH_REMATCH[1]}"
  if [[ -n "$only_step" && "$step" != "$only_step" ]]; then
    continue
  fi
  output="${results}/step-${step}.json"
  video="${results}/step-${step}.mp4"
  log="${results}/step-${step}.log"
  if [[ -s "$output" && -s "$video" ]]; then
    echo "step ${step} 已有结果，跳过。"
    continue
  fi
  rm -f -- "${results}/step-${step}.exit-code.txt"

  checkpoint_complete=1
  for required in \
    "$checkpoint/lora_adapter/adapter_model.safetensors" \
    "$checkpoint/lora_adapter/adapter_config.json" \
    "$checkpoint/action_head--${step}_checkpoint.pt" \
    "$checkpoint/proprio_projector--${step}_checkpoint.pt" \
    "$checkpoint/dataset_statistics.json"; do
    if [[ ! -s "$required" ]]; then
      echo "checkpoint ${step} 尚未完整提交：${required}；本轮跳过。"
      checkpoint_complete=0
      break
    fi
  done
  (( checkpoint_complete == 1 )) || continue

  free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if (( free_mib < min_free_mib )); then
    echo "GPU${gpu} 仅余 ${free_mib} MiB，小于门槛 ${min_free_mib} MiB；停止队列。" >&2
    exit 75
  fi

  active_scratch="${scratch}/step-${step}"
  rm -rf -- "$active_scratch"
  mkdir -p "$active_scratch"
  # Hard links avoid copying the adapter and heads. The merge only reads those
  # files and writes new model shards alongside them.
  cp -al "$checkpoint/." "$active_scratch/"
  cp "$training_record" "$active_scratch/training.json"
  mv "$active_scratch/action_head--${step}_checkpoint.pt" \
     "$active_scratch/action_head--latest_checkpoint.pt"
  mv "$active_scratch/proprio_projector--${step}_checkpoint.pt" \
     "$active_scratch/proprio_projector--latest_checkpoint.pt"

  echo "=== step ${step}: merge ===" | tee "$log"
  python3 "$packages/panthera_vla/merge_lora_checkpoint.py" \
    --run-dir "$active_scratch" \
    --base-model "$base_model" \
    --allow-running-training >>"$log" 2>&1
  python3 - "$packages/panthera_vla" "$active_scratch" <<'PY' >>"$log" 2>&1
import sys
import os
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from checkpoint import open_checkpoint

checkpoint = open_checkpoint(Path(sys.argv[2]))
actual = (
    checkpoint.policy.action_dim,
    checkpoint.policy.action_chunk,
    checkpoint.policy.proprio_dim,
    checkpoint.policy.robot_platform,
)
expected = (7, 25, int(os.environ["PANTHERA_PROPRIO_DIM"]), "PANTHERA")
if actual != expected:
    raise SystemExit(f"merged checkpoint contract {actual} != {expected}")
print(f"merged checkpoint contract: {actual}")
PY

  echo "=== step ${step}: episode ${episode} closed-loop rollout on GPU${gpu} ===" | tee -a "$log"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" ROBOT_PLATFORM=PANTHERA \
  timeout --signal=INT --kill-after=120s 2h \
  python3 "$packages/panthera_sim/rollout.py" \
    --robotwin-root "$robotwin_root" \
    --dataset-root "$dataset_root" \
    --task-config "$task_config" \
    --task-name "$task_name" \
    --episode "$episode" \
    --model "$active_scratch" \
    --execution-horizon "$execution_horizon" \
    "${ensemble_args[@]}" \
    "${offline_validation_args[@]}" \
    --allow-parity-failure \
    --trace \
    --workers 1 \
    --gpus 0 \
    --output "$output" >>"$log" 2>&1
  rollout_exit=$?
  set -e
  if [[ "$rollout_exit" != "0" ]]; then
    printf '%s\n' "$rollout_exit" >"${results}/step-${step}.exit-code.txt"
    echo "step ${step} 评测异常退出 ${rollout_exit}；停止队列，保留日志。" >&2
    exit "$rollout_exit"
  fi

  if [[ -n "$validation_reference" ]]; then
    [[ -s "$validation_reference" ]] || {
      echo "错误：离线验证参考不存在：${validation_reference}" >&2
      exit 1
    }
    python3 - "$validation_reference" "$output" <<'PY'
import json
import sys
from pathlib import Path

reference_path, output_path = map(Path, sys.argv[1:])
reference = json.loads(reference_path.read_text(encoding="utf-8"))
report = json.loads(output_path.read_text(encoding="utf-8"))
report["cases"][0]["offline_validation"] = reference["cases"][0]["offline_validation"]
report["cases"][0]["offline_validation_reused_from"] = str(reference_path)
output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
PY
  fi

  validation_loss=$(python3 - "$output" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"{report['cases'][0]['offline_validation']['normalized_l1']:.6f}")
PY
  )
  if [[ "$plot_temporal_loss" == "1" ]]; then
    python3 "$packages/panthera_sim/plot_temporal_validation_loss.py" \
      --report "$output" \
      --title "episode ${episode} | checkpoint ${step} | offline validation" \
      --output "${results}/step-${step}-temporal-validation-loss.png" \
      --csv "${results}/step-${step}-temporal-validation-loss.csv" \
      --summary "${results}/step-${step}-temporal-validation-loss-summary.json" \
      >>"$log" 2>&1
  fi
  echo "=== step ${step}: render rollout video (offline val L1 ${validation_loss}) ===" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" ROBOT_PLATFORM=PANTHERA \
  python3 "$packages/panthera_sim/render_rollout.py" \
    --robotwin-root "$robotwin_root" \
    --dataset-root "$dataset_root" \
    --task-config "$task_config" \
    --task-name "$task_name" \
    --episode "$episode" \
    --source "$output" \
    --title "episode ${episode} | checkpoint ${step}" \
    --annotation "offline val L1 (norm) ${validation_loss}" \
    "${variant_annotation_args[@]}" \
    --stride "$video_stride" \
    --fps "$video_fps" \
    --output "$video" >>"$log" 2>&1
  [[ -s "$video" ]] || { echo "step ${step} 视频为空：${video}" >&2; exit 1; }

  python3 - "$output" "$video" "$step" "${results}/summary.jsonl" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output, video, step, summary = (
    Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
)
report = json.loads(output.read_text(encoding="utf-8"))
case = report["cases"][0]
offline_validation = dict(case.get("offline_validation") or {})
offline_validation.pop("per_frame", None)
record = {
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "step": step,
    "success": bool(case.get("success")),
    "executed_actions": case.get("executed_actions"),
    "policy_queries": case.get("policy_queries"),
    "inference_timing": case.get("inference_timing"),
    "progress": case.get("progress"),
    "metrics": case.get("metrics", {}),
    "parity": case.get("parity"),
    "offline_validation": offline_validation,
    "execution_horizon": report.get("execution_horizon"),
    "temporal_ensemble": report.get("temporal_ensemble"),
    "gate_success": bool(case.get("gate_success")),
    "error": case.get("error"),
    "result": str(output),
    "video": str(video),
}
with summary.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
print(json.dumps(record, ensure_ascii=False))
PY

  if python3 - "$output" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if report.get("gate_success") else 1)
PY
  then
    success_model="${results}/successful-model-step-${step}"
    [[ ! -e "$success_model" ]] || { echo "拒绝覆盖 ${success_model}" >&2; exit 1; }
    mv "$active_scratch" "$success_model"
    active_scratch=""
    echo "step ${step} 闭环成功，完整合并模型保存在 ${success_model}。"
  else
    rm -rf -- "$active_scratch"
    active_scratch=""
  fi
  refresh_mosaic
  done

  if [[ -n "$only_step" ]]; then
    break
  fi

  if systemctl --user is-active --quiet "$training_unit"; then
    echo "当前 checkpoint 已处理完；${training_unit} 仍在训练，${poll_seconds}s 后重查。"
    sleep "$poll_seconds"
    continue
  fi
  break
done

touch "${results}/BACKLOG_COMPLETE"
echo "编号 checkpoint 积压评测完成：${results}/summary.jsonl"
