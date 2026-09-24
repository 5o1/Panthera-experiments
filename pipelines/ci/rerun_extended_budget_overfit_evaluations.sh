#!/usr/bin/env bash
# Re-evaluate every immutable checkpoint with a two-tier action budget.
# Four long-lived workers each own one GPU and process disjoint checkpoints.
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
run_root="${CI_OVERFIT_RUN_ROOT:?set CI_OVERFIT_RUN_ROOT}"
results_root="${CI_OVERFIT_EXTENDED_RESULTS:-${run_root}/extended-budget-eval}"
gpus_csv="${CI_OVERFIT_EVAL_GPUS:-0,1,2,3}"
expert_budget="${CI_OVERFIT_EVAL_EXPERT_ACTION_BUDGET:-1037}"
max_actions="${CI_OVERFIT_EVAL_MAX_ACTIONS:-2074}"
pair_runner="${workspace}/pipelines/ci/evaluate_overfit_checkpoint_preview_pair.sh"
mosaic="${workspace}/packages/panthera_sim/mosaic_videos.py"
manifest="${run_root}/checkpoints.json"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行。" >&2
  exit 1
fi
for required in "$pair_runner" "$mosaic" "$manifest"; do
  [[ -s "$required" ]] || { echo "错误：缺少 ${required}" >&2; exit 1; }
done
if [[ ! "$expert_budget" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$max_actions" =~ ^[1-9][0-9]*$ ]] || \
   (( max_actions < expert_budget )); then
  echo "错误：动作预算无效：expert=${expert_budget}, max=${max_actions}。" >&2
  exit 1
fi

IFS=',' read -r -a gpus <<<"$gpus_csv"
if [[ "${#gpus[@]}" -eq 0 ]]; then
  echo "错误：CI_OVERFIT_EVAL_GPUS 不能为空。" >&2
  exit 1
fi
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "错误：无效 GPU：${gpu}" >&2; exit 1; }
done

mapfile -t steps < <(python3 - "$manifest" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
for checkpoint in record["checkpoints"]:
    print(int(checkpoint["step"]))
PY
)
if [[ "${#steps[@]}" -eq 0 ]]; then
  echo "错误：checkpoint 清单为空。" >&2
  exit 1
fi

mkdir -p "$results_root" "$results_root/logs" "$results_root/locks"
date -u +%Y-%m-%dT%H:%M:%SZ >"${results_root}/started-at.txt"

worker() {
  local worker_index=$1
  local gpu=$2
  local position step checkpoint pair_root
  for ((position=worker_index; position<${#steps[@]}; position+=${#gpus[@]})); do
    step=${steps[$position]}
    checkpoint="${run_root}/run/overfit--${step}_chkpt"
    pair_root="${results_root}/paired-preview-step-${step}"
    if [[ -s "${pair_root}/pair-summary.json" && \
          -s "${pair_root}/step-${step}-two-mode-comparison.mp4" ]]; then
      echo "GPU${gpu}: step ${step} 已完成，跳过。"
      continue
    fi
    for required in \
      "$checkpoint/lora_adapter/adapter_model.safetensors" \
      "$checkpoint/lora_adapter/adapter_config.json" \
      "$checkpoint/action_head--${step}_checkpoint.pt" \
      "$checkpoint/proprio_projector--${step}_checkpoint.pt" \
      "$checkpoint/dataset_statistics.json"; do
      [[ -s "$required" ]] || {
        echo "错误：step ${step} checkpoint 不完整：${required}" >&2
        return 1
      }
    done
    echo "GPU${gpu}: 开始 step ${step}。"
    CI_OVERFIT_EVAL_GPU="$gpu" \
      CI_OVERFIT_EVAL_LOCK_FILE="${results_root}/locks/gpu-${gpu}.lock" \
      CI_OVERFIT_EVAL_EXPERT_ACTION_BUDGET="$expert_budget" \
      CI_OVERFIT_EVAL_MAX_ACTIONS="$max_actions" \
      CI_OVERFIT_PREVIEW_STEP="$step" \
      CI_OVERFIT_PREVIEW_RESULTS="$pair_root" \
      CI_OVERFIT_RESUME_UNIT= \
      bash "$pair_runner"
  done
}

pids=()
for index in "${!gpus[@]}"; do
  worker "$index" "${gpus[$index]}" \
    >"${results_root}/logs/gpu-${gpus[$index]}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" != "0" ]]; then
  echo "错误：至少一个并行评测 worker 失败；查看 ${results_root}/logs/。" >&2
  exit 1
fi

mapfile -t comparisons < <(
  find "$results_root" -maxdepth 2 -type f \
    -path '*/paired-preview-step-*/step-*-two-mode-comparison.mp4' -print | sort -V
)
if [[ "${#comparisons[@]}" -ne "${#steps[@]}" ]]; then
  echo "错误：期望 ${#steps[@]} 个对比视频，实际 ${#comparisons[@]} 个。" >&2
  exit 1
fi
python3 "$mosaic" \
  --input "${comparisons[@]}" \
  --fps "${CI_OVERFIT_EVAL_VIDEO_FPS:-25}" \
  --output "${results_root}/all-checkpoints-two-mode-comparison.mp4"

python3 - "$results_root" "${#steps[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
records = []
for path in sorted(
    root.glob("paired-preview-step-*/pair-summary.json"),
    key=lambda item: int(item.parent.name.rsplit("-", 1)[1]),
):
    record = json.loads(path.read_text(encoding="utf-8"))
    record["step"] = int(path.parent.name.rsplit("-", 1)[1])
    records.append(record)
if len(records) != expected:
    raise SystemExit(f"expected {expected} summaries, got {len(records)}")
(root / "summary.json").write_text(
    json.dumps({"checkpoints": records}, indent=2) + "\n", encoding="utf-8"
)
PY

date -u +%Y-%m-%dT%H:%M:%SZ >"${results_root}/finished-at.txt"
touch "${results_root}/EVALUATION_COMPLETE"
echo "扩展预算并行评测完成：${results_root}"
