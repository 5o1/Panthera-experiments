#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
activation_script="${workspace}/tools/activate_lab_vla.sh"
adapter_root="${workspace}/packages/panthera_vla"
openvla_warmup_patch="${workspace}/patches/openvla_oft_warmup_decay.patch"
openvla_constants_patch="${workspace}/patches/openvla_oft_panthera_constants.patch"
openvla_early_stopping_patch="${workspace}/patches/openvla_oft_validation_early_stopping.patch"
openvla_site_packages="${workspace}/runtime/rlinf/.venv/lib/python3.11/site-packages"
data_root="${PANTHERA_SFT_DATA_ROOT:-${workspace}/datasets/rlds/rlds_single_sft_v1}"
dataset_name="${PANTHERA_SFT_DATASET_NAME:-panthera_single_cylinder}"
dataset_version="${PANTHERA_SFT_DATASET_VERSION:-2.0.0}"
initial_model="${PANTHERA_SFT_INITIAL_MODEL:-${workspace}/models/openvla-oft-place-empty-cup}"
run_root="${PANTHERA_SFT_RUN_ROOT:-${workspace}/runs/panthera-single-openvla-sft}"
state_root="${PANTHERA_SFT_STATE_ROOT:-${workspace}/state/panthera-single-openvla-sft-state}"
early_stopping_min_delta="${PANTHERA_SFT_EARLY_STOPPING_MIN_DELTA:-0.001}"
early_stopping_patience="${PANTHERA_SFT_EARLY_STOPPING_PATIENCE:-3}"
val_freq="${PANTHERA_SFT_VAL_FREQ:-1000}"
learning_rate="${PANTHERA_SFT_LEARNING_RATE:-0.0005}"
lr_warmup_steps="${PANTHERA_SFT_LR_WARMUP_STEPS:-500}"
num_steps_before_decay="${PANTHERA_SFT_NUM_STEPS_BEFORE_DECAY:-4000}"
resume_step="${PANTHERA_SFT_RESUME_STEP:-}"
run_id="${PANTHERA_SFT_RUN_ID:-panthera-single-cylinder-sft-v1-earlystop-md1e-3-p3}"
run_dir="${run_root}/${run_id}"
dataset_marker="${PANTHERA_SFT_DATASET_MARKER:-${workspace}/state/panthera-single-sft-dataset-state/dataset.ok}"
rlds_marker="${PANTHERA_SFT_RLDS_MARKER:-${workspace}/state/panthera-single-sft-rlds-state/rlds.ok}"
smoke_marker="${PANTHERA_SFT_SMOKE_MARKER:-${workspace}/state/panthera-single-openvla-sft-smoke-state/train.ok}"
training_gpu_spec="${PANTHERA_SFT_GPUS:-0,1,2,3}"
action_chunk="${PANTHERA_ACTION_CHUNK:-5}"
robot_platform="${PANTHERA_ROBOT_PLATFORM:-BRIDGE}"
training_gpu_spec="${training_gpu_spec//,/ }"
read -r -a training_gpus <<<"$training_gpu_spec"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for value_name in early_stopping_patience val_freq; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：${value_name} 必须是正整数。" >&2
    exit 1
  fi
done
python3 - "$learning_rate" "$early_stopping_min_delta" <<'PY'
import math
import sys

learning_rate = float(sys.argv[1])
min_delta = float(sys.argv[2])
if not math.isfinite(learning_rate) or learning_rate <= 0.0:
    raise SystemExit("PANTHERA_SFT_LEARNING_RATE must be finite and positive")
if not math.isfinite(min_delta) or min_delta <= 0.0:
    raise SystemExit("PANTHERA_SFT_EARLY_STOPPING_MIN_DELTA must be finite and positive")
PY
for value_name in lr_warmup_steps num_steps_before_decay; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "错误：${value_name} 必须是非负整数。" >&2
    exit 1
  fi
done
if [[ ! "$action_chunk" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：PANTHERA_ACTION_CHUNK 必须是正整数。" >&2
  exit 1
fi
if [[ -n "$resume_step" ]]; then
  if [[ ! "$resume_step" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：PANTHERA_SFT_RESUME_STEP 必须是正整数。" >&2
    exit 1
  fi
fi
if (( ${#training_gpus[@]} == 0 )); then
  echo "错误：PANTHERA_SFT_GPUS 不能为空。" >&2
  exit 1
fi
declare -A seen_training_gpus=()
for gpu in "${training_gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "错误：PANTHERA_SFT_GPUS 只能包含非负整数 GPU 编号。" >&2
    exit 1
  fi
  if [[ -n "${seen_training_gpus[$gpu]:-}" ]]; then
    echo "错误：PANTHERA_SFT_GPUS 含重复 GPU：${gpu}" >&2
    exit 1
  fi
  seen_training_gpus[$gpu]=1
done
training_gpu_count=${#training_gpus[@]}
per_gpu_batch_size="${PANTHERA_SFT_BATCH_SIZE:-1}"
if ! [[ "$per_gpu_batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：PANTHERA_SFT_BATCH_SIZE 必须是正整数。" >&2
  exit 1
fi
training_gpu_csv=$(IFS=,; printf '%s' "${training_gpus[*]}")
for command_name in flock patch timeout nvidia-smi; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
for required in \
  "$activation_script" \
  "${adapter_root}/panthera_rlds.py" \
  "${adapter_root}/run_finetune.py" \
  "${adapter_root}/validation_early_stopping.py" \
  "$openvla_warmup_patch" \
  "$openvla_constants_patch" \
  "$openvla_early_stopping_patch" \
  "${openvla_site_packages}/vla-scripts/finetune.py" \
  "${initial_model}/config.json" \
  "${data_root}/${dataset_name}/${dataset_version}/dataset_info.json"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少正式 SFT 前置文件：${required}" >&2
    exit 1
  fi
done

finetune_source="${openvla_site_packages}/vla-scripts/finetune.py"
constants_source="${openvla_site_packages}/prismatic/vla/constants.py"
if grep -Fq \
  'if cfg.lr_warmup_steps > 0 and gradient_step_idx < cfg.lr_warmup_steps:' \
  "$finetune_source"; then
  echo "OpenVLA-OFT warmup/decay 修正已存在。"
elif patch --batch --silent --dry-run -p1 \
  -d "$openvla_site_packages" <"$openvla_warmup_patch"; then
  patch --batch --silent -p1 -d "$openvla_site_packages" <"$openvla_warmup_patch"
  echo "已应用 OpenVLA-OFT warmup/decay 修正。"
else
  echo "错误：OpenVLA-OFT warmup/decay 修正无法安全应用。" >&2
  exit 1
fi
if ! grep -Fq \
  'if cfg.lr_warmup_steps > 0 and gradient_step_idx < cfg.lr_warmup_steps:' \
  "$finetune_source"; then
  echo "错误：warmup/decay 修正分支结束后源码标记仍不存在。" >&2
  exit 1
fi
if grep -Fq 'PANTHERA_CONSTANTS = {' "$constants_source" \
  && grep -Fq "'PANTHERA': 'PANTHERA'" "$constants_source" \
  && grep -Fq 'elif ROBOT_PLATFORM == "PANTHERA":' "$constants_source"; then
  echo "OpenVLA-OFT Panthera 25x7 常量补丁已存在。"
elif patch --batch --silent --dry-run -p1 \
  -d "$openvla_site_packages" <"$openvla_constants_patch"; then
  patch --batch --silent -p1 -d "$openvla_site_packages" <"$openvla_constants_patch"
  echo "已应用 OpenVLA-OFT Panthera 25x7 常量补丁。"
else
  echo "错误：OpenVLA-OFT Panthera 25x7 常量补丁无法安全应用。" >&2
  exit 1
fi
if ! grep -Fq 'PANTHERA_CONSTANTS = {' "$constants_source" \
  || ! grep -Fq "'PANTHERA': 'PANTHERA'" "$constants_source" \
  || ! grep -Fq 'elif ROBOT_PLATFORM == "PANTHERA":' "$constants_source"; then
  echo "错误：Panthera 25x7 常量补丁分支结束后源码标记不完整。" >&2
  exit 1
fi
if grep -Fq 'Validation early stopping:' "$finetune_source" \
  && grep -Fq 'early_stopping_min_delta: float = 1e-3' "$finetune_source" \
  && grep -Fq 'Merged best validation checkpoint at:' "$finetune_source"; then
  echo "OpenVLA-OFT validation-loss 早停补丁已存在。"
elif patch --batch --silent --dry-run -p1 \
  -d "$openvla_site_packages" <"$openvla_early_stopping_patch"; then
  patch --batch --silent -p1 -d "$openvla_site_packages" <"$openvla_early_stopping_patch"
  echo "已应用 OpenVLA-OFT validation-loss 早停补丁。"
else
  echo "错误：OpenVLA-OFT validation-loss 早停补丁无法安全应用。" >&2
  exit 1
fi
if ! grep -Fq 'Validation early stopping:' "$finetune_source" \
  || ! grep -Fq 'early_stopping_min_delta: float = 1e-3' "$finetune_source" \
  || ! grep -Fq 'Merged best validation checkpoint at:' "$finetune_source"; then
  echo "错误：validation-loss 早停补丁源码标记不完整。" >&2
  exit 1
fi
for marker in "$dataset_marker" "$rlds_marker" "$smoke_marker"; do
  if [[ ! -f "$marker" ]]; then
    echo "错误：缺少前置验收标记：${marker}" >&2
    exit 1
  fi
done

resume_args=()
created_resume_links=()
cleanup_resume_links() {
  local link
  for link in "${created_resume_links[@]}"; do
    if [[ -L "$link" ]]; then
      rm -- "$link"
    fi
  done
}
trap cleanup_resume_links EXIT
if [[ -n "$resume_step" ]]; then
  for component in action_head proprio_projector; do
    latest="${initial_model}/${component}--latest_checkpoint.pt"
    numbered="${initial_model}/${component}--${resume_step}_checkpoint.pt"
    if [[ ! -e "$numbered" ]]; then
      if [[ ! -s "$latest" ]]; then
        echo "错误：续训缺少组件 checkpoint：${latest}" >&2
        exit 1
      fi
      ln -s "$(basename "$latest")" "$numbered"
      created_resume_links+=("$numbered")
    fi
  done
  resume_args=(--resume true --resume_step "$resume_step")
fi

mkdir -p "$state_root" "$run_root" "${workspace}/wandb"
exec 9>"${state_root}/train.lock"
if ! flock -n 9; then
  echo "错误：另一个正式 OpenVLA SFT 正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/train.ok" ]]; then
  python3 -m json.tool "${state_root}/train-summary.json"
  echo "正式 OpenVLA SFT 已通过，无需重复训练。"
  exit 0
fi
if [[ -e "$run_dir" ]]; then
  echo "错误：发现未带成功标记的训练输出，请先人工归档：${run_dir}" >&2
  exit 1
fi
for gpu in "${training_gpus[@]}"; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "错误：正式 SFT 选中的 GPU ${gpu} 当前有其他计算进程。" >&2
    exit 1
  fi
done

# shellcheck disable=SC1090
source "$activation_script"
command -v torchrun >/dev/null 2>&1 || {
  echo "错误：激活 Lab 环境后仍找不到 torchrun。" >&2
  exit 1
}
export PYTHONPATH="${adapter_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="$training_gpu_csv"
# 默认离线，但允许外部覆盖：WANDB_MODE=online 时直接实时同步到云端。
# 以前是无条件设成 offline，外部设 online 会被这行覆盖掉。
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${workspace}/wandb"
export TF_CPP_MIN_LOG_LEVEL=2
export ROBOT_PLATFORM="$robot_platform"
export PANTHERA_ACTION_CHUNK="$action_chunk"
export PANTHERA_RLDS_DATASET_NAME="$dataset_name"
export PANTHERA_RLDS_SCHEMA_VERSION="${PANTHERA_SFT_SCHEMA_VERSION:-3}"
export PANTHERA_RLDS_SCENE_PROFILE="${PANTHERA_SFT_SCENE_PROFILE:-}"

stamp=$(date +%Y%m%d-%H%M%S)
run_log="${state_root}/train-${stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"
started_epoch=$(date +%s)
set +e
timeout --signal=INT --kill-after=120s \
  "${PANTHERA_SFT_TIMEOUT:-7d}" \
# 在线同步时实体必须是真实存在的账号；旧的 panthera-local 只是占位名，云端
# 会返回 entity not found。离线模式下它可以随便填，所以默认改成 Lab 上已登录
# 的账号，并保留 env 覆盖。
  torchrun --standalone --nproc-per-node="$training_gpu_count" "${adapter_root}/run_finetune.py" \
    --vla_path "$initial_model" \
    --data_root_dir "$data_root" \
    --dataset_name "$dataset_name" \
    --run_root_dir "$run_root" \
    --run_id_override "$run_id" \
    "${resume_args[@]}" \
    --shuffle_buffer_size 8192 \
    --use_l1_regression true \
    --use_diffusion false \
    --num_images_in_input 1 \
    --use_proprio true \
    --batch_size "$per_gpu_batch_size" \
    --learning_rate "$learning_rate" \
    --lr_warmup_steps "$lr_warmup_steps" \
    --num_steps_before_decay "$num_steps_before_decay" \
    --grad_accumulation_steps 1 \
    --use_val_set true \
    --val_freq "$val_freq" \
    --val_time_limit 180 \
    --early_stopping_min_delta "$early_stopping_min_delta" \
    --early_stopping_patience "$early_stopping_patience" \
    --save_freq 2147483647 \
    --save_latest_checkpoint_only true \
    --image_aug true \
    --lora_rank 32 \
    --lora_dropout 0.0 \
    --merge_lora_during_training true \
    --wandb_entity "${PANTHERA_SFT_WANDB_ENTITY:-assanekowww}" \
    --wandb_project "${PANTHERA_SFT_WANDB_PROJECT:-panthera-openvla-sft}" \
    --wandb_log_freq 20 \
  2>&1 | tee "$run_log"
train_status=${PIPESTATUS[0]}
set -e
finished_epoch=$(date +%s)
printf '%s\n' "$train_status" >"${state_root}/exit-code.txt"
# 固定时间预算：timeout(1) 触发时返回 124，训练是被 SIGINT 结束的。开启
# PANTHERA_SFT_ACCEPT_TIME_BUDGET=1 时把它当作正常收尾而不是失败——用验证集 loss
# 早停当停止判据已被闭环实测否定（见 docs/10 第 9 节），预算收尾 + 定期存档 +
# 闭环 probe 选模才是当前协议。
time_budget_hit=false
if (( train_status == 124 )) && [[ "${PANTHERA_SFT_ACCEPT_TIME_BUDGET:-0}" == "1" ]]; then
  time_budget_hit=true
  echo "训练达到时间预算 ${PANTHERA_SFT_TIMEOUT:-7d}，按固定预算收尾。" >&2
fi
if (( train_status != 0 )) && [[ "$time_budget_hit" != true ]]; then
  echo "错误：正式 OpenVLA SFT 失败，退出码 ${train_status}。" >&2
  exit "$train_status"
fi
fatal_pattern='Traceback \(most recent call last\)|RuntimeError:|CUDA out of memory|nan|NaN'
if grep -Eq "$fatal_pattern" "$run_log"; then
  grep -nE "$fatal_pattern" "$run_log" >"${state_root}/fatal-errors.txt" || true
  echo "错误：正式训练日志含致命异常或非有限值。" >&2
  exit 1
fi
if [[ "$time_budget_hit" == true ]]; then
  python3 - "$run_dir" "${state_root}/time-budget.json" \
    "${PANTHERA_SFT_TIMEOUT:-7d}" "$started_epoch" "$finished_epoch" <<'PY'
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
checkpoints = sorted(
    path.name
    for path in run_dir.iterdir()
    if path.is_file() and path.suffix in (".pt", ".safetensors")
)
if not checkpoints:
    raise SystemExit("time-budget exit without any saved checkpoint")
record = {
    "status": "passed",
    "stop_reason": "time_budget",
    "budget": sys.argv[3],
    "started_epoch": int(sys.argv[4]),
    "finished_epoch": int(sys.argv[5]),
    "elapsed_s": int(sys.argv[5]) - int(sys.argv[4]),
    "checkpoints": checkpoints,
}
(run_dir / "time-budget.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
Path(sys.argv[2]).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
PY
  [[ -s "${run_dir}/time-budget.json" ]] || {
    echo "错误：未写出 time-budget.json。" >&2
    exit 1
  }
else
  early_stopping_result="${run_dir}/early-stopping.json"
  if [[ ! -s "$early_stopping_result" ]]; then
    echo "错误：训练没有生成 early-stopping.json。" >&2
    exit 1
  fi
  python3 - "$early_stopping_result" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("stop_reason") != "early_stopping":
    raise SystemExit(
        "training did not finish via validation early stopping: "
        f"{result.get('stop_reason')!r}"
    )
if result.get("best_loss") is None or result.get("best_step") is None:
    raise SystemExit("early stopping result lacks a best validation checkpoint")
PY
fi
for required in \
  "${run_dir}/config.json" \
  "${run_dir}/dataset_statistics.json"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：正式训练输出缺少文件：${required}" >&2
    exit 1
  fi
done
if ! find "$run_dir" -maxdepth 1 -type f -name '*.safetensors' -size +0c | grep -q .; then
  echo "错误：正式训练没有生成合并后的模型权重。" >&2
  exit 1
fi
if ! find "$run_dir" -maxdepth 1 -type f -name 'action_head--latest_checkpoint.pt' -size +0c | grep -q .; then
  echo "错误：正式训练没有生成 action head。" >&2
  exit 1
fi
if ! find "$run_dir" -maxdepth 1 -type f -name 'proprio_projector--latest_checkpoint.pt' -size +0c | grep -q .; then
  echo "错误：正式训练没有生成 proprio projector。" >&2
  exit 1
fi

python3 - "$run_dir" "$initial_model" \
  "$((finished_epoch - started_epoch))" "${state_root}/train-summary.json" \
  "$dataset_name" "${PANTHERA_SFT_SCHEMA_VERSION:-3}" \
  "${PANTHERA_SFT_SCENE_PROFILE:-}" "$training_gpu_count" \
  "$training_gpu_csv" "$resume_step" "$action_chunk" "$robot_platform" \
  "$learning_rate" "$lr_warmup_steps" "$num_steps_before_decay" \
  "$early_stopping_min_delta" "$early_stopping_patience" "$val_freq" \
  "$early_stopping_result" "$per_gpu_batch_size" <<'PY'
import json
from pathlib import Path
import sys

model = Path(sys.argv[1])
stats = json.loads((model / "dataset_statistics.json").read_text(encoding="utf-8"))
early_stopping = json.loads(Path(sys.argv[19]).read_text(encoding="utf-8"))
dataset_name = sys.argv[5]
if dataset_name not in stats:
    raise SystemExit(f"training output lacks {dataset_name} normalization statistics")
summary = {
    "status": "passed",
    "model": str(model),
    "initial_model": sys.argv[2],
    "initialization": (
        f"resume_from_step_{sys.argv[10]}" if sys.argv[10]
        else f"fresh_heads_from_{Path(sys.argv[2]).name}"
    ),
    "resume_step": int(sys.argv[10]) if sys.argv[10] else None,
    "stopping_rule": "validation_loss_early_stopping",
    "early_stopping_min_delta": float(sys.argv[16]),
    "early_stopping_patience": int(sys.argv[17]),
    "validation_frequency_steps": int(sys.argv[18]),
    "best_validation_loss": early_stopping["best_loss"],
    "best_validation_step": early_stopping["best_step"],
    "validation_count": early_stopping["validation_count"],
    "stop_reason": early_stopping["stop_reason"],
    "wall_seconds": int(sys.argv[3]),
    "gpu_count": int(sys.argv[8]),
    "physical_gpus": [int(value) for value in sys.argv[9].split(",")],
    "per_gpu_batch_size": int(sys.argv[20]),
    "effective_batch_size": int(sys.argv[8]) * int(sys.argv[20]),
    "objective": "l1_regression",
    "action_dimension": 7,
    "action_chunk": int(sys.argv[11]),
    "robot_platform": sys.argv[12],
    "learning_rate": float(sys.argv[13]),
    "lr_warmup_steps": int(sys.argv[14]),
    "num_steps_before_decay": int(sys.argv[15]),
    "use_proprio": True,
    "image_augmentation": True,
    "lora_rank": 32,
    "validation_enabled": True,
    "dataset": dataset_name,
    "task_schema_version": int(sys.argv[6]),
    "scene_profile": sys.argv[7] or None,
}
Path(sys.argv[4]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

for gpu in "${training_gpus[@]}"; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "错误：正式训练后 GPU ${gpu} 仍有计算进程。" >&2
    exit 1
  fi
done
printf '%s\n' "$run_dir" >"${state_root}/final-model.txt"
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/train.ok"
echo "正式 OpenVLA SFT 通过：${run_dir}"
