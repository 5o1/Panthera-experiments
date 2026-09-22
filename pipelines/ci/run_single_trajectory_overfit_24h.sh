#!/usr/bin/env bash
# Run a wall-clock-budgeted, single-trajectory overfit job without competing
# closed-loop evaluation. All three selected GPUs remain assigned to DDP; the
# numbered checkpoints can be evaluated after training has released them.
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
packages="${PANTHERA_PACKAGES:-${workspace}/packages}"
adapter_root="${packages}/panthera_vla"
dataset_root="${CI_OVERFIT_DATASET_ROOT:?set CI_OVERFIT_DATASET_ROOT}"
episode="${CI_OVERFIT_EPISODE:-2}"
dataset_name="${CI_OVERFIT_DATASET_NAME:-panthera_phone_cylinder_socket_v2}"
base_model="${CI_OVERFIT_BASE_MODEL:-${workspace}/models/openvla-oft-place-empty-cup}"
duration="${CI_OVERFIT_DURATION:-24h}"
max_steps="${CI_OVERFIT_MAX_STEPS:-1000000}"
save_freq="${CI_OVERFIT_SAVE_FREQ:-5000}"
batch_size="${CI_OVERFIT_BATCH_SIZE:-6}"
learning_rate="${CI_OVERFIT_LEARNING_RATE:-0.0005}"
train_gpus="${CI_OVERFIT_TRAIN_GPUS:-1,2,3}"
action_chunk="${CI_OVERFIT_ACTION_CHUNK:-25}"
proprio_dim="${CI_OVERFIT_PROPRIO_DIM:-7}"
wandb_entity="${CI_OVERFIT_WANDB_ENTITY:-assanekowww}"
wandb_project="${CI_OVERFIT_WANDB_PROJECT:-panthera-ci-overfit-24h}"
run_id="${CI_OVERFIT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
root="${CI_OVERFIT_ROOT:-${workspace}/ci/overfit-ep${episode}-24h/${run_id}}"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行。" >&2
  exit 1
fi
if (( save_freq <= 0 )); then
  echo "错误：CI_OVERFIT_SAVE_FREQ 必须为正整数。" >&2
  exit 1
fi
if [[ "$proprio_dim" != "7" && "$proprio_dim" != "28" ]]; then
  echo "错误：CI_OVERFIT_PROPRIO_DIM 必须为 7 或 28。" >&2
  exit 1
fi
gpu_count=$(tr ',' '\n' <<<"$train_gpus" | sed '/^$/d' | wc -l)
if [[ "$gpu_count" -ne 3 ]]; then
  echo "错误：本实验要求恰好三张训练 GPU，当前为 ${train_gpus}。" >&2
  exit 1
fi

command -v flock >/dev/null || { echo "错误：缺少 flock。" >&2; exit 1; }
mkdir -p "${workspace}/state"
if [[ -n "${CI_OVERFIT_INHERITED_LOCK_FD:-}" ]]; then
  if [[ ! "${CI_OVERFIT_INHERITED_LOCK_FD}" =~ ^[0-9]+$ ]] || \
     ! test -e "/proc/$$/fd/${CI_OVERFIT_INHERITED_LOCK_FD}"; then
    echo "错误：CI_OVERFIT_INHERITED_LOCK_FD 不是当前进程持有的文件描述符。" >&2
    exit 1
  fi
else
  exec 9>"${workspace}/state/single-trajectory-overfit.lock"
  if ! flock -n 9; then
    echo "错误：已有单轨迹过拟合任务正在运行。" >&2
    exit 1
  fi
fi

for required in \
  "$dataset_root/dataset.json" \
  "$adapter_root/build_episode_subset.py" \
  "$adapter_root/panthera_rlds.py" \
  "$adapter_root/run_finetune.py" \
  "$base_model/config.json"; do
  [[ -s "$required" ]] || { echo "错误：缺少 ${required}" >&2; exit 1; }
done

# shellcheck disable=SC1091
source "${workspace}/tools/activate_lab_vla.sh" >/dev/null 2>&1

if [[ -e "$root" ]]; then
  echo "错误：运行目录已存在，拒绝覆盖：${root}" >&2
  exit 1
fi
mkdir -p "$root"
printf '%s\n' "$run_id" >"${workspace}/state/single-trajectory-overfit-24h.latest"

subset="${root}/subset"
rlds="${root}/rlds"
run_root="${root}/run"
model="${run_root}/overfit"
setup_log="${root}/setup.log"
training_log="${root}/training.log"

finish() {
  code=$1
  printf '%s\n' "$code" >"${root}/exit-code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ >"${root}/finished-at.txt"
  trap - EXIT
  exit "$code"
}
trap 'finish $?' EXIT

{
  echo "=== 第1步：建立 episode ${episode} 自持子集 ==="
  python3 "$adapter_root/build_episode_subset.py" \
    --source-root "$dataset_root" \
    --target-root "$subset" \
    --episode "$episode" \
    --validate-on-train

  echo "=== 第2步：生成单轨迹 RLDS ==="
  export PANTHERA_RLDS_SCHEMA_VERSION="${CI_OVERFIT_SCHEMA_VERSION:-10}"
  export PANTHERA_RLDS_SCENE_PROFILE="${CI_OVERFIT_SCENE_PROFILE:-panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2}"
  export PANTHERA_RLDS_DATASET_NAME="$dataset_name"
  export PANTHERA_ACTION_CHUNK="$action_chunk"
  export PANTHERA_PROPRIO_DIM="$proprio_dim"
  export ROBOT_PLATFORM=PANTHERA
  export PANTHERA_SOURCE_DATASET_ROOT="$subset"
  python3 "$adapter_root/panthera_rlds.py" \
    --source-root "$subset" \
    --data-root "$rlds" \
    --validation-episode "$episode" \
    --summary "${root}/rlds-summary.json"
} >"$setup_log" 2>&1

export PANTHERA_RLDS_SCHEMA_VERSION="${CI_OVERFIT_SCHEMA_VERSION:-10}"
export PANTHERA_RLDS_SCENE_PROFILE="${CI_OVERFIT_SCENE_PROFILE:-panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2}"
export PANTHERA_RLDS_DATASET_NAME="$dataset_name"
export PANTHERA_ACTION_CHUNK="$action_chunk"
export PANTHERA_PROPRIO_DIM="$proprio_dim"
export ROBOT_PLATFORM=PANTHERA
export PANTHERA_SOURCE_DATASET_ROOT="$subset"
export WANDB_MODE="${CI_OVERFIT_WANDB_MODE:-online}"
export WANDB_ENTITY="$wandb_entity"
export WANDB_PROJECT="$wandb_project"

python3 - "$root/run-config.json" "$dataset_root" "$subset" "$episode" \
  "$action_chunk" "$train_gpus" "$base_model" "$duration" "$max_steps" \
  "$save_freq" "$batch_size" "$learning_rate" "$proprio_dim" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(output, source, subset, episode, chunk, train_gpus, base_model, duration,
 max_steps, save_freq, batch_size, learning_rate, proprio_dim) = sys.argv[1:]
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_dataset": source,
    "training_dataset": subset,
    "episode": int(episode),
    "action_chunk": int(chunk),
    "proprio_dim": int(proprio_dim),
    "train_gpus": train_gpus.split(","),
    "base_model": base_model,
    "wall_clock_budget": duration,
    "max_steps_guard": int(max_steps),
    "checkpoint_every_steps": int(save_freq),
    "batch_size_per_gpu": int(batch_size),
    "learning_rate": float(learning_rate),
    "closed_loop_during_training": False,
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "开始三卡 24 小时过拟合：GPU=${train_gpus}，每 ${save_freq} step 保存一次。"
date -u +%Y-%m-%dT%H:%M:%SZ >"${root}/training-started-at.txt"
set +e
CUDA_VISIBLE_DEVICES="$train_gpus" \
timeout --signal=INT --kill-after=180s "$duration" \
torchrun --standalone --nproc-per-node="$gpu_count" \
  "$adapter_root/run_finetune.py" \
  --vla_path "$base_model" \
  --data_root_dir "$rlds" \
  --dataset_name "$dataset_name" \
  --run_root_dir "$run_root" \
  --run_id_override overfit \
  --batch_size "$batch_size" \
  --learning_rate "$learning_rate" \
  --max_steps "$max_steps" \
  --use_l1_regression true \
  --use_diffusion false \
  --num_images_in_input 1 \
  --use_proprio true \
  --use_val_set false \
  --lora_rank 32 \
  --image_aug false \
  --merge_lora_during_training false \
  --save_freq "$save_freq" \
  --save_latest_checkpoint_only false \
  --wandb_entity "$wandb_entity" \
  --wandb_project "$wandb_project" \
  >"$training_log" 2>&1
train_exit=$?
set -e
printf '%s\n' "$train_exit" >"${root}/training-exit-code.txt"

if [[ "$train_exit" != "0" && "$train_exit" != "124" ]]; then
  echo "错误：训练异常退出（${train_exit}），详见 ${training_log}。" >&2
  exit "$train_exit"
fi

shopt -s nullglob
checkpoints=("${model}"--*_chkpt)
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "错误：训练没有产出任何编号 checkpoint。" >&2
  exit 1
fi

python3 - "$model/training.json" "$duration" "$train_exit" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
record["execution"] = {
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "stop_reason": "wall_clock_budget" if sys.argv[3] == "124" else "max_steps_guard",
    "wall_clock_budget": sys.argv[2],
    "training_exit_code": int(sys.argv[3]),
}
path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

for checkpoint in "${checkpoints[@]}"; do
  cp "$model/training.json" "$checkpoint/training.json"
done

python3 - "$root/checkpoints.json" "$save_freq" "${checkpoints[@]}" <<'PY'
import json
import re
import sys
from pathlib import Path

output = Path(sys.argv[1])
frequency = int(sys.argv[2])
entries = []
for raw in sys.argv[3:]:
    root = Path(raw)
    match = re.search(r"--(\d+)_chkpt$", root.name)
    if match is None:
        raise SystemExit(f"unexpected checkpoint directory: {root}")
    step = int(match.group(1))
    if step % frequency:
        raise SystemExit(f"checkpoint {root} is not aligned to {frequency} steps")
    required = [
        root / "lora_adapter" / "adapter_model.safetensors",
        root / f"action_head--{step}_checkpoint.pt",
        root / f"proprio_projector--{step}_checkpoint.pt",
        root / "dataset_statistics.json",
        root / "training.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(f"incomplete checkpoint {root}: {missing}")
    entries.append({"step": step, "path": str(root)})
entries.sort(key=lambda item: item["step"])
output.write_text(json.dumps({"checkpoints": entries}, indent=2) + "\n", encoding="utf-8")
print(f"保留 {len(entries)} 个完整 checkpoint，最后一步为 {entries[-1]['step']}。")
PY

touch "${root}/TRAINING_COMPLETE"
echo "24 小时过拟合训练结束；checkpoint 清单：${root}/checkpoints.json"
