#!/usr/bin/env bash
# Gate 2: train on one expert trajectory, evaluate on that same trajectory's
# scene, and require success.
#
# This is the smallest question the pipeline has to answer before any result
# from it means anything: can the train-to-deploy path reproduce a single
# trajectory it was fitted to, in the identical scene?  If it cannot, no amount
# of data will help, and every downstream number measures a broken pipeline
# rather than a policy.  It is deliberately an overfit test -- generalisation is
# not being asked about, so train and validation are the same episode.
#
# It failing is informative in a way the large runs are not.  A 16h run on 128
# episodes scored 0/12 on scenes it had trained on, which could have been
# covariate shift, task difficulty, or a broken pipeline; this gate separates
# the last one from the other two.
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
robotwin_root="${workspace}/runtime/robotwin"
packages="${PANTHERA_PACKAGES:-${workspace}/packages}"
adapter_root="${packages}/panthera_vla"
dataset_root="${CI_OVERFIT_DATASET_ROOT:?set CI_OVERFIT_DATASET_ROOT}"
# The scene table and the action budget come from the dataset's own snapshot.
episode="${CI_OVERFIT_EPISODE:-2}"
task_config="${CI_OVERFIT_TASK_CONFIG:-panthera_phone_cylinder_socket_v2_pilot.yml}"
task_name="${CI_OVERFIT_TASK_NAME:-place_randomized_cylinder_in_socket}"
base_model="${CI_OVERFIT_BASE_MODEL:-${workspace}/models/openvla-oft-place-empty-cup}"
dataset_name="${CI_OVERFIT_DATASET_NAME:-panthera_phone_cylinder_socket_v2}"
# The trainer saves a candidate at each validation. A separate GPU evaluates
# those candidates in the actual closed loop and writes a success sentinel;
# validation loss is telemetry, not the success criterion. The wall clock is
# only a backstop against a policy that never closes the loop.
val_freq="${CI_OVERFIT_VAL_FREQ:-250}"
min_delta="${CI_OVERFIT_MIN_DELTA:-0.00001}"
patience="${CI_OVERFIT_PATIENCE:-4}"
train_timeout="${CI_OVERFIT_TIMEOUT:-4h}"
max_steps="${CI_OVERFIT_MAX_STEPS:-40000}"
batch_size="${CI_OVERFIT_BATCH_SIZE:-4}"
learning_rate="${CI_OVERFIT_LEARNING_RATE:-0.0005}"
train_gpus="${CI_OVERFIT_TRAIN_GPUS:-${CI_OVERFIT_GPUS:-0,1,2}}"
eval_gpu="${CI_OVERFIT_EVAL_GPU:-3}"
watch_interval="${CI_OVERFIT_WATCH_INTERVAL_S:-60}"
execution_horizon="${CI_OVERFIT_EXECUTION_HORIZON:-20}"
action_chunk="${CI_OVERFIT_ACTION_CHUNK:-25}"
run_id="${CI_OVERFIT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
root="${CI_OVERFIT_ROOT:-${workspace}/ci/overfit-ep${episode}/${run_id}}"
keep="${CI_OVERFIT_KEEP:-1}"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行。" >&2
  exit 1
fi
command -v flock >/dev/null || { echo "错误：缺少 flock。" >&2; exit 1; }
mkdir -p "${workspace}/state"
exec 9>"${workspace}/state/single-trajectory-overfit.lock"
if ! flock -n 9; then
  echo "错误：已有单轨迹过拟合门禁正在运行。" >&2
  exit 1
fi
for required in "$dataset_root/dataset.json" \
                "$adapter_root/build_episode_subset.py" \
                "$adapter_root/closed_loop_watcher.py" \
                "$packages/panthera_sim/rollout.py"; do
  [[ -s "$required" ]] || { echo "错误：缺少 ${required}" >&2; exit 1; }
done

# shellcheck disable=SC1091
source "${workspace}/tools/activate_lab_vla.sh" >/dev/null 2>&1

if [[ -e "$root" ]]; then
  echo "错误：运行目录已存在，拒绝覆盖：${root}" >&2
  echo "请设置新的 CI_OVERFIT_RUN_ID，或显式指定一个不存在的 CI_OVERFIT_ROOT。" >&2
  exit 1
fi
if [[ ",$train_gpus," == *",$eval_gpu,"* && "${CI_OVERFIT_ALLOW_SHARED_GPU:-0}" != "1" ]]; then
  echo "错误：训练 GPU (${train_gpus}) 与闭环验证 GPU (${eval_gpu}) 重叠。" >&2
  echo "如确实只有一张卡，可设置 CI_OVERFIT_ALLOW_SHARED_GPU=1。" >&2
  exit 1
fi

subset="${root}/subset"
rlds="${root}/rlds"
run_root="${root}/run"
report="${root}/gate.json"
model="${run_root}/overfit"
sentinel="${model}/CLOSED_LOOP_SUCCESS"
success_model="${model}/closed-loop-best"
watcher_log="${root}/closed-loop-watcher.log"
mkdir -p "$root"

watcher_pid=""
finish() {
  code=$1
  if [[ -n "$watcher_pid" ]] && kill -0 "$watcher_pid" 2>/dev/null; then
    kill "$watcher_pid" 2>/dev/null || true
    wait "$watcher_pid" 2>/dev/null || true
  fi
  printf '%s\n' "$code" >"${root}/exit-code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ >"${root}/finished-at.txt"
  trap - EXIT
  exit "$code"
}
trap 'finish $?' EXIT

echo "=== 门禁2 第1步：单条轨迹子集（episode ${episode}）==="
python3 "$adapter_root/build_episode_subset.py" \
  --source-root "$dataset_root" \
  --target-root "$subset" \
  --episode "$episode" \
  --validate-on-train

echo "=== 门禁2 第2步：RLDS 转换 ==="
# The single episode is both train and validation: an overfit gate has nothing
# to hold out, and a held-out split would stop training on a signal that does
# not apply to the question being asked.
# The converter validates the dataset against these; they describe the data
# generation this repository targets, not a preference of the caller.
export PANTHERA_RLDS_SCHEMA_VERSION="${CI_OVERFIT_SCHEMA_VERSION:-10}"
export PANTHERA_RLDS_SCENE_PROFILE="${CI_OVERFIT_SCENE_PROFILE:-panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2}"
export PANTHERA_RLDS_DATASET_NAME="$dataset_name"
export PANTHERA_ACTION_CHUNK="$action_chunk"
export ROBOT_PLATFORM=PANTHERA
# The checkpoint records which dataset produced it, and refuses to train
# without that: weights that cannot name their dataset are not reproducible,
# and evaluation has no way to notice it is scoring the wrong pairing.
export PANTHERA_SOURCE_DATASET_ROOT="$subset"
# A gate must not depend on a network account. Offline still records the run
# locally, which is what a later investigation needs.
export WANDB_MODE="${CI_OVERFIT_WANDB_MODE:-offline}"
export WANDB_ENTITY="${CI_OVERFIT_WANDB_ENTITY:-panthera-local}"
export WANDB_PROJECT="${CI_OVERFIT_WANDB_PROJECT:-panthera-ci-overfit}"
python3 "$adapter_root/panthera_rlds.py" \
  --source-root "$subset" \
  --data-root "$rlds" \
  --validation-episode "$episode" \
  --summary "${root}/rlds-summary.json"

python3 - "$root/run-config.json" "$dataset_root" "$subset" "$episode" \
  "$action_chunk" "$train_gpus" "$eval_gpu" "$base_model" "$max_steps" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output, source, subset, episode, chunk, train_gpus, eval_gpu, base_model, max_steps = sys.argv[1:]
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_dataset": source,
    "training_dataset": subset,
    "episode": int(episode),
    "action_chunk": int(chunk),
    "train_gpus": train_gpus.split(","),
    "closed_loop_gpu": eval_gpu,
    "base_model": base_model,
    "max_steps": int(max_steps),
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "=== 门禁2 第3步：启动闭环 checkpoint 验证器 ==="
CUDA_VISIBLE_DEVICES="$eval_gpu" ROBOT_PLATFORM=PANTHERA \
python3 "$adapter_root/closed_loop_watcher.py" \
  --run-dir "$model" \
  --base-model "$base_model" \
  --dataset-root "$subset" \
  --robotwin-root "$robotwin_root" \
  --task-config "$task_config" \
  --task-name "$task_name" \
  --episode "$episode" \
  --unnorm-key "$dataset_name" \
  --action-chunk "$action_chunk" \
  --execution-horizon "$execution_horizon" \
  --interval-s "$watch_interval" \
  --scratch "${root}/watcher-scratch" \
  --success-model-dir "$success_model" \
  >"$watcher_log" 2>&1 &
watcher_pid=$!

echo "=== 门禁2 第4步：过拟合训练（闭环成功才提前停止）==="
gpu_count=$(tr ',' ' ' <<<"$train_gpus" | wc -w)
set +e
CUDA_VISIBLE_DEVICES="$train_gpus" ROBOT_PLATFORM=PANTHERA \
timeout --signal=INT --kill-after=120s "$train_timeout" \
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
  --use_val_set true \
  --val_freq "$val_freq" \
  --early_stopping_min_delta "$min_delta" \
  --early_stopping_patience "$patience" \
  --external_stop_file "$sentinel" \
  --lora_rank 32 \
  --image_aug false \
  --merge_lora_during_training false \
  --save_latest_checkpoint_only true
train_exit=$?
set -e

if [[ -n "$watcher_pid" ]] && kill -0 "$watcher_pid" 2>/dev/null; then
  kill "$watcher_pid" 2>/dev/null || true
  wait "$watcher_pid" 2>/dev/null || true
fi
watcher_pid=""

if [[ "$train_exit" != "0" && "$train_exit" != "124" ]]; then
  echo "错误：训练异常退出（${train_exit}），产物已保留在 ${root}。" >&2
  exit "$train_exit"
fi

if [[ -s "$sentinel" && -s "${success_model}/model.safetensors.index.json" ]]; then
  eval_model="$success_model"
  [[ -s "${model}/training.json" ]] && cp "${model}/training.json" "${success_model}/training.json"
else
  eval_model="$model"
fi

if [[ ! -s "${eval_model}/model.safetensors.index.json" ]]; then
  # A wall-clock stop kills the in-training merge, so do it out of band rather
  # than failing the gate for a reason that is not about the policy.
  echo "训练退出码 ${train_exit}，未见合并权重，尝试带外合并。"
  python3 "$adapter_root/merge_lora_checkpoint.py" \
    --run-dir "$model" --base-model "$base_model"
  eval_model="$model"
fi
[[ -s "${eval_model}/model.safetensors.index.json" ]] || {
  echo "错误：训练未产出合并权重：${eval_model}（退出码 ${train_exit}）" >&2; exit 1; }
python3 - "${model}/early-stopping.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("  （无 early-stopping.json，训练可能被墙钟中断）")
    raise SystemExit(0)
state = json.loads(path.read_text(encoding="utf-8"))
print(f"  过拟合到：best_loss={state['best_loss']:.8f} "
      f"best_step={state['best_step']} 验证次数={state['validation_count']} "
      f"停止原因={state.get('stop_reason')}")
PY

echo "=== 门禁2 第5步：用保留下来的候选模型做最终闭环确认 ==="
(
  cd "$packages/panthera_sim"
  # The policy contract comes from the checkpoint; the budget from the dataset.
  python3 rollout.py \
    --robotwin-root "$robotwin_root" \
    --dataset-root "$subset" \
    --task-config "$task_config" \
    --task-name "$task_name" \
    --episode "$episode" \
    --model "$eval_model" \
    --unnorm-key "$dataset_name" \
    --action-chunk "$action_chunk" \
    --execution-horizon "$execution_horizon" \
    --workers 1 \
    --gpus "$eval_gpu" \
    --output "$report"
)

python3 - "$report" "$episode" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1], encoding="utf-8").read())
case = report["cases"][0]
print(f"\nepisode {sys.argv[2]}：success={case.get('success')} "
      f"动作={case.get('executed_actions')}/{case['expert_actions']} "
      f"查询={case.get('policy_queries')}")
for key, value in sorted(case.get("metrics", {}).items()):
    print(f"    {key:32s} {value}")
if not case.get("success"):
    print(
        "\n门禁2 未通过：策略在自己过拟合的那条轨迹、完全相同的场景下仍然失败。"
        "\n这与数据量、泛化都无关——训练到部署的通路本身没有跑通，"
        "\n在修好之前，任何更大规模实验的数字都不可解读。"
    )
    raise SystemExit(1)
print("\n门禁2 通过。")
PY

touch "${root}/COMPLETE"
if [[ "$keep" == "0" ]]; then
  rm -rf "$rlds"
  echo "已清理可重建的 RLDS；单轨迹快照、模型、日志和报告均已保留。"
fi
