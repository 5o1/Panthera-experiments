#!/usr/bin/env bash
# Build and smoke-test the 28-D proprioception experiment on GPU0.
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
current_run="${CI_DYNAMICS_SOURCE_RUN:-overfit-ep2-24h-b6-20260920T140123Z}"
source_root="${CI_DYNAMICS_SOURCE_ROOT:-${workspace}/ci/overfit-ep2-24h/${current_run}/subset}"
dataset_root="${CI_DYNAMICS_DATASET_ROOT:-${workspace}/ci/dynamics-ep2/dataset}"
rlds_root="${CI_DYNAMICS_RLDS_ROOT:-${workspace}/ci/dynamics-ep2/rlds}"
state_root="${CI_DYNAMICS_STATE_ROOT:-${workspace}/state/dynamics-overfit-gpu0}"
smoke_root="${CI_DYNAMICS_SMOKE_ROOT:-${workspace}/ci/dynamics-ep2/smoke}"
robotwin_root="${workspace}/runtime/robotwin"
dataset_name="panthera_phone_cylinder_socket_v2_dynamics"
episode="${CI_OVERFIT_EPISODE:-2}"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行。" >&2
  exit 1
fi
mkdir -p "$state_root"
exec 9>"${state_root}/prepare.lock"
flock -n 9 || { echo "错误：另一个 dynamics GPU0 准备任务正在运行。" >&2; exit 1; }

# shellcheck disable=SC1091
source "${workspace}/tools/activate_lab_vla.sh" >/dev/null

python3 "${workspace}/pipelines/assemble_runtime.py" \
  --upstream robotwin \
  --source "${workspace}/externals/RoboTwin" \
  --runtime "$robotwin_root"

# The OpenVLA runtime predates the configurable proprioception width.  Upgrade
# the already-applied Panthera branch in place, but only through an exact
# replacement so an unexpected upstream/runtime state fails closed.
constants="${workspace}/envs/rlinf/lib/python3.11/site-packages/prismatic/vla/constants.py"
python3 - "$constants" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
dynamic = '"PROPRIO_DIM": int(os.environ.get("PANTHERA_PROPRIO_DIM", "7")),'
if dynamic not in text:
    old = '"PROPRIO_DIM": 7,'
    start = text.find("PANTHERA_CONSTANTS = {")
    end = text.find("\n}", start)
    if start < 0 or end < 0:
        raise SystemExit("OpenVLA Panthera constants are not in the expected state")
    block = text[start:end]
    if block.count(old) != 1:
        raise SystemExit("OpenVLA Panthera PROPRIO_DIM is not in the expected state")
    text = text[:start] + block.replace(old, dynamic) + text[end:]
    path.write_text(text, encoding="utf-8")
print("OpenVLA Panthera proprioception width is environment-selectable")
PY

if [[ ! -s "${dataset_root}/dynamics-audit.json" ]]; then
  python3 "${workspace}/packages/panthera_sim/build_dynamics_overfit_dataset.py" \
    --source-root "$source_root" \
    --target-root "$dataset_root" \
    --robotwin-root "$robotwin_root" \
    --task-config panthera_phone_cylinder_socket_v2_pilot.yml \
    --episode "$episode"
fi

export PANTHERA_RLDS_SCHEMA_VERSION=10
export PANTHERA_RLDS_SCENE_PROFILE=panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2
export PANTHERA_RLDS_DATASET_NAME="$dataset_name"
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_PROPRIO_DIM=28
export PANTHERA_SOURCE_DATASET_ROOT="$dataset_root"
export ROBOT_PLATFORM=PANTHERA

if [[ ! -s "${state_root}/rlds-summary.json" ]]; then
  python3 "${workspace}/packages/panthera_vla/panthera_rlds.py" \
    --source-root "$dataset_root" \
    --data-root "$rlds_root" \
    --validation-episode "$episode" \
    --summary "${state_root}/rlds-summary.json"
fi

if [[ ! -f "${state_root}/smoke.ok" ]]; then
  if [[ -e "$smoke_root" ]]; then
    echo "错误：smoke 目录已存在但没有通过标记，拒绝覆盖：${smoke_root}" >&2
    exit 1
  fi
  mkdir -p "$smoke_root" "${workspace}/wandb"
  export CUDA_VISIBLE_DEVICES=0
  export WANDB_MODE=offline
  export WANDB_DIR="${workspace}/wandb"
  timeout --signal=INT --kill-after=120s 45m \
    torchrun --standalone --nproc-per-node=1 \
      "${workspace}/packages/panthera_vla/run_finetune.py" \
      --vla_path "${workspace}/models/openvla-oft-place-empty-cup" \
      --data_root_dir "$rlds_root" \
      --dataset_name "$dataset_name" \
      --run_root_dir "$smoke_root" \
      --run_id_override smoke \
      --shuffle_buffer_size 128 \
      --use_l1_regression true \
      --use_diffusion false \
      --num_images_in_input 1 \
      --use_proprio true \
      --batch_size 1 \
      --max_steps 2 \
      --save_freq 999999 \
      --image_aug false \
      --lora_rank 8 \
      --merge_lora_during_training false \
      --use_val_set true \
      --val_freq 1 \
      --val_time_limit 60 \
      --early_stopping_min_delta 10.0 \
      --early_stopping_patience 1 \
      --wandb_entity panthera-local \
      --wandb_project panthera-dynamics-smoke \
      --wandb_log_freq 1 \
      >"${state_root}/smoke.log" 2>&1
  grep -q '"proprio_dim": 28' "${smoke_root}/smoke/training.json"
  touch "${state_root}/smoke.ok"
fi

date -u +%Y-%m-%dT%H:%M:%SZ >"${state_root}/completed-at.txt"
touch "${state_root}/prepare.ok"
echo "28-D dynamics 数据、RLDS 和 GPU0 训练 smoke 已通过。"
