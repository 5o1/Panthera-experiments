#!/usr/bin/env bash

# Deterministically evaluate either the frozen-SFT R0 baseline or one
# continuous-residual RL checkpoint in the release arena.  This entry point
# never performs a PPO update and never makes physical GPUs 0 or 1 visible.

set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "用法：$0 r0 | $0 {c0|c1|c2} /path/to/full_weights.pt" >&2
  exit 2
fi

eval_label="$1"
case "${eval_label}" in
  r0)
    if (( $# != 1 )); then
      echo "错误：R0 是无 RL checkpoint 的冻结 SFT 基线，不得传入权重路径。" >&2
      exit 2
    fi
    policy_variant="c0"
    checkpoint=""
    frozen_sft_baseline=true
    ;;
  c0|c1|c2)
    if (( $# != 2 )); then
      echo "错误：${eval_label} 评测必须显式传入 full_weights.pt。" >&2
      exit 2
    fi
    policy_variant="${eval_label}"
    checkpoint="$2"
    frozen_sft_baseline=false
    ;;
  *) echo "错误：未知评测组 ${eval_label}；只支持 r0、c0、c1、c2。" >&2; exit 2 ;;
esac

project_root="${PANTHERA_LAB_ROOT:-/data/lyy/panthera-vla}"
runtime="${PANTHERA_RL_RUNTIME:-${project_root}/runtime/rlinf-rl-gpu23}"
robotwin_runtime="${PANTHERA_ROBOTWIN_RUNTIME:-${project_root}/runtime/robotwin-rl-gpu23}"
python_bin="${PANTHERA_RL_PYTHON:-${project_root}/envs/rlinf/bin/python}"
config_name="robotwin_panthera_continuous_ppo"
physical_gpus="2,3"
minimum_free_mib="${PANTHERA_RL_MIN_FREE_MIB:-30000}"
poll_seconds="${PANTHERA_RL_GPU_POLL_SECONDS:-60}"
run_stamp="${PANTHERA_RL_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${PANTHERA_RL_OUTPUT_DIR:-${project_root}/ci/openvla-rl-release-eval/${run_stamp}/${eval_label}}"
run_name="${eval_label}-release-eval-${run_stamp}"
ray_tmp="/tmp/prl-eval-${run_stamp}-${eval_label}"

if [[ -n "${checkpoint}" && ! -f "${checkpoint}" ]]; then
  echo "错误：RL checkpoint 不存在：${checkpoint}" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *0* || "${CUDA_VISIBLE_DEVICES:-}" == *1* ]]; then
  echo "错误：外部 CUDA_VISIBLE_DEVICES 包含 GPU 0/1；本入口只允许物理 GPU 2,3。" >&2
  exit 2
fi
for required in \
  "${runtime}/ASSEMBLY.json" \
  "${runtime}/examples/embodiment/train_embodied_agent.py" \
  "${runtime}/examples/embodiment/config/${config_name}.yaml" \
  "${robotwin_runtime}" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "错误：缺少评测依赖：${required}" >&2
    exit 2
  fi
done
if [[ -e "${output_dir}/RUNNING" || -e "${output_dir}/COMPLETE" ]]; then
  echo "错误：评测目录不可覆盖：${output_dir}" >&2
  exit 2
fi

while true; do
  ready=1
  for gpu in 2 3; do
    free="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ ! "${free}" =~ ^[0-9]+$ ]]; then
      echo "错误：无法读取 GPU ${gpu} 的空闲显存。" >&2
      exit 2
    fi
    if (( free < minimum_free_mib )); then
      ready=0
    fi
  done
  (( ready == 1 )) && break
  printf '%s GPU 2/3 空闲显存不足 %s MiB；保持等待，不终止任何进程。\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${minimum_free_mib}"
  sleep "${poll_seconds}"
done

mkdir -p "${output_dir}" "${ray_tmp}"
date -u +%Y-%m-%dT%H:%M:%SZ >"${output_dir}/RUNNING"
"${python_bin}" - \
  "${output_dir}/eval-config.json" \
  "${checkpoint}" \
  "${eval_label}" \
  "${policy_variant}" \
  "${frozen_sft_baseline}" \
  "${PANTHERA_RL_REWARD_VARIANT:-r2}" \
  "${PANTHERA_RL_WARM_START_ACTIONS:-780}" \
  "${PANTHERA_EVAL_EXECUTION_HORIZON:-5}" \
  "${PANTHERA_RL_EVAL_ROLLOUT_EPOCHS:-4}" \
  "${PANTHERA_RL_EVAL_NUM_ENVS:-2}" \
  "${PANTHERA_RL_EVAL_EPISODE_ACTIONS:-150}" \
  "${WANDB_MODE:-offline}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output,
    checkpoint_raw,
    eval_label,
    policy_variant,
    frozen_sft_baseline,
    reward_variant,
    warm_start_actions,
    execution_horizon,
    rollout_epochs,
    num_envs,
    episode_actions,
    wandb_mode,
) = sys.argv[1:]
checkpoint = Path(checkpoint_raw).resolve() if checkpoint_raw else None
digest = hashlib.sha256()
if checkpoint is not None:
    with checkpoint.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "physical_gpus": ["2", "3"],
    "eval_label": eval_label,
    "policy_variant": policy_variant,
    "frozen_sft_baseline": frozen_sft_baseline == "true",
    "checkpoint": str(checkpoint) if checkpoint is not None else None,
    "checkpoint_size": checkpoint.stat().st_size if checkpoint is not None else None,
    "checkpoint_sha256": digest.hexdigest() if checkpoint is not None else None,
    "reward_variant": reward_variant,
    "warm_start_actions": int(warm_start_actions),
    "execution_horizon": int(execution_horizon),
    "rollout_epochs": int(rollout_epochs),
    "num_envs": int(num_envs),
    "episode_actions": int(episode_actions),
    "wandb_mode": wandb_mode,
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

finalize() {
  code=$?
  printf '%s\n' "${code}" >"${output_dir}/exit-code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ >"${output_dir}/finished-at.txt"
  if [[ -e "${output_dir}/RUNNING" ]]; then
    if (( code == 0 )); then
      mv "${output_dir}/RUNNING" "${output_dir}/COMPLETE"
    else
      mv "${output_dir}/RUNNING" "${output_dir}/FAILED"
    fi
  fi
}
trap finalize EXIT

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${physical_gpus}"
export RAY_ADDRESS=local
export RAY_TMPDIR="${ray_tmp}"
export EMBODIED_PATH="${runtime}/examples/embodiment"
export REPO_PATH="${runtime}"
export ROBOTWIN_PATH="${robotwin_runtime}"
export PYTHONPATH="${runtime}:${robotwin_runtime}:${PYTHONPATH:-}"
export ROBOT_PLATFORM=PANTHERA
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_PROPRIO_DIM=28
export PANTHERA_RL_RUNTIME="${runtime}"
export PANTHERA_ROBOTWIN_RUNTIME="${robotwin_runtime}"
export PANTHERA_RL_OUTPUT_DIR="${output_dir}"
export PANTHERA_RL_RUN_NAME="${run_name}"
export PANTHERA_RL_VARIANT="${policy_variant}"
export PANTHERA_RL_REWARD_VARIANT="${PANTHERA_RL_REWARD_VARIANT:-r2}"
export PANTHERA_RL_WARM_START=true
export PANTHERA_RL_WARM_START_ACTIONS="${PANTHERA_RL_WARM_START_ACTIONS:-780}"
export PANTHERA_RL_EVAL_ONLY=true
export PANTHERA_RL_VAL_CHECK_INTERVAL=1
export PANTHERA_RL_EVAL_FROZEN_SFT_BASELINE="${frozen_sft_baseline}"
if [[ -n "${checkpoint}" ]]; then
  export PANTHERA_RL_CHECKPOINT="${checkpoint}"
else
  unset PANTHERA_RL_CHECKPOINT
fi
export PANTHERA_EVAL_EXECUTION_HORIZON="${PANTHERA_EVAL_EXECUTION_HORIZON:-5}"
export PANTHERA_RL_EVAL_ROLLOUT_EPOCHS="${PANTHERA_RL_EVAL_ROLLOUT_EPOCHS:-4}"
export PANTHERA_RL_EVAL_NUM_ENVS="${PANTHERA_RL_EVAL_NUM_ENVS:-2}"
export PANTHERA_RL_EVAL_EPISODE_ACTIONS="${PANTHERA_RL_EVAL_EPISODE_ACTIONS:-150}"
export PANTHERA_RL_EVAL_EPOCH_ACTIONS="${PANTHERA_RL_EVAL_EPOCH_ACTIONS:-150}"
export PANTHERA_RL_HISTORY_NOISE_STD=0.0
export PANTHERA_RL_HISTORY_DROPOUT=0.0
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT=panthera-openvla-rl

cd "${robotwin_runtime}"
"${python_bin}" "${runtime}/examples/embodiment/train_embodied_agent.py" \
  --config-path "${runtime}/examples/embodiment/config" \
  --config-name "${config_name}" >"${output_dir}/run.log" 2>&1
