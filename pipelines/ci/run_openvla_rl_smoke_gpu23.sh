#!/usr/bin/env bash

# Run the continuous-residual PPO smoke matrix without touching physical
# GPUs 0 or 1.  Each variant uses physical GPUs 2 and 3 together; the variants
# run sequentially so separate Ray jobs cannot compete for either card.

set -euo pipefail

project_root="${PANTHERA_LAB_ROOT:-/data/lyy/panthera-vla}"
runtime="${PANTHERA_RL_RUNTIME:-${project_root}/runtime/rlinf-rl-gpu23}"
robotwin_runtime="${PANTHERA_ROBOTWIN_RUNTIME:-${project_root}/runtime/robotwin-rl-gpu23}"
python_bin="${PANTHERA_RL_PYTHON:-${project_root}/envs/rlinf/bin/python}"
config_name="robotwin_panthera_continuous_ppo"
physical_gpus="2,3"
minimum_free_mib="${PANTHERA_RL_MIN_FREE_MIB:-30000}"
poll_seconds="${PANTHERA_RL_GPU_POLL_SECONDS:-60}"
wandb_entity="${WANDB_ENTITY:-assanekowww}"
run_stamp="${PANTHERA_RL_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_root="${PANTHERA_RL_OUTPUT_ROOT:-${project_root}/ci/openvla-rl-smoke/${run_stamp}}"

if [[ "${CUDA_VISIBLE_DEVICES:-}" == *0* || "${CUDA_VISIBLE_DEVICES:-}" == *1* ]]; then
  echo "错误：外部 CUDA_VISIBLE_DEVICES 包含 GPU 0/1；本入口只允许物理 GPU 2,3。" >&2
  exit 2
fi

for required in \
  "${runtime}/ASSEMBLY.json" \
  "${runtime}/examples/embodiment/train_embodied_agent.py" \
  "${runtime}/examples/embodiment/config/${config_name}.yaml" \
  "${runtime}/seeds/panthera_rl_ep2_seed.json" \
  "${robotwin_runtime}" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "错误：缺少运行依赖：${required}" >&2
    exit 2
  fi
done

mkdir -p "${output_root}"
if [[ -e "${output_root}/RUNNING" || -e "${output_root}/COMPLETE" ]]; then
  echo "错误：运行目录不可覆盖：${output_root}" >&2
  exit 2
fi

wait_for_gpu_memory() {
  local gpu free
  while true; do
    local ready=1
    for gpu in 2 3; do
      free="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
      if [[ ! "${free}" =~ ^[0-9]+$ ]]; then
        echo "错误：无法读取 GPU ${gpu} 的空闲显存。" >&2
        return 2
      fi
      if (( free < minimum_free_mib )); then
        ready=0
      fi
    done
    if (( ready == 1 )); then
      return 0
    fi
    printf '%s GPU 2/3 空闲显存不足 %s MiB；保持等待，不终止任何进程。\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${minimum_free_mib}"
    sleep "${poll_seconds}"
  done
}

write_run_metadata() {
  "${python_bin}" - "${output_root}/run-config.json" "${runtime}" "${robotwin_runtime}" \
    "${physical_gpus}" "${minimum_free_mib}" "${run_stamp}" "$@" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

output, runtime, robotwin, gpus, minimum_free, stamp, *variants = sys.argv[1:]
assembly = Path(runtime, "ASSEMBLY.json")
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_stamp": stamp,
    "physical_gpus": gpus.split(","),
    "logical_gpus_seen_by_rlinf": ["0", "1"],
    "minimum_free_mib_per_gpu": int(minimum_free),
    "variants": variants,
    "runtime": runtime,
    "robotwin_runtime": robotwin,
    "assembly": json.loads(assembly.read_text(encoding="utf-8")),
    "contract": {
        "robot_platform": "PANTHERA",
        "action_chunk": 25,
        "action_dim": 7,
        "proprio_dim": 28,
        "execution_horizon": int(os.environ.get("PANTHERA_RL_EXECUTION_HORIZON", "5")),
    },
    "reward": {
        "variant": os.environ.get("PANTHERA_RL_REWARD_VARIANT", "r1"),
        "warm_start": os.environ.get("PANTHERA_RL_WARM_START", "false"),
        "warm_start_actions": int(os.environ.get("PANTHERA_RL_WARM_START_ACTIONS", "780")),
    },
    "training": {
        "max_epochs": int(os.environ.get("PANTHERA_RL_MAX_EPOCHS", "1")),
        "max_steps": int(os.environ.get("PANTHERA_RL_MAX_STEPS", "1")),
        "save_interval": int(os.environ.get("PANTHERA_RL_SAVE_INTERVAL", "1")),
        "rollout_epochs": int(os.environ.get("PANTHERA_RL_ROLLOUT_EPOCHS", "2")),
        "num_envs": int(os.environ.get("PANTHERA_RL_NUM_ENVS", "2")),
        "episode_actions": int(os.environ.get("PANTHERA_RL_EPISODE_ACTIONS", "25")),
        "epoch_actions": int(os.environ.get("PANTHERA_RL_EPOCH_ACTIONS", "25")),
        "seed": int(os.environ.get("PANTHERA_RL_SEED", "1234")),
        "wandb_mode": os.environ.get("WANDB_MODE", "offline"),
    },
}
for relative in (
    "examples/embodiment/config/robotwin_panthera_continuous_ppo.yaml",
    "rlinf/models/embodiment/openvla_oft/continuous_residual.py",
):
    path = Path(runtime, relative)
    payload.setdefault("sha256", {})[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_variant() {
  local variant="$1" consistency="$2" history_noise="$3" history_dropout="$4"
  local reward_variant="${PANTHERA_RL_REWARD_VARIANT:-r1}"
  local run_name="${reward_variant}-${variant}-${run_stamp}"
  local run_dir="${output_root}/${variant}"
  # Ray creates deeply nested Unix-domain sockets, whose Linux path limit is
  # 107 bytes.  Keep its ephemeral root under /tmp and store durable logs in
  # run_dir; placing Ray below the long CI output path fails before model load.
  local ray_tmp="/tmp/prl-${run_stamp}-${variant}"
  mkdir -p "${run_dir}" "${ray_tmp}"
  printf '%s\n' "${ray_tmp}" >"${run_dir}/ray-temp-dir.txt"
  if [[ -e "${run_dir}/RUNNING" || -e "${run_dir}/COMPLETE" ]]; then
    echo "错误：变体目录不可覆盖：${run_dir}" >&2
    return 2
  fi

  wait_for_gpu_memory
  date -u +%Y-%m-%dT%H:%M:%SZ >"${run_dir}/RUNNING"
  nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader >"${run_dir}/gpu-before.csv"

  set +e
  (
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
    export PANTHERA_RL_OUTPUT_DIR="${run_dir}"
    export PANTHERA_RL_RUN_NAME="${run_name}"
    export PANTHERA_RL_VARIANT="${variant}"
    export PANTHERA_RL_CONSISTENCY_COEF="${consistency}"
    export PANTHERA_RL_HISTORY_NOISE_STD="${history_noise}"
    export PANTHERA_RL_HISTORY_DROPOUT="${history_dropout}"
    # Keep experiment telemetry local unless the user explicitly opts in to
    # uploading it.  Setting WANDB_MODE=online at invocation remains possible.
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_ENTITY="${wandb_entity}"
    export WANDB_PROJECT="panthera-openvla-rl"
    # RoboTwin still resolves several asset files relative to the process
    # working directory.  Run from its generated runtime while keeping the
    # RLinf entry point and config absolute.
    cd "${robotwin_runtime}"
    exec "${python_bin}" "${runtime}/examples/embodiment/train_embodied_agent.py" \
      --config-path "${runtime}/examples/embodiment/config" \
      --config-name "${config_name}"
  ) >"${run_dir}/run.log" 2>&1
  local code=$?
  set -e

  printf '%s\n' "${code}" >"${run_dir}/exit-code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ >"${run_dir}/finished-at.txt"
  if (( code != 0 )); then
    mv "${run_dir}/RUNNING" "${run_dir}/FAILED"
    echo "错误：${variant} smoke 失败，详见 ${run_dir}/run.log" >&2
    return "${code}"
  fi
  mv "${run_dir}/RUNNING" "${run_dir}/COMPLETE"
}

variants=("$@")
if (( ${#variants[@]} == 0 )); then
  variants=(c0 c1 c2)
fi
for variant in "${variants[@]}"; do
  case "${variant}" in
    c0|c1|c2) ;;
    *) echo "错误：未知变体 ${variant}；只支持 c0、c1、c2。" >&2; exit 2 ;;
  esac
done

write_run_metadata "${variants[@]}"
date -u +%Y-%m-%dT%H:%M:%SZ >"${output_root}/RUNNING"

finalize_output_root() {
  local code=$?
  if [[ -e "${output_root}/RUNNING" ]]; then
    if (( code == 0 )); then
      mv "${output_root}/RUNNING" "${output_root}/COMPLETE"
    else
      mv "${output_root}/RUNNING" "${output_root}/FAILED"
    fi
  fi
  return "${code}"
}
trap finalize_output_root EXIT

for variant in "${variants[@]}"; do
  case "${variant}" in
    c0) run_variant c0 0.0 0.0 0.0 ;;
    c1) run_variant c1 0.05 0.0 0.0 ;;
    c2) run_variant c2 0.05 0.01 0.10 ;;
  esac
done

printf '全部 smoke 完成：%s\n' "${output_root}"
