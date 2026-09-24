#!/usr/bin/env bash

# Wait for the queued checkpoint evaluation, reuse its completed frozen-SFT R0
# result when available (or run R0 for an older queue), then execute the
# three-case physical reward-ordering gate. Waiting consumes no GPU. All
# executable phases expose only physical GPUs 2 and 3.

set -euo pipefail

if (( $# != 3 )); then
  echo "用法：$0 EVAL_SYSTEMD_UNIT TRAIN_OUTPUT_ROOT EVAL_OUTPUT_ROOT" >&2
  exit 2
fi

eval_unit="$1"
training_root="$2"
eval_root="$3"
project_root="${PANTHERA_LAB_ROOT:-/data/lyy/panthera-vla}"
evaluator="${project_root}/pipelines/ci/evaluate_openvla_rl_release_gpu23.sh"
summary_tool="${project_root}/packages/panthera_vla/summarize_rl_release_matrix.py"
reward_audit="${project_root}/packages/panthera_sim/audit_release_reward_replay.py"
python_bin="${PANTHERA_RL_PYTHON:-${project_root}/envs/rlinf/bin/python}"
r0_runtime="${PANTHERA_R0_RUNTIME:-${project_root}/runtime/rlinf-rl-gpu23-r0-next-20260924}"
robotwin_runtime="${PANTHERA_ROBOTWIN_RUNTIME:-${project_root}/runtime/robotwin-rl-gpu23}"
dataset_root="${PANTHERA_REWARD_REPLAY_DATASET:-${project_root}/ci/dynamics-ep2/dataset}"
post_root="${PANTHERA_RL_POST_AUDIT_ROOT:-${project_root}/ci/openvla-rl-release-post-audits/$(basename "${training_root}")}"
poll_seconds="${PANTHERA_RL_QUEUE_POLL_SECONDS:-60}"

for required in \
  "${evaluator}" \
  "${summary_tool}" \
  "${reward_audit}" \
  "${python_bin}" \
  "${r0_runtime}/ASSEMBLY.json" \
  "${robotwin_runtime}" \
  "${dataset_root}/dataset.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "错误：后置审计缺少依赖：${required}" >&2
    exit 2
  fi
done

while systemctl --user is-active --quiet "${eval_unit}"; do
  printf '%s 等待独立评测单元 %s 完成。\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${eval_unit}"
  sleep "${poll_seconds}"
done

if [[ ! -f "${training_root}/COMPLETE" ]]; then
  echo "错误：训练根目录没有 COMPLETE：${training_root}" >&2
  exit 3
fi
if [[ ! -f "${eval_root}/COMPLETE" ]]; then
  echo "错误：独立评测根目录没有 COMPLETE：${eval_root}" >&2
  systemctl --user show "${eval_unit}" \
    -p ActiveState -p SubState -p ExecMainStatus -p Result >&2 || true
  exit 3
fi
if [[ -e "${post_root}" ]]; then
  echo "错误：后置审计目录不可覆盖：${post_root}" >&2
  exit 2
fi
if [[ -e "${training_root}/summary-with-r0" ]]; then
  echo "错误：含 R0 的汇总目录不可覆盖：${training_root}/summary-with-r0" >&2
  exit 2
fi

mkdir -p "${post_root}"
date -u +%Y-%m-%dT%H:%M:%SZ >"${post_root}/RUNNING"
finalize() {
  code=$?
  printf '%s\n' "${code}" >"${post_root}/exit-code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ >"${post_root}/finished-at.txt"
  if [[ -e "${post_root}/RUNNING" ]]; then
    if (( code == 0 )); then
      mv "${post_root}/RUNNING" "${post_root}/COMPLETE"
    else
      mv "${post_root}/RUNNING" "${post_root}/FAILED"
    fi
  fi
}
trap finalize EXIT

if [[ -f "${eval_root}/r0/COMPLETE" ]]; then
  ln -s "${eval_root}/r0" "${post_root}/r0"
  printf 'reused %s\n' "${eval_root}/r0" >"${post_root}/r0-source.txt"
elif [[ -e "${eval_root}/r0" ]]; then
  echo "错误：独立评测中的 R0 存在但未完成：${eval_root}/r0" >&2
  exit 4
else
  PANTHERA_RL_RUNTIME="${r0_runtime}" \
  PANTHERA_ROBOTWIN_RUNTIME="${robotwin_runtime}" \
  PANTHERA_RL_RUN_STAMP="$(basename "${training_root}")-r0" \
  PANTHERA_RL_OUTPUT_DIR="${post_root}/r0" \
  WANDB_MODE=offline \
    "${evaluator}" r0
  printf 'executed %s\n' "${post_root}/r0" >"${post_root}/r0-source.txt"
fi

mkdir -p "${post_root}/reward-replay"
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3 \
PYTHONPATH="${project_root}/packages/panthera_sim:${robotwin_runtime}:${PYTHONPATH:-}" \
  "${python_bin}" "${reward_audit}" \
    --dataset-root "${dataset_root}" \
    --robotwin-root "${robotwin_runtime}" \
    --task-config panthera_phone_cylinder_socket_v2_pilot.yml \
    --episode 2 \
    --prefix-actions 780 \
    --output "${post_root}/reward-replay/report.json" \
    >"${post_root}/reward-replay/run.log" 2>&1

combined="${post_root}/combined-eval"
mkdir -p "${combined}"
ln -s "${post_root}/r0" "${combined}/r0"
for variant in c0 c1 c2; do
  if [[ ! -f "${eval_root}/${variant}/COMPLETE" ]]; then
    echo "错误：缺少完成的 ${variant} 独立评测：${eval_root}/${variant}" >&2
    exit 4
  fi
  ln -s "${eval_root}/${variant}" "${combined}/${variant}"
done

"${python_bin}" "${summary_tool}" "${training_root}" \
  --eval-root "${combined}" \
  --output-dir "${training_root}/summary-with-r0"

printf 'R0、奖励物理回放及四组汇总完成：%s\n' "${post_root}"
