#!/usr/bin/env bash

# Wait for one serial C0/C1/C2 training matrix, then independently evaluate
# every step-10 checkpoint.  Waiting consumes no GPU; each evaluation reuses
# the GPU2/3 admission guard in evaluate_openvla_rl_release_gpu23.sh.

set -euo pipefail

if (( $# != 2 )); then
  echo "用法：$0 SYSTEMD_UNIT TRAIN_OUTPUT_ROOT" >&2
  exit 2
fi

training_unit="$1"
training_root="$2"
project_root="${PANTHERA_LAB_ROOT:-/data/lyy/panthera-vla}"
evaluator="${project_root}/pipelines/ci/evaluate_openvla_rl_release_gpu23.sh"
eval_root="${PANTHERA_RL_EVAL_ROOT:-${project_root}/ci/openvla-rl-release-eval/$(basename "${training_root}")-step10}"
poll_seconds="${PANTHERA_RL_QUEUE_POLL_SECONDS:-60}"

if [[ ! -x "${evaluator}" ]]; then
  echo "错误：独立评测入口不存在或不可执行：${evaluator}" >&2
  exit 2
fi

while systemctl --user is-active --quiet "${training_unit}"; do
  printf '%s 等待训练单元 %s 完成。\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${training_unit}"
  sleep "${poll_seconds}"
done

if [[ ! -f "${training_root}/COMPLETE" ]]; then
  echo "错误：训练单元已停止，但训练根目录没有 COMPLETE：${training_root}" >&2
  systemctl --user show "${training_unit}" \
    -p ActiveState -p SubState -p ExecMainStatus -p Result >&2 || true
  exit 3
fi
if [[ -e "${eval_root}/RUNNING" || -e "${eval_root}/COMPLETE" ]]; then
  echo "错误：评测根目录不可覆盖：${eval_root}" >&2
  exit 2
fi

mkdir -p "${eval_root}"
date -u +%Y-%m-%dT%H:%M:%SZ >"${eval_root}/RUNNING"
finalize() {
  code=$?
  printf '%s\n' "${code}" >"${eval_root}/exit-code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ >"${eval_root}/finished-at.txt"
  if [[ -e "${eval_root}/RUNNING" ]]; then
    if (( code == 0 )); then
      mv "${eval_root}/RUNNING" "${eval_root}/COMPLETE"
    else
      mv "${eval_root}/RUNNING" "${eval_root}/FAILED"
    fi
  fi
}
trap finalize EXIT

PANTHERA_RL_RUN_STAMP="$(basename "${training_root}")-step10" \
PANTHERA_RL_OUTPUT_DIR="${eval_root}/r0" \
WANDB_MODE=offline \
  "${evaluator}" r0

for variant in c0 c1 c2; do
  mapfile -t checkpoints < <(
    find "${training_root}/${variant}" -type f \
      -path '*/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt' \
      -print
  )
  if (( ${#checkpoints[@]} != 1 )); then
    echo "错误：${variant} 应恰有一个 step-10 checkpoint，实际 ${#checkpoints[@]} 个。" >&2
    exit 4
  fi
  PANTHERA_RL_RUN_STAMP="$(basename "${training_root}")-step10" \
  PANTHERA_RL_OUTPUT_DIR="${eval_root}/${variant}" \
  WANDB_MODE=offline \
    "${evaluator}" "${variant}" "${checkpoints[0]}"
done

python_bin="${PANTHERA_RL_PYTHON:-${project_root}/envs/rlinf/bin/python}"
summary_tool="${project_root}/packages/panthera_vla/summarize_rl_release_matrix.py"
"${python_bin}" "${summary_tool}" "${training_root}" \
  --eval-root "${eval_root}" \
  --output-dir "${training_root}/summary"

printf 'R0 基线与三组 RL release-arena 评测完成：%s\n' "${eval_root}"
