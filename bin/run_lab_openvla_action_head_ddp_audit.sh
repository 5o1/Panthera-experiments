#!/usr/bin/env bash
# Run the OpenVLA-OFT action-head DDP regression audit on the Lab.
#
# The audit performs one optimizer step for both L1 and diffusion heads. It
# does not load a VLA checkpoint or start RoboTwin, and it never touches real
# hardware.

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
runtime="${workspace}/runtime/rlinf"
python="${runtime}/.venv/bin/python"
torchrun="${runtime}/.venv/bin/torchrun"
audit="${workspace}/packages/panthera_vla/audit_openvla_action_head_ddp.py"
output="${1:-${workspace}/reports/evidence/openvla_action_head_ddp_postfix_2026-09-23.json}"
log="${output%.json}.log"
gpus="${PANTHERA_DDP_AUDIT_GPUS:-0,1,2}"

[[ -x "$python" && -x "$torchrun" ]] || {
  echo "错误：OpenVLA runtime 不完整：${runtime}/.venv" >&2
  exit 1
}
[[ -f "$audit" ]] || {
  echo "错误：找不到 DDP audit：${audit}" >&2
  exit 1
}

IFS=',' read -r -a gpu_list <<< "$gpus"
if (( ${#gpu_list[@]} < 2 )); then
  echo "错误：DDP audit 至少需要两张 GPU，当前为 ${gpus}" >&2
  exit 1
fi
command -v nvidia-smi >/dev/null || {
  echo "错误：找不到 nvidia-smi，无法确认 GPU 是否空闲。" >&2
  exit 1
}

compute_apps=$(nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits)
for gpu in "${gpu_list[@]}"; do
  uuid=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
  busy=$(awk -F ', *' -v uuid="$uuid" '$1 == uuid {print}' <<< "$compute_apps")
  if [[ -n "$busy" ]]; then
    echo "错误：GPU ${gpu} 已有计算进程，拒绝启动 DDP audit：" >&2
    echo "$busy" >&2
    exit 1
  fi
done
mkdir -p "$(dirname "$output")"

export CUDA_VISIBLE_DEVICES="$gpus"
export ROBOT_PLATFORM=PANTHERA
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_PROPRIO_DIM=28
export PYTHONUNBUFFERED=1

echo "OpenVLA action-head DDP audit: GPUs=${gpus}, ranks=${#gpu_list[@]}"
"$torchrun" --standalone --nproc-per-node="${#gpu_list[@]}" \
  "$audit" --output "$output" 2>&1 | tee "$log"
echo "证据：${output}"
echo "日志：${log}"
