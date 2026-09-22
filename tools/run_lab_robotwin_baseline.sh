#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
rlinf_root="${workspace}/runtime/rlinf"
model_root="${workspace}/models/openvla-oft-place-empty-cup"
state_root="${workspace}/state/robotwin-baseline-state"
log_root="${workspace}/logs/robotwin-baseline"
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
activation_script="${script_root}/activate_lab_vla.sh"

model_repo="RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup"
model_revision="04150e05ded8b915ccf6adb7ea903ff13dc3aa27"
expected_model_bytes=15152598778

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
if [[ ! -f "${workspace}/state/bootstrap-state/verified.ok" ]]; then
  echo "错误：RLinf/RoboTwin 基础环境尚未通过验收。" >&2
  exit 1
fi
if [[ ! -f "$activation_script" ]]; then
  echo "错误：缺少 Lab 环境激活脚本：${activation_script}" >&2
  exit 1
fi

mkdir -p "$state_root" "$log_root" "${workspace}/models"
exec 9>"${state_root}/baseline.lock"
if ! flock -n 9; then
  echo "错误：另一个 RoboTwin baseline 流程正在运行。" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$activation_script"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "[1/4] 下载并验证固定提交的 OpenVLA-OFT 模型"
if [[ ! -f "${state_root}/model.ok" ]]; then
  hf download "$model_repo" \
    --revision "$model_revision" \
    --local-dir "$model_root"

  for required_file in \
    config.json \
    model.safetensors.index.json \
    model-00001-of-00004.safetensors \
    model-00002-of-00004.safetensors \
    model-00003-of-00004.safetensors \
    model-00004-of-00004.safetensors \
    proprio_projector--10000_checkpoint.pt; do
    if [[ ! -s "${model_root}/${required_file}" ]]; then
      echo "错误：模型文件缺失或为空：${required_file}" >&2
      exit 1
    fi
  done

  actual_model_bytes=$(find "$model_root" \
    -path "${model_root}/.cache" -prune -o \
    -type f -printf '%s\n' | awk '{ total += $1 } END { print total + 0 }')
  if (( actual_model_bytes < expected_model_bytes )); then
    echo "错误：模型文件总量不足：${actual_model_bytes}/${expected_model_bytes} bytes" >&2
    exit 1
  fi
  printf '%s\n' "$model_revision" >"${state_root}/model-revision.txt"
  touch "${state_root}/model.ok"
else
  echo "固定模型已下载并验证，跳过。"
fi

common_overrides=(
  "~cluster.component_placement"
  "+cluster.component_placement={env:0-3,rollout:0-3}"
  "env.eval.total_num_envs=4"
  "env.eval.rollout_epoch=1"
  "env.eval.use_fixed_reset_state_ids=true"
  "env.eval.max_episode_steps=200"
  "env.eval.max_steps_per_rollout_epoch=200"
  "rollout.model.model_path=${model_root}"
  "env.eval.assets_path=${workspace}/runtime/robotwin"
)

echo "[2/4] 组合并验证 4 卡 smoke 配置（不启动仿真）"
resolved_config="${state_root}/smoke-config.txt"
cd "$rlinf_root"
bash evaluations/run_eval.sh \
  robotwin \
  robotwin_place_empty_cup_openvlaoft_eval \
  "${common_overrides[@]}" \
  --cfg job >"$resolved_config"

grep -q '^    env: 0-3$' "$resolved_config"
grep -q '^    rollout: 0-3$' "$resolved_config"
grep -q '^    total_num_envs: 4$' "$resolved_config"
grep -q "^    model_path: ${model_root}$" "$resolved_config"
grep -q "^    assets_path: ${workspace}/runtime/robotwin$" "$resolved_config"
touch "${state_root}/config.ok"

echo "[3/4] 检查 GPU 与 Ray 运行边界"
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( gpu_count != 4 )); then
  echo "错误：预期 4 张 GPU，实际 ${gpu_count} 张。" >&2
  exit 1
fi
python -c 'from curobo.types.math import Pose; from curobo.wrap.reacher.motion_gen import MotionGen; from pytorch3d import _C'
if pgrep -u "$(id -u)" -x raylet >/dev/null 2>&1; then
  echo "错误：当前用户已有 Ray 集群，拒绝与其他任务共享或停止它。" >&2
  exit 1
fi

echo "[4/4] 执行 4 环境、200 步的 OpenVLA-OFT smoke evaluation"
if [[ ! -f "${state_root}/smoke.ok" ]]; then
  smoke_stamp=$(date +%Y%m%d-%H%M%S)
  smoke_log="${log_root}/smoke-${smoke_stamp}.log"
  printf '%s\n' "$smoke_log" >"${state_root}/smoke-log.txt"

  set +e
  timeout --signal=INT --kill-after=60s \
    "${PANTHERA_SMOKE_TIMEOUT:-45m}" \
    bash evaluations/run_eval.sh \
      robotwin \
      robotwin_place_empty_cup_openvlaoft_eval \
      "${common_overrides[@]}" 2>&1 | tee "$smoke_log"
  smoke_status=${PIPESTATUS[0]}
  set -e

  printf '%s\n' "$smoke_status" >"${state_root}/smoke-exit-code.txt"
  if (( smoke_status != 0 )); then
    echo "错误：RoboTwin smoke evaluation 失败，退出码 ${smoke_status}。" >&2
    exit "$smoke_status"
  fi

  fatal_pattern='Traceback \(most recent call last\)|ModuleNotFoundError:|ImportError:|RayTaskError\(|WorkerCrashedError:'
  if grep -Eq "$fatal_pattern" "$smoke_log"; then
    grep -nE "$fatal_pattern" "$smoke_log" \
      >"${state_root}/smoke-fatal-errors.txt" || true
    echo "错误：进程退出码虽然为 0，但日志中存在致命异常；拒绝写入成功标记。" >&2
    exit 1
  fi

  metrics_pattern="'eval/success_once':.*'eval/num_trajectories': 4"
  if ! grep -E "$metrics_pattern" "$smoke_log" \
      >"${state_root}/smoke-metrics.txt"; then
    echo "错误：未找到 4 条完整轨迹的 eval/success_once 指标；拒绝写入成功标记。" >&2
    exit 1
  fi

  evaluation_log_root=$(sed -n \
    's|.*runner.logger.log_path=\([^ ]*\).*|\1|p' "$smoke_log" | head -n 1)
  if [[ -z "$evaluation_log_root" || ! -d "${evaluation_log_root}/video/eval" ]]; then
    echo "错误：未找到本次 evaluation 的视频目录；拒绝写入成功标记。" >&2
    exit 1
  fi
  video_count=$(find "${evaluation_log_root}/video/eval" \
    -type f -name '*.mp4' -size +0c | wc -l)
  if (( video_count < 4 )); then
    echo "错误：预期至少 4 条非空轨迹视频，实际 ${video_count} 条。" >&2
    exit 1
  fi
  printf '%s\n' "$evaluation_log_root" >"${state_root}/smoke-output-root.txt"
  printf '%s\n' "$video_count" >"${state_root}/smoke-video-count.txt"
  touch "${state_root}/smoke.ok"
else
  echo "RoboTwin smoke evaluation 已通过，跳过。"
fi

date --iso-8601=seconds >"${state_root}/completed-at.txt"
echo "RoboTwin baseline smoke 完成：${state_root}"
