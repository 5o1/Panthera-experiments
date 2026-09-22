#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
rlinf_root="${workspace}/runtime/rlinf"
upstream_root="${workspace}/externals/RoboTwin"
robotwin_root="${workspace}/runtime/robotwin"
overlay_root="${workspace}/overlays/rlinf"
robotwin_overlay="${workspace}/overlays/robotwin"
state_root="${PANTHERA_EVAL_STATE_ROOT:-${workspace}/state/panthera-single-policy-eval-state}"
log_root="${PANTHERA_EVAL_LOG_ROOT:-${workspace}/logs/panthera-single-policy-eval}"
activation_script="${workspace}/tools/activate_lab_vla.sh"
task_name="${PANTHERA_EVAL_TASK_NAME:-place_cylinder_in_groove}"
scene_profile="${PANTHERA_EVAL_SCENE_PROFILE:-}"
unnorm_key="${PANTHERA_EVAL_UNNORM_KEY:-panthera_single_cylinder}"
eval_config_name="${PANTHERA_EVAL_CONFIG_NAME:-robotwin_panthera_cylinder_openvlaoft_eval}"
env_source="${PANTHERA_EVAL_ENV_SOURCE:-${overlay_root}/config/env/robotwin_place_cylinder_in_groove.yaml}"
eval_source="${PANTHERA_EVAL_CONFIG_SOURCE:-${overlay_root}/evaluations/robotwin_panthera_cylinder_openvlaoft_eval.yaml}"
seed_source="${PANTHERA_EVAL_SEED_SOURCE:-${overlay_root}/seeds/panthera_cylinder_eval_seeds.json}"
openvla_l1_eval_patch="${workspace}/overlays/rlinf/patches/rlinf_openvla_oft_l1_eval.patch"
cuda_visible_subset_patch="${workspace}/overlays/rlinf/patches/rlinf_cuda_visible_subset.patch"
eval_execution_horizon_patch="${workspace}/overlays/rlinf/patches/rlinf_eval_execution_horizon.patch"
eval_rollout_execution_horizon_patch="${workspace}/overlays/rlinf/patches/rlinf_eval_rollout_execution_horizon.patch"
openvla_constants_patch="${workspace}/patches/openvla_oft_panthera_constants.patch"
openvla_site_packages="${workspace}/runtime/rlinf/.venv/lib/python3.11/site-packages"
env_target="${PANTHERA_EVAL_ENV_TARGET:-${rlinf_root}/examples/embodiment/config/env/robotwin_place_cylinder_in_groove.yaml}"
eval_target="${PANTHERA_EVAL_CONFIG_TARGET:-${rlinf_root}/evaluations/robotwin/robotwin_panthera_cylinder_openvlaoft_eval.yaml}"
task_source="${PANTHERA_EVAL_TASK_SOURCE:-${robotwin_overlay}/envs/place_cylinder_in_groove.py}"
instruction_source="${PANTHERA_EVAL_INSTRUCTION_SOURCE:-${robotwin_overlay}/description/task_instruction/place_cylinder_in_groove.json}"
task_target="${robotwin_root}/envs/${task_name}.py"
instruction_target="${robotwin_root}/description/task_instruction/${task_name}.json"
minimum_success="${PANTHERA_EVAL_MIN_SUCCESS:-0.75}"
config_marker="${PANTHERA_EVAL_CONFIG_MARKER:-${workspace}/state/panthera-single-eval-config-state/config.ok}"
train_marker="${PANTHERA_EVAL_TRAIN_MARKER:-${workspace}/state/panthera-single-openvla-sft-state/train.ok}"
train_state="${PANTHERA_EVAL_TRAIN_STATE:-${workspace}/state/panthera-single-openvla-sft-state}"
eval_gpu_spec="${PANTHERA_EVAL_GPUS:-0,1,2,3}"
eval_gpu_spec="${eval_gpu_spec//,/ }"
read -r -a eval_gpus <<<"$eval_gpu_spec"
eval_trajectories="${PANTHERA_EVAL_TRAJECTORIES:-16}"
# One env per GPU leaves both halves of the loop idle: RoboTwin physics is CPU
# work that blocks inference, and a batch of one per rollout worker is latency
# bound on a 7B model.  RoboTwin's VectorEnv steps its sub-environments on a
# thread pool and only takes its global lock in setup/reset, so several envs per
# GPU overlap physics with each other and widen the inference batch.  Default is
# 1 so existing runs keep their exact shape.
envs_per_gpu="${PANTHERA_EVAL_ENVS_PER_GPU:-1}"
# Pin the scene each seed rebuilds.  Collection overrode the seed's posture and
# lying angle per shard and recorded only the outcome, so a dataset seed rebuilds
# a different scene about half the time -- a different posture, not just a
# different angle.  Point this at a registry (built by
# envs/panthera_scene_registry.py from a dataset's scene_info.json) to evaluate
# on the recorded scenes; leave it empty to sample fresh scenes as before.
scene_registry="${PANTHERA_EVAL_SCENE_REGISTRY:-}"
max_episode_steps="${PANTHERA_EVAL_MAX_EPISODE_STEPS:-800}"
required_initial_gripper="${PANTHERA_EVAL_REQUIRED_INITIAL_GRIPPER_OPENING:-}"
action_chunk="${PANTHERA_ACTION_CHUNK:-5}"
execution_horizon="${PANTHERA_EVAL_EXECUTION_HORIZON:-${action_chunk}}"
terminal_insertion_assist_m="${PANTHERA_TERMINAL_INSERTION_ASSIST_M:-0}"
terminal_target_assist="${PANTHERA_TERMINAL_TARGET_ASSIST:-0}"
terminal_target_assist_trigger="${PANTHERA_TERMINAL_TARGET_ASSIST_TRIGGER:-height}"
robot_platform="${PANTHERA_ROBOT_PLATFORM:-BRIDGE}"

model_root="${PANTHERA_POLICY_MODEL:-}"
if [[ -z "$model_root" && -s "${train_state}/final-model.txt" ]]; then
  model_root=$(cat "${train_state}/final-model.txt")
fi

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
if [[ -z "$model_root" ]]; then
  echo "错误：没有正式训练模型；请设置 PANTHERA_POLICY_MODEL 或先完成正式 SFT。" >&2
  exit 1
fi
if (( ${#eval_gpus[@]} == 0 )); then
  echo "错误：PANTHERA_EVAL_GPUS 不能为空。" >&2
  exit 1
fi
if [[ ! "$eval_trajectories" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：PANTHERA_EVAL_TRAJECTORIES 必须是正整数。" >&2
  exit 1
fi
if [[ ! "$max_episode_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：PANTHERA_EVAL_MAX_EPISODE_STEPS 必须是正整数。" >&2
  exit 1
fi
# The budget counts 50 Hz actions: RLinf adds chunk_actions.shape[1] per policy
# call and truncates at max_episode_steps.  A budget below what the expert's own
# trajectories need makes part of the eval unwinnable, and the run still prints a
# success rate as if it were measuring the policy.  Floors are the measured
# expert lengths for each task's own dataset.
case "$task_name" in
  place_randomized_cylinder_in_socket)
    # fixedcam 1280: upright max 1283, lying median 2211 / p95 2920 / max 5101.
    # 1600 leaves only 4.8% of lying episodes completable, capping an 18-seed
    # half-lying eval at 52.4% however good the policy is.
    minimum_episode_steps=3200
    ;;
  *)
    minimum_episode_steps=0
    ;;
esac
# A throughput or utilisation probe wants a deliberately short run, which the
# floor would otherwise block.  Allow it explicitly, and mark the result so its
# success rate is never mistaken for a measurement of the policy.
allow_short_budget="${PANTHERA_EVAL_ALLOW_SHORT_BUDGET:-0}"
budget_is_scoring=1
if (( max_episode_steps < minimum_episode_steps )); then
  if [[ "$allow_short_budget" == "1" ]]; then
    budget_is_scoring=0
    echo "警告：动作预算 ${max_episode_steps} 低于 ${task_name} 所需的 ${minimum_episode_steps}；" >&2
    echo "      本次结果仅用于吞吐探测，成功率不可解读（scoring=false）。" >&2
  else
    echo "错误：${task_name} 的动作预算至少需要 ${minimum_episode_steps}，当前为 ${max_episode_steps}；" >&2
    echo "      低于该值时专家轨迹自身都无法跑完，成功率不可解读。" >&2
    echo "      仅做吞吐探测时设 PANTHERA_EVAL_ALLOW_SHORT_BUDGET=1。" >&2
    exit 1
  fi
fi
# RLinf's budget is only one of the two limits.  In eval_mode ``_base_task``
# discards the step_lim it was handed and re-reads _eval_step_limit.yml, falling
# back to 1000 for a task the file does not list -- silently, with only a printed
# line.  Whichever limit is lower truncates, so an unlisted task caps the eval at
# 1000 actions no matter what the config says.
if (( minimum_episode_steps > 0 && budget_is_scoring == 1 )); then
  step_limit_file="${robotwin_root}/env_cfg/task_config/_eval_step_limit.yml"
  task_step_lim=$(python3 - "$step_limit_file" "$task_name" <<'PY'
import sys
import yaml

path, task = sys.argv[1], sys.argv[2]
try:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
except OSError:
    print(-1)
else:
    print(int(data.get(task, -1)))
PY
)
  if (( task_step_lim < minimum_episode_steps )); then
    echo "错误：${step_limit_file} 中 ${task_name} 的 step_lim 为 ${task_step_lim}，" >&2
    echo "      低于所需的 ${minimum_episode_steps}（缺项时 RoboTwin 会静默回落到 1000）。" >&2
    exit 1
  fi
fi
if [[ ! "$action_chunk" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：PANTHERA_ACTION_CHUNK 必须是正整数。" >&2
  exit 1
fi
if [[ ! "$envs_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：PANTHERA_EVAL_ENVS_PER_GPU 必须是正整数。" >&2
  exit 1
fi
scene_registry_overrides=()
if [[ -n "$scene_registry" ]]; then
  if [[ ! -s "$scene_registry" ]]; then
    echo "错误：场景登记表不存在或为空：${scene_registry}" >&2
    exit 1
  fi
  scene_registry_overrides=(
    "env.eval.task_config.task_randomization.scene_registry=${scene_registry}"
    "env.eval.task_config.task_randomization.scene_registry_required=true"
  )
fi
if [[ ! "$execution_horizon" =~ ^[1-9][0-9]*$ ]] \
  || (( execution_horizon > action_chunk )); then
  echo "错误：PANTHERA_EVAL_EXECUTION_HORIZON 必须在 1..${action_chunk} 范围内。" >&2
  exit 1
fi
declare -A seen_eval_gpus=()
for gpu in "${eval_gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "错误：PANTHERA_EVAL_GPUS 只能包含非负整数 GPU 编号。" >&2
    exit 1
  fi
  if [[ -n "${seen_eval_gpus[$gpu]:-}" ]]; then
    echo "错误：PANTHERA_EVAL_GPUS 含重复 GPU：${gpu}" >&2
    exit 1
  fi
  seen_eval_gpus[$gpu]=1
done
eval_gpu_count=${#eval_gpus[@]}
eval_env_count=$((eval_gpu_count * envs_per_gpu))
if (( eval_env_count > eval_trajectories || eval_trajectories % eval_env_count != 0 )); then
  echo "错误：环境总数 ${eval_env_count}（${eval_gpu_count} 卡 × 每卡 ${envs_per_gpu}）必须整除 ${eval_trajectories} 条闭环轨迹。" >&2
  exit 1
fi
eval_gpu_csv=$(IFS=,; printf '%s' "${eval_gpus[*]}")
eval_logical_last=$((eval_gpu_count - 1))
eval_logical_range="0-${eval_logical_last}"
eval_rollout_epoch=$((eval_trajectories / eval_env_count))
python3 - "$minimum_success" <<'PY'
import sys

value = float(sys.argv[1])
if not 0.0 <= value <= 1.0:
    raise SystemExit("PANTHERA_EVAL_MIN_SUCCESS must be between 0 and 1")
PY
python3 - "$terminal_insertion_assist_m" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or not 0.0 <= value <= 0.080:
    raise SystemExit("PANTHERA_TERMINAL_INSERTION_ASSIST_M must be in [0, 0.080]")
PY
if [[ "$terminal_target_assist" != 0 && "$terminal_target_assist" != 1 ]]; then
  echo "错误：PANTHERA_TERMINAL_TARGET_ASSIST 必须是 0 或 1。" >&2
  exit 1
fi
if [[ "$terminal_target_assist_trigger" != height \
  && "$terminal_target_assist_trigger" != release ]]; then
  echo "错误：PANTHERA_TERMINAL_TARGET_ASSIST_TRIGGER 必须是 height 或 release。" >&2
  exit 1
fi
for command_name in flock git patch timeout nvidia-smi; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
for required in \
  "$activation_script" "$env_source" "$eval_source" "$seed_source" "$vector_action_patch" \
  "$openvla_l1_eval_patch" \
  "$cuda_visible_subset_patch" \
  "$eval_execution_horizon_patch" \
  "$eval_rollout_execution_horizon_patch" \
  "$openvla_constants_patch" \
  "$task_source" "$instruction_source" \
  "${model_root}/config.json" "${model_root}/dataset_statistics.json"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少闭环评测前置文件：${required}" >&2
    exit 1
  fi
done
python3 - "$seed_source" "$task_name" "$eval_trajectories" <<'PY'
import json
from pathlib import Path
import sys

seed_path = Path(sys.argv[1])
task_name = sys.argv[2]
expected_count = int(sys.argv[3])
try:
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    seeds = payload[task_name]["success_seeds"]
except (json.JSONDecodeError, KeyError, TypeError) as exc:
    raise SystemExit(
        f"invalid seed file {seed_path}: expected "
        f"{{{task_name!r}: {{'success_seeds': [...]}}}} ({exc})"
    ) from exc
if not isinstance(seeds, list) or len(seeds) != expected_count:
    raise SystemExit(
        f"invalid seed file {seed_path}: expected {expected_count} seeds, "
        f"got {len(seeds) if isinstance(seeds, list) else type(seeds).__name__}"
    )
if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
    raise SystemExit(f"invalid seed file {seed_path}: every seed must be an integer")
if len(set(seeds)) != len(seeds):
    raise SystemExit(f"invalid seed file {seed_path}: duplicate seeds are not allowed")
PY
constants_source="${openvla_site_packages}/prismatic/vla/constants.py"
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
if ! find "$model_root" -maxdepth 2 -type f -name 'action_head--*_checkpoint.pt' -size +0c | grep -q .; then
  echo "错误：训练模型缺少 action head checkpoint。" >&2
  exit 1
fi

if git -C "$rlinf_root" apply --reverse --check "$openvla_l1_eval_patch" 2>/dev/null; then
  echo "RLinf OpenVLA-OFT 连续 L1 动作头评测补丁已存在。"
elif git -C "$rlinf_root" apply --check "$openvla_l1_eval_patch"; then
  git -C "$rlinf_root" apply "$openvla_l1_eval_patch"
  echo "已应用 RLinf OpenVLA-OFT 连续 L1 动作头评测补丁。"
else
  echo "错误：RLinf OpenVLA-OFT 连续 L1 动作头评测补丁无法安全应用。" >&2
  exit 1
fi

if git -C "$rlinf_root" apply --reverse --check "$cuda_visible_subset_patch" 2>/dev/null; then
  echo "RLinf CUDA 子集物理编号补丁已存在。"
elif git -C "$rlinf_root" apply --check "$cuda_visible_subset_patch"; then
  git -C "$rlinf_root" apply "$cuda_visible_subset_patch"
  echo "已应用 RLinf CUDA 子集物理编号补丁。"
else
  echo "错误：RLinf CUDA 子集物理编号补丁无法安全应用。" >&2
  exit 1
fi
if git -C "$rlinf_root" apply --reverse --check "$eval_execution_horizon_patch" 2>/dev/null; then
  echo "RLinf 评测执行视野补丁已存在。"
elif git -C "$rlinf_root" apply --check "$eval_execution_horizon_patch"; then
  git -C "$rlinf_root" apply "$eval_execution_horizon_patch"
  echo "已应用 RLinf 评测执行视野补丁。"
else
  echo "错误：RLinf 评测执行视野补丁无法安全应用。" >&2
  exit 1
fi
hf_worker="${rlinf_root}/rlinf/workers/rollout/hf/huggingface_worker.py"
sglang_worker="${rlinf_root}/rlinf/workers/rollout/sglang/sglang_embodied_worker.py"
if grep -Fq 'eval_execution_horizon = int(' "$hf_worker" \
  && grep -Fq 'eval_execution_horizon = int(' "$sglang_worker"; then
  echo "RLinf rollout worker 执行视野补丁已存在。"
elif git -C "$rlinf_root" apply --check "$eval_rollout_execution_horizon_patch"; then
  git -C "$rlinf_root" apply "$eval_rollout_execution_horizon_patch"
  echo "已应用 RLinf rollout worker 执行视野补丁。"
else
  echo "错误：RLinf rollout worker 执行视野补丁无法安全应用。" >&2
  exit 1
fi
if ! grep -Fq 'eval_execution_horizon = int(' "$hf_worker" \
  || ! grep -Fq 'eval_execution_horizon = int(' "$sglang_worker"; then
  echo "错误：rollout worker 执行视野补丁源码标记不完整。" >&2
  exit 1
fi
if ! find "$model_root" -maxdepth 1 -type f -name 'proprio_projector--*_checkpoint.pt' -size +0c | grep -q .; then
  echo "错误：训练模型缺少 proprio projector checkpoint。" >&2
  exit 1
fi
for marker in "$config_marker" "$train_marker"; do
  if [[ ! -f "$marker" ]]; then
    echo "错误：缺少前置验收标记：${marker}" >&2
    exit 1
  fi
done

mkdir -p "$state_root" "$log_root"
exec 9>"${state_root}/eval.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera 策略评测正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/eval.ok" ]]; then
  python3 -m json.tool "${state_root}/eval-summary.json"
  echo "Panthera 策略闭环评测已通过，无需重复运行。"
  exit 0
fi
for gpu in "${eval_gpus[@]}"; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "错误：闭环评测选中的 GPU ${gpu} 当前有其他计算进程。" >&2
    exit 1
  fi
done

# Assembled from the pinned upstream instead of patched into it.
python3 "${workspace}/pipelines/assemble_runtime.py" \
  --upstream robotwin \
  --source "$upstream_root" \
  --runtime "$robotwin_root"
grep -q 'action_dim' "$robotwin_root/robotwin/envs/vector_env.py" \
  || { echo "错误：装配出的运行树缺少所需改动。" >&2; exit 1; }

install -m 0644 "$env_source" "$env_target"
install -m 0644 "$eval_source" "$eval_target"
install -m 0644 "$task_source" "$task_target"
install -m 0644 "$instruction_source" "$instruction_target"

# shellcheck disable=SC1090
source "$activation_script"
export ROBOT_PLATFORM="$robot_platform"
export PANTHERA_ACTION_CHUNK="$action_chunk"
export PANTHERA_EVAL_EXECUTION_HORIZON="$execution_horizon"
export PANTHERA_TERMINAL_INSERTION_ASSIST_M="$terminal_insertion_assist_m"
export PANTHERA_TERMINAL_TARGET_ASSIST="$terminal_target_assist"
export PANTHERA_TERMINAL_TARGET_ASSIST_TRIGGER="$terminal_target_assist_trigger"
export CUDA_VISIBLE_DEVICES="$eval_gpu_csv"
export PANTHERA_CYLINDER_EVAL_SEEDS="$seed_source"
export PANTHERA_PHONE_EVAL_SEEDS="$seed_source"

resolved="${state_root}/resolved-config.yaml"
(
  cd "$rlinf_root"
  bash evaluations/run_eval.sh \
    robotwin \
    "$eval_config_name" \
    '~cluster.component_placement' \
    "+cluster.component_placement={env:${eval_logical_range},rollout:${eval_logical_range}}" \
    "env.eval.rollout_epoch=${eval_rollout_epoch}" \
    "env.eval.total_num_envs=${eval_env_count}" \
    ${scene_registry_overrides[@]+"${scene_registry_overrides[@]}"} \
    "env.eval.max_steps_per_rollout_epoch=${max_episode_steps}" \
    "env.eval.max_episode_steps=${max_episode_steps}" \
    "env.eval.task_config.step_lim=${max_episode_steps}" \
    "rollout.model.model_path=${model_root}" \
    "env.eval.assets_path=${robotwin_root}" \
    "env.eval.seeds_path=${seed_source}" \
    "rollout.model.num_action_chunks=${action_chunk}" \
    --cfg job \
    --resolve
) >"$resolved"
grep -q "^    env: ${eval_logical_range}$" "$resolved"
grep -q "^    rollout: ${eval_logical_range}$" "$resolved"
grep -q "^    rollout_epoch: ${eval_rollout_epoch}$" "$resolved"
grep -q "^    total_num_envs: ${eval_env_count}$" "$resolved"
grep -q "^      task_name: ${task_name}$" "$resolved"
grep -q "^    max_episode_steps: ${max_episode_steps}$" "$resolved"
grep -q "^    max_steps_per_rollout_epoch: ${max_episode_steps}$" "$resolved"
grep -q "^      step_lim: ${max_episode_steps}$" "$resolved"
grep -q '^      action_dim: 7$' "$resolved"
grep -q '^    action_dim: 7$' "$resolved"
grep -q '^    proprio_dim: 7$' "$resolved"
grep -q "^    num_action_chunks: ${action_chunk}$" "$resolved"
grep -q '^    use_l1_regression: true$' "$resolved"
grep -q "^    unnorm_key: ${unnorm_key}$" "$resolved"
grep -q "^    model_path: ${model_root}$" "$resolved"
if [[ -n "$required_initial_gripper" ]]; then
  grep -q "^      initial_gripper_opening: ${required_initial_gripper}$" "$resolved"
fi

if pgrep -u "$(id -u)" -x raylet >/dev/null 2>&1; then
  echo "错误：当前用户已有 Ray 集群，拒绝与其他任务共享或停止它。" >&2
  exit 1
fi

stamp=$(date +%Y%m%d-%H%M%S)
run_log="${log_root}/eval-${stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"
set +e
(
  cd "$rlinf_root"
  timeout --signal=INT --kill-after=90s \
    "${PANTHERA_POLICY_EVAL_TIMEOUT:-4h}" \
    bash evaluations/run_eval.sh \
      robotwin \
      "$eval_config_name" \
      '~cluster.component_placement' \
      "+cluster.component_placement={env:${eval_logical_range},rollout:${eval_logical_range}}" \
      "env.eval.rollout_epoch=${eval_rollout_epoch}" \
      "env.eval.total_num_envs=${eval_env_count}" \
      ${scene_registry_overrides[@]+"${scene_registry_overrides[@]}"} \
      "env.eval.max_steps_per_rollout_epoch=${max_episode_steps}" \
      "env.eval.max_episode_steps=${max_episode_steps}" \
      "env.eval.task_config.step_lim=${max_episode_steps}" \
      "rollout.model.model_path=${model_root}" \
      "env.eval.assets_path=${robotwin_root}" \
      "env.eval.seeds_path=${seed_source}" \
      "rollout.model.num_action_chunks=${action_chunk}"
) 2>&1 | tee "$run_log"
eval_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$eval_status" >"${state_root}/exit-code.txt"
if (( eval_status != 0 )); then
  echo "错误：Panthera 闭环评测失败，退出码 ${eval_status}。" >&2
  exit "$eval_status"
fi

fatal_pattern='Traceback \(most recent call last\)|ModuleNotFoundError:|ImportError:|RayTaskError\(|WorkerCrashedError:|CUDA out of memory'
if grep -Eq "$fatal_pattern" "$run_log"; then
  grep -nE "$fatal_pattern" "$run_log" >"${state_root}/fatal-errors.txt" || true
  echo "错误：闭环评测日志含致命异常。" >&2
  exit 1
fi

evaluation_root=$(sed -n 's|.*runner.logger.log_path=\([^ ]*\).*|\1|p' "$run_log" | head -n 1)
if [[ -z "$evaluation_root" || ! -d "${evaluation_root}/video/eval" ]]; then
  echo "错误：未找到闭环评测视频目录。" >&2
  exit 1
fi
video_count=$(find "${evaluation_root}/video/eval" -type f -name '*.mp4' -size +0c | wc -l)

set +e
python3 - "$run_log" "$evaluation_root" "$video_count" "$minimum_success" \
  "${state_root}/eval-summary.json" "$task_name" "$scene_profile" \
  "$unnorm_key" "$eval_gpu_count" "$eval_gpu_csv" "$eval_trajectories" \
  "$required_initial_gripper" "$action_chunk" "$execution_horizon" \
  "$robot_platform" "$terminal_insertion_assist_m" \
  "$terminal_target_assist" "$max_episode_steps" \
  "$terminal_target_assist_trigger" "$envs_per_gpu" "$eval_env_count" \
  "$eval_rollout_epoch" "$budget_is_scoring" "$minimum_episode_steps" \
  "$scene_registry" <<'PY'
import json
from pathlib import Path
import re
import sys

log_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
video_count = int(sys.argv[3])
minimum_success = float(sys.argv[4])
summary_path = Path(sys.argv[5])
text = log_path.read_text(encoding="utf-8", errors="replace")
matches = re.findall(
    r"'eval/success_once':\s*(?:array\()?([0-9.eE+-]+).*?"
    r"'eval/num_trajectories':\s*(\d+)",
    text,
)
if not matches:
    raise SystemExit("missing eval success/trajectory metrics")
success, trajectories = float(matches[-1][0]), int(matches[-1][1])
expected_trajectories = int(sys.argv[11])
if trajectories != expected_trajectories:
    raise SystemExit(
        f"expected {expected_trajectories} trajectories, got {trajectories}"
    )
if video_count < trajectories:
    raise SystemExit(f"expected at least {trajectories} videos, got {video_count}")
scoring = bool(int(sys.argv[23]))
# A probe run is not a verdict on the policy, so it neither passes nor fails the
# gate; recording it as "passed" would put an unearned green mark in the state.
passed = success >= minimum_success if scoring else None
summary = {
    "status": (
        ("passed" if passed else "below_threshold") if scoring
        else "throughput_probe_not_scored"
    ),
    "task": sys.argv[6],
    "scene_profile": sys.argv[7] or None,
    "unnorm_key": sys.argv[8],
    "trajectories": trajectories,
    "success_once": success,
    "minimum_success": minimum_success,
    "video_count": video_count,
    "gpu_count": int(sys.argv[9]),
    "physical_gpus": [int(value) for value in sys.argv[10].split(",")],
    # Throughput shape, so a run's wall clock can be read against its own
    # parallelism rather than guessed from the GPU list.
    "envs_per_gpu": int(sys.argv[20]),
    "total_num_envs": int(sys.argv[21]),
    "rollout_epoch": int(sys.argv[22]),
    # False when the action budget was deliberately below what the task needs,
    # so the success rate in this file measures throughput, not the policy.
    "scoring": scoring,
    "minimum_episode_steps": int(sys.argv[24]),
    # Which scenes were evaluated: a registry path means the recorded scenes,
    # null means freshly sampled ones.
    "scene_registry": sys.argv[25] or None,
    "initial_gripper_opening": (
        float(sys.argv[12]) if sys.argv[12] else None
    ),
    "action_chunk": int(sys.argv[13]),
    "execution_horizon": int(sys.argv[14]),
    "robot_platform": sys.argv[15],
    "terminal_insertion_assist_m": float(sys.argv[16]),
    "terminal_target_assist": bool(int(sys.argv[17])),
    "max_episode_steps": int(sys.argv[18]),
    "terminal_target_assist_trigger": sys.argv[19],
    "output_root": str(output_root),
    "log": str(log_path),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
raise SystemExit(0 if (passed or not scoring) else 2)
PY
metric_status=$?
set -e

# Ray actors can remain visible briefly after the evaluator exits. This bounded
# polling happens after the trajectory window, so it cannot create gaps in the
# recorded control stream.
eval_processes_alive() {
  pgrep -u "$(id -u)" -x raylet >/dev/null 2>&1 ||
    ps -u "$(id -u)" -o comm=,args= | awk \
      '$1 ~ /^ray::/ || ($1 ~ /^python/ && /eval_embodied_agent[.]py/) { found=1 } END { exit !found }'
}
cleanup_deadline=$((SECONDS + 90))
while eval_processes_alive && (( SECONDS < cleanup_deadline )); do
  sleep 1
done
if eval_processes_alive; then
  ps -u "$(id -u)" -o pid=,comm=,args= | awk \
    '$2 ~ /^ray::/ || $2 == "raylet" || ($2 ~ /^python/ && /eval_embodied_agent[.]py/)'
  echo "错误：闭环评测后仍有 RoboTwin/Ray 进程。" >&2
  exit 1
fi
for gpu in "${eval_gpus[@]}"; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "错误：闭环评测后 GPU ${gpu} 仍有计算进程。" >&2
    exit 1
  fi
done
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/eval.completed"
if (( metric_status != 0 )); then
  echo "错误：策略评测已完成，但成功率未达到 ${minimum_success}。" >&2
  exit "$metric_status"
fi
touch "${state_root}/eval.ok"
echo "Panthera VLA 闭环评测通过：${evaluation_root}"
