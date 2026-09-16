#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
rlinf_root="${workspace}/RLinf"
overlay_root="${workspace}/panthera-rlinf-overlay"
state_root="${workspace}/.panthera-single-eval-config-state"
activation_script="${workspace}/activate_lab_vla.sh"
env_source="${overlay_root}/config/env/robotwin_place_cylinder_in_groove.yaml"
eval_source="${overlay_root}/evaluations/robotwin_panthera_cylinder_openvlaoft_eval.yaml"
seed_source="${overlay_root}/seeds/panthera_cylinder_eval_seeds.json"
env_target="${rlinf_root}/examples/embodiment/config/env/robotwin_place_cylinder_in_groove.yaml"
eval_target="${rlinf_root}/evaluations/robotwin/robotwin_panthera_cylinder_openvlaoft_eval.yaml"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for required in "$activation_script" "$env_source" "$eval_source" "$seed_source"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少评测配置文件：${required}" >&2
    exit 1
  fi
done
if [[ ! -f "${workspace}/.panthera-embodiment-state/verified.ok" ]]; then
  echo "错误：Panthera embodiment 尚未通过验收。" >&2
  exit 1
fi

mkdir -p "$state_root"
exec 9>"${state_root}/config.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera 评测配置验收正在运行。" >&2
  exit 1
fi

install -m 0644 "$env_source" "$env_target"
install -m 0644 "$eval_source" "$eval_target"
python3 -m json.tool "$seed_source" >"${state_root}/seeds.pretty.json"

# shellcheck disable=SC1090
source "$activation_script"
export ROBOT_PLATFORM=BRIDGE
export PANTHERA_CYLINDER_EVAL_SEEDS="$seed_source"
resolved="${state_root}/resolved-config.yaml"
(
  cd "$rlinf_root"
  bash evaluations/run_eval.sh \
    robotwin \
    robotwin_panthera_cylinder_openvlaoft_eval \
    --cfg job \
    --resolve
) >"$resolved"

grep -q '^    env, rollout: 0-3$' "$resolved"
grep -q '^    total_num_envs: 4$' "$resolved"
grep -q '^    rollout_epoch: 4$' "$resolved"
grep -q '^    use_fixed_reset_state_ids: false$' "$resolved"
grep -q '^      task_name: place_cylinder_in_groove$' "$resolved"
grep -q '^      action_dim: 7$' "$resolved"
grep -q '^      embodiment:' "$resolved"
grep -q '^      - panthera$' "$resolved"
grep -q '^    max_episode_steps: 800$' "$resolved"
grep -q '^    num_action_chunks: 5$' "$resolved"
grep -q '^    action_dim: 7$' "$resolved"
grep -q '^    proprio_dim: 7$' "$resolved"
grep -q '^    unnorm_key: panthera_single_cylinder$' "$resolved"
grep -q "^    seeds_path: ${seed_source}$" "$resolved"

python3 - "$seed_source" "${state_root}/config-summary.json" <<'PY'
import json
from pathlib import Path
import sys

seeds = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))[
    "place_cylinder_in_groove"
]["success_seeds"]
if len(seeds) != 16 or len(set(seeds)) != 16:
    raise SystemExit("expected 16 unique held-out evaluation seeds")
summary = {
    "status": "passed",
    "task": "place_cylinder_in_groove",
    "embodiment": ["panthera"],
    "evaluation_seed_count": len(seeds),
    "evaluation_seed_min": min(seeds),
    "evaluation_seed_max": max(seeds),
    "parallel_environments": 4,
    "rollout_epochs": 4,
    "expected_trajectories": 16,
    "action_dimension": 7,
    "action_chunk": 5,
    "episode_action_limit": 800,
    "unnorm_key": "panthera_single_cylinder",
    "simulation_started": False,
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/config.ok"
echo "Panthera RoboTwin 闭环评测配置验收通过（未启动仿真）。"
