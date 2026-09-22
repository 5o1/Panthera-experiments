#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
rlinf_root="${workspace}/runtime/rlinf"
overlay_root="${workspace}/overlays/rlinf"
# Upstream is read-only under externals/; this runs against a runtime
# assembled from it plus the overlay plus the patches.
upstream_root="${workspace}/externals/RoboTwin"
robotwin_root="${workspace}/runtime/robotwin"
python3 "${workspace}/pipelines/assemble_runtime.py" \
  --upstream robotwin --source "$upstream_root" --runtime "$robotwin_root" >/dev/null
robotwin_overlay="${workspace}/overlays/robotwin"
state_root="${workspace}/state/panthera-v2-schema10-eval-config-state"
activation_script="${workspace}/tools/activate_lab_vla.sh"
env_source="${overlay_root}/config/env/robotwin_place_randomized_cylinder_in_socket.yaml"
eval_source="${overlay_root}/evaluations/robotwin_panthera_v2_openvlaoft_eval.yaml"
seed_source="${overlay_root}/seeds/panthera_v2_schema10_dev_seeds.json"
env_target="${rlinf_root}/examples/embodiment/config/env/robotwin_place_randomized_cylinder_in_socket.yaml"
eval_target="${rlinf_root}/evaluations/robotwin/robotwin_panthera_v2_openvlaoft_eval.yaml"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：禁止使用 root 验收评测配置。" >&2
  exit 1
fi
for required in "$activation_script" "$env_source" "$eval_source" "$seed_source" \
  "${robotwin_overlay}/envs/place_randomized_cylinder_in_socket.py" \
  "${robotwin_overlay}/envs/panthera_v2_sampling.py" \
  "${robotwin_overlay}/envs/place_vertical_cylinder_in_groove.py" \
  "${robotwin_overlay}/description/task_instruction/place_randomized_cylinder_in_socket.json"; do
  [[ -s "$required" ]] || { echo "错误：缺少配置前置文件 ${required}。" >&2; exit 1; }
done
[[ -f "${workspace}/state/panthera-v2-single-grasp-formal-state/dataset.ok" ]] || {
  echo "错误：正式数据集尚未通过。" >&2
  exit 1
}

mkdir -p "$state_root"
exec 9>"${state_root}/config.lock"
flock -n 9 || { echo "错误：评测配置验收已在运行。" >&2; exit 1; }
if [[ -f "${state_root}/config.ok" && -s "${state_root}/config-summary.json" ]]; then
  python3 -m json.tool "${state_root}/config-summary.json"
  exit 0
fi

install -m 0644 "$env_source" "$env_target"
install -m 0644 "$eval_source" "$eval_target"
install -m 0644 \
  "${robotwin_overlay}/envs/place_randomized_cylinder_in_socket.py" \
  "${robotwin_overlay}/envs/panthera_v2_sampling.py" \
  "${robotwin_overlay}/envs/place_vertical_cylinder_in_groove.py" \
  "${robotwin_root}/envs/"
install -m 0644 \
  "${robotwin_overlay}/description/task_instruction/place_randomized_cylinder_in_socket.json" \
  "${robotwin_root}/description/task_instruction/"

# shellcheck disable=SC1090
source "$activation_script"
export ROBOT_PLATFORM=PANTHERA
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_PHONE_EVAL_SEEDS="$seed_source"
resolved="${state_root}/resolved-config.yaml"
(
  cd "$rlinf_root"
  bash evaluations/run_eval.sh robotwin robotwin_panthera_v2_openvlaoft_eval \
    --cfg job --resolve
) >"$resolved"

grep -q '^    env, rollout: 0-2$' "$resolved"
grep -q '^    total_num_envs: 3$' "$resolved"
grep -q '^    rollout_epoch: 6$' "$resolved"
grep -q '^      task_name: place_randomized_cylinder_in_socket$' "$resolved"
grep -q '^      action_dim: 7$' "$resolved"
grep -q '^      - panthera_phone_symmetric$' "$resolved"
grep -q '^    max_episode_steps: 1600$' "$resolved"
grep -q '^    num_action_chunks: 25$' "$resolved"
grep -q '^    unnorm_key: panthera_phone_cylinder_socket_v2$' "$resolved"
grep -q "^    seeds_path: ${seed_source}$" "$resolved"

python3 - "$seed_source" "${state_root}/config-summary.json" <<'PY'
import json
from pathlib import Path
import sys

seeds = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))[
    "place_randomized_cylinder_in_socket"
]["success_seeds"]
if len(seeds) != 18 or len(set(seeds)) != 18:
    raise SystemExit("expected 18 unique development seeds")
summary = {
    "status": "passed",
    "task": "place_randomized_cylinder_in_socket",
    "task_schema_version": 10,
    "scene_profile": "panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2",
    "embodiment": ["panthera_phone_symmetric"],
    "camera_type": "PhoneSRT_Center4_3_Wide",
    "evaluation_seed_count": 18,
    "parallel_environments": 3,
    "rollout_epochs": 6,
    "action_dimension": 7,
    "action_chunk": 25,
    "execution_horizon": 20,
    "episode_action_limit": 1600,
    "unnorm_key": "panthera_phone_cylinder_socket_v2",
    "simulation_started": False,
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/config.ok"
echo "schema 10 三卡闭环评测配置通过（未启动仿真）。"
