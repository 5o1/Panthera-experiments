#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
source_root="${workspace}/third_party/Panthera_HT_ROS2"
source_bundle="${workspace}/third_party/Panthera_HT_ROS2-b08633.bundle"
overlay_root="${workspace}/overlays/robotwin"
sim_package="${workspace}/packages/panthera_sim"
embodiment_root="${overlay_root}/assets/embodiments/panthera"
upstream_root="${workspace}/externals/RoboTwin"
robotwin_root="${workspace}/runtime/robotwin"
state_root="${workspace}/state/panthera-embodiment-state"
result_root="${workspace}/results/panthera-embodiment-smoke"
activation_script="${workspace}/tools/activate_lab_vla.sh"
source_url="https://github.com/HighTorque-Robotics/Panthera-HT_ROS2.git"
source_commit="b08633d6c5bce89baad1821fd598243a84bc3a84"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock git; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  fi
done
for required in \
  "${sim_package}/build_panthera_embodiment.py" \
  "${overlay_root}/envs/test_panthera_embodiment.py" \
  "$activation_script"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少文件：${required}" >&2
    exit 1
  fi
done
if [[ ! -f "${workspace}/state/bootstrap-state/verified.ok" ]]; then
  echo "错误：RLinf/RoboTwin Conda 环境尚未通过验收。" >&2
  exit 1
fi

mkdir -p "$state_root" "$result_root" "${workspace}/third_party"
exec 9>"${state_root}/bootstrap.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera embodiment 构建正在运行。" >&2
  exit 1
fi

if [[ ! -d "${source_root}/.git" ]]; then
  if [[ -e "$source_root" ]]; then
    echo "错误：源码目录存在但不是 Git checkout：${source_root}" >&2
    exit 1
  fi
  if [[ -s "$source_bundle" ]]; then
    git -C "$robotwin_root" bundle verify "$source_bundle"
    git clone "$source_bundle" "$source_root"
    git -C "$source_root" remote set-url origin "$source_url"
  else
    git clone --filter=blob:none "$source_url" "$source_root"
  fi
fi
if [[ $(git -C "$source_root" remote get-url origin) != "$source_url" ]]; then
  echo "错误：Panthera 官方源码 origin 不匹配。" >&2
  exit 1
fi
if ! git -C "$source_root" cat-file -e "${source_commit}^{commit}"; then
  git -C "$source_root" fetch --depth=1 origin "$source_commit"
fi
git -C "$source_root" checkout --detach "$source_commit"
if [[ $(git -C "$source_root" rev-parse HEAD) != "$source_commit" ]]; then
  echo "错误：Panthera 官方源码提交未固定成功。" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$activation_script"
if ! command -v python >/dev/null 2>&1; then
  echo "错误：Conda 环境激活后仍缺少 python。" >&2
  exit 1
fi
python "${sim_package}/build_panthera_embodiment.py" \
  --source-root "$source_root" \
  --source-commit "$source_commit" \
  --output-root "$embodiment_root"

# Build the runtime only after the generated embodiment exists in the overlay;
# no post-assembly copy is needed and rebuilding stays deterministic.
python3 "${workspace}/pipelines/assemble_runtime.py" \
  --upstream robotwin \
  --source "$upstream_root" \
  --runtime "$robotwin_root"

robotwin_embodiment="${robotwin_root}/assets/embodiments/panthera"

registry="${robotwin_root}/env_cfg/task_config/_embodiment_config.yml"
python - "$registry" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
registry = yaml.safe_load(path.read_text(encoding="utf-8"))
expected = {"file_path": "./assets/embodiments/panthera"}
if registry.get("panthera") != expected:
    registry["panthera"] = expected
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
PY

python "${overlay_root}/envs/test_panthera_embodiment.py" \
  --embodiment-root "$robotwin_embodiment" \
  --robotwin-root "$robotwin_root" \
  --report "${result_root}/load-and-step.json" \
  2>&1 | tee "${state_root}/load-and-step.log"

python - "${result_root}/load-and-step.json" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "passed" or report.get("simulation_steps") != 500:
    raise SystemExit("Panthera embodiment report did not pass")
if report.get("wall_clock_sleep_calls") != 0:
    raise SystemExit("Panthera embodiment smoke unexpectedly used wall-clock sleep")
contract = report.get("robotwin_contract", {})
if contract.get("dual_arm_action_dimension") != 14:
    raise SystemExit("Panthera embodiment action contract is not 14-dimensional")
for side in ("left", "right"):
    plans = contract.get("plans", {}).get(side, {})
    if plans.get("current", {}).get("status") != "Success":
        raise SystemExit(f"Panthera {side} MPLib current-pose plan did not succeed")
    successes = [
        name
        for name, result in plans.items()
        if name != "current"
        and result.get("status") == "Success"
        and result.get("trajectory_steps", 0) > 0
    ]
    if not successes:
        raise SystemExit(f"Panthera {side} has no successful nonzero MPLib plan")
PY

date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/verified.ok"
echo "Panthera embodiment 静态验收通过：${result_root}/load-and-step.json"
