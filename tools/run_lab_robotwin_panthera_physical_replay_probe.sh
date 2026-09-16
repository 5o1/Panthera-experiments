#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
robotwin_root="${workspace}/RoboTwin"
overlay_root="${workspace}/panthera-robotwin-overlay"
state_root="${workspace}/.panthera-physical-replay-probe-state"
result_root="${workspace}/results/panthera-physical-replay-probe"
task_name="place_cylinder_in_groove"
task_config="panthera_single_cylinder_physical_replay_probe"
dataset_root="${workspace}/data/${task_name}/${task_config}"
activation_script="${workspace}/activate_lab_vla.sh"
expected_robotwin_commit="0008ae6800df9f75fc8de7098bacb01735fd8fd2"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock git sha256sum timeout; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
for required in \
  "$activation_script" \
  "${overlay_root}/envs/${task_name}.py" \
  "${overlay_root}/replay_panthera_dataset.py" \
  "${overlay_root}/run_oracle_smoke.py" \
  "${overlay_root}/description/task_instruction/${task_name}.json" \
  "${overlay_root}/task_config/${task_config}.yml"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少前置文件：${required}" >&2
    exit 1
  fi
done
if [[ ! -f "${workspace}/.panthera-embodiment-state/verified.ok" ]]; then
  echo "错误：Panthera embodiment 尚未通过验收。" >&2
  exit 1
fi
if [[ $(git -C "$robotwin_root" rev-parse HEAD) != "$expected_robotwin_commit" ]]; then
  echo "错误：RoboTwin checkout 不是已固定提交。" >&2
  exit 1
fi

mkdir -p "$state_root" "$result_root"
exec 9>"${state_root}/probe.lock"
if ! flock -n 9; then
  echo "错误：另一个物理回放原型正在运行。" >&2
  exit 1
fi

manifest_path="${state_root}/input.sha256"
current_manifest=$(sha256sum \
  "${overlay_root}/envs/${task_name}.py" \
  "${overlay_root}/replay_panthera_dataset.py" \
  "${overlay_root}/run_oracle_smoke.py" \
  "${overlay_root}/task_config/${task_config}.yml")
if [[ -f "${state_root}/probe.ok" && -f "$manifest_path" ]] && \
   [[ $(<"$manifest_path") == "$current_manifest" ]]; then
  python3 -m json.tool "${state_root}/probe-summary.json"
  echo "无辅助约束物理回放原型已经通过，输入未变化。"
  exit 0
fi

run_stamp=$(date +%Y%m%d-%H%M%S)
if [[ -e "$dataset_root" ]]; then
  archived_dataset="${dataset_root}.superseded-${run_stamp}"
  mv "$dataset_root" "$archived_dataset"
  echo "旧原型数据已归档：${archived_dataset}"
fi
rm -f "${state_root}/probe.ok"

install -m 0644 \
  "${overlay_root}/envs/${task_name}.py" \
  "${robotwin_root}/envs/${task_name}.py"
install -m 0644 \
  "${overlay_root}/description/task_instruction/${task_name}.json" \
  "${robotwin_root}/description/task_instruction/${task_name}.json"
install -m 0644 \
  "${overlay_root}/task_config/${task_config}.yml" \
  "${robotwin_root}/task_config/${task_config}.yml"

# shellcheck disable=SC1090
source "$activation_script"
python -m py_compile \
  "${robotwin_root}/envs/${task_name}.py" \
  "${overlay_root}/replay_panthera_dataset.py"

run_log="${state_root}/probe-${run_stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"

# The upstream collector retries failed planning with unbounded seed numbers.
# Prove one deterministic plan first so a bad pose fails once instead of
# spinning until the outer timeout.
PYTHONUNBUFFERED=1 timeout --signal=INT --kill-after=30s \
  "${PANTHERA_PHYSICAL_REPLAY_PLAN_TIMEOUT:-12m}" \
  python "${overlay_root}/run_oracle_smoke.py" \
    --robotwin-root "$robotwin_root" \
    --output-root "${result_root}/planning-check" \
    --task-config "${task_config}.yml" \
    --seeds 0 2>&1 | tee "$run_log"

set +e
(
  cd "$robotwin_root"
  PYTHONUNBUFFERED=1 timeout --signal=INT --kill-after=60s \
    "${PANTHERA_PHYSICAL_REPLAY_PROBE_TIMEOUT:-35m}" \
    python script/collect_data.py "$task_name" "$task_config"
) 2>&1 | tee "$run_log"
collect_status=${PIPESTATUS[0]}
set -e
if (( collect_status != 0 )); then
  echo "错误：单集原型采集失败，退出码 ${collect_status}。" >&2
  exit "$collect_status"
fi
if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|AssertionError:|Collect Error' "$run_log"; then
  echo "错误：单集原型采集日志含致命异常。" >&2
  exit 1
fi

python - "$dataset_root" <<'PY'
import json
from pathlib import Path
import sys

import h5py
import numpy as np

root = Path(sys.argv[1])
video_path = root / "video/episode0.mp4"
if not video_path.is_file() or video_path.stat().st_size <= 0:
    raise SystemExit("probe did not produce a nonempty episode0.mp4")
metadata = json.loads((root / "scene_info.json").read_text(encoding="utf-8"))
episode = metadata["episode_0"]["panthera_episode"]
if episode.get("schema_version") != 3:
    raise SystemExit("probe did not use single-arm task schema v3")
if episode.get("robot_count") != 1 or episode.get("action_dimension") != 7:
    raise SystemExit("probe did not use the one-Panthera 7-D contract")
if not np.allclose(episode.get("grasp_approach_axis_world"), [0, 0, -1], atol=1e-7):
    raise SystemExit("probe grasp is not top-down")
if not np.allclose(episode.get("finger_closing_axis_world"), [0, 1, 0], atol=1e-7):
    raise SystemExit("Panthera fingers do not close across the cylinder diameter")
with h5py.File(root / "data/episode0.hdf5", "r") as data:
    actions = np.asarray(data["joint_action/vector"], dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.all(np.isfinite(actions)):
        raise SystemExit("probe action contract is invalid")
PY

python "${overlay_root}/replay_panthera_dataset.py" \
  --robotwin-root "$robotwin_root" \
  --dataset-root "$dataset_root" \
  --task-config "${task_config}.yml" \
  --summary "${state_root}/replay-summary.json" \
  --diagnostic-dir "${result_root}/diagnostic" \
  --episode 0

python - "${state_root}/replay-summary.json" "${state_root}/probe-summary.json" <<'PY'
import json
from pathlib import Path
import sys

replay = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if replay.get("status") != "passed" or replay.get("passed") != 1:
    raise SystemExit("no-attachment physical replay did not pass")
case = replay["cases"][0]
if case.get("oracle_attachment_created"):
    raise SystemExit("oracle attachment was unexpectedly created")
summary = {
    "status": "passed",
    "task_schema_version": 3,
    "robot_count": 1,
    "action_dimension": 7,
    "episodes_collected": 1,
    "physical_replay_passed": 1,
    "oracle_attachment_allowed": False,
    "finger_closing_axis_world": [0.0, 1.0, 0.0],
    "replay_metrics": case["metrics"],
    "max_gripper_contact_points": case["max_gripper_contact_points"],
}
path = Path(sys.argv[2])
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
print(json.dumps(summary, indent=2))
PY

if pgrep -af 'ray::|raylet|collect_data.py|eval_policy.py' | grep -v grep; then
  echo "错误：原型结束后仍有 RoboTwin/Ray 进程。" >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "错误：原型结束后仍有 GPU 计算进程。" >&2
  exit 1
fi

printf '%s\n' "$current_manifest" >"$manifest_path"
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/probe.ok"
echo "无辅助约束物理回放原型通过：${state_root}/probe-summary.json"
