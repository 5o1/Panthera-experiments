#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
dataset_root="${workspace}/data/place_cylinder_in_groove/panthera_single_cylinder_sft_v1"
source_root="${workspace}/single_rlds_adapter_smoke_source"
data_root="${workspace}/rlds_single_adapter_smoke"
state_root="${workspace}/.panthera-single-rlds-adapter-smoke-state"
adapter_root="${workspace}/panthera-openvla-adapter"
activation_script="${workspace}/activate_lab_vla.sh"
summary_path="${state_root}/summary.json"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock timeout; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
for required in \
  "$activation_script" \
  "${adapter_root}/panthera_rlds.py" \
  "${dataset_root}/scene_info.json" \
  "${dataset_root}/data/episode0.hdf5" \
  "${dataset_root}/data/episode1.hdf5" \
  "${workspace}/panthera-robotwin-overlay/description/task_instruction/place_cylinder_in_groove.json"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少 RLDS adapter smoke 输入：${required}" >&2
    exit 1
  fi
done

mkdir -p "$state_root"
exec 9>"${state_root}/smoke.lock"
if ! flock -n 9; then
  echo "错误：另一个单臂 RLDS adapter smoke 正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/smoke.ok" ]]; then
  python3 -m json.tool "$summary_path"
  exit 0
fi
if [[ -e "$source_root" || -e "$data_root" ]]; then
  echo "错误：发现没有成功标记的既有 smoke 输出，拒绝覆盖。" >&2
  exit 1
fi

mkdir -p "${source_root}/data" "${source_root}/instructions" "$data_root"
ln "${dataset_root}/data/episode0.hdf5" "${source_root}/data/episode0.hdf5"
ln "${dataset_root}/data/episode1.hdf5" "${source_root}/data/episode1.hdf5"
cp "${workspace}/panthera-robotwin-overlay/description/task_instruction/place_cylinder_in_groove.json" \
  "${source_root}/instructions/episode0.json"
cp "${workspace}/panthera-robotwin-overlay/description/task_instruction/place_cylinder_in_groove.json" \
  "${source_root}/instructions/episode1.json"
python3 - "$dataset_root" "$source_root" <<'PY'
import json
from pathlib import Path
import sys

source = Path(sys.argv[1]) / "scene_info.json"
target = Path(sys.argv[2]) / "scene_info.json"
scene = json.loads(source.read_text(encoding="utf-8"))
subset = {key: scene[key] for key in ("episode_0", "episode_1")}
target.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
PY

# shellcheck disable=SC1090
source "$activation_script"
export PYTHONPATH="${adapter_root}${PYTHONPATH:+:${PYTHONPATH}}"
export ROBOT_PLATFORM=BRIDGE
export TF_CPP_MIN_LOG_LEVEL=2
timeout --signal=INT --kill-after=60s "${PANTHERA_SINGLE_RLDS_ADAPTER_TIMEOUT:-20m}" \
  python "${adapter_root}/panthera_rlds.py" \
    --source-root "$source_root" \
    --data-root "$data_root" \
    --summary "$summary_path" \
    --validation-episode 1

python - "$summary_path" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "passed":
    raise SystemExit("single-arm RLDS adapter smoke did not pass")
if summary.get("splits") != {"train": [0], "val": [1]}:
    raise SystemExit("single-arm RLDS adapter split mismatch")
if summary.get("action_chunk_shape") != [5, 7]:
    raise SystemExit("single-arm action chunk is not 5x7")
if summary.get("proprio_window_shape") != [1, 7]:
    raise SystemExit("single-arm proprio window is not 1x7")
if summary.get("primary_image_shape") != [1, 224, 224, 3]:
    raise SystemExit("single-arm RGB window was not decoded and resized")
PY

date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/smoke.ok"
echo "单 Panthera 7 维 RLDS adapter smoke 通过：${summary_path}"
