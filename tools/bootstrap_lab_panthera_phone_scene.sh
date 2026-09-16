#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
robotwin_root="${workspace}/RoboTwin"
overlay_root="${workspace}/panthera-robotwin-overlay"
source_embodiment="${robotwin_root}/assets/embodiments/panthera"
generated_embodiment="${overlay_root}/assets/embodiments/panthera_phone"
target_embodiment="${robotwin_root}/assets/embodiments/panthera_phone"
state_root="${workspace}/.panthera-phone-scene-state"
upstream_state="${workspace}/.panthera-single-sft-pipeline-state"
activation_script="${workspace}/activate_lab_vla.sh"
camera_profile="phone_srt_provisional_wide_v3"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi
for command_name in flock python3; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  }
done
for required in \
  "$activation_script" \
  "${overlay_root}/build_panthera_phone_embodiment.py" \
  "${overlay_root}/envs/place_vertical_cylinder_in_groove.py" \
  "${overlay_root}/task_config/panthera_phone_vertical_oracle.yml" \
  "${overlay_root}/description/task_instruction/place_vertical_cylinder_in_groove.json" \
  "${source_embodiment}/config.yml"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少 phone-SRT 场景文件：${required}" >&2
    exit 1
  fi
done
for marker in \
  "${workspace}/.bootstrap-state/verified.ok" \
  "${workspace}/.panthera-embodiment-state/verified.ok"; do
  if [[ ! -f "$marker" ]]; then
    echo "错误：缺少前置验收标记：${marker}" >&2
    exit 1
  fi
done

mkdir -p "$state_root" "$upstream_state"
exec 9>"${state_root}/scene.lock"
if ! flock -n 9; then
  echo "错误：另一个 phone-SRT 场景构建正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/scene.ok" ]]; then
  if python3 - "${state_root}/scene-summary.json" "$camera_profile" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if summary.get("camera_profile") == sys.argv[2] else 1)
PY
  then
    python3 -m json.tool "${state_root}/scene-summary.json"
    exit 0
  fi
  echo "错误：现有 phone 场景标记属于旧相机版本，请先可恢复地归档状态目录。" >&2
  exit 1
fi

# Do not replace task modules or the embodiment registry while the baseline
# collector/replay may still be importing them.  This wait is kernel-driven.
echo "等待单臂软件基线释放流水线锁……"
exec 8>"${upstream_state}/pipeline.lock"
flock 8
flock -u 8
exec 8>&-

# shellcheck disable=SC1090
source "$activation_script"
if [[ -d "$generated_embodiment" ]]; then
  generated_profile=$(python - "${generated_embodiment}/provenance.phone-srt.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
print(json.loads(path.read_text(encoding="utf-8")).get("profile", "") if path.is_file() else "")
PY
)
  if [[ "$generated_profile" != "$camera_profile" ]]; then
    echo "错误：生成目录属于旧相机版本，请先可恢复地归档：${generated_embodiment}" >&2
    exit 1
  fi
else
  python "${overlay_root}/build_panthera_phone_embodiment.py" \
    --source-root "$source_embodiment" \
    --output-root "$generated_embodiment"
fi
mkdir -p "$target_embodiment"
cp -a "${generated_embodiment}/." "$target_embodiment/"

install -m 0644 \
  "${overlay_root}/envs/place_vertical_cylinder_in_groove.py" \
  "${robotwin_root}/envs/place_vertical_cylinder_in_groove.py"
install -m 0644 \
  "${overlay_root}/description/task_instruction/place_vertical_cylinder_in_groove.json" \
  "${robotwin_root}/description/task_instruction/place_vertical_cylinder_in_groove.json"
install -m 0644 \
  "${overlay_root}/task_config/panthera_phone_vertical_oracle.yml" \
  "${robotwin_root}/task_config/panthera_phone_vertical_oracle.yml"

python - "$robotwin_root" "${state_root}/scene-summary.json" <<'PY'
import json
from pathlib import Path
import sys
import yaml

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
registry_path = root / "task_config/_embodiment_config.yml"
registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
expected = {"file_path": "./assets/embodiments/panthera_phone"}
if registry.get("panthera_phone") != expected:
    registry["panthera_phone"] = expected
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

camera_path = root / "task_config/_camera_config.yml"
cameras = yaml.safe_load(camera_path.read_text(encoding="utf-8"))
phone_camera = {"fovy": 75, "w": 320, "h": 240}
if cameras.get("PhoneSRT_Center4_3_Wide") != phone_camera:
    cameras["PhoneSRT_Center4_3_Wide"] = phone_camera
    camera_path.write_text(
        yaml.safe_dump(cameras, sort_keys=False), encoding="utf-8"
    )

config = yaml.safe_load(
    (root / "assets/embodiments/panthera_phone/config.yml").read_text(
        encoding="utf-8"
    )
)
head = config["static_camera_list"][0]
if head["name"] != "head_camera" or head["position"] != [-0.25, 0.72, 1.05]:
    raise SystemExit("phone embodiment camera profile mismatch")
if config["robot_pose"][0][1] != -0.35:
    raise SystemExit("phone embodiment robot pose mismatch")
summary = {
    "status": "passed",
    "task": "place_vertical_cylinder_in_groove",
    "robot_count": 1,
    "action_dimension": 7,
    "embodiment": "panthera_phone",
    "camera_profile": "phone_srt_provisional_wide_v3",
    "camera_type": "PhoneSRT_Center4_3_Wide",
    "camera_resolution": [320, 240],
    "camera_fovy_deg": 75,
    "camera_extrinsics_status": "provisional_from_archived_phone_view",
    "scene_geometry": "upright_yellow_cylinder_and_yellow_shallow_socket",
    "attach_on_grasp": False,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

python -m py_compile \
  "${robotwin_root}/envs/place_vertical_cylinder_in_groove.py"
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/scene.ok"
echo "phone-SRT 对齐场景已安装。"
