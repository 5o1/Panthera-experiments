#!/usr/bin/env bash
# 已弃用（2026-09-17）：128 条 schema 10 线退役，正式训练改用固定机位 1280。
# 其 4.0.0 产物已冻结、rlds.ok 仍在，本入口保留作历史证据，不要再运行。
# 版本号保持 4.0.0 以匹配磁盘上的冻结产物；当前 builder 已是 4.1.0，重跑会失败。

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
source_root="${workspace}/data/place_randomized_cylinder_in_socket/panthera_phone_cylinder_socket_v2_single_grasp_sft_v1"
data_root="${workspace}/datasets/rlds/rlds_phone_cylinder_socket_v2_sft_v1"
adapter_root="${workspace}/packages/panthera_vla"
dataset_state="${workspace}/state/panthera-v2-single-grasp-formal-state"
media_state="${workspace}/state/panthera-v2-schema10-media-audit-state"
state_root="${workspace}/state/panthera-v2-schema10-rlds-state"
summary_path="${state_root}/rlds-summary.json"
split_path="${state_root}/split.json"
dataset_name=panthera_phone_cylinder_socket_v2
dataset_version_root="${data_root}/${dataset_name}/4.0.0"
scene_profile=panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2
activation_script="${workspace}/tools/activate_lab_vla.sh"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：禁止使用 root 转换 RLDS。" >&2
  exit 1
fi
for command_name in flock timeout; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "错误：缺少命令 ${command_name}。" >&2
    exit 1
  }
done
for required in "$activation_script" "${adapter_root}/panthera_rlds.py" \
  "${dataset_state}/human-review-approval.json"; do
  [[ -s "$required" ]] || { echo "错误：缺少前置文件 ${required}。" >&2; exit 1; }
done
for marker in "${dataset_state}/dataset.ok" "${media_state}/media.ok"; do
  [[ -f "$marker" ]] || { echo "错误：缺少前置门禁 ${marker}。" >&2; exit 1; }
done

mkdir -p "$state_root" "$data_root"
exec 9>"${state_root}/rlds.lock"
flock -n 9 || { echo "错误：RLDS 转换已在运行。" >&2; exit 1; }
if [[ -f "${state_root}/rlds.ok" && -s "$summary_path" ]]; then
  python3 -m json.tool "$summary_path"
  exit 0
fi
# 走到这里说明 rlds.ok 未通过，已有 TFDS 输出无论是否完整都未经验收。对完整目录
# download_and_prepare 是 no-op，会静默复用旧字节，而下游门禁读的 splits 来自源 HDF5，
# 察觉不到复用；因此一律可恢复归档后重建。
if [[ -e "$dataset_version_root" ]]; then
  unverified="${dataset_version_root}.unverified-$(date +%Y%m%d-%H%M%S)"
  mv "$dataset_version_root" "$unverified"
  echo "已可恢复归档未经验收的 TFDS 输出：${unverified}"
fi

# 固定 16 条验证集：8 条直立 + 每个平躺角度区间各 1 条；组内优先补齐左右侧和空间格。
python3 - "$source_root/scene_info.json" "$split_path" <<'PY'
import json
from pathlib import Path
import random
import sys

scene = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rng = random.Random(20260916)
rows = []
for episode_id in range(128):
    metadata = scene[f"episode_{episode_id}"]["panthera_episode"]
    geometry = metadata["realized_geometry"]
    side = "left" if geometry["cylinder_initial_xy_m"][0] < geometry["workspace"]["robot_base_x_m"] else "right"
    rows.append({
        "episode": episode_id,
        "posture": metadata["cylinder_posture"],
        "bin": geometry["lying_angle_bin"],
        "side": side,
        "cells": [
            [geometry["cylinder_radial_bin"], geometry["cylinder_angular_bin"]],
            [geometry["socket_radial_bin"], geometry["socket_angular_bin"]],
        ],
    })

selected = []
side_counts = {"left": 0, "right": 0}
seen_cells = set()
def choose(candidates):
    rng.shuffle(candidates)
    candidates.sort(key=lambda row: (
        side_counts[row["side"]],
        -sum(tuple(cell) not in seen_cells for cell in row["cells"]),
        row["episode"],
    ))
    row = candidates[0]
    selected.append(row)
    side_counts[row["side"]] += 1
    seen_cells.update(tuple(cell) for cell in row["cells"])

for angle_bin in range(8):
    choose([row for row in rows if row["posture"] == "lying" and row["bin"] == angle_bin])
for _ in range(8):
    choose([row for row in rows if row["posture"] == "upright" and row not in selected])

validation = sorted(row["episode"] for row in selected)
training = [episode for episode in range(128) if episode not in validation]
payload = {
    "selection_seed": 20260916,
    "contract": "8 upright plus one lying episode per angle bin; greedy side/cell balance",
    "train": training,
    "val": validation,
    "val_side_counts": side_counts,
    "val_rows": sorted(selected, key=lambda row: row["episode"]),
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

# shellcheck disable=SC1090
source "$activation_script"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${adapter_root}${PYTHONPATH:+:${PYTHONPATH}}"
export ROBOT_PLATFORM=PANTHERA
export PANTHERA_ACTION_CHUNK=25
export PANTHERA_RLDS_DATASET_NAME="$dataset_name"
export PANTHERA_RLDS_SCHEMA_VERSION=10
export PANTHERA_RLDS_SCENE_PROFILE="$scene_profile"
mapfile -t validation_ids < <(python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["val"], sep="\n")' "$split_path")
validation_args=()
for episode in "${validation_ids[@]}"; do
  validation_args+=(--validation-episode "$episode")
done
stamp=$(date +%Y%m%d-%H%M%S)
run_log="${state_root}/rlds-${stamp}.log"
printf '%s\n' "$run_log" >"${state_root}/run-log.txt"
set +e
timeout --signal=INT --kill-after=60s "${PANTHERA_V2_RLDS_TIMEOUT:-120m}" \
  python "${adapter_root}/panthera_rlds.py" \
    --source-root "$source_root" \
    --data-root "$data_root" \
    --summary "$summary_path" \
    "${validation_args[@]}" 2>&1 | tee "$run_log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" >"${state_root}/exit-code.txt"
(( status == 0 )) || { echo "错误：schema 10 RLDS 转换失败。" >&2; exit "$status"; }

python - "$summary_path" "$split_path" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
split = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = {
    "status": "passed",
    "dataset_name": "panthera_phone_cylinder_socket_v2",
    "tfds_version": "4.0.0",
    "task_schema_version": 10,
    "statistics_episode_count": 128,
    "action_chunk_shape": [25, 7],
    "proprio_window_shape": [1, 7],
}
for key, value in expected.items():
    if summary.get(key) != value:
        raise SystemExit(f"RLDS {key} mismatch: {summary.get(key)!r} != {value!r}")
if summary.get("splits") != {"train": split["train"], "val": split["val"]}:
    raise SystemExit("RLDS split mismatch")
if summary.get("scene_profile") != "panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2":
    raise SystemExit("RLDS scene profile mismatch")
PY
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/rlds.ok"
echo "schema 10 TFDS/RLDS 4.0.0 转换与真实 OpenVLA 数据管线验收通过。"
