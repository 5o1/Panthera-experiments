#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
upstream_state="${workspace}/state/panthera-single-sft-pipeline-state"
state_root="${workspace}/state/panthera-single-policy-pipeline-state"
train_runner="${workspace}/bin/run_lab_openvla_sft.sh"
eval_runner="${workspace}/bin/run_lab_panthera_policy_eval.sh"
media_runner="${workspace}/bin/run_lab_panthera_single_media_audit.sh"

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
for required in "$train_runner" "$eval_runner" "$media_runner"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：缺少正式策略流水线阶段脚本：${required}" >&2
    exit 1
  fi
done

mkdir -p "$state_root" "$upstream_state"
exec 9>"${state_root}/pipeline.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera 正式策略流水线正在运行。" >&2
  exit 1
fi
if [[ -f "${state_root}/pipeline.ok" ]]; then
  python3 -m json.tool "${state_root}/pipeline-summary.json"
  echo "Panthera 正式策略流水线已通过，无需重复运行。"
  exit 0
fi

# This runner belongs to the first software baseline: a horizontal cylinder
# dropped into a blue U-shaped groove.  The archived phone view shows the real
# target as a vertical cylinder inserted into a yellow block/socket.  Keep the
# baseline artifacts, but never spend the formal four-GPU training budget on
# that visually and geometrically mismatched scene by default.
if [[ "${PANTHERA_ALLOW_UNALIGNED_BASELINE_TRAINING:-0}" != 1 ]]; then
  echo "错误：当前策略流水线是未对齐的水平圆柱基线，默认禁止正式训练。" >&2
  echo "请运行 phone-SRT 真实场景对齐版流水线；仅明确设置" >&2
  echo "PANTHERA_ALLOW_UNALIGNED_BASELINE_TRAINING=1 才允许历史对照训练。" >&2
  exit 3
fi

# 由内核文件锁事件等待上游，不使用时间轮询或 sleep。
echo "等待数据、物理回放、RLDS 和优化器 smoke 流水线释放锁……"
exec 8>"${upstream_state}/pipeline.lock"
flock 8
flock -u 8
exec 8>&-
if [[ ! -f "${upstream_state}/pipeline.ok" ]]; then
  echo "错误：上游流水线已结束，但没有通过标记；拒绝启动正式训练。" >&2
  exit 1
fi

echo "[0/2] 逐集核对 HDF5、单臂语言和连续视频"
bash "$media_runner"

python3 - "$workspace" <<'PY'
import json
from pathlib import Path
import sys

workspace = Path(sys.argv[1])
paths = {
    "dataset": workspace / ".panthera-single-sft-dataset-state/dataset-summary.json",
    "rlds": workspace / ".panthera-single-sft-rlds-state/rlds-summary.json",
    "smoke": workspace / ".panthera-single-openvla-sft-smoke-state/train-summary.json",
    "media": workspace / ".panthera-single-media-audit-state/media-summary.json",
}
values = {
    name: json.loads(path.read_text(encoding="utf-8"))
    for name, path in paths.items()
}
dataset = values["dataset"]
replay = dataset.get("no_attachment_replay", {})
if dataset.get("status") != "passed" or dataset.get("episodes") != 128:
    raise SystemExit("128-episode dataset gate did not pass")
if replay.get("status") != "passed" or replay.get("passed") != replay.get("total"):
    raise SystemExit("no-attachment physical replay gate did not pass")
rlds = values["rlds"]
if (
    rlds.get("status") != "passed"
    or rlds.get("statistics_episode_count") != 128
    or rlds.get("action_chunk_shape") != [5, 7]
):
    raise SystemExit("formal Panthera RLDS gate did not pass")
smoke = values["smoke"]
if smoke.get("status") != "passed" or smoke.get("action_chunk") != 5:
    raise SystemExit("OpenVLA optimizer smoke gate did not pass")
media = values["media"]
if (
    media.get("status") != "passed"
    or media.get("video_count") != 128
    or not media.get("all_video_frame_counts_match_hdf5")
):
    raise SystemExit("single-arm media audit gate did not pass")
print(json.dumps(values, indent=2))
PY

echo "[1/2] 运行四卡 OpenVLA 正式 SFT"
bash "$train_runner"

echo "[2/2] 运行 16 轨迹 RoboTwin 闭环评测"
bash "$eval_runner"

python3 - "$workspace" "${state_root}/pipeline-summary.json" <<'PY'
import json
from pathlib import Path
import sys

workspace = Path(sys.argv[1])
output = Path(sys.argv[2])
stages = {
    "training": workspace / ".panthera-single-openvla-sft-state/train-summary.json",
    "evaluation": workspace / ".panthera-single-policy-eval-state/eval-summary.json",
}
summary = {"status": "passed", "stages": {}}
for name, path in stages.items():
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed":
        raise SystemExit(f"stage did not pass: {name}")
    summary["stages"][name] = value
output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/pipeline.ok"
echo "Panthera 正式策略无人值守流水线通过。"
