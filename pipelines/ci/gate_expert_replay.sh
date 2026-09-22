#!/usr/bin/env bash
# Gate 1: every recorded expert trajectory must succeed when replayed through
# the evaluation executor, in the scene it was recorded in.
#
# This is the ceiling any policy number is measured against.  An episode the
# expert cannot reproduce is one where a perfect policy is also scored a
# failure, so a run that skips this reports a success rate whose maximum is
# unknown -- which is not hypothetical: an evaluation was once capped at 7.0% by
# an action budget nobody had compared against the expert's own lengths.
#
# The scene comes from the dataset's own table, and the action budget from the
# dataset's own contract, so this cannot disagree with the data it checks.
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
packages="${PANTHERA_PACKAGES:-${workspace}/packages}"
dataset_root="${CI_EXPERT_DATASET_ROOT:?set CI_EXPERT_DATASET_ROOT}"
robotwin_root="${CI_EXPERT_ROBOTWIN_ROOT:-${workspace}/runtime/robotwin}"
task_config="${CI_EXPERT_TASK_CONFIG:-panthera_phone_cylinder_socket_v2_pilot.yml}"
task_name="${CI_EXPERT_TASK_NAME:-place_randomized_cylinder_in_socket}"
report="${CI_EXPERT_REPORT:-${workspace}/reports/ci/expert-replay.json}"
workers="${CI_EXPERT_WORKERS:-12}"
gpus="${CI_EXPERT_GPUS:-0,1,2,3}"
# A stratified sample keeps the gate cheap enough to run per change; 0 replays
# every episode and is the release check.  Either way the bar is 100% of what ran.
sample="${CI_EXPERT_SAMPLE:-40}"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行。" >&2
  exit 1
fi
[[ -s "${dataset_root}/dataset.json" ]] || {
  echo "错误：${dataset_root} 没有 dataset.json；先运行 panthera_sim/backfill.py。" >&2
  exit 1
}

# shellcheck disable=SC1091
source "${workspace}/tools/activate_lab_vla.sh" >/dev/null 2>&1

echo "=== 门禁1：专家轨迹自复现 ==="
(
  cd "${packages}/panthera_sim"
  python3 replay.py \
    --dataset-root "$dataset_root" \
    --robotwin-root "$robotwin_root" \
    --task-config "$task_config" \
    --task-name "$task_name" \
    --sample "$sample" \
    --workers "$workers" \
    --gpus "$gpus" \
    --output "$report"
)

python3 - "$report" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1], encoding="utf-8").read())
failed = [c for c in report["cases"] if not c.get("success")]
for case in failed:
    metrics = case.get("metrics", {})
    reason = case.get("error") or ", ".join(
        f"{k}={v:.4f}" for k, v in sorted(metrics.items()) if isinstance(v, float)
    )
    print(f"  失败 ep{case['episode']} ({case['posture']}): {reason}")
print(f"\n专家自复现 {report['success']}/{report['total']}"
      f"  数据集 {report['dataset']} ({report['dataset_digest'][:12]}...)")
if failed:
    print("门禁1 未通过：专家轨迹必须 100% 自复现。低于 100% 时策略评测的"
          "成功率上限未知，数字不可解读。")
    raise SystemExit(1)
print("门禁1 通过。")
PY
