#!/usr/bin/env bash
# 阶段：1280 条扩充集的逐集媒体审计。tier 由 PANTHERA_V2_CHAIN_TIER 指定。
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
tier="${PANTHERA_V2_CHAIN_TIER:-expanded}"
case "$tier" in
  expanded)
    dataset_name="panthera_phone_cylinder_socket_v2_single_grasp_sft_v2"
    dataset_state="${workspace}/.panthera-v2-single-grasp-expanded-state"
    state_root="${workspace}/.panthera-v2-expanded-media-audit-state"
    ;;
  expanded_fixed)
    dataset_name="panthera_phone_cylinder_socket_v2_single_grasp_sft_v2_fixedcam"
    dataset_state="${workspace}/.panthera-v2-single-grasp-expanded_fixed-state"
    state_root="${workspace}/.panthera-v2-expanded-fixed-media-audit-state"
    ;;
  *) echo "错误：未知 tier ${tier}。" >&2; exit 1 ;;
esac

dataset_root="${workspace}/data/place_randomized_cylinder_in_socket/${dataset_name}"
activation_script="${workspace}/activate_lab_vla.sh"
auditor="${workspace}/verify_lab_panthera_single_dataset.py"
summary="${state_root}/media-summary.json"
scene_profile=panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2
workers="${PANTHERA_V2_MEDIA_WORKERS:-8}"

[[ $(id -u) -ne 0 ]] || { echo "错误：禁止使用 root 运行媒体审计。" >&2; exit 1; }
for required in "$activation_script" "$auditor" \
  "${dataset_state}/human-review-approval.json"; do
  [[ -s "$required" ]] || { echo "错误：缺少前置文件 ${required}。" >&2; exit 1; }
done
# dataset.ok 是 touch 产生的 0 字节标记，必须用 -f 而不是 -s 检查。
[[ -f "${dataset_state}/dataset.ok" ]] || {
  echo "错误：数据集尚未通过人工验收（缺少 ${dataset_state}/dataset.ok）。" >&2
  exit 1
}

mkdir -p "$state_root"
exec 9>"${state_root}/media.lock"
# 用等待锁而不是失败退出：主串流与支线可能同时触发同一阶段，先到的执行，
# 后到的等锁，拿到锁后会发现标记已存在并直接成功返回。
flock -w 21600 9 || { echo "错误：媒体审计锁等待超时。" >&2; exit 1; }
if [[ -f "${state_root}/media.ok" && -s "$summary" ]]; then
  python3 -m json.tool "$summary"
  exit 0
fi

# shellcheck disable=SC1090
source "$activation_script"
python "$auditor" \
  --dataset-root "$dataset_root" \
  --summary "$summary" \
  --expected-episodes 1280 \
  --expected-schema-version 10 \
  --expected-scene-profile "$scene_profile" \
  --dataset-name "$dataset_name" \
  --workers "$workers"
date --iso-8601=seconds >"${state_root}/completed-at.txt"
touch "${state_root}/media.ok"
echo "schema 10 ${tier} 逐集媒体审计通过。"
