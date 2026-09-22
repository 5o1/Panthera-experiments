#!/usr/bin/env bash
# 1280 随机机位数据集的训练支线：媒体审计 → RLDS → 一步 smoke → 4 卡 SFT。
# 与主串流共用同一批阶段脚本和标记文件，因此两边都不会重复执行已完成阶段。
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
export PANTHERA_V2_CHAIN_TIER=expanded
state="${workspace}/.panthera-v2-expanded-branch-state"
mkdir -p "$state"

exec 9>"${state}/branch.lock"
flock -n 9 || { echo "错误：随机机位支线已在运行。" >&2; exit 1; }

stages=(
  "run_lab_panthera_v2_expanded_media_audit.sh|${workspace}/.panthera-v2-expanded-media-audit-state/media.ok"
  "run_lab_panthera_v2_expanded_sft_rlds.sh|${workspace}/.panthera-v2-expanded-rlds-state/rlds.ok"
  "run_lab_openvla_panthera_v2_expanded_sft_step_smoke.sh|${workspace}/.panthera-v2-expanded-openvla-smoke-state/train.ok"
  "run_lab_openvla_panthera_v2_expanded_sft_earlystop.sh|${workspace}/.panthera-v2-expanded-openvla-earlystop-state/train.ok"
)

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${state}/branch.log"
}

printf '%s\n' "$BASHPID" >"${state}/branch.pid"
log "随机机位支线启动，共 ${#stages[@]} 个阶段。"
for entry in "${stages[@]}"; do
  IFS='|' read -r script marker <<<"$entry"
  name="${script%.sh}"
  if [[ -f "$marker" ]]; then
    log "跳过 ${name}：标记已存在。"
    continue
  fi
  [[ -x "${workspace}/${script}" ]] || { log "错误：缺少 ${script}，支线停止。"; exit 1; }
  stage_log="${state}/${name}-$(date +%Y%m%d-%H%M%S).log"
  printf '%s\n' "$stage_log" >"${state}/current-log.txt"
  log "启动 ${name}（日志 ${stage_log}）"
  set +e
  bash "${workspace}/${script}" >"$stage_log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" >"${state}/last-exit-code.txt"
  if [[ ! -f "$marker" ]]; then
    log "阶段 ${name} 未产出标记（退出码 ${status}），支线停止。"
    exit 1
  fi
  log "${name} 完成（退出码 ${status}）。"
done
log "随机机位支线全部完成。"
touch "${state}/branch.ok"
