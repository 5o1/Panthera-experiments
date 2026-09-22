#!/usr/bin/env bash
# Panthera v2 阶段串流编排。
#
# 本脚本不含任何阶段逻辑：每个阶段仍是独立脚本，这里只按顺序"等标记 → 触发下一个
# 阶段"。所有阶段都以幂等标记文件为准，因此中断、崩溃或整机重启后重新拉起本脚本即
# 可从任意阶段恢复，不会重复已完成的工作。
#
# 标记约定：
#   run  —— 执行阶段脚本，脚本正常返回后要求标记存在，否则判定失败并停止串流。
#   wait —— 阶段已在外部启动（例如采集任务），这里只等标记。
#   gate —— 等待人工批准文件，没有超时，也不伪造任何批准。
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
chain_state="${workspace}/.panthera-v2-stage-chain-state"
log_dir="${chain_state}/logs"
poll_seconds="${PANTHERA_V2_CHAIN_POLL_SECONDS:-60}"
mkdir -p "$chain_state" "$log_dir"

dataset_state_randomized="${workspace}/.panthera-v2-single-grasp-expanded-state"
dataset_state_fixed="${workspace}/.panthera-v2-single-grasp-expanded_fixed-state"
media_state_randomized="${workspace}/.panthera-v2-expanded-media-audit-state"
media_state_fixed="${workspace}/.panthera-v2-expanded-fixed-media-audit-state"
rlds_state_randomized="${workspace}/.panthera-v2-expanded-rlds-state"
rlds_state_fixed="${workspace}/.panthera-v2-expanded-fixed-rlds-state"
smoke_state_randomized="${workspace}/.panthera-v2-expanded-openvla-smoke-state"
smoke_state_fixed="${workspace}/.panthera-v2-expanded-fixed-openvla-smoke-state"
sft_state_randomized="${workspace}/.panthera-v2-expanded-openvla-earlystop-state"
sft_state_fixed="${workspace}/.panthera-v2-expanded-fixed-openvla-earlystop-state"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${chain_state}/chain.log"
}

state_dir() {
  case "$1:$2" in
    expanded:media)       echo "$media_state_randomized" ;;
    expanded:rlds)        echo "$rlds_state_randomized" ;;
    expanded:smoke)       echo "$smoke_state_randomized" ;;
    expanded:sft)         echo "$sft_state_randomized" ;;
    expanded_fixed:media) echo "$media_state_fixed" ;;
    expanded_fixed:rlds)  echo "$rlds_state_fixed" ;;
    expanded_fixed:smoke) echo "$smoke_state_fixed" ;;
    expanded_fixed:sft)   echo "$sft_state_fixed" ;;
  esac
}

# 每个阶段一个独立脚本，按 tier 复用同一份实现。
stage_script() {
  case "$2" in
    media) echo "${workspace}/run_lab_panthera_v2_expanded_media_audit.sh" ;;
    rlds)  echo "${workspace}/run_lab_panthera_v2_expanded_sft_rlds.sh" ;;
    smoke) echo "${workspace}/run_lab_openvla_panthera_v2_expanded_sft_step_smoke.sh" ;;
    sft)   echo "${workspace}/run_lab_openvla_panthera_v2_expanded_sft_earlystop.sh" ;;
    *) echo "错误：未知阶段 ${2}" >&2; return 1 ;;
  esac
}

# tier|stage|mode|marker
stages=(
  "expanded_fixed|generation|wait|${dataset_state_fixed}/automated-audit.ok"
  "expanded|approval|gate|${dataset_state_randomized}/human-review-approval.json"
  "expanded|dataset_ok|file|${dataset_state_randomized}/dataset.ok"
  "expanded|media|run|${media_state_randomized}/media.ok"
  "expanded|rlds|run|${rlds_state_randomized}/rlds.ok"
  "expanded|smoke|run|${smoke_state_randomized}/train.ok"
  "expanded|sft|run|${sft_state_randomized}/train.ok"
  "expanded_fixed|approval|gate|${dataset_state_fixed}/human-review-approval.json"
  "expanded_fixed|dataset_ok|file|${dataset_state_fixed}/dataset.ok"
  "expanded_fixed|media|run|${media_state_fixed}/media.ok"
  "expanded_fixed|rlds|run|${rlds_state_fixed}/rlds.ok"
  "expanded_fixed|smoke|run|${smoke_state_fixed}/train.ok"
  "expanded_fixed|sft|run|${sft_state_fixed}/train.ok"
)

exec 9>"${chain_state}/chain.lock"
flock -n 9 || { echo "错误：阶段串流编排已在运行。" >&2; exit 1; }
printf '%s\n' "$BASHPID" >"${chain_state}/chain.pid"
log "串流编排启动，共 ${#stages[@]} 个阶段。"

for entry in "${stages[@]}"; do
  IFS='|' read -r tier stage mode marker <<<"$entry"
  name="${tier}:${stage}"
  if [[ -f "$marker" ]]; then
    log "跳过 ${name}：标记已存在。"
    continue
  fi
  export PANTHERA_V2_CHAIN_TIER="$tier"
  case "$mode" in
    gate|file)
      log "等待 ${name}：需要人工产出的 ${marker}（不自动伪造）。"
      until [[ -f "$marker" ]]; do sleep "$poll_seconds"; done
      log "${name} 已就绪。"
      ;;
    wait)
      log "等待 ${name}：等待外部任务产出 ${marker}。"
      until [[ -f "$marker" ]]; do sleep "$poll_seconds"; done
      log "${name} 完成。"
      ;;
    run)
      script=$(stage_script "$tier" "$stage")
      [[ -x "$script" ]] || { log "错误：缺少阶段脚本 ${script}，串流停止。"; exit 1; }
      stamp=$(date +%Y%m%d-%H%M%S)
      stage_log="${log_dir}/${tier}-${stage}-${stamp}.log"
      log "启动 ${name}：${script}（日志 ${stage_log}）"
      printf '%s\n' "$stage_log" >"${chain_state}/current-stage-log.txt"
      set +e
      bash "$script" >"$stage_log" 2>&1
      status=$?
      set -e
      printf '%s\n' "$status" >"${chain_state}/last-exit-code.txt"
      if [[ ! -f "$marker" ]]; then
        log "阶段 ${name} 未产出标记（退出码 ${status}），串流停止，等待人工处理。"
        exit 1
      fi
      log "${name} 完成（退出码 ${status}）。"
      ;;
  esac
done

log "串流全部完成。"
touch "${chain_state}/chain.ok"
