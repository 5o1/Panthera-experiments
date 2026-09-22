#!/usr/bin/env bash
# 查看阶段串流编排状态。
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
chain_state="${workspace}/.panthera-v2-stage-chain-state"
pid=""
[[ -s "${chain_state}/chain.pid" ]] && pid=$(<"${chain_state}/chain.pid")
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  echo "串流进程：运行中（PID=${pid}）"
else
  echo "串流进程：未运行"
fi
[[ -f "${chain_state}/chain.ok" ]] && echo "串流状态：已全部完成"
[[ -s "${chain_state}/current-stage-log.txt" ]] \
  && echo "当前阶段日志：$(<"${chain_state}/current-stage-log.txt")"
[[ -s "${chain_state}/last-exit-code.txt" ]] \
  && echo "上一阶段退出码：$(<"${chain_state}/last-exit-code.txt")"
echo "--- chain.log 末尾 ---"
[[ -s "${chain_state}/chain.log" ]] && tail -n 12 "${chain_state}/chain.log" || echo "(暂无日志)"
echo "--- 各门禁标记 ---"
for marker in \
  "${workspace}/.panthera-v2-single-grasp-expanded_fixed-state/automated-audit.ok" \
  "${workspace}/.panthera-v2-single-grasp-expanded-state/human-review-approval.json" \
  "${workspace}/.panthera-v2-expanded-media-audit-state/media.ok" \
  "${workspace}/.panthera-v2-expanded-rlds-state/rlds.ok" \
  "${workspace}/.panthera-v2-expanded-openvla-smoke-state/train.ok" \
  "${workspace}/.panthera-v2-expanded-openvla-earlystop-state/train.ok" \
  "${workspace}/.panthera-v2-single-grasp-expanded_fixed-state/human-review-approval.json" \
  "${workspace}/.panthera-v2-expanded-fixed-media-audit-state/media.ok" \
  "${workspace}/.panthera-v2-expanded-fixed-rlds-state/rlds.ok" \
  "${workspace}/.panthera-v2-expanded-fixed-openvla-smoke-state/train.ok" \
  "${workspace}/.panthera-v2-expanded-fixed-openvla-earlystop-state/train.ok"; do
  if [[ -f "$marker" ]]; then
    echo "  [x] $marker"
  else
    echo "  [ ] $marker"
  fi
done
