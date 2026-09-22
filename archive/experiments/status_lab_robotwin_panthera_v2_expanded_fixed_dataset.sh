#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state_root="${workspace}/.panthera-v2-single-grasp-expanded_fixed-state"
shard_root="${workspace}/data_phone_v2_single_grasp_expanded_fixed_shards/place_randomized_cylinder_in_socket"
pid=""
[[ -s "${state_root}/launcher.pid" ]] && pid=$(<"${state_root}/launcher.pid")
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  echo "进程：运行中（PID=${pid}）"
else
  echo "进程：未运行"
fi
completed_shards=0
committed_episodes=0
if [[ -d "$shard_root" ]]; then
  completed_shards=$(find "$shard_root" -mindepth 2 -maxdepth 2 -path '*/.episode_commits' -type d -print0 2>/dev/null \
    | xargs -0 -r -I{} sh -c 'test "$(find "{}" -maxdepth 1 -name "episode*.json" | wc -l)" -eq 8 && echo 1' \
    | wc -l)
  committed_episodes=$(find "$shard_root" -path '*/.episode_commits/episode*.json' -type f 2>/dev/null | wc -l)
fi
echo "完成分片：${completed_shards}/160"
echo "原子提交轨迹：${committed_episodes}/1280"
if [[ -f "${state_root}/automated-audit.ok" ]]; then
  echo "状态：自动审计通过，等待人工验收"
elif [[ -f "${state_root}/awaiting-human-review.txt" ]]; then
  echo "状态：等待人工验收"
else
  echo "状态：生成或审计尚未完成"
fi
if [[ -s "${state_root}/current-log.txt" ]]; then
  log=$(<"${state_root}/current-log.txt")
  echo "日志：${log}"
  [[ -s "$log" ]] && tail -n 12 "$log"
fi
