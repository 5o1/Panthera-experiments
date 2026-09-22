#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
starter="${workspace}/bin/start_lab_panthera_v2_unattended_pipeline.sh"
marker="# panthera-v2-schema10-autoresume"
entry="@reboot ${starter} ${marker}"
command -v crontab >/dev/null 2>&1 || {
  echo "错误：Lab 没有 crontab，无法安装普通用户重启恢复。" >&2
  exit 1
}
[[ -x "$starter" ]] || { echo "错误：缺少启动器 ${starter}。" >&2; exit 1; }

temporary=$(mktemp)
trap 'rm -f "$temporary"' EXIT
crontab -l 2>/dev/null | grep -Fv "$marker" >"$temporary" || true
printf '%s\n' "$entry" >>"$temporary"
crontab "$temporary"
echo "已安装普通用户 @reboot 恢复入口；质量门终态不会重复运行。"
