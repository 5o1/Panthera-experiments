#!/usr/bin/env bash

set -uo pipefail

port="${1:-9000}"
output_dir="${PANTHERA_SRT_OUTPUT_DIR:-/tmp/panthera-srt-camera}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "错误：未找到 ffmpeg。" >&2
  exit 1
fi

mkdir -p "$output_dir"
echo "SRT 摄像头接收器：srt://0.0.0.0:${port}?mode=listener"
echo "最新画面：${output_dir}/latest.jpg"
echo "循环缓存：${output_dir}/buffer_00.mkv .. buffer_11.mkv"

while true; do
  ffmpeg \
    -nostdin \
    -nostats \
    -hide_banner \
    -loglevel level+warning \
    -y \
    -fflags +genpts \
    -i "srt://0.0.0.0:${port}?mode=listener&latency=200000" \
    -map 0:v:0 \
    -map '0:a?' \
    -c copy \
    -f segment \
    -segment_time 10 \
    -segment_wrap 12 \
    -reset_timestamps 1 \
    "${output_dir}/buffer_%02d.mkv" \
    -map 0:v:0 \
    -an \
    -vf fps=2 \
    -q:v 2 \
    -update 1 \
    "${output_dir}/latest.jpg"

  exit_code=$?
  if (( exit_code == 130 || exit_code == 143 )); then
    exit "$exit_code"
  fi
  echo "SRT 连接已断开（ffmpeg=${exit_code}），1 秒后重新监听。" >&2
  sleep 1
done
