#!/usr/bin/env bash

# Record one reproducible vision-teleoperation run.  This script records only
# ROS topics; it never enables or starts the robot.

set -e

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/tools/activate_panthera.sh"
# shellcheck source=tools/lib/vision_bag_topics.sh
source "${PROJECT_ROOT}/tools/lib/vision_bag_topics.sh"

if ! command -v ros2 >/dev/null 2>&1; then
    printf '错误：ros2 不可用，请先安装并构建 Panthera 环境。\n' >&2
    exit 1
fi

if ! ros2 bag --help >/dev/null 2>&1; then
    printf '错误：ros2 bag 不可用，请安装 ros-humble-rosbag2。\n' >&2
    exit 1
fi

label="${1:-vision_teleop}"
if [[ ! "$label" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf '错误：记录名称只能包含字母、数字、下划线和连字符。\n' >&2
    exit 2
fi

output="${PROJECT_ROOT}/bags/${label}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PROJECT_ROOT}/bags"
printf '开始记录：%s\n按 Ctrl+C 结束录制；这不会启动或控制机械臂。\n' "$output"

exec ros2 bag record -o "$output" "${VISION_BAG_TOPICS[@]}"
