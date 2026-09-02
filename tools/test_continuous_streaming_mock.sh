#!/usr/bin/env bash

# Pure-software test for the Host-style latest-only streaming backend.

set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_LOG_DIR="$(mktemp -d /tmp/panthera-streaming-mock-log.XXXXXX)"
LAUNCH_LOG="${ROS_LOG_DIR}/launch-output.log"
export ROS_LOG_DIR ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID="$((220 + ($$ % 20)))"

# shellcheck source=tools/lib/process_cleanup.sh
source "${PROJECT_ROOT}/tools/lib/process_cleanup.sh"
launch_pid=""
cleanup() {
    trap - EXIT INT TERM
    stop_process_group "$launch_pid" TERM
}
trap cleanup EXIT INT TERM

set +u
source /opt/ros/humble/setup.bash
source /home/assaneko/Panthera_HT_ROS2/install/setup.bash
source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
set -u
export AMENT_PREFIX_PATH="${PROJECT_ROOT}/ros2_ws/install/panthera_vision_teleop:${AMENT_PREFIX_PATH}"

setsid ros2 launch panthera_vision_teleop continuous_control_mock.launch.py \
    use_continuous_backend:=true >"${LAUNCH_LOG}" 2>&1 &
launch_pid=$!
deadline=$((SECONDS + 20))
until rg -q 'Configured and activated.*arm_controller' "${LAUNCH_LOG}"; do
    kill -0 "$launch_pid" 2>/dev/null || { printf '模拟 launch 提前退出：%s\n' "$LAUNCH_LOG" >&2; exit 1; }
    (( SECONDS < deadline )) || { printf '等待模拟控制器超时：%s\n' "$LAUNCH_LOG" >&2; exit 1; }
done

output="$(timeout 20 ros2 run panthera_vision_teleop mock_continuous_cartesian_probe)"
rg -q '^PASS: latest-only 50 Hz Cartesian stream reached target and stale-held$' <<<"$output"
printf '%s\n' "$output"
printf 'PASS：Host 风格连续 backend 的纯软件链路通过。\n'
