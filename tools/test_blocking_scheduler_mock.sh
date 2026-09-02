#!/usr/bin/env bash

# Verify the compatibility scheduler for the official blocking /pos_cmd path.

set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d /tmp/panthera-blocking-scheduler.XXXXXX)"
launch_pid=""
# shellcheck source=tools/lib/process_cleanup.sh
source "${PROJECT_ROOT}/tools/lib/process_cleanup.sh"
cleanup() { stop_process_group "$launch_pid" TERM; }
trap cleanup EXIT INT TERM

set +u
source /opt/ros/humble/setup.bash
source /home/assaneko/Panthera_HT_ROS2/install/setup.bash
source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
set -u
export AMENT_PREFIX_PATH="${PROJECT_ROOT}/ros2_ws/install/panthera_vision_teleop:${AMENT_PREFIX_PATH}"
export ROS_DOMAIN_ID="${PANTHERA_SCHEDULER_DOMAIN_ID:-$((60 + ($$ % 160)))}"
export ROS_LOCALHOST_ONLY=1
export ROS_LOG_DIR="${TEST_DIR}/ros-log"

setsid ros2 launch panthera_vision_teleop vision_teleop.launch.py \
    use_websocket:=false use_debug:=true debug_scenario:=xyz_cycle \
    fake_robot_feedback:=true fake_robot_follow_commands:=true \
    fake_robot_speed_m_s:=0.01 publish_robot_commands:=true \
    profile_config:="${PROJECT_ROOT}/ros2_ws/src/panthera_vision_teleop/config/teleop_safe.yaml" \
    >"${TEST_DIR}/launch.log" 2>&1 &
launch_pid=$!

output="$(timeout 15 ros2 run panthera_vision_teleop mock_scheduler_probe)"
grep -q '^PASS: direct scheduler sent only while idle and produced no rapid FIFO burst$' <<<"$output" || { printf '%s\n' "$output" >&2; exit 1; }
printf '%s\n' "$output"
