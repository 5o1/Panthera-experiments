#!/usr/bin/env bash

# End-to-end synthetic vision test. No camera, SDK, serial port or robot command publisher.

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d /tmp/panthera-vision-mock.XXXXXX)"
LAUNCH_LOG="${TEST_DIR}/launch.log"
HUD_OUTPUT="${TEST_DIR}/hud.txt"
launch_pid=""
bag_pid=""

# shellcheck source=tools/lib/process_cleanup.sh
source "${PROJECT_ROOT}/tools/lib/process_cleanup.sh"
cleanup() {
    stop_process_group "$bag_pid" INT
    stop_process_group "$launch_pid" TERM
}
trap cleanup EXIT INT TERM

set +u
source /opt/ros/humble/setup.bash
source /home/assaneko/Panthera_HT_ROS2/install/setup.bash
source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
set -u
export AMENT_PREFIX_PATH="${PROJECT_ROOT}/ros2_ws/install/panthera_vision_teleop:${AMENT_PREFIX_PATH}"
export ROS_DOMAIN_ID="${PANTHERA_VISION_MOCK_DOMAIN_ID:-$((40 + ($$ % 180)))}"
export ROS_LOCALHOST_ONLY=1
export ROS_LOG_DIR="${TEST_DIR}/ros-log"

if ss -ltn 2>/dev/null | grep -qE '[:.]8765[[:space:]]'; then
    printf '错误：端口 8765 已被占用。\n' >&2
    exit 1
fi

setsid ros2 launch panthera_vision_teleop vision_teleop.launch.py \
    fake_robot_feedback:=true publish_robot_commands:=false \
    profile_config:="${PROJECT_ROOT}/ros2_ws/src/panthera_vision_teleop/config/teleop_safe.yaml" \
    >"${LAUNCH_LOG}" 2>&1 &
launch_pid=$!

for _ in $(seq 1 100); do
    ss -ltn 2>/dev/null | grep -qE '[:.]8765[[:space:]]' && break
    kill -0 "$launch_pid" 2>/dev/null || { printf '错误：视觉 launch 提前退出：%s\n' "$LAUNCH_LOG" >&2; exit 1; }
    sleep 0.1
done
ss -ltn 2>/dev/null | grep -qE '[:.]8765[[:space:]]' || { printf '错误：bridge 未监听 8765。\n' >&2; exit 1; }

setsid ros2 bag record -o "${TEST_DIR}/bag" \
    /teleop/human_pose /teleop/debug_target \
    /teleop/diagnostics/human_delta_camera \
    /teleop/diagnostics/filtered_robot_offset \
    /teleop/diagnostics/motion_limited_robot_offset \
    /teleop/transport_extra_lag_ms >"${TEST_DIR}/bag.log" 2>&1 &
bag_pid=$!
sleep 0.5
timeout 8 ros2 topic echo /teleop/hud std_msgs/msg/String --once >"${HUD_OUTPUT}" &
echo_pid=$!
"${PROJECT_ROOT}/.venv-wsl-vision/bin/python" \
    "${PROJECT_ROOT}/windows_vision/vision_sender.py" \
    --synthetic --duration 2.0 --fps 15 --no-preview
wait "$echo_pid"
stop_process_group "$bag_pid" INT
bag_pid=""

grep -q 'TRACKING' "$HUD_OUTPUT" || { printf '错误：HUD 未进入 TRACKING：%s\n' "$HUD_OUTPUT" >&2; exit 1; }
grep -q 'target_delta' "$HUD_OUTPUT" || { printf '错误：HUD 缺少目标偏移。\n' >&2; exit 1; }
if ros2 topic list --no-daemon 2>/dev/null | grep -qx '/pos_cmd'; then
    topic_info="$(ros2 topic info /pos_cmd --no-daemon 2>/dev/null || true)"
    grep -Eq 'Publisher count:[[:space:]]*0' <<<"$topic_info" || { printf '错误：dry-run 中出现 /pos_cmd publisher。\n%s\n' "$topic_info" >&2; exit 1; }
fi

/usr/bin/python3 "${PROJECT_ROOT}/tools/analyze_vision_teleop.py" "${TEST_DIR}/bag" --no-plot
grep -q 'transport_extra_lag_ms' "${TEST_DIR}/bag/analysis/summary.json" || { printf '错误：离线摘要缺少传输延迟。\n' >&2; exit 1; }
grep -q 'motion_limited_robot_offset' "${TEST_DIR}/bag/analysis/summary.json" || { printf '错误：离线摘要缺少运动限幅数据。\n' >&2; exit 1; }

printf 'PASS：synthetic→WebSocket→ROS→mapper→HUD 完整链路可用，且没有 /pos_cmd publisher。\n'
printf '测试日志：%s\n' "$TEST_DIR"
