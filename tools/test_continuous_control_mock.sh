#!/usr/bin/env bash

# Exercise the proposed continuous joint-command path against GenericSystem.
# This script cannot load the Panthera SDK or access serial devices.

set -eo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_LOG_DIR="$(mktemp -d /tmp/panthera-mock-control-log.XXXXXX)"
LAUNCH_LOG="${ROS_LOG_DIR}/launch-output.log"
export ROS_LOG_DIR
if [[ -n "${PANTHERA_MOCK_DOMAIN_ID:-}" ]]; then
    export ROS_DOMAIN_ID="${PANTHERA_MOCK_DOMAIN_ID}"
else
    # Give every run its own DDS domain so a recently stopped test cannot be
    # mistaken for the new controller manager during discovery cleanup.
    export ROS_DOMAIN_ID="$((20 + ($$ % 200)))"
fi
export ROS_LOCALHOST_ONLY=1

source /opt/ros/humble/setup.bash
source /home/assaneko/Panthera_HT_ROS2/install/setup.bash
source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
# The current separated ament_python install does not prepend its package
# prefix to AMENT_PREFIX_PATH, so make the package index explicit here.
export AMENT_PREFIX_PATH="${PROJECT_ROOT}/ros2_ws/install/panthera_vision_teleop:${AMENT_PREFIX_PATH}"
set -u

launch_pid=""
cleanup() {
    if [[ -n "${launch_pid}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
        kill -INT -- "-${launch_pid}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${launch_pid}" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "${launch_pid}" 2>/dev/null; then
            kill -TERM -- "-${launch_pid}" 2>/dev/null || true
        fi
        wait "${launch_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

setsid ros2 launch panthera_vision_teleop continuous_control_mock.launch.py \
    use_cartesian_backend:=true \
    >"${LAUNCH_LOG}" 2>&1 &
launch_pid="$!"

for _ in $(seq 1 200); do
    if rg -q 'Configured and activated.*arm_controller' "${LAUNCH_LOG}"; then
        break
    fi
    if ! kill -0 "${launch_pid}" 2>/dev/null; then
        printf '错误：模拟控制 launch 提前退出。日志：%s\n' "${LAUNCH_LOG}" >&2
        exit 1
    fi
    sleep 0.1
done

if ! rg -q 'Configured and activated.*arm_controller' "${LAUNCH_LOG}"; then
    printf '错误：arm_controller 未进入 active。日志：%s\n' "${LAUNCH_LOG}" >&2
    exit 1
fi

probe_output="$(timeout 20s ros2 run panthera_vision_teleop mock_trajectory_probe)"
if ! rg -q '^PASS: trajectory success, named feedback, cancel, and post-cancel hold$' \
    <<<"${probe_output}"; then
    printf '错误：模拟连续控制探针没有成功。\n%s\n' "${probe_output}" >&2
    exit 1
fi

cartesian_output="$(timeout 20s ros2 run panthera_vision_teleop mock_cartesian_probe)"
if ! rg -q '^PASS: lightweight FK/IK backend reached target, stale-held, fault-locked, and reset$' \
    <<<"${cartesian_output}"; then
    printf '错误：轻量笛卡尔 backend 探针没有成功。\n%s\n' "${cartesian_output}" >&2
    exit 1
fi

printf '%s\n' "${probe_output}"
printf '%s\n' "${cartesian_output}"
printf 'PASS：GenericSystem 连续控制、取消保持、失败锁定和显式恢复均可用。\n'
printf '模拟日志：%s\n' "${LAUNCH_LOG}"
