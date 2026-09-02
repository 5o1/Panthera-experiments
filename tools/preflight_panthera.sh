#!/usr/bin/env bash

# Read-only environment checks. This script never launches a driver or sends a command.

set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-mock}"
failures=0
warnings=0

pass() { printf 'PASS  %s\n' "$1"; }
warn() { printf 'WARN  %s\n' "$1"; warnings=$((warnings + 1)); }
fail() { printf 'FAIL  %s\n' "$1"; failures=$((failures + 1)); }

case "$MODE" in
    mock|camera|hardware|robot) ;;
    *) printf '用法：%s [mock|camera|hardware|robot]\n' "$0" >&2; exit 2 ;;
esac

if [[ -r /opt/ros/humble/setup.bash ]]; then
    # Do not enable nounset until after ROS setup: Humble reads optional variables.
    set +u
    source /opt/ros/humble/setup.bash
    [[ -r /home/assaneko/Panthera_HT_ROS2/install/setup.bash ]] && source /home/assaneko/Panthera_HT_ROS2/install/setup.bash
    [[ -r "${PROJECT_ROOT}/ros2_ws/install/setup.bash" ]] && source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
    project_prefix="${PROJECT_ROOT}/ros2_ws/install/panthera_vision_teleop"
    if [[ -d "${project_prefix}/share/ament_index" ]]; then
        export AMENT_PREFIX_PATH="${project_prefix}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
    fi
    set -u
else
    fail '缺少 /opt/ros/humble/setup.bash'
fi

command -v ros2 >/dev/null 2>&1 && pass "ros2: $(command -v ros2)" || fail 'ros2 命令不可用'
[[ -x "${PROJECT_ROOT}/.venv-wsl-vision/bin/python" ]] && pass '视觉虚拟环境存在' || fail '缺少 .venv-wsl-vision'
[[ -f "${PROJECT_ROOT}/docs/references/models/pose_landmarker_full.task" ]] && pass 'Pose 模型存在' || fail '缺少 Pose 模型'
[[ -f "${PROJECT_ROOT}/docs/references/models/gesture_recognizer.task" ]] && pass 'Gesture 模型存在' || fail '缺少 Gesture 模型'

if [[ -x "${PROJECT_ROOT}/.venv-wsl-vision/bin/python" ]]; then
    if "${PROJECT_ROOT}/.venv-wsl-vision/bin/python" -c 'import cv2, mediapipe, websockets, numpy' >/dev/null 2>&1; then
        pass '视觉 Python 依赖可导入'
    else
        fail '视觉 Python 依赖导入失败'
    fi
fi

if command -v ros2 >/dev/null 2>&1; then
    ros2 pkg prefix panthera_vision_teleop >/dev/null 2>&1 && pass 'panthera_vision_teleop 已构建' || fail 'panthera_vision_teleop 未构建或未 source'
    ros2 pkg prefix panthera_ht_config >/dev/null 2>&1 && pass '官方 Panthera description/config 可见' || fail 'panthera_ht_config 不可见'
    ros2 pkg prefix joint_trajectory_controller >/dev/null 2>&1 && pass 'JointTrajectoryController 插件包可见' || fail 'JointTrajectoryController 插件包不可见'
fi

if [[ "$MODE" == camera || "$MODE" == robot ]]; then
    if compgen -G '/dev/video*' >/dev/null; then
        videos=(/dev/video*)
        pass "WSL 摄像头节点：${videos[*]}"
        [[ -r "${videos[0]}" ]] && pass "当前用户可读 ${videos[0]}" || fail "当前用户不可读 ${videos[0]}（需 video 组）"
    else
        fail 'WSL 中没有 /dev/video*'
        warn '在 Windows 管理员 PowerShell 运行 usbipd list；随后 usbipd bind --busid BUSID 和 usbipd attach --wsl --busid BUSID'
    fi
    id -nG | tr ' ' '\n' | grep -qx video && pass '当前会话具有 video 组' || fail '当前会话没有 video 组'
fi

if [[ "$MODE" == hardware || "$MODE" == robot ]]; then
    if command -v lsusb >/dev/null 2>&1 && lsusb | grep -qi 'caf1:ffff'; then
        pass '检测到 caf1:ffff Livelybot 通信板'
    else
        fail '未检测到 caf1:ffff Livelybot 通信板'
        warn '若 usbipd list 显示 Shared 而不是 Attached，请在 Windows PowerShell 重新执行 usbipd attach --wsl --busid BUSID'
    fi
    serial_count=0
    for index in 0 1 2 3 4 5 6; do
        device="/dev/ttyACM${index}"
        if [[ -c "$device" ]]; then
            serial_count=$((serial_count + 1))
            [[ -r "$device" && -w "$device" ]] || fail "当前用户不可读写 ${device}（需 dialout 组）"
        else
            fail "缺少 ${device}"
        fi
    done
    [[ "$serial_count" -eq 7 ]] && pass '七个 Panthera 串口均存在' || true
    id -nG | tr ' ' '\n' | grep -qx dialout && pass '当前会话具有 dialout 组' || fail '当前会话没有 dialout 组'
    if pgrep -af 'arm_control_node|ros2_control_node' | grep -v "preflight_panthera" >/dev/null 2>&1; then
        fail '检测到可能占用机械臂的旧驱动进程'
        pgrep -af 'arm_control_node|ros2_control_node' || true
    else
        pass '没有检测到旧机械臂驱动进程'
    fi
fi

printf '\n预检结果：%d 项失败，%d 项警告（模式：%s）。\n' "$failures" "$warnings" "$MODE"
if (( failures > 0 )); then
    exit 1
fi
