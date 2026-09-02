#!/usr/bin/env bash

# Panthera 真机位置往返测试（单终端）：
#   positionpark -> position0 -> positionpark
# position0：从 Host 前端采集并确认的默认启动关节位置。
# positionpark：六个关节角全为 0 的停放位置。
# 注意：本脚本会让真实机械臂运动。物理急停必须触手可及。

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '错误：请直接运行 ./run_manual_target_test.sh，不要使用 source。\n' >&2
    return 2
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
driver_pid=""

POSITION0_JOINTS="[0.0, 0.6016, 0.844, 0.0, 0.0, 0.0]"
POSITIONPARK_JOINTS="[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"

cleanup() {
    trap - EXIT
    if [[ -n "$driver_pid" ]] && kill -0 "$driver_pid" 2>/dev/null; then
        kill -INT "$driver_pid" 2>/dev/null || true
        wait "$driver_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'printf "\n测试已中断，正在安全关闭驱动……\n" >&2; exit 130' INT TERM

fail() {
    printf '错误：%s\n' "$1" >&2
    exit 1
}

wait_for_driver() {
    local deadline topics services
    deadline=$((SECONDS + 30))

    while (( SECONDS < deadline )); do
        if ! kill -0 "$driver_pid" 2>/dev/null; then
            wait "$driver_pid" || true
            fail '官方驱动在就绪前退出。'
        fi

        topics="$(ros2 topic list 2>/dev/null || true)"
        services="$(ros2 service list 2>/dev/null || true)"
        if grep -qx '/arm_status' <<<"$topics" \
            && grep -qx '/end_pose_euler' <<<"$topics" \
            && grep -qx '/move_to_joint' <<<"$services"; then
            return 0
        fi
        sleep 0.25
    done

    fail '等待机械臂话题和服务超时（30 秒）。'
}

if ! compgen -G '/dev/ttyACM*' >/dev/null; then
    fail 'WSL 中没有发现 /dev/ttyACM* 设备。'
fi

if ! id -nG | tr ' ' '\n' | grep -qx dialout; then
    if [[ "${PANTHERA_DIALOUT_REEXEC:-0}" == "1" ]]; then
        fail '无法自动取得 dialout 组权限；请确认当前用户已加入 dialout 组。'
    fi

    printf '当前终端尚未激活 dialout 权限，正在自动切换……\n'
    printf -v reexec_command 'PANTHERA_DIALOUT_REEXEC=1 bash %q' "${BASH_SOURCE[0]}"
    sg dialout -c "$reexec_command"
    exit $?
fi

# shellcheck source=tools/activate_panthera.sh
source "${PROJECT_ROOT}/tools/activate_panthera.sh"

printf '\n启动官方机械臂驱动（最大速度 0.1 rad/s）……\n'
ros2 launch panthera_arm_control arm_control.launch.py max_velocity:=0.1 &
driver_pid=$!
wait_for_driver

printf '\n当前机械臂状态：\n'
arm_status="$(ros2 topic echo /arm_status --once)"
printf '%s\n' "$arm_status"
if ! grep -q '^arm_enabled: true$' <<<"$arm_status"; then
    fail '机械臂未使能，不执行往返。'
fi
if ! grep -q '^motion_status: 0$' <<<"$arm_status"; then
    fail '机械臂当前不是空闲状态，不执行往返。'
fi
if ! grep -q "^error_message: ''$" <<<"$arm_status"; then
    fail '机械臂报告了错误，不执行往返。'
fi

printf '\n当前末端位姿：\n'
ros2 topic echo /end_pose_euler --once

printf '\n即将执行真实运动：\n'
printf '  positionpark -> position0\n'
printf '  position0    -> positionpark\n'
printf 'position0    = %s\n' "$POSITION0_JOINTS"
printf 'positionpark = %s\n' "$POSITIONPARK_JOINTS"
printf '\n确认机械臂周围无人、无障碍，且物理急停触手可及。\n'
read -r -p '输入 park-position0-park 才会开始：' confirmation
if [[ "$confirmation" != 'park-position0-park' ]]; then
    printf '已取消，没有发送运动命令。\n'
    exit 0
fi

printf '\n[1/2] 前往 position0……\n'
position0_response="$(
    ros2 service call /move_to_joint panthera_interfaces/srv/MoveToJoint \
        "{joint_angles: ${POSITION0_JOINTS}, velocity_scaling: 1.0, acceleration_scaling: 0.3}"
)"
printf '%s\n' "$position0_response"
if ! grep -Eq 'success[=:][[:space:]]*[Tt]rue' <<<"$position0_response"; then
    fail 'position0 动作未成功，已取消返回动作。'
fi

printf '\n[2/2] 返回 positionpark……\n'
park_response="$(
    ros2 service call /move_to_joint panthera_interfaces/srv/MoveToJoint \
        "{joint_angles: ${POSITIONPARK_JOINTS}, velocity_scaling: 1.0, acceleration_scaling: 0.3}"
)"
printf '%s\n' "$park_response"
if ! grep -Eq 'success[=:][[:space:]]*[Tt]rue' <<<"$park_response"; then
    fail '返回 positionpark 失败。'
fi

printf '\n往返完成。最终末端位姿：\n'
ros2 topic echo /end_pose_euler --once
