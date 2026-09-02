#!/usr/bin/env bash

# Explicit recovery command: move the real arm to the user-defined all-zero park pose.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '请直接执行 %s，不要 source。\n' "$0" >&2
    return 2
fi
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
driver_pid=""
started_driver=0

cleanup() {
    trap - EXIT INT TERM
    if [[ "$started_driver" == 1 && -n "$driver_pid" ]] && kill -0 -- "-$driver_pid" 2>/dev/null; then
        kill -TERM -- "-$driver_pid" 2>/dev/null || true
        wait "$driver_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if ! id -nG | tr ' ' '\n' | grep -qx dialout; then
    if [[ "${PANTHERA_PARK_REEXEC:-0}" == 1 ]]; then
        printf '错误：无法取得 dialout 权限。\n' >&2
        exit 1
    fi
    printf -v command 'PANTHERA_PARK_REEXEC=1 bash %q' "$0"
    exec sg dialout -c "$command"
fi

"${PROJECT_ROOT}/tools/preflight_panthera.sh" hardware
source "${PROJECT_ROOT}/tools/activate_panthera.sh"
printf '即将把真实机械臂缓动到 positionpark（六关节全 0）。\n'
printf '请确认周围无人、无障碍，物理急停触手可及。\n'
read -r -p '输入 PARK-PANTHERA 才会继续：' confirmation
[[ "$confirmation" == PARK-PANTHERA ]] || { printf '已取消。\n'; exit 1; }

if ! ros2 node list --no-daemon 2>/dev/null | grep -qx '/arm_control_node'; then
    setsid ros2 launch panthera_arm_control arm_control.launch.py max_velocity:=0.1 &
    driver_pid=$!
    started_driver=1
fi

deadline=$((SECONDS + 35))
until ros2 service list --no-daemon 2>/dev/null | grep -qx '/move_to_joint'; do
    [[ "$started_driver" == 0 ]] || kill -0 "$driver_pid" 2>/dev/null || { printf '错误：驱动提前退出。\n' >&2; exit 1; }
    (( SECONDS < deadline )) || { printf '错误：等待 /move_to_joint 超时。\n' >&2; exit 1; }
    sleep 0.25
done

response="$(timeout 25 ros2 service call /move_to_joint panthera_interfaces/srv/MoveToJoint \
    '{joint_angles: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], velocity_scaling: 1.0, acceleration_scaling: 0.3}')"
printf '%s\n' "$response"
grep -Eq 'success[=:][[:space:]]*[Tt]rue' <<<"$response" || { printf '错误：positionpark 未成功。\n' >&2; exit 1; }
printf 'positionpark 已到达。\n'
