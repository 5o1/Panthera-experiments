#!/usr/bin/env bash

# Guarded real-hardware stages for the ros2_control candidate backend.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '请直接执行 %s，不要 source。\n' "$0" >&2
    return 2
fi
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-}"
launch_pid=""
bag_pid=""
launch_log=""

case "$STAGE" in hold|micro|cancel|suite|remaining) ;; *) printf '用法：%s hold|micro|cancel|suite|remaining\n' "$0" >&2; exit 2 ;; esac
# shellcheck source=tools/lib/process_cleanup.sh
source "${PROJECT_ROOT}/tools/lib/process_cleanup.sh"
cleanup() {
    trap - EXIT INT TERM
    stop_process_group "$bag_pid" INT
    stop_process_group "$launch_pid" TERM
}
trap cleanup EXIT INT TERM

if ! id -nG | tr ' ' '\n' | grep -qx dialout; then
    if [[ "${PANTHERA_HW_REEXEC:-0}" == 1 ]]; then
        printf '错误：无法取得 dialout 权限。\n' >&2
        exit 1
    fi
    printf -v command 'PANTHERA_HW_REEXEC=1 bash %q %q' "$0" "$STAGE"
    exec sg dialout -c "$command"
fi

"${PROJECT_ROOT}/tools/preflight_panthera.sh" hardware
source "${PROJECT_ROOT}/tools/activate_panthera.sh"

printf '真实 ros2_control 实验阶段：%s\n' "$STAGE"
printf '此入口与 arm_control_node 互斥，不会启动视觉；请让物理急停始终触手可及。\n'
printf '官方 hardware plugin 的 deactivate 尚未显式发送 stop，本阶段结束前会保持当前位置，然后关闭控制器。\n'
required="RUN-PANTHERA-${STAGE^^}"
read -r -p "输入 ${required} 才会继续：" confirmation
[[ "$confirmation" == "$required" ]] || { printf '已取消。\n'; exit 1; }

output="${PROJECT_ROOT}/bags/hardware_${STAGE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PROJECT_ROOT}/bags"
launch_log="${output}.launch.log"
setsid ros2 bag record -o "$output" \
    /joint_states \
    /dynamic_joint_states \
    /arm_controller/controller_state \
    /arm_controller/follow_joint_trajectory/_action/status \
    /teleop/experiment_marker &
bag_pid=$!

setsid ros2 launch panthera_vision_teleop continuous_control_hardware.launch.py \
    confirmation:=ENABLE-PANTHERA-ROS2-CONTROL >"$launch_log" 2>&1 &
launch_pid=$!
deadline=$((SECONDS + 45))
until rg -q 'Configured and activated.*arm_controller' "$launch_log"; do
    if rg -q 'Motor connection disconnected|目标位置: \[999|Serial error|Unable to open port' "$launch_log"; then
        printf '错误：硬件初始化出现电机断连、999 无效反馈或串口故障；拒绝发送 action。\n' >&2
        tail -n 80 "$launch_log" >&2
        exit 1
    fi
    kill -0 "$launch_pid" 2>/dev/null || { printf '错误：hardware launch 提前退出。\n' >&2; exit 1; }
    (( SECONDS < deadline )) || { printf '错误：等待 arm_controller active 超时。\n' >&2; exit 1; }
    # Startup-only polling: no trajectory exists yet and rosbag remains one
    # continuous file. Sampling phases themselves never use sleep boundaries.
    sleep 0.1
done

if [[ "$STAGE" == suite ]]; then
    stages=(hold micro cancel)
elif [[ "$STAGE" == remaining ]]; then
    stages=(micro cancel)
else
    stages=("$STAGE")
fi

for index in "${!stages[@]}"; do
    current_stage="${stages[$index]}"
    ros2 run panthera_vision_teleop hardware_trajectory_probe "$current_stage"
    printf '阶段 %s 已通过，bag 持续录制中。\n' "$current_stage"
done

stop_process_group "$bag_pid" INT
bag_pid=""
printf '连续硬件 case 结束，完整 bag：%s\n' "$output"
printf 'hardware launch 日志：%s\n' "$launch_log"
