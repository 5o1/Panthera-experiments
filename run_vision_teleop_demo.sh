#!/usr/bin/env bash

# One-terminal runner for the monocular Panthera demo.
#   ./run_vision_teleop_demo.sh synthetic   # no camera, no robot (default)
#   ./run_vision_teleop_demo.sh camera      # real camera, fake robot feedback, dry-run
#   ./run_vision_teleop_demo.sh robot       # legacy blocking driver
#   ./run_vision_teleop_demo.sh continuous  # Host-style non-blocking real control
# Optional second argument: safe (default), normal/extended/envelope (dry-run only),
# or one of the dedicated depth/orientation dry-run profiles.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '请直接执行 ./run_vision_teleop_demo.sh，不要 source。\n' >&2
    return 2
fi

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-synthetic}"
PROFILE="${2:-safe}"
PROFILE_CONFIG="${ROOT}/ros2_ws/src/panthera_vision_teleop/config/teleop_${PROFILE}.yaml"
VISION_PYTHON="${ROOT}/.venv-wsl-vision/bin/python"
pipeline_pid=""
driver_pid=""
hardware_pid=""
hardware_log=""
interrupted=0
normal_visual_exit=0
POSITION0='[0.0, 0.6016, 0.844, 0.0, 0.0, 0.0]'
POSITIONPARK='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
DRIVER_MAX_VELOCITY=0.1
MOVE_VELOCITY_SCALING=1.0

# shellcheck source=tools/lib/process_cleanup.sh
source "${ROOT}/tools/lib/process_cleanup.sh"

fail() {
    printf '错误：%s\n' "$1" >&2
    exit 1
}

stop_process() {
    local pid="${1:-}"
    # Background jobs inherit SIGINT ignored from Bash. SIGTERM is handled by
    # ROS launch/rclcpp. ``setsid`` makes pid the process-group id.
    stop_process_group "$pid" TERM
}

move_joint() {
    local name="$1" joints="$2"
    printf '机械臂缓动到 %s……\n' "$name"
    local response
    response="$(timeout 25 ros2 service call /move_to_joint panthera_interfaces/srv/MoveToJoint \
        "{joint_angles: ${joints}, velocity_scaling: ${MOVE_VELOCITY_SCALING}, acceleration_scaling: 0.3}")" || return 1
    printf '%s\n' "$response"
    grep -Eq 'success[=:][[:space:]]*[Tt]rue' <<<"$response"
}

move_joint_continuous() {
    local name="$1" joints="$2"
    printf '机械臂缓动到 %s……\n' "$name"
    ros2 run panthera_vision_teleop joint_trajectory_move \
        --target "${joints//[\[\] ]/}" \
        --max-velocity 0.6 --max-acceleration 2.0 --tolerance 0.05
}

cleanup() {
    # Once shutdown starts, a second Ctrl+C must not interrupt the park/driver
    # cleanup half-way and leave arm_control_node owning the serial ports.
    trap - EXIT
    trap '' INT TERM
    stop_process "$pipeline_pid"
    local park_verification_failed=0
    if [[ "$MODE" == continuous && "$normal_visual_exit" == 1 && -n "$hardware_pid" ]] && kill -0 -- "-$hardware_pid" 2>/dev/null; then
        move_joint_continuous positionpark "$POSITIONPARK" || park_verification_failed=1
    elif [[ "$MODE" == continuous && -n "$hardware_pid" ]]; then
        printf '异常/中断退出：不追加 positionpark；正在关闭连续控制器。\n' >&2
    elif [[ "$MODE" == robot && "$normal_visual_exit" == 1 && -n "$driver_pid" ]] && kill -0 -- "-$driver_pid" 2>/dev/null; then
        if /usr/bin/python3 "${ROOT}/tools/wait_arm_idle.py" --timeout 25; then
            if move_joint positionpark "$POSITIONPARK"; then
                /usr/bin/python3 "${ROOT}/tools/verify_joint_target.py" \
                    --target '0,0,0,0,0,0' --tolerance 0.05 --stable-samples 5 --timeout 5 \
                    || park_verification_failed=1
            else
                printf '警告：未能自动返回 positionpark；请检查机械臂状态。\n' >&2
                park_verification_failed=1
            fi
        else
            printf '警告：机械臂未在期限内进入无故障空闲状态，不追加 positionpark。\n' >&2
            park_verification_failed=1
        fi
    elif [[ "$MODE" == robot && -n "$driver_pid" ]]; then
        printf '异常/中断退出：为避免意外续动，不再下发 positionpark；正在关闭驱动并抱闸。\n' >&2
    fi
    stop_process "$driver_pid"
    stop_process "$hardware_pid"
    if [[ "$normal_visual_exit" == 1 && "$park_verification_failed" == 1 ]]; then
        printf '错误：positionpark 反馈未通过容差验收；本次 case 判定失败。\n' >&2
        exit 1
    fi
}

trap cleanup EXIT
trap 'interrupted=1; exit 130' INT TERM

case "$MODE" in
    synthetic|camera|robot|continuous) ;;
    *) fail '模式只能是 synthetic、camera、robot 或 continuous。' ;;
esac

case "$PROFILE" in
    safe|normal|extended|envelope|depth_world|orientation_roll|orientation_pitch|orientation_yaw) ;;
    *) fail '配置档只能是 safe、normal、extended、envelope、depth_world、orientation_roll、orientation_pitch 或 orientation_yaw。' ;;
esac
[[ -f "$PROFILE_CONFIG" ]] || fail "找不到配置档：${PROFILE_CONFIG}"
if [[ "$MODE" == robot || "$MODE" == continuous ]] && [[ "$PROFILE" != safe ]]; then
    if [[ "$PROFILE" != normal || "${PANTHERA_ALLOW_NORMAL_ROBOT:-0}" != 1 || -z "${PANTHERA_EXPERIMENT_PLAN:-}" ]]; then
        fail 'robot 模式的非 safe 配置只能由已审查的 normal 数据采集 case 启动。'
    fi
fi
if [[ "$MODE" == robot && "$PROFILE" == normal ]]; then
    # Teleoperation uses the upstream driver's 0.5 rad/s default. Startup and park retain the previous
    # 0.1 rad/s effective speed through the service scaling below.
    DRIVER_MAX_VELOCITY=0.5
    MOVE_VELOCITY_SCALING=0.2
fi

[[ -x "$VISION_PYTHON" ]] || fail "找不到视觉环境：${VISION_PYTHON}"

if [[ "$MODE" != synthetic ]] && ! compgen -G '/dev/video*' >/dev/null; then
    fail 'WSL 中没有 /dev/video*；请先用 usbipd attach 摄像头，再重试。'
fi

if [[ "$MODE" != synthetic ]] && ! id -nG | tr ' ' '\n' | grep -qx video; then
    if [[ "${PANTHERA_VIDEO_REEXEC:-0}" == 1 ]]; then
        fail '无法取得 video 设备权限；请确认当前用户已加入 video 组。'
    fi
    printf '当前终端尚未激活 video 权限，正在自动切换……\n'
    printf -v command 'PANTHERA_VIDEO_REEXEC=1 bash %q %q %q' "$0" "$MODE" "$PROFILE"
    exec sg video -c "$command"
fi

export MPLCONFIGDIR="${ROOT}/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
"$VISION_PYTHON" -c 'import cv2, mediapipe, websockets' || fail '视觉 Python 依赖不完整。'

if [[ "$MODE" == robot || "$MODE" == continuous ]] && ! id -nG | tr ' ' '\n' | grep -qx dialout; then
    if [[ "${PANTHERA_DIALOUT_REEXEC:-0}" == 1 ]]; then
        fail '无法取得 dialout 设备权限。'
    fi
    printf -v command 'PANTHERA_DIALOUT_REEXEC=1 bash %q %q %q' "$0" "$MODE" "$PROFILE"
    exec sg dialout -c "$command"
fi

# shellcheck source=tools/activate_panthera.sh
source "${ROOT}/tools/activate_panthera.sh"
preflight_mode="$MODE"
[[ "$MODE" == synthetic ]] && preflight_mode=mock
[[ "$MODE" == continuous ]] && preflight_mode=robot
"${ROOT}/tools/preflight_panthera.sh" "$preflight_mode"
export ROS_LOG_DIR="${ROOT}/log/vision_demo"
mkdir -p "$ROS_LOG_DIR"

publish_robot=false
fake_robot=true
use_continuous_backend=false
sender_args=(--server ws://127.0.0.1:8765)
if [[ -n "${PANTHERA_EXPERIMENT_PLAN:-}" ]]; then
    [[ -f "$PANTHERA_EXPERIMENT_PLAN" ]] || fail "找不到实验计划：${PANTHERA_EXPERIMENT_PLAN}"
    sender_args+=(--experiment-plan "$PANTHERA_EXPERIMENT_PLAN")
    [[ "$MODE" == robot || "$MODE" == continuous ]] && sender_args+=(--experiment-require-enabled)
fi

if ss -ltn 2>/dev/null | grep -qE '[:.]8765[[:space:]]'; then
    fail '端口 8765 已被占用；请先关闭旧的视觉 Demo，避免连接到错误的 bridge。'
fi

if [[ "$MODE" == synthetic ]]; then
    sender_args+=(--synthetic)
elif [[ "$MODE" == camera ]]; then
    printf 'CAMERA DRY-RUN：会读取真实摄像头，但 /pos_cmd 发布者不会被创建。\n'
else
    [[ -e /dev/ttyACM0 ]] || fail '找不到 Panthera /dev/ttyACM0。'
    printf '\n即将运行真实机械臂视觉遥操作：\n'
    printf '  启动后先缓动到 position0；正常退出后返回 positionpark。\n'
    printf '  摄像头窗口中按 Space 才启用跟随，R 重新标定，Esc 正常退出。\n'
    printf '  Ctrl+C 属于中断退出：停止发新目标并关闭驱动，不自动走停放轨迹。\n'
    printf '确认机械臂周围无人、无障碍，且物理急停触手可及。\n'
    read -r -p '输入 ENABLE-VISION-ROBOT 才会继续：' confirmation
    [[ "$confirmation" == ENABLE-VISION-ROBOT ]] || fail '已取消。'

    existing_nodes="$(ros2 node list --no-daemon 2>/dev/null || true)"
    grep -qx '/arm_control_node' <<<"$existing_nodes" && fail '已有 arm_control_node 在运行，请先关闭，避免争用串口。'
    if [[ "$MODE" == continuous ]]; then
        hardware_log="${ROS_LOG_DIR}/continuous_hardware.log"
        setsid ros2 launch panthera_vision_teleop continuous_control_hardware.launch.py \
            confirmation:=ENABLE-PANTHERA-ROS2-CONTROL >"$hardware_log" 2>&1 &
        hardware_pid=$!
        deadline=$((SECONDS + 45))
        until rg -q 'Configured and activated.*arm_velocity_limit_controller' "$hardware_log"; do
            if rg -q 'Motor connection disconnected|目标位置: \[999|Serial error|Unable to open port' "$hardware_log"; then
                tail -n 80 "$hardware_log" >&2
                fail '连续硬件后端初始化失败。'
            fi
            kill -0 "$hardware_pid" 2>/dev/null || fail '连续硬件后端在就绪前退出。'
            (( SECONDS < deadline )) || fail "等待连续控制器就绪超时；日志：${hardware_log}"
            sleep 0.1
        done
        move_joint_continuous position0 "$POSITION0" || fail '未能到达 position0，不启动视觉遥操作。'
        use_continuous_backend=true
        fake_robot=false
    else
        setsid ros2 launch panthera_arm_control arm_control.launch.py max_velocity:="$DRIVER_MAX_VELOCITY" &
        driver_pid=$!
        if ! /usr/bin/python3 "${ROOT}/tools/wait_ros_service.py" \
            /move_to_joint panthera_interfaces/srv/MoveToJoint --timeout 35; then
            kill -0 "$driver_pid" 2>/dev/null || fail '机械臂驱动在就绪前退出。'
            fail '等待机械臂驱动服务超时。'
        fi
        kill -0 "$driver_pid" 2>/dev/null || fail '机械臂驱动在服务就绪后意外退出。'
        move_joint position0 "$POSITION0" || fail '未能到达 position0，不启动视觉遥操作。'
        publish_robot=true
        fake_robot=false
    fi
fi

setsid ros2 launch panthera_vision_teleop vision_teleop.launch.py \
    use_websocket:=true use_debug:=false \
    fake_robot_feedback:="$fake_robot" \
    profile_config:="$PROFILE_CONFIG" \
    publish_robot_commands:="$publish_robot" \
    use_continuous_backend:="$use_continuous_backend" \
    publish_gripper_commands:=false &
pipeline_pid=$!

kill -0 "$pipeline_pid" 2>/dev/null || fail 'ROS 视觉链路启动失败。'

printf '\n视觉发送端启动。\n'
if [[ "$MODE" == synthetic ]]; then
    printf '这是合成数据；按 Ctrl+C 结束。另开终端可观察 /teleop/debug_target。\n'
elif [[ "$MODE" == camera && -n "${PANTHERA_EXPERIMENT_PLAN:-}" ]]; then
    printf '相机采集不控制机械臂；确认 pose OK 和 RECORDER READY 后，只按一次 Enter。\n'
else
    printf 'Space：启用/禁用 | R：重新标定 | Esc：正常退出\n'
fi
"$VISION_PYTHON" "${ROOT}/windows_vision/vision_sender.py" "${sender_args[@]}"
normal_visual_exit=1
