#!/usr/bin/env bash

# One long, continuously recorded vision case with frame-driven phase markers.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '请直接执行 %s，不要 source。\n' "$0" >&2
    return 2
fi
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_NAME="${1:-}"
bag_pid=""

case "$CASE_NAME" in
    camera_characterization)
        MODE=camera
        PROFILE=safe
        PLAN="${PROJECT_ROOT}/config/experiments/camera_characterization.json"
        ;;
    camera_yz_envelope)
        MODE=camera
        PROFILE=envelope
        PLAN="${PROJECT_ROOT}/config/experiments/camera_yz_envelope.json"
        ;;
    robot_yz_acceptance)
        MODE=robot
        PROFILE=safe
        PLAN="${PROJECT_ROOT}/config/experiments/robot_yz_acceptance.json"
        ;;
    robot_yz_normal_acceptance)
        MODE=robot
        PROFILE=normal
        PLAN="${PROJECT_ROOT}/config/experiments/robot_yz_normal_acceptance.json"
        ;;
    robot_yz_fast_response)
        # This is the single final acceptance case for the Host-style
        # latest-only ros2_control path; it is not a speed-search step.
        MODE=continuous
        PROFILE=normal
        PLAN="${PROJECT_ROOT}/config/experiments/robot_yz_fast_response.json"
        ;;
    *)
        printf '用法：%s camera_characterization|camera_yz_envelope|robot_yz_acceptance|robot_yz_normal_acceptance|robot_yz_fast_response\n' "$0" >&2
        exit 2
        ;;
esac

# shellcheck source=tools/lib/process_cleanup.sh
source "${PROJECT_ROOT}/tools/lib/process_cleanup.sh"
# shellcheck source=tools/lib/vision_bag_topics.sh
source "${PROJECT_ROOT}/tools/lib/vision_bag_topics.sh"
cleanup() {
    trap - EXIT INT TERM
    stop_process_group "$bag_pid" INT
}
trap cleanup EXIT INT TERM

source "${PROJECT_ROOT}/tools/activate_panthera.sh"
output="${PROJECT_ROOT}/bags/${CASE_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PROJECT_ROOT}/bags"
setsid ros2 bag record -o "$output" "${VISION_BAG_TOPICS[@]}" &
bag_pid=$!

printf '单例采集 case：%s\n一次启动会完成全部阶段并连续写入同一个 bag：%s\n' "$CASE_NAME" "$output"
if [[ "$MODE" == camera ]]; then
    printf '只需为第一阶段按一次 Enter；之后倒计时 6 秒自动换阶段。正式采集按有效视觉帧计数，不使用 shell sleep。\n'
else
    printf '每阶段只按一次 Space；程序会原子装载阶段并启用跟随。预热/采集按有效视觉帧推进；失跟会暂停计数，不使用 shell sleep。\n'
fi
export PANTHERA_EXPERIMENT_PLAN="$PLAN"
if [[ "$PROFILE" == normal && ( "$MODE" == robot || "$MODE" == continuous ) ]]; then
    export PANTHERA_ALLOW_NORMAL_ROBOT=1
fi
"${PROJECT_ROOT}/run_vision_teleop_demo.sh" "$MODE" "$PROFILE"

stop_process_group "$bag_pid" INT
bag_pid=""
cp -- "$PLAN" "$output/experiment_plan.json"
cp -- "$PROJECT_ROOT/ros2_ws/src/panthera_vision_teleop/config/teleop_${PROFILE}.yaml" \
    "$output/profile_used.yaml"
/usr/bin/python3 "${PROJECT_ROOT}/tools/analyze_vision_teleop.py" "$output"
printf 'case 完成；bag 和分析结果位于：%s\n' "$output"
