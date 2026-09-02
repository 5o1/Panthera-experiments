#!/usr/bin/env bash

# Activate the ROS 2 Humble + Panthera workspaces.
#
# Run directly to open a ready-to-use Panthera shell:
#   ./tools/activate_panthera.sh
#
# Or source it to activate the current shell:
#   source tools/activate_panthera.sh

set -e

ROS_SETUP="/opt/ros/humble/setup.bash"
PANTHERA_DRIVER_SETUP="${HOME}/Panthera_HT_ROS2/install/setup.bash"
PROJECT_SETUP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/ros2_ws/install/setup.bash"
PROJECT_PACKAGE_PREFIX="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/ros2_ws/install/panthera_vision_teleop"

for setup_file in "$ROS_SETUP" "$PANTHERA_DRIVER_SETUP" "$PROJECT_SETUP"; do
    if [[ ! -f "$setup_file" ]]; then
        printf '错误：找不到环境文件：%s\n' "$setup_file" >&2
        printf '请先完成对应工作区的 colcon build。\n' >&2
        return 1 2>/dev/null || exit 1
    fi
done

# ROS 2 Humble 的 setup.bash 不兼容 nounset；加载后恢复调用者的设置。
had_nounset=0
[[ $- == *u* ]] && had_nounset=1
set +u
source "$ROS_SETUP"
source "$PANTHERA_DRIVER_SETUP"
source "$PROJECT_SETUP"
# This workspace currently has one isolated ament_python package.  colcon's
# generic Python develop task installs its ament marker but does not add the
# isolated prefix to AMENT_PREFIX_PATH, so make ROS package discovery explicit.
if [[ -d "$PROJECT_PACKAGE_PREFIX/share/ament_index" ]]; then
    case ":${AMENT_PREFIX_PATH:-}:" in
        *":${PROJECT_PACKAGE_PREFIX}:"*) ;;
        *) export AMENT_PREFIX_PATH="${PROJECT_PACKAGE_PREFIX}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}" ;;
    esac
fi
(( had_nounset )) && set -u

export PANTHERA_ENV_ACTIVE=1

printf 'Panthera 环境已激活：ROS %s | %s\n' "${ROS_DISTRO:-unknown}" "$PROJECT_SETUP"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '已进入 Panthera 终端；输入 exit 返回原终端。\n'
    export PS1="(panthera) ${PS1:-\u@\h:\w\$ }"
    exec bash --noprofile --norc -i
fi
