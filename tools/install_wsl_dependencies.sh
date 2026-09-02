#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROS_DISTRO="humble"
readonly ROS_APT_SOURCE_VERSION="1.2.0"
readonly ROS_APT_SOURCE_DEB="/tmp/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.jammy_all.deb"
readonly ROS_APT_SOURCE_URL="https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.jammy_all.deb"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VISION_VENV="${REPO_ROOT}/.venv-wsl-vision"

if [[ "$(uname -s)" != "Linux" ]] || ! grep -qi microsoft /proc/sys/kernel/osrelease; then
  echo "Error: this installer must run inside WSL 2." >&2
  exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_CODENAME:-}" != "jammy" ]]; then
  echo "Error: ROS 2 Humble binary packages require Ubuntu 22.04 (Jammy)." >&2
  exit 1
fi

if [[ "${EUID}" -eq 0 ]]; then
  echo "Error: run this script as your normal user; it will invoke sudo when needed." >&2
  exit 1
fi

echo "[1/7] Requesting sudo once"
sudo -v

echo "[2/7] Installing repository prerequisites"
sudo env DEBIAN_FRONTEND=noninteractive apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates curl locales software-properties-common
sudo locale-gen en_US en_US.UTF-8

echo "[3/7] Configuring the official ROS 2 apt repository"
curl -fL "${ROS_APT_SOURCE_URL}" -o "${ROS_APT_SOURCE_DEB}"
sudo dpkg -i "${ROS_APT_SOURCE_DEB}"
sudo env DEBIAN_FRONTEND=noninteractive apt-get update

echo "[4/7] Installing ROS 2 Humble, development tools, and camera utilities"
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ros-humble-desktop \
  ros-dev-tools \
  ros-humble-moveit \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-controller-manager \
  ros-humble-robot-state-publisher \
  ros-humble-rviz2 \
  ros-humble-xacro \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-gazebo-ros-pkgs \
  libyaml-cpp-dev \
  libserialport-dev \
  libeigen3-dev \
  libboost-all-dev \
  liburdfdom-dev \
  python3-numpy \
  python3-scipy \
  python3-websockets \
  python3-opencv \
  python3-pip \
  python3-venv \
  python3-pytest \
  v4l-utils \
  ffmpeg \
  usbutils

echo "[5/7] Initializing rosdep"
if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi
rosdep update

echo "[6/7] Creating the WSL vision environment"
/usr/bin/python3 -m venv --system-site-packages "${VISION_VENV}"
"${VISION_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VISION_VENV}/bin/python" -m pip install -r "${REPO_ROOT}/windows_vision/requirements.txt" pytest
# Jammy's websockets 9.1 still passes asyncio's removed loop= argument on
# Python 3.10. ROS nodes use /usr/bin/python3, so install the compatible
# release for that interpreter as well as inside the vision venv.
/usr/bin/python3 -m pip install --user 'websockets>=10.4,<12'

echo "[7/7] Granting the current user camera and serial-device access"
sudo usermod -aG video,dialout "${USER}"

# ROS setup scripts intentionally read optional environment variables. Temporarily
# disable nounset so this installer can keep its stricter setting everywhere else.
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
ros2 --help >/dev/null
colcon --help >/dev/null
"${VISION_VENV}/bin/python" -c \
  'import cv2, mediapipe, numpy, scipy, websockets; print("WSL vision imports: OK")'

echo
echo "Installation complete."
echo "ROS setup: source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "Vision env: source ${VISION_VENV}/bin/activate"
echo "Restart the WSL session before using newly granted video/dialout groups."
