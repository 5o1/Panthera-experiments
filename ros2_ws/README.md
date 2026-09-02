# ROS 2 学习工作区

本目录是这个 GitHub 仓库自己的 ROS 2 overlay workspace，只保存 `panthera_vision_teleop`，不包含 Panthera 官方驱动，也不需要把本包复制到官方仓库。

第一次学习不要直接启动视觉遥操。先完成仓库根目录的 `docs/00_ros2_manual_motion.md`，实现一个默认 dry-run、只允许单次发布的微小位移节点。

构建顺序：

1. 在 WSL Ubuntu 22.04 中单独解压或克隆并构建 Panthera 官方 Humble 仓库。
2. source `/opt/ros/humble/setup.bash` 和官方工作区的 `install/setup.bash`，使它成为 underlay。
3. 回到本仓库的 `ros2_ws`，用 `rosdep` 检查依赖，再执行 `colcon build --symlink-install`。
4. source 本工作区的 `install/setup.bash`。每个新终端都必须保持 ROS → 官方 Panthera → 本项目的 source 顺序。

```bash
source /opt/ros/humble/setup.bash
cd ~/Panthera_HT_ROS2
colcon build
source install/setup.bash

cd ~/panthera/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

这种布局使本项目拥有独立的仓库、README、Issue、提交历史和 CI，同时仍能解析 `panthera_interfaces` 并与官方驱动通信。

本地参考资料：

```text
docs/references/source/Panthera-HT_ROS2-humble.zip
docs/references/docs/ros2_humble_install_ubuntu.html
docs/references/docs/ros2_create_package.html
docs/references/docs/ros2_python_pub_sub.html
```
