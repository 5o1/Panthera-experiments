# colcon 源码目录

这里始终只维护自编写的 `panthera_vision_teleop`。Panthera 官方包不应放入本仓库，也不需要位于同一个 `src/`；它们由已经构建并 source 的官方 underlay 提供：

```text
underlay: Panthera_HT_ROS2/install
overlay:  panthera/ros2_ws/install
```

构建本 overlay 之前必须先执行 `source ~/Panthera_HT_ROS2/install/setup.bash`，否则 colcon 无法解析 `panthera_interfaces`。官方 Humble 源码归档：`docs/references/source/Panthera-HT_ROS2-humble.zip`。
