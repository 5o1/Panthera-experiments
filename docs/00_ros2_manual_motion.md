# 阶段 0：用 ROS 2 让 Panthera 微小移动一次

这一阶段只验证一件事：自己写的 ROS 2 节点能读取当前末端位姿，并向 Panthera 官方直接驱动节点发送一次微小的相对位置目标。

暂时不接摄像头、WebSocket、人体姿态、夹爪和连续遥操。

## 1. 本阶段文件

```text
ros2_ws/src/panthera_vision_teleop/
├── panthera_vision_teleop/
│   └── manual_target_node.py  # 你要填写的 ROS 2 节点
├── config/
│   └── manual_test.yaml       # 默认 dry-run 的参数
├── setup.py                   # ros2 run 入口
└── package.xml                # ROS 2 依赖
```

数据流：

```text
panthera_arm_control
  ├── /end_pose_euler ──> manual_target_node
  ├── /arm_status ───────> manual_target_node
  └── /pos_cmd <───────── manual_target_node（显式允许后只发一次）

manual_target_node
  └── /teleop/manual_target_preview（dry-run 始终可以查看）
```

## 2. 开始前

- 机械臂必须牢固固定，运动范围内没有人和障碍物。
- 物理停止手段必须触手可及，另开一个终端准备调用 `/stop_srv`。
- 使用 Ubuntu 22.04、ROS 2 Humble 和已经构建的 Panthera 官方工作区。
- 第一次只移动一个轴 1 cm，不测试姿态，不测试连续发送。
- 软件停止不能替代物理急停。

确认 WSL 能看到设备：

```bash
ls -l /dev/ttyACM*
```

## 3. 启动官方驱动

终端 A：

```bash
source /opt/ros/humble/setup.bash
source ~/Panthera_HT_ROS2/install/setup.bash

ros2 launch panthera_arm_control arm_control.launch.py max_velocity:=0.1
```

这里把关节速度上限从官方默认的 `0.5 rad/s` 降到 `0.1 rad/s`。不要在首次试动中使用默认速度。

终端 B：

```bash
source /opt/ros/humble/setup.bash
source ~/Panthera_HT_ROS2/install/setup.bash

ros2 topic echo /arm_status --once
ros2 topic echo /end_pose_euler --once
ros2 topic info /pos_cmd -v
ros2 interface show panthera_interfaces/msg/PosCmd
```

只有在状态反馈存在、没有故障，并且你理解了 `PosCmd` 字段后继续。

停止命令放在独立终端备用：

```bash
ros2 service call /stop_srv std_srvs/srv/Trigger "{}"
```

当前官方 `arm_control_node` 使用单线程执行器，而到点运动回调会阻塞等待。因此运动进行时，`/stop_srv` 可能无法立刻得到处理。它不是安全级急停，也不能作为唯一停止手段；必须优先准备机械臂或控制盒的物理停止方式。

## 4. 手动填写节点

打开 `manual_target_node.py`，按文件顶部的 TODO 依次实现。最小职责只有四项：

1. 订阅 `/end_pose_euler` 和 `/arm_status`。
2. 用当前 XYZ 加上 `delta_xyz`，当前 RPY 原样复制。
3. 把目标发布到 `/teleop/manual_target_preview`。
4. 只有 `send_robot_command=true` 且状态合法时，向 `/pos_cmd` 发布一次。

不要在这一阶段加入键盘控制、timer 连续发布、夹爪控制或视觉输入。

## 5. 构建 overlay

终端 C：

```bash
source /opt/ros/humble/setup.bash
source ~/Panthera_HT_ROS2/install/setup.bash

cd ~/panthera/ros2_ws
colcon build --symlink-install --packages-select panthera_vision_teleop
source install/setup.bash
```

## 6. 先做 dry-run

保持 `manual_test.yaml` 中：

```yaml
send_robot_command: false
delta_xyz: [0.01, 0.0, 0.0]
```

运行节点：

```bash
ros2 run panthera_vision_teleop manual_target_node \
  --ros-args \
  --params-file ~/panthera/ros2_ws/src/panthera_vision_teleop/config/manual_test.yaml
```

另一个终端查看预览：

```bash
ros2 topic echo /teleop/manual_target_preview --once
```

手工确认：

- 只有一个位置轴改变 `0.01 m`。
- RPY 与 `/end_pose_euler` 完全一致。
- `gripper` 是 `-1.0`。
- `mode1` 是 `0`。
- `/pos_cmd` 没有本节点的 publisher。

## 7. 允许一次真机发布

先明确使能：

```bash
ros2 service call /enable_srv panthera_interfaces/srv/Enable \
  "{enable_request: true}"
```

确认现场安全后，用命令行覆盖参数，不修改 YAML 的安全默认值：

```bash
ros2 run panthera_vision_teleop manual_target_node \
  --ros-args \
  --params-file ~/panthera/ros2_ws/src/panthera_vision_teleop/config/manual_test.yaml \
  -p send_robot_command:=true
```

节点必须只发送一次，然后记录 `SENT` 并拒绝再次发送。观察机械臂是否完成约 1 cm 的移动，再查看：

```bash
ros2 topic echo /end_pose_euler --once
ros2 topic echo /arm_status --once
```

## 8. 完成条件

- dry-run 目标计算正确。
- 默认运行不会发送 `/pos_cmd`。
- 真机模式每次进程最多发送一条命令。
- 机械臂只在一个方向移动约 1 cm。
- 失能、运动中或故障状态下不发送命令。
- 你能解释 subscriber、publisher、parameter 和 `--once`/单次发送的区别。

完成后再进入 `debug_pose_publisher.py` 和 `teleop_mapper_node.py`，不要直接跳到 10 Hz 连续遥操。
