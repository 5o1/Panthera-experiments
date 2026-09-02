# panthera_vision_teleop

ROS 2 包内的职责边界：

- `vision_bridge_node`：WebSocket JSON → 已验证的 `/teleop/*` 观测。
- `teleop_mapper_node`：标定、映射、滤波、限幅以及 dry-run/真机发布边界。
- `debug_pose_publisher`：不依赖相机或真机的确定性输入与假反馈。
- `manual_target_node`：阶段 0 的单次小位移实验。
- `vision_protocol`、`rotation_utils`、`teleop_logic`：没有 ROS 依赖的可测试核心。
- `kinematics`、`cartesian_trajectory_backend`：可替换的轻量 FK/Jacobian/IK 连续候选。

直接启动 ROS 合成输入：

```bash
ros2 launch panthera_vision_teleop vision_teleop.launch.py \
  use_websocket:=false use_debug:=true fake_robot_feedback:=true
```

启动真实视觉 bridge、但仍使用假机器人反馈：

```bash
ros2 launch panthera_vision_teleop vision_teleop.launch.py \
  use_websocket:=true use_debug:=false fake_robot_feedback:=true \
  publish_robot_commands:=false publish_gripper_commands:=false
```

`publish_robot_commands=false` 时 mapper 不会创建 `/pos_cmd` publisher。夹爪还有独立的
`publish_gripper_commands` 安全开关；只有两个参数都为 `true` 时才创建
`/gripper_cmd`。第一版真机 Demo 固定关闭夹爪命令，因为实测自然放松手可能被通用
模型识别成 `Open_Palm`。

映射 profile 通过 launch 的 `profile_config` 叠加，不修改基础配置。safe/normal、单目深度和
三个单轴姿态档均位于 `config/`；除 safe 外都只允许 dry-run。连续控制的 mock 与硬件
overlay 分开，硬件 launch 还要求确认词，且绝不能与 `arm_control_node` 同时运行。
