# Panthera-HT 单目视觉遥操作 Demo

这个项目已实现一条可 review、可离线测试、默认不触碰真机的链路：

```text
WSL 摄像头 → MediaPipe Pose/Gesture → WebSocket 最新状态
           → vision_bridge_node → /teleop/*
           → teleop_mapper_node → /teleop/debug_target
                                → continuous_cartesian_backend
                                → 50 Hz 最新关节目标
                                → 200 Hz ros2_control → SDK iswait=false
                                → /pos_cmd（保留的旧阻塞式对照路径）
                                → /gripper_cmd（额外独立开关，首版关闭）
```

第一版真机跟随只启用左右 Y 和上下 Z。真实相机验收发现单目深度会让机器人 X
前后轴明显抖动，因此 X 默认锁定；激活瞬间的机械臂朝向也保持不变。旋转数学已经
实现并有测试，但配置中的 `enable_orientation` 默认是 `false`，应在位置控制稳定后
单独开启。

## 最快开始

先运行完全不需要相机和机械臂的端到端合成 Demo：

```bash
cd ~/panthera
./run_vision_teleop_demo.sh synthetic
```

它会启动 WebSocket bridge、假机器人反馈、mapper 和合成视觉发送器。默认 `publish_robot_commands=false`，ROS 图中不会创建 `/pos_cmd` 发布者。另一个终端可以观察：

```bash
source ~/panthera/tools/activate_panthera.sh
ros2 topic echo /teleop/debug_target
ros2 topic echo /teleop/debug_gripper
```

假机器人默认只提供固定反馈，因此不会伪装成已经执行真实命令。需要专门检查“一次只发
一条、到达后再发下一条”的调度逻辑时，可用下面的纯软件模式；它会创建 `/pos_cmd`，
但订阅方是假机器人，不会启动官方驱动或访问串口：

```bash
ros2 launch panthera_vision_teleop vision_teleop.launch.py \
  use_websocket:=false use_debug:=true fake_robot_feedback:=true \
  publish_robot_commands:=true fake_robot_follow_commands:=true
```

预览 HUD 和 `/teleop/diagnostics/*` 会显示/发布状态机、人体位移、目标位移、workspace
裁剪、速度限制和姿态保持。需要保留一次实验时，在 Demo 运行期间另开终端执行：

```bash
cd ~/panthera
./tools/record_vision_teleop.sh my_test
```

录制结果保存到 `bags/`；该脚本本身不会启动或控制机械臂。

录制结束后生成统计、CSV 和曲线：

```bash
./tools/analyze_vision_teleop.py bags/实际目录名
```

需要连续采完多个动作而不反复 review/启动时，使用单例 case：

```bash
./tools/run_data_collection_case.sh camera_characterization
./tools/run_data_collection_case.sh robot_yz_acceptance
```

相机 case 每阶段按 Enter；真机 case 再按 Space 显式启用。之后由有效帧计数推进；阶段完成
会自动禁用输出，但驱动、视觉模型和 rosbag 都保持运行，一个 case 只生成一个带 phase
marker 的 bag。

摄像头已经通过 usbipd 挂进 WSL、能够看到 `/dev/video*` 后：

```bash
./run_vision_teleop_demo.sh camera safe
```

这仍是 dry-run。预览窗口按键：Space 启用/禁用，R 重新标定，X 进入/退出深度 clutch，
Esc 正常退出。可选 dry-run profile 为 `normal`、`extended`、`depth_world`、`orientation_roll`、
`orientation_pitch`、`orientation_yaw`；脚本明确禁止它们进入 robot 模式。

旧阻塞驱动仍可用于已经完成的 safe 对照实验：

```bash
./run_vision_teleop_demo.sh robot safe
```

真机模式还要求输入完整确认词。它启动官方驱动，先缓动到 `position0`；发送器初始保持禁用，必须在窗口按 Space 才开始跟随。Esc 正常退出会返回 `positionpark`。Ctrl+C 被视作异常中断，不再追加停放动作，而是停止视觉链路并关闭驱动。清理开始后会忽略第二次 Ctrl+C，避免停放或驱动关闭被中途打断。

日常连续控制不再靠逐档真机试速。项目采用 Host 已使用的参数：常用关节速度
`0.6 rad/s`、硬件上限 `1.0 rad/s`、加速度上限 `2.0 rad/s²`。视觉只更新最新目标，
项目 backend 以 50 Hz 生成受限关节流，`ros2_control` 以 200 Hz 调用 SDK 的
`iswait=false` 非阻塞接口。完整真机验收使用一个连续录包 case：

```bash
./tools/run_data_collection_case.sh robot_yz_fast_response
```

它只在开始时要求一次总确认；不是逐档寻找速度的实验。

## 建议的 review 顺序

1. `vision_protocol.py`：JSON 的版本、类型、有限值和递增序号检查。
2. `rotation_utils.py`：肩肘腕坐标系、ROS 四元数、RPY、Slerp 和限角。
3. `teleop_logic.py`：零点标定、坐标轴映射、增益、硬限幅、EMA、手势去抖。
4. `vision_bridge_node.py`：网络线程和 ROS executor 如何隔离，断线 0.5 秒后如何禁用。
5. `teleop_mapper_node.py`：dry-run/真机边界以及“最多一个在途命令”。
6. `windows_vision/vision_sender.py`：相机帧怎样经过两个 MediaPipe Tasks 模型变成协议包。
7. `run_vision_teleop_demo.sh`：进程生命周期、设备权限、position0/positionpark。

实时遥操作刻意使用“latest-only”而不是 FIFO 动作队列。连续路径不会等待一个点完成；
目标过期后发送当前位置保持并要求重新启用。旧 `/pos_cmd` 路径仍维持“一次一个在途点”，
只作为历史对照和回退。

## 构建与测试

```bash
source /opt/ros/humble/setup.bash
source ~/Panthera_HT_ROS2/install/setup.bash
cd ~/panthera/ros2_ws
colcon build --symlink-install --packages-select panthera_vision_teleop

cd ~/panthera
PYTHONPATH=ros2_ws/src/panthera_vision_teleop \
  /usr/bin/python3 -m pytest -q ros2_ws/src/panthera_vision_teleop/test
```

未来连续控制 backend 的纯软件通路可以单独验证。下面的脚本硬编码使用
`mock_components/GenericSystem`，不会加载 Panthera SDK 或访问串口：

```bash
./tools/test_continuous_control_mock.sh
./tools/test_continuous_streaming_mock.sh
```

它同时测试标准 FollowJointTrajectory 和项目内轻量 URDF FK/Jacobian/IK backend。当前不把
MoveIt Servo 作为依赖；若以后需要其碰撞能力，可以重新以同一组 mock 指标比较。

完整纯软件验收还包括：

```bash
./tools/preflight_panthera.sh mock
./tools/test_process_cleanup.sh
./tools/test_vision_pipeline_mock.sh
./tools/test_blocking_scheduler_mock.sh
```

参数集中在 `ros2_ws/src/panthera_vision_teleop/config/teleop.yaml`。完整设计、坐标约定和分阶段真机验收见：

- [ROS 2 最小试动](docs/00_ros2_manual_motion.md)
- [下一阶段 TODO](docs/01_next_stage_vision_teleop_todo.md)
- [开源遥操作实现调查](docs/02_open_source_teleoperation_research.md)
- [开源参考采纳矩阵与驱动审计](docs/03_reference_adoption_matrix.md)
- [真机实验检查表](docs/04_real_hardware_experiment_checklist.md)
- [无需真机的软件验收基线](docs/05_software_acceptance.md)
- [连续数据采集单例 Case 设计](docs/06_data_collection_cases.md)
- [单目视觉实验计划](<docs/Panthera-HT 单目视觉手臂与手势遥操作实验计划.md>)
- [依赖列表](<docs/Panthera-HT 单目视觉遥操作依赖列表.md>)

这仍是教学 Demo，不是工业安全控制器。视觉失联会锁定后续目标，恢复跟踪后仍需重新
启用；但官方驱动正在执行的单条阻塞命令无法由该视觉锁立即中断，因此不能替代物理急停。
