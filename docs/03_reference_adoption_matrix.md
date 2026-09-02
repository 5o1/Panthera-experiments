# Panthera 开源参考采纳矩阵

更新日期：2026-09-02

## 1. 使用原则

开源项目在本项目中只提供三类信息：已经被代码验证过的设计方法、已知失败模式，以及可
供比较的实现。它们不是需求、依赖或架构约束。每一项都必须重新回答：它解决的是不是我
们的问题，输入条件是否相同，是否适合 Panthera 驱动，是否能通过我们的测试。答案改变
时可以立即修改或弃用。

矩阵中的状态含义如下：

- **保留**：本项目已经采用，且当前测试或真机现象支持它。
- **准备采用**：适合当前问题，但必须先在 dry-run/模拟中完成验收。
- **研究分支**：有潜力，尚未证明收益大于复杂度，不进入默认真机链路。
- **暂缓**：条件不足，后续增加传感器或底层接口后再判断。
- **弃用**：行为与当前目标或安全原则冲突，不复制。

## 2. 参考方案到本项目的映射

| 前人工作中的方法 | 解决的问题 | 本项目对应位置 | 当前决定 | 采用或弃用依据 |
|---|---|---|---|---|
| MoveIt Servo 的新鲜命令超时和平滑停止 | 输入中断后不执行旧速度 | 连续控制 backend | 当前弃用实现，仅保留语义 | Servo 未安装；项目内轻量 FK/IK backend 已通过相同 mock 门槛，尚无收益证据支持增加依赖 |
| `openarm_teleoperation` 的腕位移→Y/Z 速度 | 避免离散目标逐点停顿 | `teleop_mapper_node` 后的运动生成层 | 研究分支 | 相对速度适合单目 Y/Z，但回中才能停止，可能产生漂移；需与相对位置控制对比 |
| `openarm_teleoperation` 的逐轴 deadband | 静止视觉噪声不触发运动 | `teleop_logic.MappingConfig` | 保留 | 已实现连续边界死区并有单元测试 |
| 失跟自动回 home、重新检测后自动恢复 | 原项目试图恢复可操作状态 | 启停流程 | 弃用 | 失跟时环境信息最差；当前要求锁存禁用并显式重新 enable |
| MIRROR 的 30/500/1000 Hz 分频循环 | 视觉帧率不能直接承担电机控制 | 视觉、目标生成器、驱动三层 | 准备采用原则 | 具体频率由 Panthera 实测决定，不照搬数值 |
| MIRROR 的 INIT/IDLE/TRACKING/STOP 状态机 | 状态转换和动作生成混在回调中难以保证安全 | 未来 teleop FSM | 准备采用结构，自行定义状态 | 我们需要 DISABLED/CALIBRATING/ARMED/TRACKING/STALE_LOCK/FAULT/PARKING |
| MIRROR 的人物 ID 锁定 | 多人画面中控制对象突然切换 | 摄像头感知层 | 暂缓 | 当前 MediaPipe 配置只取一个 pose 且没有稳定 ID；多人支持前必须另做身份策略 |
| MIRROR 的大跳变低权吸收 | 减少关键点瞬时跳跃 | 感知滤波 | 弃用原行为 | 错误样本会缓慢污染目标；真机更适合拒绝并在连续异常后锁定 |
| humanoid-arm-retarget 的 base/tool 两级标定 | 传感器坐标轴和夹爪工具轴不一致 | `position_axis_matrix`、未来 TF/工具标定 | 准备扩展 | 当前只有固定轴矩阵，需把相机安装和工具朝向分成两个明确变换 |
| 优化 IK 的上一帧 warm start 和连续性代价 | IK 在多个可行解之间跳动 | 未来 IK/Servo 层 | 准备采用原则 | 官方 KDL 已用当前反馈作 seed，但没有显式关节变化代价；先观察 Servo/MoveIt 行为 |
| 单帧最大关节变化拒绝 | 感知或 IK 异常导致关节跳跃 | 连续 backend 输出门 | 准备采用 | 阈值必须按时间换算成 rad/s、rad/s²，不能复制其他机器人的固定数值 |
| Engage/Detach clutch 和冷却时间 | 操作者明确取得/释放控制权 | Space enable、未来深度/姿态 clutch | 保留并扩展 | 当前显式 enable 已存在；X 和姿态应有独立 clutch 而非自动开启 |
| dex-retargeting 的相对关键点向量 | 人手和机器人手大小、位置不同 | 夹爪输入 | 准备采用简化版 | 使用归一化 thumb-index `pinch_ratio`，不引入完整手部优化器 |
| 双阈值迟滞、保持时间和冷却 | 自然手势造成夹爪反复/误触 | `GestureDebouncer` 替代方案 | 准备采用 | 真机夹爪仍关闭；先做数据录制和 5 分钟 dry-run |
| mediapipe_ros2_suite 的 ROS Image 感知架构 | 感知可录制、回放、替换 | Windows/WSL WebSocket 边界 | 暂缓整体迁移 | 当前隔离解决了 Python ABI 和设备问题；只借鉴源时间戳、原始关键点和回放能力 |
| AnyTeleop 的 RGB 弱透视深度网络 | 从单 RGB 近似全局三维腕位置 | X 前后轴 | 研究分支 | 需要额外模型和本机数据验证；在此之前 X 保持关闭或使用 clutch 的相对 2.5D |
| AnyTeleop 的 25 Hz 目标→120 Hz 碰撞约束轨迹 | 稀疏感知目标变成连续安全运动 | 连续 backend | 准备采用原则 | 频率和生成器可更换，但关节限位、碰撞和时间连续性必须有等价实现 |
| 直接把人体角度映射到机器人少数关节 | 不依赖图像尺度且无需 IK | 可选“动作模仿”模式 | 暂缓 | 不满足当前末端跟随目标；可作为教学/故障降级模式，而非主控制方式 |

## 3. Panthera 官方驱动审计

### 3.1 已确认能力

当前使用的 `panthera_arm_control/arm_control_node.cpp` 在
`executeJointTarget()` 中调用：

```cpp
robot_->posVelMaxTorque(joint_values, velocities, max_torque_, true, ...);
```

最后的 `true` 要求 SDK 一直等待目标到达，所以 `/pos_cmd`、`arm_joint_cmd` 和相关服务
回调会阻塞。阻塞不是 SDK 的硬限制。SDK 的 `Panthera::posVelMaxTorque()` 默认
`is_wait=false`，非等待模式在发送一次电机命令后立即返回。

官方仓库还包含 `panthera_hardware/PantheraHardwareInterface`。它导出关节 position、
velocity 和 effort 状态，在 `ros2_control` 的 `write()` 中使用
`posVelMaxTorque(..., false)` 或 MIT 五参数接口。官方 controller manager 配置的更新率
为 100 Hz，并提供 `JointTrajectoryController` 和 50 Hz 关节状态发布。因此，Panthera
存在真正连续控制的底层基础，不必永远受 `/pos_cmd` 限制。

相关源码位于：

- `/home/assaneko/Panthera_HT_ROS2/src/panthera_arm_control/src/arm_control_node.cpp`
- `/home/assaneko/Panthera_HT_ROS2/src/hightorque_robot/include/panthera/Panthera.hpp`
- `/home/assaneko/Panthera_HT_ROS2/src/hightorque_robot/src/panthera/Panthera.cpp`
- `/home/assaneko/Panthera_HT_ROS2/src/panthera_hardware/src/panthera_hardware_interface.cpp`

### 3.2 不能直接上真机的原因

审计和模拟启动发现以下问题：

1. 官方 `demo.launch.py` 生成的 spawner 请求 `arm_controller`，但其默认
   `ros2_controllers.yaml` 定义的是 `panthera_arm_controller`。实测 mock 启动因此报
   `type param was not defined for arm_controller`。
2. `hardware.launch.py` 使用的非夹爪 ros2_control xacro 只声明 position command，配套
   `ros2_controllers_hardware.yaml` 却要求 position 和 velocity，控制器可能无法激活。
   带夹爪的另一份 xacro 才同时声明这两个接口。
3. `PantheraHardwareInterface::on_deactivate()` 只记录日志，没有发送 stop 或当前位置保持。
4. hardware interface 没有独立的“上游命令过期”看门狗；若控制器退出、DDS 断开或命令
   停止，必须验证实际保持/停止行为，不能从 100 Hz 更新率推断安全。
5. MoveIt `joint_limits.yaml` 没有启用加速度限制；硬件 xacro 中的 1.0 rad/s 最大速度与
   MoveIt 文件中的 3.7–5.0 rad/s 也不是同一层含义，需要统一并采用较保守值。
6. 当前机器安装了 MoveIt move_group，但没有安装 `moveit_servo`。Servo 仍只是候选。
7. `arm_control_node` 与 `ros2_control_node` 都会独占同一组串口，不能同时启动。

这些问题说明官方仓库提供了能力原型，但没有提供已经通过本项目安全验收的连续真机链路。

## 4. 本次建立的独立验证

本项目新增了一个硬编码假硬件的测试分支：

- `continuous_control_mock.launch.py` 只加载官方 mock URDF 中的
  `mock_components/GenericSystem`，没有真实硬件开关和 SDK 配置参数。
- `continuous_control_mock.yaml` 统一使用 `arm_controller` 和六个 joint name，只导出
  position command。
- `tools/test_continuous_control_mock.sh` 启动隔离 DDS domain，等待控制器激活，发送一条
  `FollowJointTrajectory`，要求结果同时满足 `error_code: 0` 和 `SUCCEEDED`，最后清理
  整个 launch 进程组。

2026-09-01 的扩展测试结果为 PASS：标准 action 成功、不完整关节目标不得成功、取消后
保持；项目内 URDF FK/Jacobian/阻尼 IK backend 也能到达小笛卡尔目标，并在目标 stale 后
取消/保持。这只证明 GenericSystem 通路；没有证明 Panthera hardware plugin、串口时序、
停止延迟或真机关节反馈可用。

本机 Host 仓库进一步给出了可直接采用的运行基线：主位置控制循环 200 Hz，通常使用
0.6 rad/s，配置的每关节速度上限为 1.0 rad/s、加速度上限为 2.0 rad/s²，并持续调用
`Joint_Pos_Vel(..., iswait=False)`。因此本项目已停止通过真机逐档搜索速度。

项目现新增 `continuous_cartesian_backend`：它以 50 Hz 将最新笛卡尔目标变成受限关节流，
controller manager 以 200 Hz 写入硬件。Panthera SDK 的 `vel` 参数语义是正的最大速度，
不是有符号的关节速度，因此 position trajectory 和六关节 0.6 rad/s velocity-cap 分别由
两个 controller 占用不同 command interface，避免错误地把轨迹导数传给 SDK。

## 5. 当前候选架构和否决门槛

曾考虑候选 A：

```text
视觉相对目标 → MoveIt Servo → JointTrajectory → arm_controller
             → PantheraHardwareInterface → SDK 非阻塞命令
```

当前优先候选 B：

```text
视觉相对目标 → 自研轻量差分 IK/目标生成器 → JointTrajectory → 同一硬件层
```

当前不安装 Servo；若以后需要自碰撞/场景碰撞或轻量 IK 的可达性不足，再以相同 mock 指标
重新比较。两者共用 `ros2_control` 硬件层。出现以下
任一情况，可以直接放弃 `ros2_control`/Servo 路线并改造直接 SDK 节点：

- hardware plugin 无法稳定保持 100 Hz，或串口请求/反馈造成明显周期阻塞；
- action cancel/命令过期不能在规定时间内停止；
- 控制器切换、退出或异常时不能确定地保持当前位置或制动；
- 官方关节状态顺序、单位或方向与 URDF 不一致；
- 为修复硬件 plugin 所需改动超过直接实现专用 latest-only 驱动的复杂度；
- MoveIt Servo 的计算、依赖或调参成本没有带来可测量的平滑性和安全收益。

在真机接入前必须依次通过：mock trajectory、MoveIt/Servo mock、hardware plugin 只读反馈、
当前位置保持、极小单关节轨迹、取消/超时、故障恢复，最后才允许视觉目标进入该 backend。
