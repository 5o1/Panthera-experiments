# 连续数据采集单例 Case 设计

目标是减少反复启动、Codex/权限 review 和设备重新初始化造成的延迟，同时不把互斥驱动或
未经验收的自由度塞进同一次真机运动。采集窗口内不使用 shell `sleep`。

## 1. 可以合并的三类长流程

| 单例 case | 一次启动内包含 | 阶段推进依据 | 驱动/输出 |
|---|---|---|---|
| `camera_characterization` | 静止、前后、左右、上下、屈伸、roll/pitch/yaw、捏合 | 第一阶段按一次 Enter；之后每阶段间倒计时 6 秒并自动推进 | camera dry-run；无 `/pos_cmd` |
| `camera_yz_envelope` | 静止、最大幅度左右、最大幅度上下、最大 YZ 圆 | 第一阶段按一次 Enter；之后倒计时 6 秒自动推进 | 1:1 Y/Z、无实际限幅/滤波的 camera dry-run；无 `/pos_cmd` |
| `robot_yz_acceptance` | 静止、Y 往返、Z 往返、YZ 小圆、停止保持 | 每阶段一次 Space 原子装载并启用；只有启用且有效的 pose 帧才计数 | 旧 `arm_control_node`，仅 safe Y/Z，夹爪/深度/姿态关闭 |
| `robot_yz_normal_acceptance` | normal 静止、较大 Y、较大 Z、较大 YZ 圆、停止保持 | 每阶段一次 Space 原子装载并启用；正常退出验证 park 关节反馈 | 旧 `arm_control_node`，normal Y/Z，仍关闭夹爪/深度/姿态 |
| `robot_yz_fast_response` | 自然速度的大 YZ 圆、停止保持 | 两阶段各一次 Space；正常退出验证 park 关节反馈 | normal 工作空间；50 Hz latest-only backend；200 Hz ros2_control；Host 0.6/1.0 rad/s、2.0 rad/s²边界 |
| `ros2_control suite` | 当前位保持、joint1 0.005 rad 往返、0.015 rad 慢轨迹取消 | ROS action 成功/取消结果；阶段结束后人工输入继续词 | 新 ros2_control 候选；不启动视觉 |

计划中的有效帧预算如下。这里的秒数只用于帮助操作者预估体力，不用于代码切段：

| case | 正式采集帧 | 预热帧 | 15 Hz 下纯有效帧时间 |
|---|---:|---:|---:|
| `camera_characterization` | 2460 | 285 | 约 3 分 3 秒 |
| `robot_yz_acceptance` | 1380 | 150 | 约 1 分 42 秒 |

实际总时长还包括操作者在阶段边界的准备时间；这些等待也保留在同一个 bag 中。帧率下降或
短暂失跟只会延长总时长，不会减少阶段内样本数。

对应命令：

```bash
./tools/run_data_collection_case.sh camera_characterization
./tools/run_data_collection_case.sh camera_yz_envelope
./tools/run_data_collection_case.sh robot_yz_acceptance
./tools/run_data_collection_case.sh robot_yz_normal_acceptance
./tools/run_data_collection_case.sh robot_yz_fast_response
./tools/run_continuous_hardware_experiment.sh suite
```

前两个 case 从开始到结束只创建一个 rosbag。每个视觉包携带
`/teleop/experiment_marker`，形式为 `phase:STATE:remaining_valid_frames`。离线分析器自动按
`RECORDING` 区间分段，输出每阶段统计。warmup 和 recording 只消耗满足置信度阈值的有效
帧，所以失跟、帧率波动或操作者尚未入位不会偷偷耗完固定时间。
case 正常结束后还会把本次实际使用的 JSON 复制为 bag 内的 `experiment_plan.json`，避免
以后修改默认计划后无法还原旧数据的分段条件。

发送端还要求 bridge 的 marker publisher 已发现 rosbag 订阅者，HUD 必须显示
`RECORDER READY` 才允许开始阶段（相机 case 用 Enter，真机 case 用 Space）。录包订阅者中途消失会暂停帧计数并把真机输出锁回
`DISABLED`；硬件 action 探针在 START marker 无订阅者时直接拒绝运动。

每个真机视觉阶段完成时发送端自动锁回 `DISABLED`，但不关闭驱动、不停止 bridge、也不
切断 rosbag。操作者看完上一阶段状态后，为下一阶段只按一次 Space；发送端先确认 pose 与
录包均就绪，再在同一次按键事件中装载阶段并启用跟随。这样阶段内部是连续轨迹，阶段边界
也不会出现“机械臂已经运动、实验 marker 却还没开始”的未标记区间。

camera-only case 不存在机器人输出，因此采用自动阶段推进：完成一个阶段后 HUD 显示下一
动作说明并倒计时 6 秒。倒计时在帧循环中用单调时钟检查，不调用 `sleep`，所以视频与 rosbag
持续运行。全程只需首次 Enter，不用十次返回终端或重复确认。该设置只存在于 camera JSON，
robot JSON 不启用。正式阶段仍按有效 pose 帧计数，以保证每段可用样本数一致。

“正常双手打字”不在当前验收范围：唯一的摄像头位于笔记本屏幕顶部，正常打字时手腕必然离开画面，
无法同时满足姿态跟踪的右肩/右肘/右腕可见性要求。这是当前硬件几何下的不适用项，不要求添购摄像头。

连续硬件 suite 只启动一次 `ros2_control_node`，并把三个 action 阶段写入同一 bag。每个
阶段的 START/PASS/FAIL 由探针直接发布 marker；只有前一阶段自动 PASS 才会进入下一阶段，
中途异常立即退出。整个 suite 只在开始时要求一次总确认；已通过 hold 后也可用
`remaining` 将 micro/cancel 合并为一次会话。

## 2. 不能合并的部分

- `arm_control_node` 与 `ros2_control_node` 会争用同一组串口，必须是两个独立 case，中间
  确认前一个进程完全退出。
- camera 表征数据和 robot Y/Z 数据不能只采一份：前者需要自由地做深度、伸直、打字等
  动作，后者必须保持 safe profile 和急停专注度。
- X、roll/pitch/yaw、夹爪的**真实输出**不能并入首次 Y/Z case。它们可以在 camera case
  一次采齐观测数据，但必须在离线审查后分别建立单自由度真机 case。
- USB 重挂、断电重启、故障恢复会改变外部状态，本来就必须形成独立 case，不能为了减少
  review 而隐藏在正常轨迹中。
- position0 启动和 positionpark 正常退出可以合并在每个长视觉 case 的首尾；异常退出不
  自动追加 park，避免在未知状态下续动。

## 3. 哪些等待不再使用 sleep

- 采集长度：有效视觉帧计数。
- 失跟：帧计数暂停，状态机等待新消息并要求显式 re-arm。
- 机械臂动作完成：`FollowJointTrajectory` result 或 `/arm_status` 空闲消息。
- stale 停止：目标 monotonic 时间戳和 ROS timer，超时后 action cancel/hold。
- 阶段切换：camera case 首次 Enter、真机视觉 case 单次 Space、硬件 suite 的 action result。
- rosbag：在整个 case 中持续运行，不在每个动作前后启动/停止。
- rosbag 就绪/掉线：DDS subscription count，未就绪不允许进入采集或硬件动作。

旧驱动的 `/move_to_joint` 就绪、动作结束和退出前空闲状态均通过 ROS graph/message 事件
等待，不再用 shell `sleep`。仍可能在启动/退出辅助代码中看到很短的轮询等待（例如
controller manager CLI 探测或给进程组优雅退出）；这些发生在轨迹采集窗口之外，不决定
样本边界，也不会切断连续轨迹。若对应组件以后提供 ready/shutdown 事件，可再替换掉这些
轮询。
