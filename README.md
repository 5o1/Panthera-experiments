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
- [历史双臂探索记录（不得作为当前任务定义）](docs/07_dual_arm_vla_sim_to_real.md)
- [单 Panthera 圆柱入槽：Lab 实验阶段报告（2026-09-13）](docs/08_lab_experiment_progress_2026-09-13.md)
- [单 Panthera 圆柱入槽：仿真与真实场景对齐](docs/09_sim_real_scene_alignment.md)
- [OpenVLA 闭环失败分析与续训门禁](docs/10_openvla_closed_loop_failure_analysis.md)
- [随机圆柱入槽 v2 单侧 Pilot（已否决）](docs/11_panthera_v2_dataset_pilot_2026-09-16.md)
- [随机圆柱入槽 v2 左右对称 Pilot（顿挫，已否决）](docs/12_panthera_v2_symmetric_dataset_pilot_2026-09-16.md)
- [随机圆柱入槽 v2 连续轨迹 Pilot（实际 qpos 复核后已否决）](docs/13_panthera_v2_smooth_dataset_pilot_2026-09-16.md)
- [随机圆柱入槽 v2 单次抓取、连续搬运与直接释放修复](docs/14_panthera_v2_single_grasp_direct_release_2026-09-16.md)
- [随机圆柱入槽 v2 schema 10 正式数据集](docs/15_panthera_v2_formal_dataset_2026-09-16.md)
- [schema 10 仿真数据集生成过程报告（HTML）](docs/panthera_v2_dataset_generation_report_zh.html)
- [schema 10 仿真数据集生成过程报告（PDF）](docs/panthera_v2_dataset_generation_report_zh.pdf)
- [随机圆柱入槽 v2 无人值守训练流水线](docs/16_panthera_v2_unattended_training_pipeline_2026-09-16.md)
- [随机圆柱入槽 v2 扩充数据集与异步落盘流水线](docs/17_panthera_v2_expanded_dataset_pipeline_2026-09-17.md)
- [闭环评测台缺陷排查](docs/18_eval_harness_defects_2026-09-19.md)
- [配置与目录架构](docs/19_architecture_plan_2026-09-19.md)
- [评测相机与训练观测错配](docs/20_eval_observation_mismatch_2026-09-19.md)
- [RoboTwin 上游迁移到 6dde571](docs/21_upstream_migration_2026-09-20.md)
- [单轨迹过拟合门禁收口](docs/22_single_trajectory_overfit_gate_hardening_2026-09-20.md)
- [OpenPI π0.5 迁移与同口径基线](docs/23_openpi_pi05_migration_baseline_2026-09-21.md)
- [OpenVLA 28 维动力学 proprio 过拟合实验](docs/24_openvla_dynamics_proprioception_overfit_2026-09-21.md)
- [单目视觉实验计划](<docs/Panthera-HT 单目视觉手臂与手势遥操作实验计划.md>)
- [依赖列表](<docs/Panthera-HT 单目视觉遥操作依赖列表.md>)

这仍是教学 Demo，不是工业安全控制器。视觉失联会锁定后续目标，恢复跟踪后仍需重新
启用；但官方驱动正在执行的单条阻塞命令无法由该视觉锁立即中断，因此不能替代物理急停。

## 外部实验台摄像头

WSL 中可启动 SRT listener：

```bash
./tools/receive_srt_camera.sh 9000
```

最新解码画面位于 `/tmp/panthera-srt-camera/latest.jpg`，最近约两分钟保存在 12 个循环
MKV 分段中。当前实验网络已将 Windows `192.168.31.30:9000/udp` 限定开放给
`192.168.31.0/24`，并转发到 WSL listener。摄像头 caller 使用：

```text
srt://192.168.31.30:9000?mode=caller&latency=200000
```

当前没有额外的 VLA 专用相机，手机 SRT 被选为唯一策略相机候选并兼作监控相机。固定安装
和重新标定完成前，它只能用于离线开发和监看。`latest.jpg` 是 2 FPS 的监控输出，不是
策略接口；在线 VLA 必须使用连续解码、内存 latest-only、帧龄检查和断流锁存，详见
[仿真与真实场景对齐](docs/09_sim_real_scene_alignment.md)。

当前 Lab 的 VPN 路由只存在于 Windows 网络栈。WSL/Codex 通过统一入口执行远程命令：

```bash
./tools/lab_ssh.sh nvidia-smi
```

该入口使用普通用户 `lyy`。Lab 上的日常安装、开发和计算不得使用管理员账户；只有驱动、
系统包、重启或权限修复确实需要提权时，才临时使用 `mole`。

Lab 的无 Docker VLA 环境固定放在 `/data/lyy/panthera-vla/`。Miniforge 安装在
`/data/lyy/tools/miniforge3/`，RLinf 使用其中独立的 Python 3.11 prefix。可恢复构建和
登录后激活入口分别是：

当前装配任务严格定义为**单台 Panthera、单个夹爪、一个圆柱插入凹槽**。竖直插入任务
使用 schema v4，对外是 7 维状态/绝对动作（六关节 + 夹爪）。第一版 phone-SRT 相机已经
通过物理 oracle、但因构图只显示中央末端而被视觉门禁否决；它的 128 集只作诊断并已归档，
不进入正式训练。宽视角 v3 已通过固定与随机化 oracle 各 4/4 以及抽帧构图审计；正式 128 集
已完成采集、合并、schema/随机化/视频验收和 episode 0/112 无附着回放，TFDS 3.0.0、独立
归一化统计与一步优化器 smoke 也已通过。首个 5000 步 SFT 已完成，但修正 RLinf 未加载
连续 L1 action head 的评测错误后，训练种子与未见种子双轨闭环仍为 0/2。遥测证明执行层
正常。无损续训到 20000 步后验证 L1 最低为 `0.04076`，但双轨仍为 0/2，并收敛在演示
开头的静止段；把 reset 夹爪预置为 0.9 也没有解除该固定点。数据审计确认 5 帧动作窗口在
该段看不到后续手臂运动。`25x7` RLDS、一步优化器 smoke 和 GPU1–3 上的 10000 步训练
均已完成；短执行视野扫描把失败定位到搬运倾斜和槽口释放。当前候选采用 `18/25` 重规划：
VLA 完成抓取和初始抬升，末端升到标定插入高度后，由只读取标定槽位、夹爪开度与末端
反馈的分阶段技能接管。技能按“短抬升—槽外横移—净空抬升—定姿—对齐—插入—稳定—
松爪—退夹”执行；已知规划/插入失败 seed 的定向回归为 4/4。30k 高学习率续训发生策略
坍缩，开发集只有 1/16，已否决。15k 低学习率模型的纯 VLA 和延迟释放接管分别为 4/16、
11/16；固定偏置 7 mm、分段松爪和浅插入候选均未增加成功 seed，已恢复 4.5 mm 基线。
从稳定 15k 模型以 `1e-5` 续训到 20k 已完成，验证 Next-Actions L1 从 `0.03578` 降到
`0.02537`，但闭环 dev16 只有 2/16，说明离线 loss 改善没有转化为闭环能力。门禁已阻止
final seed 300001–300016 运行。2026-09-15 按用户要求暂停后续训练、调参和评测，等待人工
改良；不能越级进入真机。
schema v3 水平圆柱以及
下列双 Piper、双 Panthera 和 14 维命令都只是历史对照。

旧左右对称 v2 和 `v2_symmetric_smooth_pilot` 都已否决：前者逐段停车，后者虽然规划速度
审计通过，但 HDF5 实际 `qpos` 与视频仍有停顿，而且平躺圆柱会落桌后二次抓取。schema 9
的单次抓取方案去掉了二次抓取，但全局 TOPP 在高曲率路点附近仍会减速。当前 schema 10
使用三次样条几何路径和全局弧长五次时间律；正式 128 集已经完成，直立/平躺各 64，8 个
平躺角度区间各 8。实际 `qpos` 在几何进度 5%–95% 内的 ≥80 ms 停顿为 0，数据、分层
10 分钟审阅视频和 `dataset.ok` 均保存在 Lab。未转换 RLDS，未启动训练。可恢复生成入口是：

```bash
bash /data/lyy/panthera-vla/run_lab_robotwin_panthera_v2_pilot.sh
```

WSL 仓库中的同一入口为 `tools/run_lab_robotwin_panthera_v2_pilot.sh`。该入口已改用新目录，
不会复用被否决的数据，并以 HDF5 实际关节状态而非只看规划速度进行连续性验收。实现、
结果和 Lab 产物见
[v2 schema 10 正式数据集](docs/15_panthera_v2_formal_dataset_2026-09-16.md)。

用户已审阅并批准 schema 10 的分层视频。Lab 已启动不依赖 agent 在线值守的长流程：
8-worker 媒体审计、TFDS/RLDS 4.0.0、一步优化器 smoke、GPU1–3 单次连续 10k SFT、
三卡 18 条开发集评测，以及仅在开发成功率达到 75% 后才运行的 18 条 final。普通用户
`@reboot` 入口会从阶段标记恢复；不会在训练中途调参或自动枚举实验。详见
[无人值守训练流水线](docs/16_panthera_v2_unattended_training_pipeline_2026-09-16.md)。

正式扩充集 1280 条已生成两版：随机机位版（2026-09-17，已人工批准）和固定机位版
`..._sft_v2_fixedcam`（2026-09-17 19:57 完成，机器审计通过，用户已审阅 10 分钟视频）。
两版都使用有界异步整集编码/原子提交，并用几何投影和 SAPIEN actor segmentation 双门禁
保证机械臂、圆柱和槽的全部可见像素位于中央 80% 区域；固定机位版复用同一套门禁但不再
逐集重采机位。**2026-09-17 决定弃用 v3 `panthera_phone_vertical_sft_v1`，正式训练只采用
固定机位 1280**；随机机位版保留为证据。两次大规模生成都因少数轨迹的重定时峰值加速度
越过 `2.002 rad/s²` 门禁而需要定点重采，根因是规划器 scale 空间容差与审计值空间门禁
不一致，尚未修复。

合并审计另补了一项实际 `qpos` 速度连续性判据：原判据的速度/加速度上界只加在重定时
**规划**轨迹上，实际 `qpos` 只查停顿，测不到速度跳变（schema 9 即由此漏过）。新判据以
`a_limit * dt` 为预算度量逐关节速度跳变，门禁比 `1.5`；固定机位全量 1280 最差 1.0673，
已否决的 schema 9 为 17.4060。设计、对比数据和 Lab 路径见
[v2 扩充数据集与异步落盘流水线](docs/17_panthera_v2_expanded_dataset_pipeline_2026-09-17.md)。

后续 SFT 曾改为验证集 L1 早停（最小有效下降 `1e-3`、耐心 3）。2026-09-17 的实测推翻了
该判据的前提：25x7 的 20k checkpoint 在全部离线指标上都优于 15k，其 release-gate dev16
却是 2/16 对 11/16（Fisher 精确检验 p = 0.0032）。离线指标在专家状态分布上度量，无法
预测闭环，**不得作为选模或停止判据**；该早停配置本身也因 `min_delta` 是绝对阈值而会在
远未收敛处停下。详见
[OpenVLA 闭环失败分析](docs/10_openvla_closed_loop_failure_analysis.md)。

```bash
# 在 Lab 上运行；重复执行会跳过已验证阶段
bash /data/lyy/panthera-vla/bootstrap_lab_vla.sh

# 进入已完成的 RLinf + RoboTwin 环境
source /data/lyy/panthera-vla/tools/activate_lab_vla.sh

# 固定模型提交，续传下载并运行 4 卡、4 环境的官方基线 smoke
bash /data/lyy/panthera-vla/run_lab_robotwin_baseline.sh

# 历史对照：运行双 Piper 圆柱入槽 scripted-oracle 四种子验收
bash /data/lyy/panthera-vla/run_lab_robotwin_cylinder_oracle.sh

# 从固定的官方 ROS 2 提交生成并验收 Panthera RoboTwin embodiment
bash /data/lyy/panthera-vla/bootstrap_lab_panthera_embodiment.sh

# 历史对照：运行双 Panthera 圆柱入槽 scripted-oracle 四种子验收
bash /data/lyy/panthera-vla/run_lab_robotwin_panthera_oracle.sh

# 历史对照：采集并核验双 Panthera 4-episode 上游格式 smoke 数据集
bash /data/lyy/panthera-vla/run_lab_robotwin_panthera_dataset_smoke.sh

# 历史对照：双臂正式契约 smoke
bash /data/lyy/panthera-vla/run_lab_robotwin_panthera_contract_smoke.sh

# 当前任务：单 Panthera、7 维契约、无 D6 附着的单集物理回放探针
bash /data/lyy/panthera-vla/run_lab_robotwin_panthera_physical_replay_probe.sh

# 历史对照：双臂 RLDS 25x14 smoke
bash /data/lyy/panthera-vla/run_lab_panthera_rlds_smoke.sh

# 当前任务：无人值守采集 128 个单臂、带视觉/几何随机化的 SFT v1 episode
bash /data/lyy/panthera-vla/run_lab_robotwin_panthera_sft_dataset.sh

# 当前任务：单臂数据验收、7 维 RLDS 转换和最小训练 smoke
bash /data/lyy/panthera-vla/run_lab_panthera_sft_pipeline.sh

# 当前任务：只解析单臂 16-seed 闭环评测配置，不启动仿真
bash /data/lyy/panthera-vla/run_lab_panthera_eval_config_smoke.sh

# 当前任务：上游通过后执行四卡 5000 步 SFT 和 16 轨迹闭环评测
bash /data/lyy/panthera-vla/run_lab_panthera_policy_pipeline.sh

# 当前正式任务：phone-SRT 对齐场景 oracle 与 128 集数据
bash /data/lyy/panthera-vla/run_lab_panthera_phone_oracle.sh
bash /data/lyy/panthera-vla/run_lab_robotwin_panthera_phone_sft_dataset.sh

# 当前正式任务：对齐数据 → TFDS 3.0.0 → 配置验收 → 一步优化器 smoke
bash /data/lyy/panthera-vla/run_lab_panthera_phone_sft_pipeline.sh

# 当前正式任务：媒体审计 → SFT → 16 条 held-out 闭环评测
bash /data/lyy/panthera-vla/run_lab_panthera_phone_policy_pipeline.sh

# 5000 步闭环失败后的无损续训；仅使用物理 GPU1–3
bash /data/lyy/panthera-vla/start_lab_openvla_phone_sft_continue_20k.sh

# 无人值守等待 20k 产物门，随后执行 2/2 诊断和通过门控后的 16 条正式评测
bash /data/lyy/panthera-vla/start_lab_panthera_phone_post20k_pipeline.sh

# 当前：25x7 一步优化器 smoke（已通过）
bash /data/lyy/panthera-vla/run_lab_openvla_phone_sft_25_step_smoke.sh

# 已完成：GPU1–3 训练 25x7 模型；旧的纯 VLA 后置门禁未通过
bash /data/lyy/panthera-vla/start_lab_openvla_phone_sft_25x7_10k.sh
bash /data/lyy/panthera-vla/start_lab_panthera_phone_post25x7_pipeline.sh

# 当前候选：先跑 seed 0/200001 的 2/2 门禁，通过后自动跑 16 条留出评测
bash /data/lyy/panthera-vla/start_lab_panthera_phone_target_assist_gate_pipeline.sh

# 已完成且在 dev16 停止：15k 以 1e-5 续训到 20k，再执行开发/最终 16-seed 门禁
bash /data/lyy/panthera-vla/start_lab_panthera_phone_25x7_low_lr_20k_pipeline.sh
```

各正式脚本接受显式 GPU 列表。本实验可使用全部四张卡，训练默认
`PANTHERA_SFT_GPUS=0,1,2,3`；评测运行器包含 RLinf
局部 GPU rank 到启动器物理 GPU 掩码的修补，避免逻辑 `0,1` 错占物理 GPU0。正式评测按
两个环境各运行八轮，仍产生固定的 16 条轨迹，不减少评测样本。

构建脚本固定 RLinf/RoboTwin 提交、把缓存和临时文件限制在 `/data/lyy/`，并在
`/data/lyy/panthera-vla/state/bootstrap-state/` 记录阶段结果；当前 `environment.ok`、
`assets.ok` 和 `verified.ok` 均已通过。基线脚本在 `.robotwin-baseline-state/` 记录模型
revision、最终组合配置、日志、退出码、指标和视频目录。2026-09-12 的四环境/200 步
`place_empty_cup` smoke 得到 `success_once=0.75`（3/4），四条非空视频齐全，日志无致命
异常，退出后 Ray 与 GPU worker 均已清理。这些脚本都不会安装或调用 Docker，也不会
使用管理员账户。

以下双臂结果只保留为历史探索证据，**不是当前任务定义，也不能作为当前训练集**。当前
目标是单台 Panthera 用一个夹爪抓取圆柱并插入凹槽，外部 action/state 均为 7 维。

同日，程序化 `place_cylinder_in_groove` 第一版 scripted oracle 在双 Piper embodiment
上通过 4/4 固定种子：每条轨迹包含 13 个事件阶段，最终横向误差不超过 0.32 mm、X
误差不超过 0.57 mm、圆柱轴误差不超过 0.62°，释放后线/角速度均为 0，四张非空阶段图
齐全且无残留 GPU/Ray 进程。该结果证明任务几何、动作流程和状态成功判据可运行；它仍
不是 Panthera embodiment 或 VLA 策略验收。双 Piper 独立规划器缺少共享物体闭链，当前
oracle 在确认双夹爪接触后采用显式 attach-on-grasp D6 约束，释放前恢复真实碰撞；此
建模选择必须在后续 Panthera 仿真和真机夹持实验中重新验证。

Panthera 的第一版 RoboTwin/MPLib embodiment 也已从官方 ROS 2 提交 `b08633d6...`
生成并验收。SAPIEN 识别六个手臂旋转关节和两个夹爪直线关节，官方限位、对称夹爪目标、
双臂 14 维顺序以及左右六个 `±1 cm` IK 方向均通过；双 Panthera 的同一圆柱入槽 oracle
随后通过 4/4 固定种子，每条保留 13 个阶段，最终最大 X/横向/轴误差分别为
0.707 mm、0.296 mm 和 0.074°。结果位于
`/data/lyy/panthera-vla/results/panthera-cylinder-oracle-smoke/`。当前仍使用
attach-on-grasp 建模假设，且只验收了 MPLib 路径；CuRobo 碰撞球和 VLA 策略尚未验收。

同一 oracle 已经通过 RoboTwin 官方 `collect_data.py` 采集为 4 个可回放 episode，位置在
`/data/lyy/panthera-vla/data/place_cylinder_in_groove/panthera_cylinder_dataset_smoke/`。
每集包含 628 个 RGB/双臂 14 维样本和一条 628 帧视频，四集均通过有限值、官方关节限位、
动作形状、帧数和清理验收；成功路径不使用墙钟 `sleep`。这只是上游格式 smoke：四集当前
几何相同，HDF5 没有时间戳，`joint_action` 是控制目标而非实际关节反馈，视频固定为 30 FPS，
自然语言生成器还不支持该绝对保存路径。因此它仅保留为上游兼容性对照，不作为 VLA
训练集。

正式契约 smoke 位于相邻的 `panthera_cylinder_contract_smoke/`。4 个 seed 均使用确定性
毫米级圆柱/凹槽位置扰动，统一时钟重采后每集 701–710 帧；HDF5 同时保存 14 维 action
target、独立的 14 维实际 articulation qpos/夹爪开度，以及每帧精确的 simulation
step/time。相邻有效样本最大间隔为 5 个 250 Hz 物理步，包含抓取沉降和释放后 1 秒落槽
过程；每集还生成 7 条
语言指令并记录源码提交、实际几何、随机化参数和 attach-on-grasp 标记。旧的 250 步采样
空洞版本已可恢复地归档为 `panthera_cylinder_contract_smoke.v1-noncontinuous/`。这 4 集只
用于契约验收，数量和变化范围仍不足以训练或证明 VLA 策略。

正式契约随后按统一全局仿真时钟重采：运动、夹爪和释放沉降循环都以
`simulation_step_count % 5` 决定 50 Hz 样本。阶段首尾诊断帧仍可保留在原始 HDF5；RLDS
转换只取全局 5-step 网格并按 step 去重。由于 RoboTwin 在应用 drive target、推进物理后
才拍照，RLDS 把图像和实测 qpos 对齐到下一网格动作，而不是复用同一行已经执行过的目标。
`panthera_cylinder` TFDS 1.0.0 已生成 train `[0,1,2]` / val `[3]`，OpenVLA-OFT 原生数据
管线已读出 `1x224x224x3` 图像、`1x14` proprio 和 `25x14` 绝对关节动作块；归一化统计
来自 Panthera 数据，不再复用 ALOHA/Piper 数值范围。这仍只是四集格式 smoke。
