# Panthera 项目持续上下文

本文件是后续 coding agent 的入口。开始分析或修改本仓库前，先读取下列文档，不要仅依
赖会话上下文推测项目状态：

1. [根目录 README](README.md)：当前可运行链路、启动模式、安全边界和测试命令。
2. [下一阶段 TODO](docs/01_next_stage_vision_teleop_todo.md)：真机实验结论、未解决问题、
   优先级和验收标准。这是任务状态的主要来源。
3. [开源遥操作实现调查](docs/02_open_source_teleoperation_research.md)：相似项目的代码级
   技术分析，以及推荐给 Panthera 的架构方向。
4. [开源参考采纳矩阵](docs/03_reference_adoption_matrix.md)：哪些方法保留、准备采用、研究、
   暂缓或弃用，以及 Panthera 连续控制驱动审计。
5. [ROS 2 最小试动记录](docs/00_ros2_manual_motion.md)：官方驱动接口和最小真机操作。
6. [真机实验检查表](docs/04_real_hardware_experiment_checklist.md)：分级命令、否决条件和记录模板。
7. [软件验收基线](docs/05_software_acceptance.md)：无需真机的完整测试入口和覆盖范围。
8. [连续采集单例 Case](docs/06_data_collection_cases.md)：哪些实验合并、哪些隔离，以及
   事件/有效帧驱动规则。
9. [单目视觉实验计划](docs/Panthera-HT%20单目视觉手臂与手势遥操作实验计划.md)：最初
   的系统设计、坐标约定和分阶段实验方案。
10. [OpenVLA 闭环失败分析](docs/10_openvla_closed_loop_failure_analysis.md)：连续动作头、
    GPU 映射、训练/未见双轨遥测结论和当前续训门禁。
11. [仿真与真实场景对齐](docs/09_sim_real_scene_alignment.md)：真实工作台影像归档位置、
    策略相机/监控相机边界、必须对齐的任务几何和 sim-to-real 门槛。
12. [v2 单次抓取与直接释放修复](docs/14_panthera_v2_single_grasp_direct_release_2026-09-16.md)：
    schema 9 oracle、实际 qpos 连续性门禁和历史 PhysX 接触抖动。
13. [v2 schema 10 正式数据集](docs/15_panthera_v2_formal_dataset_2026-09-16.md)：128 条正式
    轨迹、连续性验收、Lab 产物和异步落盘任务池设计边界。
14. [v2 无人值守训练流水线](docs/16_panthera_v2_unattended_training_pipeline_2026-09-16.md)：
    用户审阅批准、长任务阶段、GPU1–3 并行策略、质量门和重启恢复入口。

## 当前边界

- 当前 VLA 装配任务是**一台 Panthera 机械臂、一个夹爪，将一个圆柱插入凹槽**。不是
  双机械臂协作任务。仿真的外部 observation/action 契约为 7 维：`joint1..joint6` 加
  一个归一化夹爪量。仓库中双 Piper、双 Panthera 和 14 维数据仅是 2026-09-12 以前的
  历史探索/兼容性证据，不得用于当前训练、评测或真机方案。
- 当前默认候选链路是单 RGB MediaPipe → WebSocket → mapper → 50 Hz latest-only IK/关节流
  → 200 Hz ros2_control → Panthera SDK 非阻塞命令。旧 `/pos_cmd` 阻塞路径仅保留作对照。
- 真机默认只开放 Y/Z 相对位置；X 深度和末端姿态仍默认关闭，夹爪真机发布也关闭。
- 连续路径采用 Host 的 0.6 rad/s 常用速度、1.0 rad/s 硬件上限和 2.0 rad/s² 加速度上限；
  不再用逐档真机实验搜索速度。目标过期后必须保持，物理急停仍是必要条件。
- 失跟、低置信度、断线、来源帧过期后必须锁存为禁用；数据恢复不得自动恢复运动。
- 实时视觉目标保持 latest-only，禁止改成 FIFO 动作队列。
- 不得在没有用户明确要求、真机确认和急停准备的情况下启动官方驱动或发布真机命令。
- 开源项目只提供设计候选、实现证据和失败案例，不构成本项目的硬约束。应按 Panthera
  驱动能力、单目可观测性、测试结果和安全目标独立取舍，任何参考方案都可以弃用。

## 维护规则

完成一个阶段或得到新的真机结论后，应同步更新
`docs/01_next_stage_vision_teleop_todo.md`；形成架构结论时更新对应设计/调查文档，并在
README 的文档入口中加入链接。不要把关键结论只留在聊天记录中。
