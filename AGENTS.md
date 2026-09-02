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

## 当前边界

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
