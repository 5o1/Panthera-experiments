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
15. [v2 扩充数据集流水线](docs/17_panthera_v2_expanded_dataset_pipeline_2026-09-17.md)：
    固定机位 1280 条正式数据、异步落盘、连续性门禁和早停结论的历史演变。
16. [闭环评测台缺陷排查](docs/18_eval_harness_defects_2026-09-19.md)：动作预算、场景复现、
    专家重放基线以及两道 CI 门禁。
17. [配置与目录架构](docs/19_architecture_plan_2026-09-19.md)：产物自持配置、只读上游、
    overlay、runtime 装配和当前目录职责。
18. [评测观测错配](docs/20_eval_observation_mismatch_2026-09-19.md)：固定相机恢复、RGB/BGR
    契约、旧闭环数字作废及单轨迹过拟合现状。
19. [RoboTwin 上游迁移](docs/21_upstream_migration_2026-09-20.md)：从 `0008ae6` 迁移到
    `6dde571`、新版目录兼容和补丁取舍。
20. [单轨迹过拟合门禁收口](docs/22_single_trajectory_overfit_gate_hardening_2026-09-20.md)：
    25×7 契约、checkpoint lineage、闭环选模和不可覆盖运行目录。
21. [OpenPI π0.5 迁移与同口径基线](docs/23_openpi_pi05_migration_baseline_2026-09-21.md)：
    单相机/7 维适配、LeRobot 转换、统一评测 backend 和公平比较门禁。
22. [OpenVLA 28 维动力学 proprio 过拟合实验](docs/24_openvla_dynamics_proprioception_overfit_2026-09-21.md)：
    qpos/qvel/qacc/effort 合同、GPU0 smoke、三卡后继训练和验收口径。
23. [OpenVLA-OFT 多卡 action head 同步修复](docs/25_openvla_action_head_ddp_sync_fix_2026-09-23.md)：
    DDP forward 绕过的三卡复现、L1/diffusion 修复、历史 checkpoint 影响和复现门禁。

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
- 当前正式仿真数据为固定机位 schema 10 的 1280 条数据。旧随机相机策略评测数字因观测
  错配作废；专家重放和单轨迹闭环两道门禁仍为红色，不得越级恢复真机实验。
- 专家状态分布上的验证 L1 已被证明不能预测闭环成功率，不得作为正式训练的选模或停止
  判据。正式训练使用固定计算预算、周期 checkpoint，并以闭环 rollout 选模。
- RoboTwin 上游固定为 `main@6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755`。上游检出必须
  保持干净；运行树只能由 `pipelines/assemble_runtime.py` 从 pin、overlay 和 patch 生成。
- 修复前的 OpenVLA-OFT 多卡连续动作头训练绕过了 DDP `forward`，各 rank action head
  会分叉；旧三卡 checkpoint 只能保留作诊断，不能作为正确多卡训练基线。任何新多卡训练
  必须先通过 `bin/run_lab_openvla_action_head_ddp_audit.sh`。

## 维护规则

完成一个阶段或得到新的真机结论后，应同步更新
`docs/01_next_stage_vision_teleop_todo.md`；形成架构结论时更新对应设计/调查文档，并在
README 的文档入口中加入链接。不要把关键结论只留在聊天记录中。
