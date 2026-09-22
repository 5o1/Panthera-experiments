# 24 · OpenVLA 28 维动力学 proprio 过拟合实验（2026-09-21）

## 1. 实验目的

step 65000 的单帧 OpenVLA-OFT 已能把圆柱送到正确直接释放位，却仍保持夹爪闭合。记录帧
上它可以在专家释放位置输出开爪，说明问题不是夹爪 action 头完全没有学会，而是自主闭环
状态下的释放阶段难以区分。本实验只改变 proprioception，比较当前 7 维输入和以下 28 维
输入：

```text
[qpos(7), qvel(7), qacc(7), effort(7)]
```

每个 7 维块均按 `joint1..joint6, gripper` 排列；action 仍为原来的 25×7 绝对关节目标，
RGB、语言指令、episode 2、基础模型、LoRA rank、batch、学习率、24 小时预算和每 5000 step
保存 checkpoint 均保持不变。没有加入“已稳定多少秒”、任务阶段、槽口坐标或强制松爪规则，
因此实验检验的是通用机器人动力学状态是否改善阶段可观测性，而不是用任务专用状态机解题。

## 2. 状态语义与数据来源

RoboTwin 在线状态由 `place_cylinder_in_groove._actual_robot_state()` 生成。位置、速度和加速度
来自 articulation 的 `get_qpos()`、`get_qvel()` 和 `get_qacc()`。`effort` 使用 PhysX 的
incoming joint-frame load：六个旋转关节取 joint-frame torque-x，夹爪直线关节取 force-x。
它表示父链接传给子链接的关节载荷，包含驱动、接触等动力学影响，不应误写成只含电机命令
的“纯执行器扭矩”。实现刻意不用 `get_qf()`，因为当前 RoboTwin 控制器会把 `qf` 写成重力/
科氏补偿项。PhysX 对该量的定义见
[Link incoming joint force](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html#link-incoming-joint-force)。

历史 schema 10 HDF5 只保存了 qpos。为了不重录并改变已批准的 episode 2，
`build_dynamics_overfit_dataset.py` 保留原始 RGB、7 维 qpos 和 7 维 action，并在同一场景中
确定性重放动作，补出 qvel、qacc 和 effort。重放 qpos 与记录 qpos 的中位/P99 绝对误差
门槛分别为 0.02/0.10 rad；超过门槛会删除未完成目标目录并失败。这个派生方法只服务于当前
单轨迹诊断。后续正式多轨迹数据应在采集时直接记录四个状态块，避免把重放误差扩散到训练集。

## 3. 已完成的 GPU0 门禁

Lab 产物：

```text
/data/lyy/panthera-vla/ci/dynamics-ep2/dataset
/data/lyy/panthera-vla/ci/dynamics-ep2/rlds/panthera_phone_cylinder_socket_v2_dynamics/5.0.0
/data/lyy/panthera-vla/ci/dynamics-ep2/smoke/smoke
/data/lyy/panthera-vla/state/dynamics-overfit-gpu0
```

派生集含 1037 个 50 Hz 样本。重放与记录 qpos 的中位误差为 `0.0002734 rad`，P99 为
`0.0072121 rad`，均通过门槛。四块最大绝对值依次为 qpos `1.9072`、qvel `2.6620`、
qacc `719.087`、effort `29.4789`；夹爪位置被归一化后，它的加速度数值尺度明显大于旋转
关节，因此必须沿用 RLDS 的逐维统计归一化，不能直接把原始 28 维值送入模型。

RLDS 5.0.0 已验证 state 为 28 维、action 为 25×7。GPU0 随后完成两次真实 optimizer step，
训练合同记录 `proprio_dim=28`，证明 28 维状态能够进入新建的 proprio projector 并参与
前向、反向和参数更新。通过标记为：

```text
/data/lyy/panthera-vla/state/dynamics-overfit-gpu0/prepare.ok
```

这只证明数据和优化链路可运行，不证明闭环释放已经改善。

## 4. 正式任务排队状态

原 7 维 24 小时训练继续独占 GPU1–3，没有被本次修改或 GPU0 smoke 中断。28 维任务通过
`single-trajectory-overfit.lock` 排在它后面；等待脚本持有并继承同一文件描述符，锁释放到
新训练启动之间没有可被其他同类任务插入的空窗。前序任务只有写出 `TRAINING_COMPLETE`
时才会启动后继任务，异常退出会写 `blocked` 并停止。

排队入口和日志为：

```text
/data/lyy/panthera-vla/pipelines/ci/queue_dynamics_overfit_after_current.sh
/data/lyy/panthera-vla/state/dynamics-overfit-queue/queue.log
```

后继任务使用 GPU1–3，每卡 batch 6，LoRA rank 32，24 小时墙钟预算，每 5000 step 保存
不可覆盖 checkpoint，并在线同步到 W&B project
`assanekowww/panthera-ci-overfit-dynamics-24h`。运行目录将位于：

```text
/data/lyy/panthera-vla/ci/overfit-ep2-dynamics-24h/dynamics-28d-ep2-24h-<UTC时间>/
```

## 5. 自动后续验收

2026-09-22 已补上独立的自动评测队列。与此前 7 维编号-checkpoint评测相同，它立即在
GPU0 上消费已经完整保存的 checkpoint，同时训练继续独占 GPU1–3；两者不需要串行等待。
评测器每轮处理当前积压后继续轮询新 checkpoint，直到训练写出 `TRAINING_COMPLETE` 且
最后一个 checkpoint 也处理完毕，才关闭队列并生成总拼接视频。入口和日志为：

```text
/data/lyy/panthera-vla/pipelines/ci/queue_dynamics_overfit_evaluation.sh
/data/lyy/panthera-vla/state/dynamics-overfit-queue/evaluation-queue.log
```

评测会处理每个 5000-step 编号 checkpoint，同时生成 h20 与 h1 temporal ensemble 视频、
策略推理耗时、完整 25×7 prediction trace、逐时间步离线 L1 和闭环阶段遥测；最后把所有
checkpoint 的双模式视频拼成一个总画面。评测器不再硬编码 7 维，而是从 checkpoint 的
`training.json` 读取 proprio 宽度并设置模型运行时，未知宽度直接失败。编号 checkpoint
仍使用既有评测锁，避免与其他 GPU0 checkpoint 评测重叠。

比较时固定相机、scene、episode、动作预算及 checkpoint 选择规则，只允许 7D/28D proprio
成为实验变量。核心判据仍是自主闭环是否稳定抓取、到达直接释放位、实际输出开爪并使圆柱
落槽；训练 loss 或记录帧 L1 下降不能代替 Gate 2。完成标记与总视频分别为：

```text
/data/lyy/panthera-vla/state/dynamics-overfit-queue/evaluation.complete
/data/lyy/panthera-vla/ci/overfit-ep2-dynamics-24h/<run-id>/all-checkpoints-two-mode-comparison.mp4
```
