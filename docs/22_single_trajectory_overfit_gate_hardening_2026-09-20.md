# 22 · 单轨迹过拟合门禁收口（2026-09-20）

本文记录 Gate 2 的代码收口，不记录新的训练结论。本轮没有启动 GPU 长任务；“单轨迹能否
闭环复现”仍是红色门禁，只有 Lab 实际运行成功后才能改为通过。

## 1. 门禁回答的问题

Gate 2 只回答一个问题：OpenVLA-OFT 在 episode 2 上训练后，能否在该 episode 记录的
同一场景、同一相机和同一动作预算中完成圆柱入槽。它不测试泛化，因此训练集和验证集
有意使用同一条轨迹。

固定输入为正式 fixedcam schema 10 数据的 episode 2：seed `10000002`、直立圆柱、
1042 条记录动作。Lab 源目录为：

```text
/data/lyy/panthera-vla/data/place_randomized_cylinder_in_socket/
panthera_phone_cylinder_socket_v2_single_grasp_sft_v2_fixedcam
```

门禁会先建立只含该 episode 的自持 subset，再由 subset 生成 RLDS。训练和闭环评测都读取
该 subset；checkpoint 的 `training.json` 也记录 subset 的名称、摘要、路径与父数据 lineage，
不再把 1280 条父数据误写成实际训练集。

## 2. 本轮修复

### 2.1 动作契约不再静默错配

门禁显式导出 `PANTHERA_ACTION_CHUNK=25`。RLDS 转换器、OpenVLA 常量和数据集 contract
会在训练前交叉核对动作维度与 chunk；当前合法值是 `(25, 7)`。任何一处回落到旧默认
5-step 或 LIBERO 的 8 维契约都会立即失败。

### 2.2 停止条件改为闭环成功，保留固定计算上限

验证 L1 只作为遥测，不再决定正式门禁是否成功。训练每 250 step 保存候选 checkpoint；
GPU 3 上的 `closed_loop_watcher.py` 对完整 checkpoint 快照做同场景 rollout。成功候选被
原子保留到 `closed-loop-best/`，随后写 `CLOSED_LOOP_SUCCESS`；训练进程在下一次验证时
读取该信号并正常退出。

训练默认使用 GPU 0–2，验证使用 GPU 3，脚本拒绝两组 GPU 重叠。`max_steps=40000` 是
可复现的计算预算，4 小时 timeout 仅是挂死保护；OpenVLA 补丁已恢复 `max_steps` 分支并
将它列为合法停止原因。若到预算仍没有闭环成功，训练可以正常结束，但最终 Gate 2 仍会
因 rollout 失败而保持红色。

### 2.3 checkpoint 快照与成功模型完整性

watcher 对 LoRA adapter、action head、proprio projector、统计文件和 `training.json` 的
mtime/size 组成完整指纹。复制前后指纹不同则丢弃快照并重试，避免合并训练过程中被撕裂
的 checkpoint。成功模型使用不可覆盖目录；成功 sentinel 通过临时文件替换原子写入。
合并脚本只阻止属于同一 run 的活跃训练，不再被机器上无关的 finetune 进程误伤。

OpenVLA 的验证/停止逻辑和 warmup 后允许学习率衰减的修复现由同一个 `finetune.py`
补丁持有；bootstrap 检测补丁集合变化时会先逆序撤销完整旧集合，再应用完整新集合，避免
把两个补丁世代混装到同一环境。

Lab 激活入口统一为仓库实际跟踪的 `tools/activate_lab_vla.sh`；当前 gate 与活跃脚本不再
引用仅在旧 Lab checkout 中存在、但 WSL 仓库无法部署的 `bin/activate_lab_vla.sh`。
bootstrap 也不再读取 NAS：旧 `place_empty_cup` walkthrough 只是参考资料，不属于当前
任务依赖，不能让它的挂载状态阻塞 runtime 或训练环境构建。

装配器调用 `git apply` 时设置 runtime 父目录为 Git 搜索上界。Lab 的 runtime 位于本仓库
内部；若不设置该边界，Git 会向上发现 Panthera 的 `.git`，可能返回成功却把路径相对到
源码仓库，导致生成树没有真正应用补丁。新增回归测试直接在仓库内创建临时 runtime。
同一 Git 搜索上界也用于 site-packages 中的 OpenVLA 补丁；否则 Lab 环境目录同样位于
Panthera checkout 内，常量补丁可能被报告为成功但实际仍使用 LIBERO 的 8-step 契约。

### 2.4 运行目录与产物保留

默认目录包含 UTC 时间和 PID，已存在即拒绝运行，不再 `rm -rf` 旧实验。全局 `flock`
阻止两个 Gate 2 同时运行。退出时总会写 `exit-code.txt` 与 `finished-at.txt`；默认保留
subset、RLDS、模型、日志和报告。只有显式设置 `CI_OVERFIT_KEEP=0` 才删除可重建的 RLDS，
不会删除模型或证据。

## 3. Lab 运行方式

先完成 runtime/bootstrap，再以普通用户设置正式数据集根目录运行：

```bash
export CI_OVERFIT_DATASET_ROOT=/data/lyy/panthera-vla/data/place_randomized_cylinder_in_socket/panthera_phone_cylinder_socket_v2_single_grasp_sft_v2_fixedcam
bash /data/lyy/panthera-vla/pipelines/ci/gate_single_trajectory_overfit.sh
```

一次运行只有一次外层启动，不需要人工中途确认。`run-config.json` 固化 episode、模型、
25-step chunk、GPU 分配和 40000-step 上限；`closed-loop-validation.jsonl` 连续记录每个候选
的阶段、接触、抬升、槽距离与专家轨迹偏差。

## 4. 验收状态

本地验收包括 187 项 `packages/` 与 `pipelines/` 测试、74 项 ROS 测试和 Shell 语法检查。
Lab 已建立持久环境 `/data/lyy/panthera-vla/envs/rlinf`，两棵 runtime 从固定上游重新装配，
并完成下列无训练验收：

- Lab 同一套 187 项源码测试全部通过；
- OpenVLA 实际解析为 `(ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM) = (7, 25, 7)`；
- curobo 固定为 `v0.7.8`，`curobo.types.math` 与 `MotionGen` 可导入；
- PyTorch `2.11.0+cu130` 可见 4 张 GPU；
- RoboTwin `6dde571...` 与 RLinf `a3816b...` 两个上游 checkout 保持干净。

首次环境安装发生在装配器路径缺陷修复前，因此一度安装了 curobo main 的新扁平 API；
runtime 补丁真正生效后已原位替换为固定 tag `v0.7.8`。TensorFlow 2.15 会报告无法加载
CUDA 13 GPU 库，但这里只承担 RLDS/输入流水线；模型训练使用的 PyTorch CUDA 与 4 卡
可见性已经独立验收通过。

以上只证明门禁代码与 Lab 环境可以装配和执行，不证明策略成功。

下一步只能是：在 Lab 的干净 runtime 中执行一次 Gate 2，保留完整运行目录并检查最终
`gate.json`。Gate 2 通过前，不恢复 1280 条正式训练，也不进入真机复现。

## 5. 24 小时过拟合运行（进行中）

2026-09-20 22:01（Lab 本地时间）启动 episode 2 的训练-only 长任务。GPU0 已被其他用户
占用，因此 GPU1–3 全部用于 DDP，不在 GPU3 上并发闭环 watcher；这轮训练结束后再释放
GPU 并逐 checkpoint 运行同场景闭环。配置为每卡 batch 6（有效 batch 18）、LoRA rank 32、
25×7 连续动作、24 小时墙钟预算，每 5000 step 保存一个不可覆盖的编号 checkpoint。W&B
在线 run 为 `assanekowww/panthera-ci-overfit-24h/adeiyrzm`。

用户级 systemd unit：

```text
panthera-overfit-ep2-24h-b6-20260920T140123Z.service
```

运行目录：

```text
/data/lyy/panthera-vla/ci/overfit-ep2-24h/
overfit-ep2-24h-b6-20260920T140123Z/
```

启动后实测 GPU1–3 显存约 40.3、40.3、41.2 GiB，瞬时利用率 100%、86%、100%；没有
OOM。该状态只说明训练已健康运行，**不表示 Gate 2 已通过**。

这次 training-only 运行显式设置了 `use_val_set=false`，因此 W&B 没有历史
`VLA Val/Loss`，不得把训练 loss 改名后写进 checkpoint 对比视频。编号 checkpoint 的
评测改为在保存后计算 `offline deterministic validation L1`：对 episode 2 全部 50 Hz
记录帧构造与训练相同的 25×7、尾部保持填充动作块，在 bounds-q99 归一化空间计算平均
绝对误差。结果同时记录总 L1、当前动作、后续动作、六关节、夹爪和物理动作 MAE，并显式
写入 `recorded_during_training=false`。该离线值用于横向观察 checkpoint，不能冒充训练时
遥测，也不能替代闭环 Gate 2。

GPU0 上的长驻评测 unit 为
`panthera-overfit-numbered-eval-v3-20260921.service`。它依次合并不可变的编号 checkpoint，
计算上述离线 L1、运行同场景闭环、生成每个 checkpoint 的带标注视频，并在每项完成后刷新
`numbered-eval/all-checkpoints-comparison.mp4`。训练仍活动时只轮询新 checkpoint，不持有
合并模型；训练停止且积压清空后才写 `BACKLOG_COMPLETE`。

首个完整产物 step 5000 已落盘。1037 个唯一 50 Hz 记录帧（原始文件的 1042 条记录中，
阶段边界重复采样按训练规则保留最后一条）得到归一化总 L1 `0.04261852`；六关节 L1
`0.03548626`、夹爪 L1 `0.08541212`。闭环为 0/1；相机渲染一致性通过，但记录帧/渲染帧
上的模型中位误差分别为 12.20/10.75 mrad，均未通过 10 mrad 模型门槛，因此视频只作
诊断，`gate_success=false`。单视频与总拼接视频已目视确认同时显示 checkpoint step 和
`offline val L1 0.042619`。

step 10000 视频显示策略已把圆柱搬到槽口上方但未松爪。为区分模型缺失与执行视野截断，
诊断重跑在 JSON 中新增 `prediction_trace`，保存 53 次查询各自完整的 25×7 原始输出。
结果排除了“松开落在每个 chunk 的第 21–25 步、因只执行前 20 步而永远被截掉”：策略在
第 140 步由 `0.61394` 降到 `0.46953`、首次进入闭合区间后，所有后续完整预测块的夹爪
最大值仍只有 `0.46953`，大于等于 0.5 的松开 action 为 0 个，尾部五步同样为 0 个。
从已经到达插入阶段的查询 row 480 起，各块夹爪最大值约为 0.292–0.306，接近专家闭合
目标 0.3。专家夹爪目标实际从 row 790 的 `0.30704` 开始连续增大；row 801 从
`0.48291` 上升到 `0.50050` 越过了诊断所用的 0.5 分界；执行器也用同一中点识别“首次
重开”并配置释放接触求解，但实际夹爪命令始终是连续开度，不是二元控制，0.5 也不是释放
动作的起点。因此当前直接原因是 step 10000 模型没有输出连续开爪轨迹，不是执行器吞掉了
已经生成的松爪指令。诊断产物在
`gripper-diagnostic-step10000/step-10000.json`。

## 6. 双执行模式预览与真实工作台延迟估计

step 10000 增加两种同场景闭环预览。基线仍为每次输出 25 个动作、执行 20 个后重规划；
候选版则在每个控制步都重新输出 25 个动作，并对所有覆盖当前时间步的当前/历史预测做等权
平均，即 `execution_horizon=1`、`temporal_ensemble=0.0`。后者实现的是动作块 temporal
ensemble，而不是对已经执行的关节反馈做移动平均。

计时边界刻意从“解码后的 RGB 帧和机器人状态已经在内存中”开始，到动作数组返回 CPU
结束，包含图像预处理、H2D、模型前向和 D2H。仿真渲染、物理步进、视频渲染/编码均不在
计时内；真实相机采集/传输/解码、ROS 传输与机器人控制器同样尚未加入。因此这个数字是
迁移到现实工作台时的策略侧下界，不是完整端到端延迟。

实测基线 53 次查询的中位延迟为 `128.77 ms/query`，按实际执行跨度摊销为
`6.45 ms/action`，相当于 50 Hz、20 ms 动作预算的 `0.32×`。逐步等权版 1042 次查询的
中位/P95/最大延迟为 `128.59/130.73/147.12 ms/action`，实时负载为
`6.43×/6.54×/7.36×`，1042 次全部错过 20 ms 截止时间。两版闭环均未成功：基线到达
插入阶段，逐步等权版只到达搬运到槽附近。该结果说明等权 temporal ensemble 可以作为
行为诊断，但按当前模型和单 GPU 推理路径不能直接作为 50 Hz 真机控制策略。

最终并排视频、两份完整 trace 和机器可读摘要位于：

```text
/data/lyy/panthera-vla/ci/overfit-ep2-24h/
overfit-ep2-24h-b6-20260920T140123Z/paired-preview-step-10000/
```

视频每帧显示 `ms/query`、按本帧所属查询摊销的 `ms/action`、相对 50 Hz 的实时负载以及
`OK/LATE`。后续编号 checkpoint 从 step 30000 起由
`panthera-overfit-numbered-paired-eval-v2-20260921.service` 自动生成这两版和并排视频；
入口为 `pipelines/ci/evaluate_numbered_overfit_checkpoint_pairs.sh`。旧单版本队列已停止。

## 7. step 45000 的时间损失与释放可观测性

离线验证现保留每个 50 Hz 记录帧的 25-step 总/关节/夹爪 L1、当前步 L1，以及专家和模型
当前夹爪目标，并生成 PNG/CSV。step 45000 的整体归一化 L1 为 `0.018706`。释放窗口
row 777–825 的 25-step 夹爪 L1 均值/最大值为 `0.03201/0.07966`，全局夹爪均值为
`0.01955`；当前步夹爪 L1 均值/最大值为 `0.03398/0.09577`，全局当前步夹爪均值为
`0.01632`。释放确实是夹爪头最难的局部区间，但 25×7 总 L1 被六关节平均后没有升高，
因此聚合 loss 不能揭示该问题。

更重要的是，在记录的专家 observation 上，模型当前夹爪目标于 row 803 越过 0.5，专家
于 row 801 越过，只晚 40 ms；模型已经学会在专家状态分布上松爪。自主闭环却产生两种
相反失败。h20 不是“过早松爪导致掉落”：它在夹爪仍张开时已经因剧烈动作碰撞并撞走圆柱，
根本没有稳定夹住。接触发生于 row 79–113，期间夹爪命令仍约 `0.89–0.91`；圆柱在
row 111/120 已偏移 `3.68/10.02 cm`，而夹爪直到 row 140 才首次低于 0.5，row 160 的
重新张开只是碰撞失效后的后续事件。h1 等权 temporal ensemble 则真正夹住并到达插入
阶段，但 row 800 后所有完整原始预测块的夹爪值仍只有 `0.301–0.310`，没有任何松爪输出。
这不是执行器截断，也不是“夹爪头完全没学会”，而是策略进入不同闭环状态后无法稳定识别
任务阶段。

2026-09-24 的用户视频复核进一步收紧了上述归因：抓取前的剧烈抖动会在闭合完成前撞倒、
推离圆柱，所以“稳定抓取和抬升”本身仍未完成；这不是抓住后搬运阶段的误差。反过来，
在已经形成稳定夹持的已审阅 rollout 中，后续搬运均能完成并到达槽上方。当前证据因此支持
“抓取前近物体控制不稳定 → 条件稳定的持物搬运 → 到位后释放不触发”这一分段，而不支持
“抓取、搬运、释放都受到同一种偏离”的概括。这里的条件稳定性限于已审阅 rollout，不能
外推为未见场景的必然成功保证。

训练与评测相机位姿误差只有 `3.36e-8 m`，可排除相机错位。相机视角本身仍可能造成
专家 row 800→801 整幅图平均变化仅 `0.20/255`，proprio 状态差仅 `4.65e-5`；但两帧的
夹爪目标也是连续释放斜坡上的相邻值，而不是语义相反的动作。每个训练标签还是 25 步未来
动作块，相邻样本共享其中 24 步。这组数值只能说明当前观测在释放斜坡附近变化很小，不能
单独证明“同一观测对应冲突标签”或部分可观测。已经确认的直接事实仍是闭环协变量偏移后
模型不再输出开爪轨迹；时间历史、接触或显式阶段变量是否能改善，必须通过对照实验验证。

## 8. step 65000 滑动平均版的人工审阅结论

用户对 step 65000 视频的人工审阅表明：逐控制步生成动作块并做历史覆盖预测滑动平均的
版本，已经能够稳定、准确地把圆柱移动到槽点正上方，但仍然没有松开夹爪。这说明关节运动
和平移阶段已经不再是该版本的首要失败点；剩余失败集中在终端释放。

2026-09-21 已经通过 Windows SSH 代理读取现有 trace 并运行
`packages/panthera_sim/analyze_temporal_gripper.py`，不需要重跑 rollout。row 750--1041 的
newest 原始夹爪预测范围为 `0.305469--0.312305`，等权 ensemble 为
`0.302734--0.307820`；两者都没有达到开爪阈值 `0.5`，且全轨迹中
`suppressed_open_rows=0`。因此已排除“模型已经要求松爪，只是夹爪维被滑动平均压回阈值
以下”。仅对六关节做 ensemble、夹爪采用 newest 并不能修复该 checkpoint。

rollout 最终圆柱中心 `z=0.885201 m`；schema 10 的直接释放目标由 socket top
`0.840 m`、圆柱半高 `0.060 m` 和 bottom clearance `-0.014 m` 得到 `z=0.886 m`，误差
仅约 `0.8 mm`。所以 `height_error=25.0 mm` 是松爪后靠重力落到槽底前的预期高度差，不能
解释成策略还没有下降到释放位。它已经到达正确的直接释放位，却持续输出闭合。

重新审计后必须撤回“相邻帧具有语义相反标签”的表述。保留后的轨迹 row 789 仍为
`0.300000`，row 790 开始以 `0.307035` 连续打开，row 801 才越过事后分析使用的 0.5
分界，row 824 达到 `0.905025`；整个开爪过程持续约 34 个 50 Hz 步。相邻 25 步动作块
还共享 24 个目标。因此 `0.482915 -> 0.500503` 只是连续开度轨迹的相邻点，不能作为阶段
混叠的直接证据。整幅 RGB 平均变化 `0.199852/255` 和 7 维 state L2 `4.29e-5` 仍说明
局部观测变化很小；同一 checkpoint 在记录帧上也沿连续斜坡输出并最终越过 0.5，而在自主
闭环中停在约 0.30。当前可确认的是闭环分布偏移与开爪轨迹未触发；是否因为缺少时间历史
或显式阶段，仍是需要消融验证的假设。不得用槽口几何规则强制开爪来冒充 VLA 成功。
