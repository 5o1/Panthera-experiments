# OpenVLA 闭环失败分析与续训门禁

## 1. 结论

当前任务仍是一台 Panthera、一个夹爪，将竖直黄色圆柱插入黄色槽座。v3 数据、TFDS 3.0.0
和 5000 步 SFT 均有效，但策略尚未通过闭环。不得据此恢复实验台或启动真机。

第一次 16 条未见轨迹的 0/16 不能用于评价模型，因为 RLinf 官方评测路径只加载了
proprio projector，遗漏了训练输出目录中独立保存的连续 L1 action head，实际执行的是
离散 token fallback。项目补丁 `tools/patches/rlinf_openvla_oft_l1_eval.patch` 已修正加载与
批量推理，离线 episode 0 五点探针得到平均绝对误差 0.04275、最大绝对误差 0.17059。

修正后合并运行训练 seed 0 和未见 seed 200001，结果仍为 0/2。每条轨迹完整执行 160 个
五步动作块；动作末值与执行后关节的最大差分别约 0.080 和 0.030，说明命令没有被执行层
吞掉。两条轨迹最后都收敛到近似
`[1.296, 1.65, 1.64, -1.58, 0, 1.359, 0.395]`，没有完成抓取或插入。

训练 seed 的首个动作块结束后，机械臂关节已经最接近专家 episode 0 的第 84 帧，而不是
第 5 帧；臂关节距离仍约 0.143 rad。夹爪空抓后观测值降至约 0.395，而成功示范中的实测
夹爪范围为 0.689–0.992，后续状态因此落到行为克隆训练分布之外。训练 seed 的圆柱累计
移动约 1.65 m 并跌离桌面；未见 seed 的圆柱累计移动约 0.05 m，均未靠近槽中心。

5000 步训练使用物理 GPU1–3、每卡 batch 1，有效样本量约 15000；训练集为 112 个 episode、
约 98000 帧，因此尚不足一轮。当前优先假设是欠拟合叠加闭环误差累积，而不是动作契约、
RoboTwin TOPP 执行或 L1 action head 文件损坏。

## 2. 20000 步续训结果

5000 步合并模型、action head 和 proprio projector 已无损续训到总计 20000 步：

- 输入模型：`/data/lyy/panthera-vla/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-5000steps`
- 输出模型：`/data/lyy/panthera-vla/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-20000steps`
- 状态目录：`/data/lyy/panthera-vla/.panthera-phone-openvla-sft-20k-state/`
- 启动入口：`/data/lyy/panthera-vla/start_lab_openvla_phone_sft_continue_20k.sh`
- 计算资源：仅物理 GPU1–3；GPU0 禁止本实验使用。
- 最佳验证 L1：`0.0407615`（step 19000）
- 最终验证 L1：`0.0480221`（step 20000）

官方续训入口要求带步数的组件文件名；运行器只为已有
`action_head--latest_checkpoint.pt` 和 `proprio_projector--latest_checkpoint.pt` 创建同目录
符号链接，然后使用 `--resume true --resume_step 5000`。原 5000 步文件未被覆盖。训练外层
shell 在模型保存后、状态文件写入前消失；恢复脚本对四个权重分片、索引、action head、
proprio projector、日志终点和 GPU 清理逐项验收后，明确以
`original_wrapper_exit_observed=false` 记录恢复完成，未伪造原包装器退出码。

20000 步模型的 seed 0/200001 双轨仍为 0/2。每条 160 个五步动作块，最大执行误差约
0.015 rad，说明仿真执行层正常；六关节只在复位附近变化，圆柱净位移为 0。训练 seed
最终稳定映射到专家 episode 0 的索引 42。把 reset 后夹爪预置为 0.9 的独立对照同样 0/2，
最终仍停在索引 42，因此排除了“只因初始夹爪值不匹配”的假设。

## 3. GPU 映射修正

RLinf 会发现全机四卡，然后把外层 `CUDA_VISIBLE_DEVICES=1,2` 内的 local rank `0,1`
再次写进 worker 环境，导致错误占用物理 GPU0/1。错误启动在模型进入有效轨迹前终止并归档，
GPU0 已回到 16 MiB 基线。

`tools/patches/rlinf_cuda_visible_subset.patch` 将计算 worker 的 local rank 通过父进程掩码
翻译回物理编号，同时保留无 GPU 管理 worker 的全机声明。单元验证为 `0→1`、`1→2`，
运行时 `nvidia-smi` 也确认两个模型 PID 只位于物理 GPU1/2。

## 4. 学习率调度审计

9000 总步数时 W&B 仍记录学习率 `5e-4`。检查实际执行的 OpenVLA-OFT `finetune.py` 后确认，
warmup 分支在 `gradient_step_idx >= lr_warmup_steps` 后仍然每步把 optimizer 学习率直接写回
原值；`MultiStepLR` 在第 4000 个续训 update 做出的 10 倍衰减因此会在下一步被覆盖。20k
运行加载的是旧代码，故完整运行仍保持 `5e-4`；其离线 loss 不被当成闭环成功。
`tools/patches/openvla_oft_warmup_decay.patch` 已应用到后续训练入口：warmup 完成后停止覆盖
optimizer 学习率，使调度器衰减能够保持。

第一次 25x7 训练启动时又发现了补丁状态检测缺陷：GNU `patch --reverse --batch --dry-run`
面对尚未应用的补丁会打印 `Unreversed patch detected`，但仍返回退出码 0，旧入口因而把
未修源码误判为已修。该次运行在约 step 2787 主动停止，没有生成或冒充正式 checkpoint；
状态、W&B 文件和日志归档在
`action-horizon-25/invalid-lr-patch-detection-20260913-210809/`。训练/评测入口现改为核对
明确的修复后源码标记，并在分支结束后再次断言标记完整；实际文件第 1080 行已确认包含
`gradient_step_idx < cfg.lr_warmup_steps` 后才重新启动 10000 步训练。

## 5. 5 帧固定点与 25 帧方案

128 集共 105662 个样本。相邻专家关节动作范数中位数为 `0.00310 rad`，30.1% 不超过
`1e-4 rad`。episode 0 的索引 0–47 只在调整夹爪，六个手臂关节保持原位；在策略反复落入的
索引 42，未来第 1 和第 5 个目标仍不动，而未来第 25 个目标已经包含明确的手臂位移。

因此不降低 50 Hz 控制频率，而把一次预测窗口从 5 帧（0.1 秒）扩为 25 帧（0.5 秒）。
`tools/patches/openvla_oft_panthera_constants.patch` 增加独立的 `PANTHERA` 平台常量：动作块
25、动作 7 维、本体状态 7 维、Q99 归一化。现有 TFDS 无需重建；真实数据管线已读出
`(25,7)` action、`(1,7)` proprio 和 `(1,224,224,3)` RGB，一步优化器 smoke 已通过。

从 20k 合并 VLA 权重初始化的 25x7 action head 与 proprio projector 已在物理 GPU1–3
完成 10000 步训练；最佳验证 L1 为 `0.033623`（step 8000），最终为 `0.034562`，GPU0
未用于本实验。全执行 seed 0/200001 双轨仍为 0/2，因此没有越级运行 16 个 held-out seed。

训练 seed 的 32 个动作块已覆盖 31 个不同的最近专家索引，并从开场推进到约索引 756，
排除了旧的索引 42 固定点。圆柱净移动约 0.17 m、累计路径约 0.27 m，夹爪实际最小开度约
0.679（与成功示范受圆柱阻挡后的约 0.689 接近），说明策略确实完成接触、夹持和搬运。
失败主要发生在搬运中逐步倾斜/滑移并侧向释放：seed 0 最终圆柱轴误差约 55.9°；未见
seed 200001 最终仍直立，但位置和高度均未达到插入条件。

有界诊断保持模型预测 25 帧，只执行前 10/15/20 帧便重新观察和规划，从而减少
0.5 秒开环误差。首次 horizon 10 试运行暴露出 RLinf 的两个 worker 使用不同循环次数：
EnvWorker 按 10 帧计划 80 次，HuggingFace rollout worker 仍按 25 帧只发送 32 次，运行
因此等待而非完成。该无效结果已归档且明确标记；两侧现统一按 execution horizon 计算，
修正扫描仍限制为 800 个真实仿真控制步。粗扫 `10/15/20` 与细扫 `18/19/21/22` 全部为
0/1；`20` 最接近任务成功，`18` 的水平对中最好，但两者均在槽口释放时产生约 12–13°
倾斜。该结果把失败定位为接触丰富的末端插入，而不是长程抓取和搬运，故停止继续枚举
execution horizon。

## 6. 末端技能与 GRPO 取舍

RLinf 的官方 RoboTwin GRPO 配置面向离散动作 token。当前 OpenVLA-OFT 使用连续 L1
action head；其连续推理分支将 `chunk_logprobs` 置零，因而直接运行官方 GRPO 不会对该
动作头形成有效策略梯度。除非先实现连续分布及可训练 log-prob，不把该路径列为当前实验。

现阶段采用可部署的混合控制：VLA 负责视觉抓取和初始抬升；末端技能只使用预先标定的槽位
位姿、夹爪开度及机器人自身末端反馈，不读取仿真圆柱真值。25/40/55 mm 相对下插、释放时
晚触发以及一次性斜向对准均失败。分阶段技能加入固定工具偏置、槽外定姿、释放后退夹后，
seed 200001/200003 为 2/2；但晚接管 16-seed 只有 6/16，失败抓取在接管前已因长程搬运
倾斜。离线审计显示 11 个可靠抓取中，末端刚升到 0.98 m 时 10 个圆柱倾角不超过 1.8°，
最差为 4.34°，因此改为达到标定插入高度即接管。

直接从抓取点抬到 60 mm 槽口净空会触及 Panthera 工作空间上边界。最终路径先抬 30 mm，
横移到槽外 85 mm 接近点，再升到 50 mm 净空，随后定姿、水平对齐、5 mm 分段插入、
推进 100 个物理步稳定、松爪并上退 30 mm。此前失败的 seed 200007/200011/200013/200015
回归为 4/4，尾部连续成功观测块为 49–52。物理推进属于同一连续 episode，不使用墙钟
`sleep`；圆柱位姿仅写入诊断轨迹，不参与控制。

延长窗口时还发现 RLinf/RoboTwin 有三处独立上限：
`env.eval.max_steps_per_rollout_epoch`、`env.eval.max_episode_steps` 和
`env.eval.task_config.step_lim`。只改前一或两处会在第 800 步结束或自动 reset；两个无效
对照均已保留并明确标注，不计入结论。运行器现把三者绑定为同一个显式参数。

## 7. 证据位置

- 首次错误 action head 的 0/16：
  `/data/lyy/panthera-vla/reports/panthera-phone-vertical-sft-v3/eval-v1-failure/`
- L1 离线探针：
  `/data/lyy/panthera-vla/reports/panthera-phone-vertical-sft-v3/l1-inference-probe.json`
- 修正后训练/未见双轨摘要、完整 JSONL、日志和视频：
  `/data/lyy/panthera-vla/reports/panthera-phone-vertical-sft-v3/eval-l1-diagnostic-train-vs-heldout/`
- 20k 双轨失败（含训练曲线摘要、两条视频、轨迹和校验清单）：
  `/data/lyy/panthera-vla/reports/panthera-phone-vertical-sft-v3/post20k/diagnostic-failure-20260913-195236/`
- reset 夹爪 0.9 对照失败：
  `/data/lyy/panthera-vla/reports/panthera-phone-vertical-sft-v3/post20k/pregrip09-failure-20260913-202356/`
- 25x7 全执行双轨失败：
  `/data/lyy/panthera-vla/reports/panthera-phone-vertical-sft-v3/action-horizon-25/diagnostic-20260913-225033/`
- 执行视野 worker 不一致的无效运行：
  `/data/lyy/panthera-vla/reports/panthera-phone-vertical-sft-v3/action-horizon-25/invalid-execution-horizon-worker-mismatch-20260913-231700/`
- 执行视野粗扫与细扫：
  `/data/lyy/panthera-vla/reports/panthera-phone-vertical-sft-v3/action-horizon-25/execution-horizon-sweep-20260913-232514/`
  和 `execution-horizon-sweep-20260913-234418/`
- 相对下插与标定末端技能诊断：同一父目录下的 `terminal-assist-sweep-20260914-001122/`、
  `target-assist-late-trigger-20260914-003600/`、`target-assist-diagonal-entry-20260914-004200/`
  与 `target-assist-staged-entry-20260914-004832/`
- 立即接管开发集 7/16 与工作空间失败回归：同一父目录下的
  `immediate-handoff-staged-route-dev16-7of16-20260914-041321/` 和
  `height-gated-direct-lift-unreachable-regression-1of4-20260914-043116/`
- 最终分段路径 4/4 回归：同一父目录下的
  `lift-height-handoff-route-regression-4of4-20260914-043858/`
- 30k 高学习率模型 dev16 坍缩与 10k 路径对照：同一父目录下的
  `30k-resume-high-lr-dev16-1of16-20260914-081811/`、
  `30k-stabilized-route-regression-1of4-20260914-083537/` 和
  `10k-stabilized-route-regression-4of4-20260914-085645/`
- 15k 低学习率模型的延迟释放接管 dev16（11/16）：同一父目录下的
  `15k-release-gate-dev16-11of16-20260914-103700/`
- 已否决的末端几何候选：同一父目录下的
  `15k-release-calibration-7mm-1of4-20260915-164300/`、
  `15k-guided-release-2of4-20260915-165500/` 和
  `15k-shallow-release-2of4-20260915-170400/`

## 8. 后续门禁

30k 高学习率续训的 dev16 只有 1/16，已否决；10k 原模型在同一路径的定向回归仍为 4/4，
因此不是执行层回归。15k 低学习率模型在不加技能时为 4/16；延迟到 VLA 已形成可靠抓取、
进入标定槽位邻域并主动请求松爪后再接管，得到 11/16。失败 seed 中 200002、200004、
200016 未形成可靠抓取；200006、200014 接管后仍因物体相对夹爪偏置/倾斜失败。

对末端技能的三项有界对照均未新增成功：固定 Y 偏置从 4.5 mm 改到 7 mm 为 1/4；先命令
夹爪到 0.82 再完全松爪为 2/4，而且碰撞已把实际开度顶到约 0.94，该命令实际形成重新夹紧；
高 30 mm 松爪的浅插入为 2/4，200014 不再掉出台面但以约 66.8° 横躺在槽口。三项都已
否决并恢复 4.5 mm 深插入基线，后续不再扫描末端几何常数。

从稳定 15k 模型以 `1e-5`、无 warmup 在物理 GPU1–3 续训到 20k 已完成。验证
Next-Actions L1 依次为 15000 `0.03578`、16000 `0.02905`、17000 `0.02689`、
18000 `0.02580`、19000 `0.02508`、20000 `0.02537`。训练产物完整，但延迟释放技能下的
dev16 仅 2/16；成功 seed 为 200004、200009，只有 200003、200004、200009、200015
触发末端接管。该结果再次表明离线 L1 下降不能替代闭环门禁。dev 未达到 12/16，因此
final seed 300001–300016 从未启动。

完整训练状态、W&B、16 条轨迹/视频、日志、源码快照和校验清单位于同一父目录下的
`20k-lr1e5-release-gate-dev16-2of16-20260915-181400/`；53 项清单已通过 SHA-256 校验，
运行后无 Ray、训练、评测或 GPU 计算进程残留。2026-09-15 用户要求本组结束后暂停，
因此不得自动启动新的训练、调参或评测，等待人工改良。
