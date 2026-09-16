# 单 Panthera 圆柱入槽 VLA：Lab 实验阶段报告（2026-09-13）

## 1. 目标与当前结论

当前目标是在 RoboTwin + RLinf 中让**一台 Panthera 机械臂、一个夹爪**抓取圆柱并插入
凹槽，随后才进入真机 shadow mode。任务不是双机械臂协作。外部 observation/action 契约
为 7 维：六个关节加一个夹爪量。

单臂单集规划、连续采集和无辅助约束物理回放已经通过。schema v4 竖直插入场景也已通过
固定场景与随机化多 seed oracle，但第一版相机在视觉审计中被否决。宽视角 v3 的 128 集、
TFDS 3.0.0、一步优化器 smoke 和首个 5000 步 SFT 已完成；修正评测端连续动作头加载后，
训练/未见种子双轨闭环仍为 0/2。续训到 20000 步后双轨依然 0/2，夹爪预置 0.9 的对照也
没有解除策略固定点，因而仍不能报告为策略成功。当前已把动作窗口从 `5x7` 改为 `25x7`，
真实 RLDS batch 和一步优化器 smoke 均通过。新的 25x7 模型已在 GPU1–3 完成 10000 步
训练，但 seed 0/200001 双轨仍为 0/2。短执行视野扫描完成后，混合控制改为 `18/25`：
VLA 负责抓取和初始抬升，达到标定高度后由末端技能接管搬运与插入。已知失败 seed 的
路径回归通过 4/4；10k 模型的可靠抓取覆盖仍只有 11/16，因此 GPU1–3 正在从 10k 无损
续训到总计 30k。旧双臂数据与报告只保留为历史诊断证据。

## 2. 当前单臂已通过阶段

- RLinf/RoboTwin 无 Docker 环境位于 `/data/lyy/panthera-vla/`，四张 RTX 4090 均已验收。
- 官方 `place_empty_cup` 基线在 4 个环境中成功 3/4；评测日志和 4 条视频位于
  `/data/lyy/panthera-vla/RLinf/logs/20260912-03:12:35-robotwin_place_empty_cup_openvlaoft_eval/`。
- `place_cylinder_in_groove` schema v3 已加载一台 Panthera、一个普通圆柱和一个凹槽；
  对外保存 7 维实际状态与 7 维绝对动作。
- 单 seed scripted 规划成功，随后官方采集器生成 734 个 50 Hz 网格动作样本；HDF5 和
  连续 MP4 分别约 21 MB 和 395 KB。
- 同一条数据通过 RoboTwin 真实 `Base_Task.gen_sparse_reward_data` 路径回放，未创建 D6
  attach-on-grasp。回放 1/1 成功，最大夹爪接触点 8；最终 X/横向/高度误差约为
  0.036/0.504/0.091 mm，轴误差 0.190°，夹爪已打开。
- 探针完成后没有 RoboTwin、Ray 或 GPU 计算进程残留。结构化证据位于
  `/data/lyy/panthera-vla/.panthera-physical-replay-probe-state/`，数据位于
  `/data/lyy/panthera-vla/data/place_cylinder_in_groove/panthera_single_cylinder_physical_replay_probe/`。
- 正式随机化预检首轮为 2/4：两个失败 seed 在初始帧已经与凹槽近侧导轨接触，规划本身
  都成功。把圆柱初始 Y 基准从 `-0.06 m` 移到 `-0.02 m` 后，同一 seed 0–3 通过 4/4。
  失败日志和阶段图保留在 `.panthera-single-sft-dataset-state/planning-preflight/`。
- 使用两个已完成 episode 的独立 RLDS adapter smoke 已通过：TFDS 2.0.0 注册名为
  `panthera_single_cylinder`，OpenVLA 真实数据管线读出 `5x7` action chunk、`1x7`
  proprio 和 `224x224` RGB。摘要位于
  `.panthera-single-rlds-adapter-smoke-state/summary.json`。第一次运行暴露并修正了 TFDS
  builder 仍注册为旧名称的问题，失败输出已可恢复地归档。

## 3. Phone-SRT 对齐版竖直插入场景

- `place_vertical_cylinder_in_groove` 使用一台 Panthera、一个夹爪、竖直黄色圆柱和黄色
  浅槽；外部状态/动作均为 7 维，任务元数据为 schema v4。任务语义与插入方向已经对齐。
- 夹爪从上方接近，工具局部 X 映射到世界 `-Z`，手指闭合方向映射到世界 `-X`。规划和
  回放不创建 D6 attach-on-grasp；手指碰撞体使用独立高摩擦垫，圆柱与场景保持较低摩擦，
  避免把高摩擦错误施加到圆柱后产生初始抖动。
- 固定场景 seed 0–3 的 scripted oracle 通过 4/4；随机化 seed
  0、10000、20000、30000 的预检也通过 4/4。成功判据同时要求位置、竖直轴、速度、
  夹爪打开以及夹爪接触冲量归零。
- OpenVLA 一步优化器 smoke 已修复并通过，证明真实训练入口可读取单臂 7 维 RLDS；正式
  phone 数据仍必须生成独立 TFDS 3.0.0 和独立归一化统计，不能复用旧横向场景统计。
- 从首个 episode 的开场、中段、终态视频抽帧后发现，v1 仿真只在画面中央显示末端，机座
  和大部分臂体被上沿裁掉；真实归档帧则是手机广角、Panthera 位于左侧、黄色任务区位于
  中央。v1 相机因此没有通过视觉门禁，不能进入正式训练。
- 128 集 v1 诊断数据完成了四分片采集、合并与 episode 0/112 无附着回放。首次合并暴露
  语言模板缺少 `{A}`/`{B}` 占位符，导致 instruction JSON 被全部过滤；修正模板后复用
  已完成分片生成了 128/128 条指令，没有重复采集。由于视觉门禁失败，整套数据和状态已
  可恢复地移动到
  `/data/lyy/panthera-vla/archives/phone-srt-centered-4x3-v1-20260913/`，不用于训练。
- v2 先后尝试任务区 X `-0.30 m` 和 `-0.20 m`：前者四个固定 seed 都在初始顶部抓取 IK
  失败，后者在槽边抬升阶段到达工作空间边界，因此均已弃用并归档。
- 宽视角 v3 保留中心 4:3 策略裁剪和 `320×240` 契约，将垂直 FOV 设为 75°，机器人基座
  Y 设为 `-0.35 m`，任务区与相机中心 X 设为 `-0.25 m`。固定 seed 0–3 与随机化 seed
  0/10000/20000/30000 均通过 4/4 物理 oracle；开场、槽边抬升和插入终态抽帧通过
  “机器人在左、任务在中”的构图门禁。它是基于归档图像的临时构图对齐，不是度量标定。
- v3 正式 128 集已完成四分片采集、合并、schema/随机化/视频验收及 episode 0/112 无附着
  回放；TFDS 3.0.0 以 112/16 划分训练/验证集，动作块为 `5x7`，一步优化器 smoke 已通过。
- 5000 步 SFT 在 GPU1–3 完成。首次 16 条闭环评测为 0/16，但评测代码当时错误使用离散
  token 输出，未加载训练产生的独立 L1 action head；该结果只作为失败证据，不能评价模型。
- 修正 action head 后，训练 seed 0 与未见 seed 200001 的合并诊断仍为 0/2。两条轨迹均执行
  160 个动作块且执行误差小，却在空抓后收敛到近似相同的固定关节姿态。训练 seed 的圆柱
  累计移动约 1.65 m 并跌离桌面，未见 seed 累计移动约 0.05 m；两者均未接近槽中心。5000 步、有效 batch 3
  只覆盖约 1.5 万样本，少于约 9.8 万训练帧的一轮，当前无损续训到总计 20000 步。
- 20000 步续训完整保存，最佳验证 L1 为 `0.0407615`（step 19000），最终为 `0.0480221`。
  但双轨仍为 0/2：两条各执行 160 个动作块，最大执行误差约 0.015 rad，六关节只在复位
  附近微动，圆柱净位移为 0。训练 seed 的状态最终反复对应专家索引 42。
- 独立的 reset 夹爪 0.9 对照同样为 0/2，说明问题不是单纯初始夹爪命令不一致。全数据
  审计显示专家索引 42 的未来 5 帧仍是静止目标，但未来 25 帧已包含手臂运动；因此建立
  Panthera 专属 `25x7` 常量，不改变 50 Hz 控制频率。RLDS 读取得到 `(25,7)` action、
  `(1,7)` proprio、`(1,224,224,3)` 图像，一步优化器 smoke 已完成前向、反向和更新。
- 首次 25x7 正式启动暴露了 GNU `patch --reverse --batch` 的假阳性：warmup/decay 修复并未
  真正写入源码。该无效运行在约 step 2787 主动停止，未产生正式 checkpoint；全部状态和
  日志保存于 `reports/panthera-phone-vertical-sft-v3/action-horizon-25/`
  `invalid-lr-patch-detection-20260913-210809/`。入口现用明确源码标记做前后置断言，确认
  实际训练文件包含修复条件后已从头重启。
- 修正后的 25x7 训练完整跑到 step 10000，最佳验证 L1 为 `0.033623`（step 8000），最终为
  `0.034562`；step 4000 后实际学习率保持为 `5e-5`，证明 warmup/decay 修补生效。模型、
  四个非空权重分片、action head 和 proprio projector 位于
  `/data/lyy/panthera-vla/runs/panthera-phone-openvla-sft/`
  `panthera-phone-wide-v3-vertical-sft-25x7-10000steps/`。
- 25 帧全执行双轨仍为 0/2，失败证据位于
  `/data/lyy/panthera-vla/reports/panthera-phone-vertical-sft-v3/action-horizon-25/`
  `diagnostic-20260913-225033/`。训练 seed 的 32 次预测覆盖 31 个不同的最近专家索引并推进
  到约索引 756，说明旧的索引 42 固定点已解除；圆柱也被夹持并移动约 0.27 m。失败发生在
  搬运后的渐进倾斜/滑移与侧向释放：seed 0 最终轴误差约 55.9°，未见 seed 则没有形成插入。
- 已完成“预测 25 帧、只执行部分帧后重新观察”的 receding-horizon 诊断。首次
  horizon 10 运行发现 EnvWorker 计划 80 次而 HuggingFace rollout worker 仍只发送 32 次，
  已停止并完整归档到 `invalid-execution-horizon-worker-mismatch-20260913-231700/`，不计入
 评测结果。两侧循环现统一按执行视野计算，修正后的每项扫描都保留 800 个真实控制步。
- 粗扫 `10/15/20` 和细扫 `18/19/21/22` 均为 0/1。`20` 最接近成功：最终 X/Y 误差
  2.17/6.65 mm、插入深度 5.75 mm、轴误差 13.16°且夹爪已打开；`18` 的水平中心更好
  （0.33/0.90 mm），但轴误差仍为 12.11°。其余候选没有可靠释放，因此不再盲扫执行视野。
  两轮视频、逐块 JSONL 和摘要分别保存在 `execution-horizon-sweep-20260913-232514/` 与
  `execution-horizon-sweep-20260913-234418/`。
- 官方 RoboTwin GRPO 配置不能直接解决当前连续动作头：OpenVLA-OFT 的连续 L1 分支在
  rollout 中返回零 `chunk_logprobs`，而 GRPO 更新依赖离散 token log-prob。因此没有把
  一个不会优化当前 action head 的官方命令冒充强化学习实验。
- 纯相对向下的 25/40/55 mm 末端辅助、晚触发和一次性斜向入槽均失败。分阶段技能先后
  加入 60 mm 净空、固定工具/圆柱 Y 偏置、槽外定姿、释放后 30 mm 退夹，seed 200001/
  200003 达到 2/2。晚接管开发集为 6/16，轨迹证明可靠抓取后的 18–19 个 VLA 重规划块会
  累积物体倾斜；可靠抓取代理本身为 11/16，且代理首次成立时 10 个 seed 倾角小于 1°、
  最差为 4.809°。
- 立即接管的直接抬升会在工作空间上边界失败。最终条件改为抓取代理成立、夹爪仍在受阻
  开度带且末端升到标定插入高度；路径先短抬 30 mm，横移到槽外 85 mm 接近点，再升到
  50 mm 净空后定姿、对齐和插入。插入后先推进 100 个物理步稳定，再松爪和退夹；这不是
  墙钟 `sleep`。已知规划/插入失败 seed 200007/200011/200013/200015 回归为 4/4，结束时
  分别保留 51/49/52/51 个连续成功观测块。
- GPU0 禁止用于本实验。RLinf 原实现会把外层 `CUDA_VISIBLE_DEVICES=1,2` 的局部 rank
  重新写成物理 `0,1`；已加入局部 rank 到父掩码的映射修补，并实测模型 PID 只落在物理
  GPU1/2。训练继续使用 GPU1–3。
- v3 视觉审计报告和配对参考图保存在
  `/data/lyy/panthera-vla/reports/panthera-phone-vertical-sft-v3/visual-audit/`。

## 4. 历史双臂探索及否决证据

以下内容属于已经被当前单臂任务取代的双臂探索，不能继续训练或部署。双臂 SFT v1 在
新场景中通过 RoboTwin 的真实 `Base_Task.gen_sparse_reward_data` 动作路径回放，
并显式禁止 scripted oracle 的 D6 附着。episode 0 和 episode 112 均失败（0/2），所以
`dataset.ok` 与上游 `pipeline.ok` 没有生成，第二级训练流水线也拒绝启动。

episode 0 的细粒度诊断表明：双夹爪完成闭合，接触点最多 28 个，并将圆柱抬起约 29 mm；
进入横向搬运后接触丢失，圆柱停留在起始区域，最小目标距离约 87.8 mm。原始回放报告在
`/data/lyy/panthera-vla/.panthera-sft-dataset-state/replay-summary.json`，逐动作块图位于
`/data/lyy/panthera-vla/results/replay-diagnostic/episode-0/`。

进一步核对 Panthera URDF 后发现，夹爪直线关节沿工具局部 Y 轴运动，而旧双臂抓取四元数将
该方向映射到了圆柱长轴世界 X；横向搬运世界 Y 因而主要依赖摩擦。新的直径夹持姿态已经
证明 MPLib 可规划，并完成 13 个 oracle 阶段，但当前释放后圆柱停在凹槽上方约 30 mm。
该原型未通过，所以只保存诊断报告和逐阶段图：

- `/data/lyy/panthera-vla/.panthera-physical-replay-probe-state/`
- `/data/lyy/panthera-vla/results/panthera-physical-replay-probe/`

## 5. 视频与图像证据

2026-09-13 盘点 `/data/lyy/panthera-vla/` 得到 152 个非空 MP4，总大小约 85.8 MB。
其中 128 条属于 SFT v1，其他来自正式契约、上游格式 smoke、历史归档和官方基线评测。
其中旧双臂 SFT v1 的 128 条不能用于当前训练。当前单臂探针另保存了非空连续 MP4：
`panthera_single_cylinder_physical_replay_probe/video/episode0.mp4`。正式单臂数据集必须继续
要求每个 episode 都有非空视频，并将视频数量纳入通过门禁。

## 6. 下一步门禁

1. [已完成] 单个固定 seed 依次通过 scripted 规划、1-episode 连续采集、无 D6 新场景
   回放、成功判据、非空 MP4 与进程清理；未使用无界 seed 重试。
2. [已完成] 将 RLDS/OpenVLA 数据适配器改为单臂 7 维契约，使用独立 schema/version、
   统计和 `unnorm_key`，并通过两集真实读取 smoke。
3. [已完成] 竖直任务语义、v3 固定/随机化物理 oracle 和基于真实归档帧的视觉构图门禁
   已通过；真实尺寸与最终相机外参仍待恢复实验台后测量。
4. [已完成] 128 集 v3 对齐数据、合并、schema/随机化/视频核验，以及 episode 0/112
   无附着物理回放；旧双臂和旧横向单臂数据均未进入正式训练。
5. [已完成] phone 数据专属 RLDS、归一化统计、一步优化器 smoke 和 5000 步 SFT。
6. [已完成但未通过闭环] 20000 步模型、双轨和夹爪预置对照均已归档；两次双轨均为 0/2。
7. [已完成但未通过闭环] 25x7 模型已完成 10000 步训练；全执行双轨为 0/2，但旧固定点已
   解除，失败收敛为搬运中的物体倾斜/滑移和释放对准误差。日志、逐块遥测和视频均已归档。
8. [已完成但未通过闭环] 执行视野 10/15/20 与 18/19/21/22 均已留存视频和逐块遥测；
   `20/25` 最优但仍是 0/1，不再继续盲扫。
9. [已完成] 标定槽位末端技能的路径/稳定性回归通过 4/4；动作决策未读取仿真圆柱位姿。
10. [进行中] 25x7 模型从 10k 无损续训到总计 30k。完成后自动评测开发 seed
    200001–200016；成功率至少 75% 才运行未用于调参的 final seed 300001–300016。
    两组均达到 75%、视频/遥测一致且进程清理通过，才完成仿真门禁并通知恢复真实实验台。

本报告中的路径均在 Lab 的 `/data/lyy/`，日常实验使用普通用户 `lyy`，未使用 Docker
或管理员账户，也没有启动真实机械臂。
