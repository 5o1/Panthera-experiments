# OpenVLA release arena 强化学习链路与首轮 smoke

日期：2026-09-24

## 1. 本阶段回答了什么

本阶段把 DDP 修复后的 28 维 OpenVLA-OFT `step 75000` 接入 RLinf 的连续动作 PPO，固定
使用物理 GPU 2、3，不触碰 GPU 0、1。目标不是用一次更新证明策略已经学会插入，而是先
验证下列工程事实：

1. 25×7 连续动作能够采样并重新计算 action-level log-prob；
2. 两卡 FSDP 能对残差策略和 critic 产生有限、非零且同步的梯度；
3. chunk 实际只执行前 5 步时，reward、done、bootstrap 和 log-prob 的时间维仍一致；
4. R2 物理效果奖励能够按动作进入 rollout buffer，而不被 RLinf 的 termination reward
   覆盖；
5. episode 2 的专家前缀可以只用于构造 release arena reset，不进入策略 observation 或
   reward。

截至本文记录时，这五点已经通过单测和一次 R2-C0 PPO 更新。`success_once` 仍为 0，不能
据此声称释放问题已经解决。

## 2. 实现

连续 L1 头保持冻结；新增的小型 tanh-squashed Gaussian residual policy 围绕冻结 SFT
均值输出有界随机残差。C0 不使用历史，C1 增加六个机械臂维度的对齐 chunk-overlap loss，
C2 再输入上一 chunk 的有效尾部和最近实际执行动作，并对历史施加噪声与 dropout。夹爪维
不参加 overlap loss，避免平滑项压制接触模式切换。

R1 只给稳定插入 `+1`、不可恢复失败 `-1`。R2 在 R1 上加入物理效果势函数差分、很小的
时间代价和六关节动作变化惩罚；不读取专家帧号，不因夹爪目标跨过某个阈值加分。reward
evaluator 对专家完成、提前释放、到位仍夹持、槽外丢失、反复开合、成功吸收态和夹爪不
参与平滑共七类情况做了脚本测试。

release arena 当前从 episode 2 HDF5 的前 780 个 50 Hz 专家动作重放出“已夹持、接近
释放区”的物理状态。重放只构造 reset，随后清空 episode action 计数、reward 历史和成功
latch。策略获得的是重放后的 RGB 与 28 维 proprio，不获得专家索引或物体真值。

成功不是单帧判定：圆柱必须在合法释放体积内解除接触，持续至少 250 个 250 Hz 物理步，
再连续通过三次 50 Hz 的位置、姿态、速度、夹爪和接触检查。

## 3. 先发现并修复的集成问题

在获得首个 PPO 更新前依次修复了：Ray Unix socket 路径、双 env seed 数量、RoboTwin 新版
兼容包、运行工作目录、语言指令来源、FSDP 冻结/可训练参数混合、28 维 proprio 选择、
5/25 horizon 的 done 与 log-prob 对齐，以及 bf16 log-prob 必须升为 float32。

首个 R2 release-arena 运行
`ci/openvla-rl-release-arena/20260924T053644Z/` 虽然获得非零 return，却产生
`entropy_loss=inf`、`policy_loss=nan` 和 `grad_norm=nan`。原因是 release 段的冻结 SFT
动作会精确饱和到 `±1`：在 bfloat16 中，`1 - 1e-6` 仍舍入为 `1.0`，所以先 clamp 再
`atanh` 仍得到无穷大。

修复后 `_atanh` 先把输入提升到 float32，再施加开区间 clamp。新增回归用精确 `±1` 的
bf16 SFT 动作验证 location、采样、log-prob 和反向梯度全部有限。同时 actor 在 backward
前检查 log-prob、entropy、value 和总 loss；非有限值会直接失败，不再静默保存坏
checkpoint。

首轮串行 `R2-C1/C2` 中，C1 正常完成，C2 在第一个 rollout 的历史编码器处失败。C2 的
rollout worker 为了稳定保存上一 chunk 和已执行动作而使用 float32，而 OpenVLA 残差模块
已经转为 bf16；`history_encoder` 的首个线性层因输入和权重 dtype 不同而报
`mat1 and mat2 must have the same dtype`。后续 `ray.kill`、Gloo 断连和 actor died 都是
主异常触发的清理级联，不是外部杀进程或显存不足。修复是在进入可训练历史编码器前把派生
历史特征显式对齐到 observation feature 的 device/dtype，并新增“bf16 C2 + float32 rollout
history + 带噪声/dropout + backward”回归。Lab 正式 runtime 的连续策略测试由 9 项增加到
10 项，全部通过；真实 C2 rollout 随后也跨过原故障点并完成 PPO 更新。

## 4. 已完成实验

### 4.1 R1 完整 reset 工程 smoke

| 组别 | Lab 目录 | reward / success | grad norm | 结论 |
|---|---|---:|---:|---|
| R1-C0 | `ci/openvla-rl-smoke/20260924T050849Z/` | 0 / 0 | 0.732 | 连续 PPO 基线可更新 |
| R1-C1 | `ci/openvla-rl-smoke/20260924T051657Z/c1/` | 0 / 0 | 0.835 | overlap loss 实际进入总损失 |
| R1-C2 | `ci/openvla-rl-smoke/20260924T051657Z/c2/` | 0 / 0 | 0.877 | 历史条件分支可更新 |

三组各执行一次 PPO update 并写出约 15.4 GB 的 `global_step_1` checkpoint。由于完整 reset
只有 25 个动作预算，三组没有终局任务奖励；这些结果只证明工程链路，不比较任务学习效果。

### 4.2 R2 release arena C0

修复后的同口径运行位于
`ci/openvla-rl-release-arena/20260924T055017Z/`。内部 checkpoint 目录仍带旧的
`r1-c0` 前缀，但 `run-config.json` 和 Hydra 展开配置明确记录 `reward.variant=r2`；后续
launcher 已把 reward variant 纳入 run name。

一次 PPO update 的关键指标为：

- `episode_len=150`，`num_trajectories=3`；
- `env/return=0.07097497`，`env/reward=0.00047316644`；
- rollout `rewards=0.0024`，证明动作对齐的 R2 shaping 已进入 buffer；
- `actor/policy_loss=0.021`，`actor/entropy_loss=-2.785`；
- `actor/grad_norm=0.833`，`actor/total_loss=0.0031`；
- `success_once=0.0`；
- `global_step_1/full_weights.pt` 为 15,425,324,245 bytes。

因此 R2-C0 已通过“非零 reward、有限 loss/gradient、非空 checkpoint”集成门槛，但未通过
“任务成功”门槛。

### 4.3 R2 release arena C1/C2

三种变体的一步 smoke 使用相同的 episode 2 前 780 动作 warm start、150 个动作上限、两轮
rollout、seed 1234 和冻结 `step 75000` 基线。C1 位于
`ci/openvla-rl-release-arena/20260924T060250Z/c1/`；修复后的 C2 位于
`ci/openvla-rl-release-arena/20260924T062231Z/c2/`。旧目录
`20260924T060250Z/c2/` 是 dtype 故障证据，不含有效 PPO checkpoint，不能用于比较。

| 变体 | env return | success | overlap loss | policy / total loss | grad norm | checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.07097 | 0 | — | 0.021 / 0.0031 | 0.833 | 15,425,324,245 B |
| C1 | 0.12410 | 0 | 0.042 | 0.045 / 0.0077 | 1.006 | 15,425,324,245 B |
| C2 | 0.08633 | 0 | 0.044 | 0.060 / 0.0078 | 1.137 | 15,425,327,925 B |

三组都获得非零 shaping return、有限更新和完整 checkpoint，但三组任务成功仍全部为零。
单次 PPO 更新下的 return 排名不能作为选模依据：各组采样到的 trajectory 数也分别为
3、1、2，随机 rollout 方差足以掩盖一致性方法差异。因此下一步不是凭这一行数字选择 C1，
而是让 C0/C1/C2 在同一预算下各运行 10 次更新，再比较成功、释放失败类型和时序指标。

## 5. 自动验证与运行边界

正式生成 runtime 上当前测试为：RoboTwin reward/执行器 `11 passed`，连续残差 PPO
`10 passed`。RLinf 与 RoboTwin 固定上游仍为 clean。运行入口只暴露
`CUDA_VISIBLE_DEVICES=2,3`；RLinf 内部逻辑 rank 0/1 分别映射物理 GPU 2/3。物理 GPU 1
上的其他用户进程未被读取、终止或复用。

launcher 默认 `WANDB_MODE=offline`，只有调用方显式设置 `WANDB_MODE=online` 才允许上传。
`run-config.json` 现在同时记录 max epochs/steps、保存间隔、rollout 数、env 数、episode/epoch
动作预算、seed 和 W&B 模式，避免长任务只凭服务命令推测实际训练合同。

为避免用训练时 stochastic return 选模，新增
`pipelines/ci/evaluate_openvla_rl_release_gpu23.sh`：它要求显式给出变体和 `full_weights.pt`，
以 deterministic policy 在相同 release arena 做 checkpoint-only rollout，保存视频和评测
指标，且 runner 在该模式下不会执行 PPO update。补丁先装配到独立临时 runtime 并通过
Python 编译和 Hydra 完整解析；解析结果确认 `only_eval=true`、checkpoint、C2 完整模型
合同、5-step execution horizon、780-step warm start 和视频开关均生效。当前训练完成后，
`queue_openvla_rl_release_eval_gpu23.sh` 会检查根目录 `COMPLETE` 和每组唯一的 step-10
checkpoint，再串行评测 C0/C1/C2；等待阶段不占 GPU。

新增 `packages/panthera_vla/summarize_rl_release_matrix.py` 从 TensorBoard event 读取每一次
PPO update，而不是解析异步终端表或只使用 W&B 最后一行。它导出逐步 CSV、包含有限性和
极值/均值的 JSON，以及 return、成功、policy loss、gradient norm、overlap loss 和 critic
loss 六联曲线 PNG。该工具已在正在增长的 C0 event 上验证，成功导出前 5 次更新且所有
记录值均为有限数。

基础评测只能回答 success/return/episode length，不能区分“提前掉落”“已到位仍夹持”，
也不能量化 chunk 边界抖动。为此任务环境又增加只走 `info` 通道的特权诊断：是否形成可靠
夹持、持物到达释放体积、合法／非法释放、硬失败、终态仍夹持、释放延迟及其 censor 标记，
以及六关节目标的一阶变化、二阶差分和 chunk 边界二阶差分。它们不进入 RGB/28 维 proprio，
也不参与 reward。RLinf wrapper 只接受每个环境恰好一个标量的 `panthera_*` 字段并把它们
写入 episode metrics；错误宽度会立即失败。RoboTwin 相关回归为 `12 passed`，RLinf 聚合
合同和汇总器测试也通过。增强版隔离 runtime 为
`runtime/robotwin-rl-metrics-next-20260924/` 与
`runtime/rlinf-rl-metrics-next-20260924/`，两份固定上游在装配后仍保持 clean。

后续又补齐了冻结 SFT 的 R0 同口径入口。R0 不加载 RL checkpoint，而是在相同的连续策略
wrapper 中使用零初始化 residual head 和 deterministic mean；CPU 回归验证该输出在数值容差
内等于 SFT mean。runner 要求 `eval_frozen_sft_baseline=true` 与 `ckpt_path` 严格二选一，
防止 checkpoint 拼错时静默退化成未训练策略。R0/C0--C2 launcher 元数据与完成标记测试为
`5 passed`，零残差/连续策略测试为 `10 passed`，含 R0 的 TensorBoard 汇总测试为
`2 passed`。新补丁已装配并通过 Hydra 解析于独立临时 runtime
`runtime/rlinf-rl-gpu23-r0-next-20260924/`；实际 R0 rollout 尚未运行，也未替换当前训练或
已排队 C0--C2 评测使用的 runtime。

### 5.1 十更新短程矩阵的中途结果

R2-C0 已完成 10/10 次更新并以退出码 0 写出
`global_step_10/actor/model_state_dict/full_weights.pt`，文件大小为 15,425,324,245 bytes。
TensorBoard 的 10 条 return、success、policy loss、gradient norm 和 critic loss 均为有限值；
训练 rollout 的 `success_once` 全部为 0，return 均值为 `-0.13965`，范围为
`[-1.14015, 0.11616]`。这证明短程训练和保存链路完整，不证明策略成功。

R2-C1 随后也完成 10/10 次更新、以退出码 0 写出同样大小的 step-10 checkpoint。它的
10 条 TensorBoard 指标全部有限：return 均值为 `-0.26046`，范围为
`[-0.54507, 0.62572]`；`chunk_overlap_loss` 从最高 `0.09825` 降到最后的 `0.02261`；
`grad_norm` 范围为 `[0.76139, 1.72571]`。第 2 次更新的两条 stochastic 训练轨迹中有一条
成功，记录为 `success_once=0.5`，其余九次更新均为 0。因此 C1 只能证明短训练中曾采到
一次成功，不能替代独立 deterministic 评测。R2-C2 已自动接续，展开配置核验为
`variant=c2`、R2 reward、history noise `0.01`、history dropout `0.10`、10 次更新；主进程
和各 Ray worker 仍只映射到物理 GPU 2、3。

## 6. 尚未完成

- R2-C0/C1/C2 的一步同口径 smoke 已完成，但任务成功均为零；10 次 PPO 更新的短程对照中
  C0、C1 已完整完成，C2 正在运行，根目录为
  `ci/openvla-rl-release-arena-10step/20260924T063243Z/`，只使用物理 GPU 2、3。
  对应的增强独立闭环评测与视频已排队到
  `ci/openvla-rl-release-eval/20260924T063243Z-step10/`，服务
  `panthera-rl-r2arena-metrics-eval-after-10step-20260924T063243Z.service` 会在训练完整成功后
  用上述隔离 runtime 串行运行 R0/C0/C1/C2；等待阶段不占 GPU。
- 冻结 SFT R0 的无 checkpoint 评测入口、临时 runtime 和回归测试已完成，但实际 R0 rollout
  尚未执行；它和专家／立即开爪／始终不开爪三种 PhysX reward replay 已排在 C0--C2
  独立评测之后自动执行，服务为
  `panthera-rl-r2arena-metrics-post-audits-20260924T063243Z.service`。后置 launcher、汇总与 reward
  replay 合计 `9 passed`，等待阶段不占 GPU，执行阶段只使用物理 GPU 2、3。同一 release
  arena 的稀疏 R1 训练对照仍未执行。
- 尚未使用三个训练 seed，也没有执行反向课程的“合法释放位／近但不合法／搬运途中”三层
  混采。
- 尚未获得 release arena 或完整任务成功，不得进入真机，也不得把非零 shaping return
  当作任务完成。
- 只有在 C0/C1/C2 同时报告任务成功、提前释放、到位仍夹持、overlap error、边界 jerk 和
  释放延迟后，才能选择长训练候选。
