# OpenVLA-OFT 多卡 action head 不同步：复现、修复与影响报告

日期：2026-09-23

适用范围：Panthera 单机械臂圆柱入槽、OpenVLA-OFT 连续动作头、多 GPU 数据并行训练

修复分支：`fix/openvla-ddp-action-head-sync`

## 1. 结论

Panthera 之前使用的 OpenVLA-OFT 多卡训练路径存在一个已经在 Lab 上复现的分布式训练错误。训练程序把连续动作头包装成 PyTorch `DistributedDataParallel`（DDP）后，却通过
`action_head.module.predict_action(...)` 或 `action_head.module.predict_noise(...)` 直接调用底层模块。这个调用没有经过 DDP 包装器的 `forward` 入口，所以各卡 action head 的梯度没有执行跨卡归并。

三张 RTX 4090 上的一步优化器对照实验显示：保持初始化、每卡 batch size、学习率和输入形状不变，只把调用方式从底层模块改为 DDP `forward`，L1 action head 的跨卡参数最大绝对差异由 `0.001000002` 变为 `0`，同一输入的跨卡输出最大绝对差异由 `0.171851695` 变为 `0`。修复后的正式回归又覆盖了 diffusion action head；其跨卡参数和输出差异也都为 `0`。完整数值保存在 [修复前证据](evidence/openvla_action_head_ddp_prefix_2026-09-22.json) 和 [修复后证据](evidence/openvla_action_head_ddp_postfix_2026-09-23.json)。

修复已经进入受管 overlay，并装配到 Lab 的实际 OpenVLA-OFT 环境。它同时覆盖 L1 与 diffusion 训练，但没有改变单卡推理接口、动作维度、动作块长度、损失函数或 checkpoint 文件格式。

这个结果不能直接证明历史闭环抖动完全由 DDP 错误造成。它能证明的是：此前使用三卡训练的连续 action head 在每个 rank 上已经分叉，最终 checkpoint 只保存 rank 0 的版本。因此，修复前的多卡训练结果不能继续作为“正确多卡 SFT”的基线；必须用修复后的训练器重新训练或至少做严格的成对续训对照，才能量化该错误对抖动、松爪和闭环成功率的贡献。

## 2. 任务和软件边界

当前任务是一台 Panthera 机械臂使用一个夹爪把圆柱插入凹槽。策略输出仍是每步 7 个值：六个关节目标和一个归一化夹爪目标。当前 OpenVLA-OFT 训练一次预测 `25 × 7` 的动作块；28 维动力学 proprio 只改变输入状态，不改变动作宽度。

本文所说的连续动作头（action head），是接在视觉—语言模型隐藏特征之后、把特征转换为 `25 × 7` 连续动作数值的神经网络。proprio 指机器人自身状态；当前 28 维输入由关节位置、速度、加速度和力矩四个 7 维块组成。多卡训练为每张 GPU 启动一个进程，分布式训练把每个进程称为一个 rank。

本次问题位于训练器内部，不涉及 RoboTwin 物理仿真、真实机械臂驱动或相机输入。修复和验证均未启动真机控制。

实际 Lab 软件版本如下；OpenVLA-OFT 的仓库和完整提交来自已安装 Python distribution 的 `direct_url.json` [3]：

| 项目 | 版本或提交 |
|---|---|
| OpenVLA-OFT Python 包 | `0.0.1` |
| OpenVLA-OFT 安装来源 | `RLinf/openvla-oft@c9f0f3d31f438d98f6137936ea78a47f0b2ab087` |
| PyTorch | `2.11.0+cu130` |
| GPU | 3 × NVIDIA GeForce RTX 4090（回归使用 GPU 0、1、2） |
| 动作合同 | action chunk `25`，action dimension `7` |
| 测试 batch | 每卡 `6`，全局 `18` |
| 测试学习率 | `5e-4` |

## 3. DDP 原本应当完成什么

DDP 为每个 GPU 进程保留一份相同的模型。每张卡读取不同的局部 batch，分别计算局部梯度；反向传播期间，DDP 对对应参数的梯度做跨 rank 归并，使所有进程在执行相同优化器步骤后仍持有相同参数。PyTorch 官方文档明确把 DDP 构造、`forward` 和对其输出求导列为分布式同步点，并说明 DDP 通过同步梯度维持各副本一致 [1]。

设第 `r` 张卡的 action head 参数为 `θ`，局部 batch 产生梯度 `g_r`。正确的数据并行更新使用所有 rank 的归并梯度：

```text
g = (g_0 + g_1 + ... + g_(N-1)) / N
θ' = optimizer(θ, g)
```

如果每张卡直接调用 `DDP.module`，每个优化器看到的仍是自己的局部梯度：

```text
rank 0: θ'_0 = optimizer(θ, g_0)
rank 1: θ'_1 = optimizer(θ, g_1)
rank 2: θ'_2 = optimizer(θ, g_2)
```

因为三个 batch 不同，`g_0`、`g_1`、`g_2` 通常不同，一步之后三份 action head 就不再相同。

## 4. 根因

### 4.1 故障调用路径

安装环境中的 `vla-scripts/finetune.py` 原来执行：

```python
predicted_actions = action_head.module.predict_action(actions_hidden_states)
noise_pred = action_head.module.predict_noise(actions_hidden_states)
```

`action_head` 是 DDP 包装器，`action_head.module` 是被包装的普通 `nn.Module`。上面两行绕过了 DDP 自己的 `__call__`/`forward` 路径。OpenVLA-OFT 上游 issue #160 描述了相同调用和影响，但在本项目决定修改前，我们没有把 issue 当作事实，而是在实际 Lab 版本上做了独立对照实验 [2]。

### 4.2 为什么训练没有立即报错

底层 action head 仍是合法的 PyTorch 模块，局部前向、局部反向传播和优化器更新都能运行。错误表现为各 rank 静默分叉，而不是必然抛异常。VLA 主干仍通过自己的 DDP 包装器同步，但它接收到的是经过不同 rank-local action head 计算的反向信号。训练 loss 继续下降、rank 0 也能保存 checkpoint，所以只观察“训练是否运行”或一条 loss 曲线无法发现这个问题。

checkpoint 保存逻辑只在 `distributed_state.is_main_process` 为真时执行，并保存
`action_head.state_dict()`。因此，分叉后只有 rank 0 的 action head 进入 checkpoint；其他 rank 的版本随进程结束而丢失。

## 5. 修复前实验：验证 issue 是否真实影响 Panthera

### 5.1 设计

诊断程序直接导入当前环境的 `L1RegressionActionHead`，使用三张 GPU 和三个进程。每个 rank 以相同随机种子创建模型，DDP 初始化后模型参数相同；随后每个 rank 用不同随机种子生成局部 batch。这个最小实验没有加载完整 VLA 或 RLDS，但保留了判断 action head 是否需要梯度归并的关键条件：各 rank 初始参数相同，局部输入和局部梯度不同。

为了让诊断在数十秒内完成，L1 头使用输入宽度 `64`、隐藏宽度 `128`，diffusion 头使用输入和隐藏宽度 `64`；两者仍直接实例化实际安装包中的类，动作块和动作宽度保持 `25 × 7`。模型种子为 `20260922`，rank-local batch 种子为 `1000 + rank`，三卡共同验证输入的种子为 `424242`。这些缩小的隐藏宽度不用于报告策略性能，只用于检查同一组实际参数是否被 DDP 归并。

实验只改变一项：

- 故障组调用 `wrapped.module.predict_action(hidden)`；
- 对照组为同一个 action head 临时补充 `forward`，调用 `wrapped(hidden)`。

两组都执行一次 loss、`backward()` 和 AdamW `step()`。实验记录三个量：

1. 三张卡同一层梯度之间的最大绝对差异；
2. 一步更新后同一层参数之间的最大绝对差异；
3. 更新后给三张卡相同输入时，输出之间的最大绝对差异。

### 5.2 结果

| 调用路径 | 梯度跨卡最大差异 | 参数跨卡最大差异 | 同一输入输出最大差异 |
|---|---:|---:|---:|
| 原训练路径：绕过 DDP | `0.083215743` | `0.001000002` | `0.171851695` |
| 对照路径：经过 DDP `forward` | `0` | `0` | `0` |

两个路径使用相同初始权重、三个相同规模的 rank-local batch、相同 L1 loss 和相同 `5e-4` 学习率。差异只来自是否经过 DDP `forward`。这个对照足以确认 issue 所述机制在 Panthera 的实际软件组合中发生。

## 6. 实现修改

### 6.1 为两种 action head 提供标准 `forward`

补丁在 `prismatic/models/action_heads.py` 中分别加入：

```python
def forward(self, actions_hidden_states):
    return self.predict_action(actions_hidden_states)
```

以及：

```python
def forward(self, actions_hidden_states):
    return self.predict_noise(actions_hidden_states)
```

原有 `predict_action` 和 `predict_noise` 保留，因此推理代码及已有 checkpoint 加载逻辑无需改接口。`forward` 只把训练调用引入 DDP 能识别的标准入口。

### 6.2 训练调用经过 DDP 包装器

`run_forward_pass` 中的两条训练路径改为：

```python
predicted_actions = action_head(actions_hidden_states)
noise_pred = action_head(actions_hidden_states)
```

diffusion 反向采样函数中仍有一处 `action_head.module.predict_noise(...)`。该代码位于生成验证动作的无梯度采样循环，不参与 action head 参数的 `backward()`，所以本次没有为了形式统一而引入多余的 DDP 训练通信。静态契约测试只禁止参数化训练路径重新使用 `.module.predict_*`。

### 6.3 受管补丁与实际环境

OpenVLA-OFT 安装在 RLinf 虚拟环境的 `site-packages`，不是本仓库的普通源码检出。正式修改因此保存在：

```text
overlays/openvla/patches/openvla_oft_finetune_and_ddp.patch
```

这里的 overlay 是 Panthera 仓库保存的自有修改层。bootstrap 先准备固定版本的上游软件，再把 overlay 中的补丁应用到运行环境；这样不需要直接改写上游检出，也能重新生成相同 runtime。

原 `openvla_oft_validation_early_stopping.patch` 也修改 `finetune.py`。仓库契约规定同一个上游文件只能由一个补丁拥有，否则补丁顺序会使后续装配失效。因此本次把早停、学习率修复和 DDP 修复合并为同一个 `finetune` 补丁，没有新增第二个同时修改该文件的补丁。

应用补丁时还发现，Lab 实际 `constants.py` 曾被改成支持 `PANTHERA_PROPRIO_DIM`，但 bootstrap 状态目录仍记录更旧的固定 7 维补丁。bootstrap 正确拒绝了混合状态。处理方式是从固定上游文件与实际安装文件生成精确逆补丁，恢复上游基线；将失效记录移到
`state/bootstrap-state/stale-openvla-patches-20260923/`；再重新生成可正向、反向应用的 constants 补丁。第二次 bootstrap 能识别两个补丁均已应用，记录副本与 overlay 的 SHA-256 也完全一致。

## 7. 修复后回归

正式回归脚本为：

```text
packages/panthera_vla/audit_openvla_action_head_ddp.py
bin/run_lab_openvla_action_head_ddp_audit.sh
```

它同时测试 L1 与 diffusion action head。每类 action head 都运行两个 case：

- `module_bypass_negative_control`：故意绕过 DDP，确认测试能观察到分叉；
- `ddp_forward`：使用修复后的正式路径，要求梯度、参数和输出的跨卡最大绝对差异均不超过 `1e-7`。

| action head | case | 梯度差异 | 参数差异 | 同一输入输出差异 |
|---|---|---:|---:|---:|
| L1 | 绕过 DDP 的负对照 | `0.083215743` | `0.001000002` | `0.171851695` |
| L1 | 修复后的 DDP 路径 | `0` | `0` | `0` |
| Diffusion | 绕过 DDP 的负对照 | `0.130688921` | `0.001000002` | `0.098186493` |
| Diffusion | 修复后的 DDP 路径 | `0` | `0` | `0` |

回归最终写出 `status: pass`。负对照仍然分叉，说明“全零”不是因为三个 rank 恰好拿到相同 batch，也不是测试失去检测能力。

补丁还通过以下检查：

- runtime bootstrap 首次应用成功，随后幂等复跑成功；
- 实际安装的 L1 与 diffusion 类都存在新 `forward`；
- 实际训练调用位于 DDP 包装器，只有无梯度 diffusion sampling 保留 `.module`；
- 聚焦仓库契约测试为 `2 passed`；
- 状态目录和 overlay 中两个 OpenVLA 补丁的 SHA-256 分别一致。

完整仓库契约测试初次运行时另外暴露了一个仓库边界问题：Lab 的 README 和 AGENTS 引用
WSL 技术文档，Lab 工作树还跟踪了整套 `docs/`。短暂补齐缺失文档后虽得到 `96 passed`，
但那是在错误边界上让测试变绿。最终修法是让技术报告只由 WSL 源仓库持有，Lab
`gpu_node` 只保留最小运行说明；仓库角色合同在 Lab 上反向要求 `docs/` 不存在。DDP
审计的机器可读结果和原始日志仍保存在 Lab 的 `reports/evidence/`。

### 7.1 独立的 OpenVLA-OFT PR 分支

Panthera overlay 用于复现实验环境，但不是向上游提交 PR 的分支。上游修复另建在独立
checkout：

```text
Lab checkout: /data/lyy/panthera-vla/third_party/openvla-oft-pr
upstream:     https://github.com/moojink/openvla-oft.git
base:         upstream/main@e4287e94541f459edc4feabc4e181f537cd569a8
branch:       fix/ddp-action-head-forward-sync
fix commit:   bf0b44c53cab576fe8c2e3d5f38dc06666e75528
review fix:   2c767d928fbfdf57f100b8840b6e3b4a9d5e5559
```

该提交只包含上游可接受的最小修改：两个 action head 的 `forward()`、训练路径经过 DDP
wrapper，以及独立的两 rank CPU/Gloo 回归测试。它不包含 Panthera 专用常量、早停、学习率、
外部停止文件、数据合同或报告。测试对 L1 和 diffusion 分别使用不同 rank-local batch，
执行一步 AdamW 后要求所有参数逐元素完全相同。Copilot review 指出测试注入的合成
`prismatic` 包会残留在 `sys.modules`；后续提交在 `finally` 中逐项恢复原模块状态，并增加
导入隔离回归。最终结果为 `3 passed, 2 subtests passed`。
新增测试通过 Ruff lint/format，完整 diff 通过 `git diff --check`。上游原有两个修改文件
自身存在与本次无关的旧 Ruff 告警，因此没有借本 PR 重排或清理那些文件。

该分支已推送到 `5o1/openvla-oft`，并向 `moojink/openvla-oft:main` 创建
[PR #162](https://github.com/moojink/openvla-oft/pull/162)。PR 为非 draft 状态，包含功能修复
和 review 后的测试隔离修复；创建时上游没有配置自动 status check。

## 8. 对历史 checkpoint 的影响

`dynamics-28d-ep2-24h-20260921T140150Z` 明确使用 GPU 1、2、3，每卡 batch size `6`、学习率 `5e-4`、L1 continuous action head，并每 5000 步保存 checkpoint。它走过本报告复现的故障路径，因此该运行中的三个 action head 在训练期间并不相同，保存产物是 rank 0 的版本。

这带来四个结论：

1. 该运行仍可作为“故障训练器产生过什么行为”的诊断材料，视频也不需要删除。
2. 它不能作为正确三卡训练的性能基线，不能用其结果否定或确认 28 维 proprio 的价值。
3. step 15000、40000、65000 等 checkpoint 的成功或抖动现象是真实观察，但不能把原因只归结为数据、时间窗口、摄像机或 loss；训练器本身已经违反多卡一致性前提。
4. 本次一步实验没有量化修复后长训练的抖动幅度，也没有证明 DDP 分叉是抖动的唯一原因。要回答该问题，必须保持数据、seed、学习率、batch、checkpoint 和评测口径不变，重新训练后做成对比较。

单卡训练不存在“rank 之间不同步”这一故障。它仍可能有数据覆盖、闭环分布偏移或动作连续性问题，但不能与本 DDP 根因混为一谈。

## 9. 复现方法

以下命令都在 Lab 普通用户 `lyy`、目录 `/data/lyy/panthera-vla` 下运行，不使用管理员账户，也不访问真机：

```bash
cd /data/lyy/panthera-vla

# 从固定上游、overlay 和补丁重建 runtime；重复执行也应成功。
bash tools/bootstrap_lab_vla.sh

# 默认使用 GPU 0、1、2，运行 L1 和 diffusion 的三卡回归。
bash bin/run_lab_openvla_action_head_ddp_audit.sh

# 只检查与本修复直接相关的仓库契约。
runtime/rlinf/.venv/bin/python -m pytest -q \
  pipelines/test_repository_contract.py \
  -k "openvla_patch_keeps_the_hard_budget or openvla_action_heads_train_through_ddp_forward"
```

验收条件是 JSON 的顶层 `status` 为 `pass`，且两个 action head 的 `ddp_forward` 三项跨卡差异都不超过 `1e-7`。负对照必须至少有一项大于 `1e-7`；否则测试没有证明它能识别同步缺失。

一键入口会用 `nvidia-smi` 检查所选 GPU 上的计算进程；任一卡已被占用就直接退出，不会抢占或结束现有进程。通过 `PANTHERA_DDP_AUDIT_GPUS` 选择至少两张空闲卡后可以重试：

```bash
PANTHERA_DDP_AUDIT_GPUS=2,3 \
  bash bin/run_lab_openvla_action_head_ddp_audit.sh
```

## 10. 后续门禁

本次没有自动启动新的 24 小时训练。下一次正式 OpenVLA 多卡训练应满足：

1. 先运行本报告的 DDP audit 并保存 `status: pass` 的 JSON；
2. 从共同基线重新训练，作为判断修复影响的主实验；从旧 rank 0 checkpoint 续训只能作为恢复能力的附加实验；
3. 保持 episode 2、`25 × 7` action、28 维 proprio、每卡 batch `6` 和学习率 `5e-4`，使结果能与旧运行成对比较；
4. 继续每 5000 步保存 checkpoint，并以同一闭环评测和视频口径比较 step 15000、40000、65000；
5. 分别报告闭环成功、松爪、复位和动作抖动，不能用验证 L1 单独选模；
6. 在修复后的成对训练完成前，不把旧三卡 checkpoint 用作真机迁移依据。

### 10.1 修复后重训已启动

2026-09-23 启动前，在 GPU 1、2、3 上重新执行 action-head DDP audit；L1 和 diffusion
修复路径的梯度、参数以及共同输入输出跨 rank 最大差异均为 `0`，机器可读证据为：

```text
/data/lyy/panthera-vla/reports/evidence/openvla_action_head_ddp_preflight_20260922T173641Z.json
```

随后从共同基础模型启动 28 维 episode 2 三卡重训，保持 `25 × 7` action、每卡 batch 6、
学习率 `5e-4`、24 小时预算和每 5000 步 checkpoint。后台单元、运行目录与 W&B run 为：

```text
panthera-overfit-ddpfix-20260922T173901Z.service
/data/lyy/panthera-vla/ci/overfit-ep2-dynamics-24h/ddpfix-dynamics-28d-ep2-24h-20260922T173901Z/
https://wandb.ai/assanekowww/panthera-ci-overfit-dynamics-24h/runs/1s35hyv6
```

启动核查时 GPU 1–3 均为 100% 利用率并已越过首批优化步骤。该记录只证明任务使用修复
训练器正常启动；抖动、松爪、复位和闭环成功仍须由后续编号 checkpoint 评测判定。
`panthera-overfit-ddpfix-eval-20260922T173901Z.service` 已同时启动并绑定本次运行；它在 GPU0
等待每个 5000-step checkpoint，生成 h1/h20 双模式预览，训练结束后生成总拼接视频。

### 10.2 训练收口与扩展预算重评

修复后训练按 24 小时墙钟预算正常结束，训练子进程的 `124` 是 `timeout` 到达预算的预期
状态；外层流水线验收 checkpoint 后以 `0` 收口并写出 `TRAINING_COMPLETE`。最终保存
step 5000–85000 共 17 个完整 checkpoint。

原评测把 episode 2 的 1037 帧专家轨迹长度同时当成准时门禁和强制终止上限。这只能回答
“是否不慢于专家”，不能区分“策略不会完成”和“策略需要恢复时间”。新口径在同一次 rollout
中保留 `expert_action_budget=1037`，但把 `evaluation_action_budget` 扩展到 2074，并记录：

- `on_time_success`：不超过 1037 步成功，计入原严格门禁；
- `delayed_success`：1038–2074 步成功，只计入诊断成功；
- `failure`：2074 步内仍未成功。

旧自动队列还暴露了一个独立竞态：step 80000 目录出现但权重尚未写完时，底层评测选择
“本轮跳过”，上层仍立即拼接不存在的视频并退出。队列现先验证 adapter、action head、
proprio projector 和统计文件全部非空，再启动双模式评测。

全部 17 个 checkpoint 的扩展预算重评已写入新的、不可覆盖的结果目录：

```text
/data/lyy/panthera-vla/ci/overfit-ep2-dynamics-24h/
  ddpfix-dynamics-28d-ep2-24h-20260922T173901Z/
  extended-budget-eval-20260923T214752Z/
```

启动时 GPU0、1 曾被同一 Lab 用户的另一个 `patchconvmoe` 实验占用，因此 Panthera 没有
抢占；该任务结束后改为 GPU0–3 四卡并行。当前后台单元为
`panthera-ddpfix-extended-eval-20260923T220819Z.service`。完成标志应为
`EVALUATION_COMPLETE`，并应有 17 个双模式视频和一个总拼接视频；在这些条件满足前不把
重评标记为完成。

第一次双卡启动还发现 GPU 编号契约缺失：空闲检查采用 `nvidia-smi` index，而 CUDA 未固定
设备枚举顺序，名义上的 `2,3` 实际撞到物理 GPU0、1 上的外部任务并 OOM。该次服务在产生
正式 rollout 结果前已停止。评测器现强制 `CUDA_DEVICE_ORDER=PCI_BUS_ID`；分别在 CUDA
ordinal 2、3 做短暂显存分配后，`nvidia-smi` 观察到的 UUID 与物理 index 2、3 完全一致，
但随后的进程审计又发现两个独立覆盖点：rollout 的 multiprocessing 初始化会按 `--gpus`
重写 `CUDA_VISIBLE_DEVICES`，而 checkpoint 合并此前根本没有设置设备。评测器现把 worker
ordinal 同时传给 `--gpus`、rollout 环境、渲染环境和合并环境。重启后，GPU0、1、2 的
活跃合并进程分别出现在对应物理 UUID 上，GPU3 worker 已携带
`CUDA_VISIBLE_DEVICES=3` 进入闭环 rollout。此前错误服务均未留下正式完成结果。

## 11. 参考资料

[1] PyTorch, “DistributedDataParallel,” 官方文档，访问于 2026-09-23：<https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html>。

[2] TJ12342, “Action head gradients are not synchronized in multi-GPU training because DDP forward is bypassed,” OpenVLA-OFT issue #160, 2026-07-29，访问于 2026-09-23：<https://github.com/moojink/openvla-oft/issues/160>。

[3] RLinf OpenVLA-OFT 安装来源提交 `c9f0f3d31f438d98f6137936ea78a47f0b2ab087`：<https://github.com/RLinf/openvla-oft/tree/c9f0f3d31f438d98f6137936ea78a47f0b2ab087>。

## 12. 项目内证据索引

- 修复补丁：`overlays/openvla/patches/openvla_oft_finetune_and_ddp.patch`
- 运行回归：`packages/panthera_vla/audit_openvla_action_head_ddp.py`
- Lab 一键入口：`bin/run_lab_openvla_action_head_ddp_audit.sh`
- 静态契约：`pipelines/test_repository_contract.py`
- 修复前 JSON：`docs/evidence/openvla_action_head_ddp_prefix_2026-09-22.json`
- 修复后 JSON：`docs/evidence/openvla_action_head_ddp_postfix_2026-09-23.json`
- Lab 原始修复后日志：`/data/lyy/panthera-vla/reports/evidence/openvla_action_head_ddp_postfix_2026-09-23.log`
- OpenVLA-OFT PR checkout：`/data/lyy/panthera-vla/third_party/openvla-oft-pr`
- OpenVLA-OFT PR 提交：`bf0b44c53cab576fe8c2e3d5f38dc06666e75528`
- OpenVLA-OFT PR：<https://github.com/moojink/openvla-oft/pull/162>
- 受影响历史运行配置：`/data/lyy/panthera-vla/ci/overfit-ep2-dynamics-24h/dynamics-28d-ep2-24h-20260921T140150Z/run-config.json`
