# 23 · OpenPI π0.5 迁移与同口径基线（2026-09-21）

## 1. 决策与当前状态

OpenVLA-OFT 不被删除，继续作为已经投入训练和评测的基线；π0.5 作为新的候选策略。
两者只比较低层策略能力，不把任务专用状态机、额外相机或不同执行器混进对照实验。

本轮只完成代码与本地合同测试，没有占用 Lab GPU、没有停止或重启正在运行的 OpenVLA
24 小时训练，也没有宣称 π0.5 已经训练成功。WSL 到 Lab 的 SSH 通道在本轮末尾超时，
所以后台 unit 的即时状态尚未重新读取。

## 2. 固定比较合同

两种模型都读取当前单帧 RGB、当前 7 维 proprio 和同一语言指令，并输出 25×7 动作块。
外部七维顺序固定为 `joint1..joint6 + gripper`，语义固定为六关节绝对位置加绝对夹爪量。

π0.5 的内部变换为：

1. 手机/头部相机放入 `base_0_rgb`；两个腕部相机槽填黑并将 mask 设为 false；
2. 先在物理 7 维上计算 normalization statistics；
3. 前六个绝对关节目标相对当前 state 转成 delta，夹爪保持绝对；
4. OpenPI 官方 `PadStatesAndActions` 再把 state/action 补到模型的 32 维；
5. 推理输出先恢复为绝对关节目标，再裁掉 padding，只把 7 维交给 Panthera executor。

因此，三相机是 OpenPI 的固定输入槽，不是三台物理相机要求；当前单手机相机输入合法。
也没有把 DROID 的关节速度 action 语义套到 Panthera 数据上。

## 3. 已落地代码

- `packages/panthera_vla/openpi_adapter.py`：单相机 mask、输入校验和 7 维输出裁剪；
- `packages/panthera_vla/panthera_lerobot.py`：从正式 snapshot 数据生成 LeRobot 数据；
- `packages/panthera_vla/openpi_config.py`：运行时构造 π0.5 full/LoRA 配置，不修改 OpenPI；
- `packages/panthera_vla/run_openpi_finetune.py`：normalization stats 与 JAX/PyTorch 训练入口；
- `packages/panthera_sim/policy_backends.py`：OpenVLA-OFT/π0.5 统一预测接口；
- `packages/panthera_sim/rollout.py`：两种 backend 共用同一个场景、executor 和成功判据；
- `packages/panthera_sim/parity.py`：记录帧/渲染帧一致性门禁改为 backend 无关；
- `packages/panthera_sim/compare_policy_reports.py`：只允许同数据摘要、同 episode、同预算、
  同 execution horizon 和同 temporal ensemble 的报告进入比较。

LeRobot 转换没有复制时间对齐算法，而是复用 RLDS 的全局 50 Hz 网格和“当前 observation
对应下一网格 target”规则。输出目录已存在时会拒绝覆盖；转换 sidecar 会保存源数据摘要、
action 语义和 Panthera episode 到 LeRobot episode 的映射。

## 4. Lab 执行顺序

以下是恢复 Lab 连接后的顺序，不应与当前 OpenVLA 训练争抢 GPU。OpenPI checkout 和环境
必须先固定到明确提交；所有可写数据仍放在 `/data/lyy`。

```bash
export PANTHERA_ROOT=/data/lyy/panthera-vla
export LEROBOT_HOME=$PANTHERA_ROOT/lerobot
export REPO_ID=panthera/schema10

python packages/panthera_vla/panthera_lerobot.py \
  --source-root /data/lyy/panthera-vla/data/place_randomized_cylinder_in_socket/panthera_phone_cylinder_socket_v2_single_grasp_sft_v2_fixedcam \
  --output-root "$LEROBOT_HOME/$REPO_ID" \
  --repo-id "$REPO_ID"

python packages/panthera_vla/run_openpi_finetune.py stats \
  --openpi-root /data/lyy/openpi \
  --lerobot-home "$LEROBOT_HOME" --repo-id "$REPO_ID" \
  --checkpoint-base-dir /data/lyy/panthera-vla/openpi-checkpoints \
  --assets-base-dir /data/lyy/panthera-vla/openpi-assets

python packages/panthera_vla/run_openpi_finetune.py train \
  --openpi-root /data/lyy/openpi \
  --lerobot-home "$LEROBOT_HOME" --repo-id "$REPO_ID" \
  --checkpoint-base-dir /data/lyy/panthera-vla/openpi-checkpoints \
  --assets-base-dir /data/lyy/panthera-vla/openpi-assets \
  --exp-name schema10-lora --lora --action-horizon 25 \
  --save-interval 5000 --keep-period 5000
```

训练脚本仍使用固定计算预算和周期 checkpoint；不会恢复以 validation L1 下降幅度作为正式
停止条件。多卡 JAX 训练由可见设备与 `--fsdp-devices` 控制；PyTorch 多卡则应由 OpenPI
官方 `torchrun` 入口承载，本 wrapper 可在各 rank 中调用同一 `train_loop`。

## 5. 验收门禁

正式启动 π0.5 长训练前依次要求：

- OpenPI 固定提交、环境和基础 checkpoint 下载完成；
- 转换后的 episode 数、帧数、源摘要和随机抽样图像/action 与 RLDS 一致；
- normalization stats smoke 和单 batch 前向/反向通过；
- episode 2 小计算预算 overfit 能生成 checkpoint；
- 记录帧和渲染帧 parity 都通过后，才运行同场景闭环；
- 通过单轨迹门禁后，才在相同 schema 10 train/eval split 上与 OpenVLA-OFT 对比；
- 比较报告必须由 `compare_policy_reports.py` 验证合同相同，最终仍以闭环成功率选模。

当前未完成项是 OpenPI 上游 pin/环境、LeRobot 正式转换、π0.5 smoke、单轨迹训练和闭环结果。
这些未完成前，π0.5 只是已经接好接口的候选 backend，不是新的默认 baseline。
