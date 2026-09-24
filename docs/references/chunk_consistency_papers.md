# 动作 chunk 历史条件与时序一致性论文索引

检索日期：2026-09-24

本索引记录为 Panthera 单臂圆柱入槽任务调查“把上一 chunk 加入策略，并在训练中保持
跨 chunk 一致性”时筛选的原始论文。PDF 位于被 Git 忽略的
`docs/references/papers/chunk_consistency/`；索引、来源和 SHA-256 由 Git 跟踪。

| 年份 | 论文 | 本地 PDF | 原始来源 | SHA-256 |
|---|---|---|---|---|
| 2025 | Real-Time Execution of Action Chunking Flow Policies | [PDF](papers/chunk_consistency/2025_Black_RTC.pdf) | [NeurIPS/arXiv](https://arxiv.org/abs/2506.07339) | `dd6d602be1f50b894d88517bf68916c4272885b2e29d96cffc426dacff41898f` |
| 2026 | Action-Prior Denoising for Smooth Real-Time Chunking | [PDF](papers/chunk_consistency/2026_Liu_Soft_RTC.pdf) | [arXiv](https://arxiv.org/abs/2605.25537) | `7fed2b487b899706ea2e752c3dfa31d9e9552571df8d704e7642da7eb1b3e303` |
| 2026 | SEAM: Smooth Execution of Action-Chunked Motion for Vision-Language-Action Policies | [PDF](papers/chunk_consistency/2026_Zhan_SEAM.pdf) | [arXiv](https://arxiv.org/abs/2607.04609) | `39eea20983bf492624ae0861e9de1bf037024ab71702faec375eed28623ca1ef` |
| 2026 | ChunkFlow: Towards Continuity-Consistent Chunked Policy Learning | [PDF](papers/chunk_consistency/2026_Yang_ChunkFlow.pdf) | [arXiv](https://arxiv.org/abs/2607.12992) | `dcb450b15b50f7d41e86433607c91845a8cd27a357e7c4aca301ecda2b4fd258` |

四份文件均通过 `file` 识别为 PDF。RTC 是 NeurIPS 2025 论文；另外三份是 2026 年预印本，
应视为实现候选而不是已经普遍成立的定论。RTC、Soft RTC 和 SEAM 主要针对 diffusion/flow
动作头；Panthera 当前是连续 L1/残差头，不能直接复制其 denoising/inpainting 算法。
ChunkFlow 的已执行动作历史、对齐重叠边界损失、动作一阶/二阶连续性、历史扰动和在 RL
阶段保留这些正则的结构与当前方案最接近。项目适用性分析见
[OpenVLA 释放切换强化学习奖励设计](../27_openvla_rl_release_reward_design_2026-09-24.md)。
