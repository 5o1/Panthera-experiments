# 强化学习释放切换与奖励设计论文索引

检索日期：2026-09-24

本索引记录为 Panthera 单臂圆柱入槽任务的强化学习奖励设计筛选的原始论文。PDF 位于被
Git 忽略的 `docs/references/papers/rl_reward_design/`；索引、来源和 SHA-256 由 Git 跟踪。
这些论文分别提供势函数奖励塑形、稀疏奖励探索、反向课程、示范状态重置、显式奖励机、
延迟信用分配、残差强化学习，以及 VLA 强化微调中的过程奖励、阶段奖励和自采成功轨迹
辅助行为克隆证据。

| 年份 | 论文 | 本地 PDF | 原始来源 | SHA-256 |
|---|---|---|---|---|
| 1999 | Policy Invariance Under Reward Transformations | [PDF](papers/rl_reward_design/1999_Ng_Harada_Russell_Policy_Invariance_Reward_Shaping.pdf) | [作者主页](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | `a641bf38a15f97d0ab182cb99a8257eafb9abca0d011434fa0ce1904b322d4e3` |
| 2017 | Hindsight Experience Replay | [PDF](papers/rl_reward_design/2017_Andrychowicz_Hindsight_Experience_Replay.pdf) | [NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html) | `0494b3b65ee3246f9b23225147b7820a0f985145b11f1962538562947bddbc67` |
| 2017 | Reverse Curriculum Generation for Reinforcement Learning | [PDF](papers/rl_reward_design/2017_Florensa_Reverse_Curriculum_Generation.pdf) | [PMLR](https://proceedings.mlr.press/v78/florensa17a.html) | `a33975824bd6f34af0984494d4678c16d28fd3daf533b7dbe56eb566933a24ef` |
| 2018 | Using Reward Machines for High-Level Task Specification and Decomposition | [PDF](papers/rl_reward_design/2018_Icarte_Reward_Machines.pdf) | [PMLR](https://proceedings.mlr.press/v80/icarte18a.html) | `8eb5064ea43f07ecaf4521379ca451346822273f33199605a89b4488219373c0` |
| 2018 | Overcoming Exploration in Reinforcement Learning with Demonstrations | [PDF](papers/rl_reward_design/2018_Nair_Overcoming_Exploration_with_Demonstrations.pdf) | [arXiv](https://arxiv.org/abs/1709.10089) | `d28ea1cd0d2644378a576d825d9e1ca9af1d23714bc5cde1f5b5ff8ff2fe09a1` |
| 2019 | RUDDER: Return Decomposition for Delayed Rewards | [PDF](papers/rl_reward_design/2019_Arjona_Medina_RUDDER.pdf) | [NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html) | `1f266002ebc945fabd153292c29f236f81363115f74a3c8258296c488c2d1167` |
| 2019 | Residual Reinforcement Learning for Robot Control | [PDF](papers/rl_reward_design/2019_Johannink_Residual_Reinforcement_Learning_Robot_Control.pdf) | [arXiv](https://arxiv.org/abs/1812.03201) | `c6279120f9e0a799a9d7a943a5624a5738250e588f99c5fc8a9fac28e2039641` |
| 2025 | VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable RL | [PDF](papers/rl_reward_design/2025_Lu_VLA_RL.pdf) | [arXiv](https://arxiv.org/abs/2505.18719) | `8016399c5af6084fde140d8744df0a49248f69e4af207b15f5b90049f06c056a` |
| 2025 | STARE-VLA: Progressive Stage-Aware Reinforcement for Fine-Tuning VLA Models | [PDF](papers/rl_reward_design/2025_Xu_STARE_VLA.pdf) | [arXiv](https://arxiv.org/abs/2512.05107) | `b76fbbd8a943ab679df56bd42bd538da6c6725bd19ac231fbdce89e114ff3da3` |
| 2025 | VLA Model Post-Training via Action-Chunked PPO and Self Behavior Cloning | [PDF](papers/rl_reward_design/2025_action_chunked_ppo_self_behavior_cloning.pdf) | [arXiv](https://arxiv.org/abs/2509.25718) | `544b1d56dbc5bea2d1817a389952c665791e9959eafd223e138d1dc314d2b8b6` |
| 2026 | Foresight Residual RL for Long-Horizon Robot Manipulation with VLA Models | [PDF](papers/rl_reward_design/2026_Liu_Foresight_Residual_RL.pdf) | [arXiv](https://arxiv.org/abs/2607.16506) | `1d7a7a1e1bb487cd4006fa10fd320753efcc6ef9bd0c5310203f54a9cc97491a` |
| 2026 | RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training | [PDF](papers/rl_reward_design/2026_Zang_RLinf_VLA.pdf) | [arXiv](https://arxiv.org/abs/2510.06710) | `94cfba7349722afb9aeb34571593e8179381b8ba5e3a37859a850474e19f7818` |

十二份文件均通过 `file` 识别为 PDF。当前 WSL 的系统代理会使 arXiv TLS 连接提前断开；
重新下载 arXiv 文件时应显式使用 `curl --noproxy '*'`，其他来源可使用默认网络。完整的
项目适用性分析和实验方案见
[OpenVLA 释放切换强化学习奖励设计](../27_openvla_rl_release_reward_design_2026-09-24.md)。
