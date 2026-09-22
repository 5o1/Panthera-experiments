# 18. 闭环评测台缺陷排查（2026-09-19）

## 0. 触发

16 小时子集训练（128 条、4 卡 batch 6、51000+ 步）结束：验证损失最好 **0.00802**
（第 48000 步），合并后 18 seed 评测 **0/18**。与 round 0（3 小时）的 0/18 完全一致——
训练时长增加 5 倍、损失降到 8e-3，闭环成功率没有任何变化。

排查结论：**在策略质量被讨论之前，这个评测台的理论成功率上限只有 7.0%。**

## 1. 动作预算：真正生效的是 1000，不是配置里的 1600

两处限制同时存在，取先触发者：

- RLinf：`robotwin_env.py` 的 `self._elapsed_steps += chunk_actions.shape[1]`，
  超过 `max_episode_steps` 即 truncate。配置写的是 1600。
- RoboTwin：`_base_task.py` 在 `eval_mode` 下**丢弃调用方传入的 `step_lim`**，改从
  `task_config/_eval_step_limit.yml` 重读；本任务不在该文件里，于是
  `print(f"{self.task_name} not in step limit file, set to 1000")`，**静默降到 1000**。

两者单位相同，都是 50 Hz 动作数。所以真正的预算是 **1000**。

对照固定机位 1280 专家轨迹的实测长度（立姿中位 1076 / 最长 1283，躺姿中位 2211 /
p95 2920 / 最长 5101）：

| 预算 | 立姿可完成 | 躺姿可完成 |
|---|---|---|
| **1000（实际生效）** | **13.9%** | **0.0%** |
| 1283 | 100% | 0.0% |
| 1600（配置值） | 100% | 4.8% |
| 2400 | 100% | 65.0% |
| **3200（已改为）** | **100%** | **96.2%** |
| 5200 | 100% | 100% |

评测 seed 为 200001–200018，姿态按 `seed % 2` 交替，即 9 立 9 躺。
**预算 1000 下 18 条评测的理论成功率上限 = 7.0%。** 0/18 完全在预期之内。

这条同时解释了"立姿那一半预算够却也全挂"——立姿在 1000 下同样只有 13.9% 可完成。

### 已落地

- `task_config/_eval_step_limit.yml` 增加 `place_randomized_cylinder_in_socket: 3200`；
  当前由 `overlays/robotwin/task_config/_eval_step_limit.yml` 整份持有，不再用补丁追加。
- `robotwin_place_randomized_cylinder_in_socket.yaml` 的
  `max_steps_per_rollout_epoch` / `max_episode_steps` / `step_lim` 改为 3200。
- `tools/run_lab_panthera_policy_eval.sh` 增加两道拒绝：预算低于按任务写死的下限时
  拒跑；`_eval_step_limit.yml` 中该任务缺项或值不足时拒跑（这正是静默降级的入口）。

## 2. 相机视角：训练单一固定机位，评测随机机位

数据集 `scene_info.json` 记录 `pose_mode: fixed`、方位 90°、高 0.60 m、距 0.55 m，
1280 条共用一个机位。而评测 env 配置的 `task_randomization` 下**没有
`camera_randomization` 块**，`CameraRandomizationSpec` 的默认值是
`pose_mode = "randomized"`，于是每条 episode 重新采样。实测四个评测 seed：

```
seed 10000001  方位 101.3°  高 0.786 m  距 1.055 m
seed 10000003  方位  83.2°  高 0.853 m  距 0.945 m
seed 10870007  方位  78.1°  高 0.758 m  距 0.796 m
seed 10990005  方位 127.1°  高 0.652 m  距 0.818 m
```

距离接近训练值的两倍，物体在画面里只有一半大。用同一个 16h 模型在两种图像上做离线
审计（`audit_copycat_baseline.py`，走评测侧的 `get_model` + `center_crop_image`）：

| 图像来源 | 手臂 L1 (rad) | copy 基线 | skill_score | 方向余弦 |
|---|---|---|---|---|
| 固定机位（训练视角） | 0.00976 | 0.02300 | **0.576** | 0.605 |
| 随机机位（评测视角） | 0.01891 | 0.01989 | **0.049** | 0.345 |

`skill_score = 1 - model_l1 / copy_l1`，0.049 意味着策略退化到与"照抄本体感受、原地
不动"几乎无差别，对应接触图上观察到的"策略从不去抓"。

**但修正相机后重跑仍是 0/18**，因为第 1 节的预算问题当时尚未发现。相机错配是真实
缺陷且必须修，但不是 0/18 的主因。

### 已落地

env 配置补上完整 `camera_randomization` 块（`pose_mode: fixed` 及数据集的固定位姿），
并把 `domain_randomization.random_head_camera_dis` 从 0.015 改回 0——采集侧显式关闭了
这项未记录的各向同性抖动，评测侧却打开了。

## 3. TOPP 对抖动动作的时长膨胀

`SubEnv.step` 把整个 chunk 交给 `gen_sparse_reward_data`，后者用 TOPP 重定时。实测
（`audit_topp_feasibility.py`，16h 模型，4 条 episode × 6 个锚点）：

| 变体 | TOPP 后物理步数中位 |
|---|---|
| 专家 chunk | 156.5 |
| 策略 chunk 原始 | **1028.0（6.6×）** |
| 策略 chunk + 时序集成 | 448.0 |
| 策略 + 集成 + Chaikin | 357.0 |

即每次策略调用换来的不是 100 个物理步而是上千个，与第 1 节的动作预算相乘。

时序集成（ACT 式：每步查询、对同一时刻的多次重叠预测做指数加权平均）把膨胀砍掉
一半以上。OpenVLA-OFT 自身默认不做时序集成（其取向是靠并行解码 + 整块开环执行拉
吞吐），当前配置是照该取向来的；但本项目卡在成功率而非吞吐，且执行器对抖动的惩罚
远比一般设置严重。实现见 `audit_topp_feasibility.py` 的 `--temporal-ensemble`。

## 4. 线程级环境并行是负收益

`run_lab_panthera_policy_eval.sh` 原先把 `total_num_envs` 写死等于 GPU 数。解耦出
`PANTHERA_EVAL_ENVS_PER_GPU` 后实测（18 条轨迹，3 卡）：

| 每卡环境数 | 环境总数 | 墙钟 | step time |
|---|---|---|---|
| 1 | 3 | 5:27 | 327.8 s |
| 2 | 6 | 9:03 | 543.8 s |
| 3 | 9 | 11:36 | 696.5 s |

**单调变慢**，n=3 比 n=1 慢 2.1 倍。RoboTwin 的 `VectorEnv` 用 `ThreadPoolExecutor`
（`global_lock` 只在 setup/reset 持有，step 不持锁），但 SAPIEN 物理步不释放 GIL，
线程之间没有并行、只有争抢。旋钮保留（默认 1，行为不变），**结论是不要调它**。

可用的吞吐杠杆只剩进程级并行（多用一张卡）和消除第 3 节的 TOPP 膨胀。

## 5. 评测成本

吞吐/占用测量不需要计分跑。新增 `tools/probe_lab_panthera_eval_throughput.sh`：
3 条轨迹、200 步预算、边跑边采样 GPU 与负载，约一分钟。为此给预算下限开了显式逃生口
`PANTHERA_EVAL_ALLOW_SHORT_BUDGET=1`，并在摘要里写 `scoring: false`、状态
`throughput_probe_not_scored`，且不写 `eval.ok`——短跑的成功率不会被误读为策略成绩。

## 6. 离线审计：泛化正常，方向性不足

同一 16h 模型，训练视角图像，按训练集内 / 验证 / 完全未见分组：

| 组 | 手臂 L1 (rad) | skill_score | 方向余弦 |
|---|---|---|---|
| 训练集内 | 0.00774 | 0.651 | 0.641 |
| 验证 | 0.00976 | 0.576 | 0.605 |
| 完全未见 | 0.01035 | 0.497 | 0.601 |

- 损失量纲交叉验证通过：归一化 loss 0.008 × 前 6 关节量程均值 2.4113 / 2 ≈ **9.65 mrad**，
  与实测 9.8 mrad 吻合。
- **不是过拟合**：训练集内 7.7 mrad → 完全未见 10.4 mrad，仅差 34%。
- 但 `skill_score` 0.50–0.65、方向余弦 0.60，即比"原地不动"只好一倍、运动方向只有中等
  相关。这是在评测台被修好之后仍需回答的问题。

## 7. 三项全修之后：仍然 0/18

预算 3200 + 固定机位 + `_eval_step_limit.yml` 补齐，18 seed 重跑：

```
episode_len=3200.0   num_trajectories=18   success_once=0.0   success_at_end=0.0
```

18 条全部跑满 3200 步，无一成功，墙钟 10:52。

**三个缺陷都是真的、都必须修**——修之前该评测在数学上最高只能到 7.0%，任何成功率
数字都不可解读——**但修完之后策略依然一次都没成功**。问题落回策略本身，与第 6 节的
离线数据自洽：方向余弦 0.60、skill 0.58，即运动幅度大致正确而方向只有中等相关。

自此评测台产出的数字才是可解读的，0/18 第一次真正意味着"策略做不到"，而不是
"评测台不允许它做到"。

## 8. 训练 seed 不能直接拿去评测：姿态也不复现

为回答"训练集内的场景能不能成功"，需要用训练 episode 的 seed 重建场景。但采集时
**姿态是按分片强制的**（`forced_posture`），而评测不设强制，`sample_scene(seed)` 用
`posture = "upright" if seed % 2 == 0 else "lying"` 决定。两者无关：

```
训练子集 112 条中，采集姿态与 seed 推导姿态不一致 = 57 条（50.9%）
```

即一半以上的训练 seed 在评测里重建出的是**另一种姿态的场景**。这比第 1 节记过的
"躺姿角度扇区差 56°" 更根本——那只是角度，这是姿态本身。

可安全复用的只有"采集为立姿且 seed 为偶数"的交集，训练子集里共 28 条。取其中 18 条
（`seeds/panthera_v2_trainset_upright18.json`），逐条核验重建场景与
`scene_info.json` 一致：圆柱 xy、槽 xy、朝向、姿态四项全部吻合，不一致 0/18。

**任何"在训练数据上评测"的对照都必须先做这一步核验**，否则测的是另一批场景。

## 9. 训练集场景上同样 0/18

用第 8 节筛出并逐条核验过的 18 条训练场景（全部在 `subset-manifest.json` 的 `train`
列表里，模型在 51000 步里反复见过；全部立姿，3200 预算下 100% 可完成；相机固定到训练
机位）重跑：

```
trajectories=18   success_once=0.0   max_episode_steps=3200
```

**在自己训练过的场景上也是 0/18。** 排除了泛化，也排除了评测台——对这 18 条而言预算、
相机、场景三项都已确认正确。

与第 6 节的离线数据自洽：训练 episode 上 L1 只有 7.7 mrad，但 `skill_score` 0.651、
运动方向余弦 0.641。L1 衡量单步预测贴合度，任务成败取决于连续多步的方向一致性；每次
策略调用开环执行 20 步，0.64 的方向相关度会迅速积累成偏离。低 L1 与零成功率之间没有
矛盾，二者度量的根本不是同一件事。

## 10. 场景复现重构（替代逐处打补丁）

第 8 节的坑和第 1 节的静默降级都属于同一类问题：场景的真源在 `scene_info.json`，而
环境只能从 seed 加隐藏的强制标志重建，于是每个需要复现场景的调用方都要自己拼一遍契约
再自己写一遍校验。改为把场景做成数据：

- 新增 `envs/panthera_scene_registry.py`：seed → 完整 `realized_geometry` 的登记表。
  因为 `self.realized_geometry = scene_sample` 本就是 `_sample_task_scene` 的返回值，
  登记表原样返回即可，不引入新的中间表示。
- 环境新增两个 `task_randomization` 键：`scene_registry`（命中则按记录摆）与
  `scene_registry_required`（未命中直接报错，不静默回落采样）。每条 episode 记录
  `scene_source` 作为溯源。
- **seed 仍是寻址键**，所以 RLinf 的 `env_seeds` 与 RoboTwin 的 `VectorEnv` 一行未改；
  变的只是映射关系从隐式函数变成显式数据。这是该方案"廉价"的来源。
- `audit_dense_execution.py` 删除手工拼契约的 `_restore_scene_contract`，改为查表。
- 评测台新增 `PANTHERA_EVAL_SCENE_REGISTRY`，摘要记录 `scene_registry` 字段。

验证：

- `test_panthera_scene_registry.py` 17 项通过，覆盖实际踩过的失败形态（姿态与角度不符、
  四元数差一个 π/8 扇区即那个 56°、轴向与角度不符、槽位置与 socket 不一致、seed 重复、
  往返序列化）。
- 端到端等价：40 条专家重放走登记表得 39/40（躺姿 19/20、立姿 20/20），失败仍是
  ep896，与手工契约路径逐条相同。
- 已从固定机位 1280 生成登记表：1280 个场景，640 立 / 640 躺，全部通过校验。

## 11. 遗留
- 时序集成尚未做进评测执行器，目前只在离线审计里验证过。
- `PANTHERA_EVAL_MIN_SUCCESS=0.75` 这个门槛是在预算 1000（上限 7.0%）下设的，须在
  预算修正后重新确定。
- 第 6 节的方向性不足：128 条子集、51000 步是否根本不足以学出方向性，尚未回答。
