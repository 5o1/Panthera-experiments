# Panthera v2 扩充数据集与异步落盘流水线

日期：2026-09-17

状态：1280 条已于 2026-09-17 生成完毕并通过全量机器审计，当前停在人工验收门前，等待
人工审阅 10 分钟分层视频。生成任务 01:34 启动（PID `1665537`），160/160 分片、
1280/1280 原子提交；首次合并审计 03:51 因 3 条轨迹的重定时加速度越过门禁而中止，定点
重采修复后 04:48 通过全量审计并写出 `automated-audit.ok` 与 `awaiting-human-review.txt`
（见第 6、7 节）。尚未转换 RLDS、训练或评测。

## 1. 数据集定义

新数据集名为 `panthera_phone_cylinder_socket_v2_single_grasp_sft_v2`，仍严格对应一台
Panthera、一个夹爪、一个圆柱插入一个槽，外部状态/动作契约仍为六关节加一个归一化夹爪
量，共 7 维。目标为 1280 条成功轨迹：直立 640 条、平躺 640 条；平躺角度 8 个区间各
80 条。后续通过人工验收后，预定划分为 1024 条训练、128 条验证 ID 和 128 条冻结组合
测试；当前生成脚本不执行该划分，也不触发训练。

采集分为 160 个可恢复分片，每片 8 条。前 80 片生成直立圆柱，后 80 片按每个平躺角度
区间 10 片分层。默认使用 GPU0–3，每张卡 3 个采集器；每个分片最多自动重试 3 次。seed
候选总数、连续失败和连续异常均有上限，编程错误不会无限快速换 seed。

## 2. 异步编码与写盘

仿真线程仍按顺序产生完整轨迹及逐帧 cache；一条轨迹结束后，cache 视为不可变快照并提交
给独立进程。每个采集器默认 1 个 writer、最多 2 条待处理轨迹且待写 cache 不超过 2 GiB；
任一数量或字节边界达到上限时才反压仿真线程。

writer 在隐藏的 partial 路径完成 JPEG/HDF5 和 MP4 编码，验证动作、RGB、视频帧数一致后
才分别原子替换正式文件，最后写入 `.episode_commits/episodeN.json`。只有 HDF5、MP4 和
提交标记三者一致时才能恢复跳过该 episode。子进程异常会回传主采集器；正常退出前必须
drain 所有任务。逐帧原始 pickle cache 仍由仿真进程顺序写入，异步化的是耗时的整集读取、
图像编码、HDF5/MP4 生成和最终提交，因此不改变帧顺序或状态/动作/图像时间对齐。

微型集成测试已验证两条成功轨迹的有界提交，并注入畸形 cache 验证异常回传、partial 清理
及“不产生完成标记”。

## 3. 受约束相机随机化

每条轨迹独立采样前方半球俯视机位：方位角 25°–155°，相机离桌高度 0.55–0.95 m，水平
距离 0.65–1.10 m，最小俯视角 28°。相机始终无滚转地朝向任务区域；RoboTwin 原有的未
记录各向同性相机抖动已关闭。

几何门禁把机械臂初始各 link、75% 可达任务体积、圆柱边界和槽边界投影到 320×240、垂直
视场 75° 的图像中，要求全部关键点位于中央 80% 区域。随后使用 SAPIEN actor segmentation
做渲染门禁：机械臂、圆柱、槽分别必须至少有 200、6、20 个可见像素，且三类对象的全部
可见像素都必须位于中央 80% 区域。不满足条件时重采样机位，最多 64 个可见性候选。

1280 个纯几何 seed 验收覆盖 25.0°–155.0° 方位和 0.551–0.950 m 高度。最终平躺圆柱
端到端 smoke 的相机离桌 0.901 m、俯视 41.8°；机械臂、圆柱、槽分别有 736、102、230
个可见像素，且全部位于中央区域。该轨迹 2182 帧，HDF5/MP4 原子提交通过。

### 3.1 机位固定/随机开关（`pose_mode`）

`task_randomization.camera_randomization.pose_mode` 现在可选 `randomized`（默认，每集按
seed 采样）或 `fixed`（全部 episode 复用同一个世界位姿）。固定模式要求同时给出
`fixed_azimuth_deg`、`fixed_height_above_table_m`、`fixed_horizontal_distance_m`、
`fixed_look_at_xyz_m` 四项，忽略 seed、`task_center_xyz` 与 `target_xy_jitter_m`，并且
固定值必须落在同一配置里的采样区间内，否则启动即报错。两种模式都要通过几何和渲染
两道可见性门禁；固定模式不再重采样，不满足时直接以具体原因报错。

```yaml
  camera_randomization:
    enabled: true
    pose_mode: fixed
    fixed_azimuth_deg: 90.0            # 机械臂基座正前方
    fixed_height_above_table_m: 0.60
    fixed_horizontal_distance_m: 0.55
    fixed_look_at_xyz_m: [0.0013, -0.0929, 0.8851]
    minimum_horizontal_distance_m: 0.55   # 必须容纳固定值
    target_xy_jitter_m: 0.0
```

用现有 1280 条轨迹的真实关键点做纯几何复算验收：

- `randomized` 模式逐位复现原数据集：1280/1280 的 `position_xyz_m` 与 `forward_xyz`
  完全相同，因此该开关不改变既有数据的可复现性。
- 固定模式候选中，方位 90°、离桌 0.60 m、水平距 0.55 m 时 1280/1280 通过门禁，圆柱
  17.7 像素（画面高度 7.4%）、槽 30.3 像素（12.6%），最小中央余量 2.0%；离桌 0.75 m、
  水平距 0.45 m 时圆柱 14.8 像素，但最小中央余量只剩 0.76%，对工作区随机范围变化没有
  余量。更近的机位（水平距 0.30 m）会因 75% 可达包络出画而全部被拒。

可见性门禁里的"75% 可达任务体积"八点包络是限制机位拉近的主要因素；它保证整条可达
半圆都在画面内，比真机固定相机的实际需要更保守。固定机位下若要让物体在画面中更接近
真机比例，需要同时收窄工作区随机范围或放宽该包络要求。

2026-09-17 用推荐参数做了 12 条试采（直立 3 条 + 平躺区间 0/4/7 各 3 条，seed
3000000–3300004，GPU0–3 上四路并行）：12 条轨迹共用唯一一个相机位姿（`position_xyz_m`
与 `forward_xyz` 各只有一个取值），零门禁拒绝，契约与重定时门禁全部通过。渲染可见像素
中位数（现有 1280 条 → 固定机位试采）：机械臂 1231 → 2279，圆柱 119 → 245（最小值
6 → 180），槽 266 → 561。试采产物在 `data_phone_v2_fixed_camera_trial/`，配置为
`RoboTwin/task_config/panthera_fixed_camera_trial_*.yml`，视频为
`reports/dataset-review/panthera_fixed_camera_trial/fixed-camera-trial-12ep-2x-1024x768.mp4`
（378 s、1024×768）。该试采不进入正式数据集，也不触发 RLDS 或训练。

## 4. 完成门禁和人工验收

生成完成后脚本会合并 1280 条轨迹并全量复核：单臂/7 维/单次抓取、真实 `qpos` 连续性、
速度和加速度、左右及空间覆盖、姿态和角度平衡、唯一 seed、HDF5 有限值，以及逐条相机
投影与渲染可见性。之后生成约 10 分钟、2 倍速、1024×768 的平衡分层审阅视频。

通过机器门禁只会创建 `automated-audit.ok` 和 `awaiting-human-review.txt`，不会创建可供
训练流水线消费的 `dataset.ok`。人工明确批准前，RLDS、训练和闭环评测都不会启动。

## 5. 人工验收后的训练停止规则

后续正式 SFT 不再以固定训练步数作为正常结束条件。每 1000 个优化步计算一次验证集完整
动作块 L1 loss；相对当前最佳值的下降至少达到 `1e-3` 才算有效改善。下降不足、持平或
上升都会消耗一次耐心，连续 3 次无有效改善即停止，并保留对应的最佳 LoRA、连续动作头、
proprio projector 和合并模型。三卡验证 loss 会先做分布式汇总，再由主进程广播保存/停止
决定，避免各 rank 分叉。

训练循环已删除固定步数退出；7 天 timeout 只作为进程异常失控保护。若没有触发上述验证集
早停，包装器会把训练判为失败，不会写 `train.ok`。训练入口为
`run_lab_openvla_panthera_v2_sft_earlystop.sh`。扩充集仍必须先完成人工审阅、RLDS 转换和
一步优化器 smoke，当前不得执行该入口。

### 5.1 `--max_steps` 已被早停补丁失效（影响一步 smoke）

早停补丁删除了训练循环里的固定步数退出分支：

```python
-            if log_step == cfg.max_steps:
-                print(f"Max step {cfg.max_steps} reached! Stopping training...")
-                break
```

并在收尾处加了 `if stop_reason != "early_stopping": raise RuntimeError(...)`。因此
`--max_steps N` **不再能结束训练**。一步优化器 smoke 原本靠 `--max_steps 1` +
`--use_val_set false` 退出，这个组合在补丁后**没有任何出口**，只会跑到 45 分钟超时并以
非零码失败，该门禁因而永远无法通过。

2026-09-18 的修法是让 smoke 走受支持的出口：`--use_val_set true`、`--val_freq 1`、
`--val_time_limit 60`、`--early_stopping_min_delta 10.0`、`--early_stopping_patience 1`。
第一次验证建立基线并存档，第二次因 min_delta 极大必然不算改善，patience=1 立即以
`early_stopping` 正常结束。副作用是 smoke 顺带覆盖了验证与存档路径，覆盖面反而更广。

遗留项：`--max_steps` 作为硬上限被静默忽略仍是回归，正确修法是在补丁里恢复该分支并把
`max_steps` 列为合法的 `stop_reason`。本次未做。

### 5.2 基座模型的 `config.json` 是被训练入口就地修改的共享可变产物

训练入口每次启动都会修改
`models/openvla-oft-place-empty-cup/config.json` 并留下一份
`config.json.back.<时间戳>`。2026-09-18 该目录下已累积 15 份备份，对应历次启动。

当前所有训练都是串行的，因此没有出过问题。但这是一个**被全部训练入口共享的可变文件**：
两个训练若并发启动，会互相覆盖对方写入的 `config.json`，而该文件决定模型结构相关字段，
后果可能是静默加载错误的配置而不是显式报错。在引入并行训练（例如同时训练多个数据集
配比、或 DAgger 的多轮并行）之前必须先处理，可选方案是把基座复制到每次运行自己的目录，
或改为不写回基座而通过参数传递。

同一目录下 `proprio_projector--10000_checkpoint.pt` 存在但**没有 action head 文件**，
因此每次从该基座起训时 action head（约 1.51 亿参数）都是随机初始化，只有 VLM 骨干和
proprio projector 是预训练的。这解释了为什么"已有预训练模型"并不意味着已具备抓取能力：
把视觉隐状态映射到 7 维绝对关节目标的那部分没有任何先验，且基座 `place_empty_cup` 的
动作维度与本任务不同，即便存在也无法加载。

## 6. 首次合并审计失败与定点重采修复

2026-09-17 03:39 生成完成，160/160 分片、1280/1280 原子提交。03:51 首次合并审计在
episode 744 中止，退出信息为 `episode 744 retiming contract mismatch`。只读全量复算确认
越界范围只有 3 条，且全部是重定时峰值加速度越过 `2.0 + 0.002 rad/s²` 门禁：

| 全局 id | 分片/槽位 | 原 seed | 原峰值加速度 | 新 seed | 新峰值加速度 |
| --- | --- | --- | --- | --- | --- |
| 744 | shard 93 / 0 | 930000 | 2.002952 | 2093000 | 1.916914 |
| 1004 | shard 125 / 4 | 1250006 | 2.003075 | 2125006 | 1.931296 |
| 1180 | shard 147 / 4 | 1470006 | 2.003266 | 2147005 | 1.935709 |

根因是容差量纲不一致：`_retime_joint_path_by_arc_length` 的收敛判据是 scale 空间的
`scale <= 1.001`，对应加速度值空间上限 `2 × 1.001² = 2.004002`；审计门禁则是值空间的
`2.002`。落在 `(2.002, 2.004002]` 的轨迹会被规划器接受却被审计否决。128 条正式集最大
值为 2.001620，恰好压在门禁之下，因此当时未暴露；扩到 1280 条后长尾三次越界。速度方向
相反（规划器上限 0.6006 < 门禁 0.601），不会触发同类失败。

修复采用定点重采：以对应分片的 `forced_posture` 与 `forced_lying_angle_bin` 生成
`episode_num: 1` 的单条配置，seed 改到空闲号段（`2093000`/`2125000`/`2147000` 起），
逐条校验契约后原位替换分片槽位，同步分片 `scene_info.json` 与 `seed.txt`，再让管线重新
合并。替换后的 episode 元数据带 `repair` 字段记录被替换 seed 与原加速度值。

- 单条重采配置：`RoboTwin/task_config/panthera_v2_single_grasp_repair_shard{93,125,147}.yml`
- 重采产物：`data_phone_v2_single_grasp_repair/`（保留作证据）
- 被替换轨迹隔离：`.panthera-v2-single-grasp-expanded-state/quarantine-replaced-episodes-20260917-044505`
- 修复前合并目录留存：`data/place_randomized_cylinder_in_socket/panthera_phone_cylinder_socket_v2_single_grasp_sft_v2.pre-repair-20260917`

04:48 重新合并与全量审计通过：`maximum_retime_joint_acceleration_radps2` 为 2.001881，
真实 `qpos` 近零速比例和 ≥80 ms 停顿均为 0，左右 619/661 与 636/644，空间覆盖 36/36，
机位互异 1280，审阅视频 603.767 s、1024×768。

## 7. 遗留项

- 规划器收敛判据应改为与审计门禁一致的值空间判据（`peak_acceleration <= 2.002`、
  `peak_velocity <= 0.601`）。本次只替换了数据，没有改规划器，因此今后任何规模的生成
  仍会在同一窄带上重复失败。
- 定点重采未计入 `collector_candidates` / `collector_rejected_candidates`，这两个字段
  仍只反映分片采集日志。

## 8. RLDS 转换的性能与并行开关

2026-09-17 对 1280 条的 TFDS/RLDS 转换做性能排查，得到两条结论，第二条推翻了最初
的假设。

**一、逐帧解码加重新编码是纯浪费。** 转换器原先对每帧调用 `_decode_rgb`，把 HDF5 里
的 JPEG 解成像素数组，TFDS 的 `Image` 特征（`encoding_format="jpeg"`）再编码回 JPEG
写盘。读 TFDS 实现后确认 `_ImageEncoder.encode_image_or_path` 对 `np.ndarray` 走编码
分支、对 `bytes` 则原样返回；而 HDF5 里存的本来就是 JPEG（magic `ffd8ffe0`，平均
22.9 KB/帧）。改为透传原始字节后，单帧成本从 0.770 ms 降到 0.043 ms（17.8 倍），整体
速率从 0.65 条/秒升到 3.2 条/秒，1280 条由约 33 分钟降到约 7 分钟。透传绕过了原先
`.convert("RGB")` 的通道归一化，因此补了逐帧只读文件头校验，要求每帧都是 RGB 320×240。

**二、并行不划算，默认保持串行。** 转换器支持多进程按 episode 并行（有界、保序），
`PANTHERA_RLDS_WORKERS` 显式数字强制档位，`auto` 按 CPU 核数与可用内存自动推算（预留
一个核给写盘、按每 worker 约 1.5 GB 夹逼，`PANTHERA_RLDS_WORKER_GB` 可调），不设则
为 1。200 条子集实测吞吐（条/秒）：

| workers | 1 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- |
| 条/秒 | **1.93** | 1.56 | 1.26 | 1.41 | 1.50 |

每一档并行都比串行慢 20–40%：`GeneratorBasedBuilder` 在父进程串行序列化并写盘，这条
关键路径不被 worker 加速，worker 反而为每条 episode 增加约 46 MB 的进程间拷贝。并行
路径的正确性已验证——6 条 episode 在串行与 8 worker 下的图像字节、动作、状态逐条一致。
要真正提速需绕开 TFDS 单进程写入器（按分片并行写再合并），会改变 RLDS 产物结构，未做。

**四、Beam 并行写入已验证不可行（负面结果，代码已删）。** 2026-09-17 实现并实测了用 TFDS
的 Beam 后端绕开单进程写入器的方案，结论是否定的，不要再尝试 DirectRunner 这条路。

先修掉了实现本身的两个真实缺陷：其一，`_generate_examples` 同时含 `return <PTransform>` 和
`yield`，而 Python 中只要函数体出现 `yield` 整个函数即为生成器函数，`return` 的值被塞进
`StopIteration` 传不出去；TFDS `split_builder.py:340` 又是先判 `Iterable` 再判 `PTransform`，
于是拿到一个空可迭代对象，最终在 writer 抛 `AssertionError: No examples were yielded`。其二，
`beam.Create` 只发出一个 bundle 而 DirectRunner 按 bundle 分发，不加 `Reshuffle` 时无论配多少
worker 都只有一个在干活。

两个缺陷修复后实测（64 集真实数据，112 核机器）：

| 方案 | 64 集耗时 | 投影 1280 集 |
| --- | --- | --- |
| 串行（当前） | 21.4 s（0.3349 s/ep） | **7.14 min** |
| Beam DirectRunner | 约 7 min 未跑完 | 约 140 min |
| Beam + Reshuffle | 更差：2:25 仅写出 28 MB，RSS 10.4 GB | 更差 |

两种配置都**没有派生出任何并行 worker**：始终只有一个 Python 进程占 105% CPU，112 核机器
load average 仅 5.35。`Reshuffle` 还会把整个 PCollection 物化进内存。根因是 DirectRunner 本就是
Beam 官方定位的测试 runner，每元素附带额外编解码、pickle 与不可变性检查，且 `multi_processing`
模式在这条含 TFDS 写 sink 的 pipeline 上不生效。换 Flink/Spark 需要 JDK、job server jar，且默认
`environment_type=DOCKER` 与本项目不用 Docker 的约束冲突，而 TFDS 的 beam writer 在 portable
runner 上是未经测试的组合——为一个 7 分钟、每数据集只跑一次的操作引入这一套不成比例。

依赖冲突另有一条可复用的结论：apache-beam 2.76 与现有 protobuf 4.25.9 / TF 2.15 / tfds 4.9.3
并不冲突，冲突的是 uv 会顺带把 `tensorflow-metadata` 解析到 1.21.0（需要 protobuf ≥ 5.27 才有的
`runtime_version`）；能工作的组合是 `tensorflow-metadata==1.17.3`。

若今后确有频繁重转的需求，应做的是**分片并行 + 合并**（N 个独立 `build_dataset` 进程各写各的
data_dir，再重编分片号并重写 `dataset_info.json`）：纯 Python、零新依赖、不涉及 JVM 或 Docker，
正确性可用串行产物逐样本比对验证。本次未实施。

**三、透传需要补回完整性校验，并升版本号。** 只读文件头的校验不触碰熵编码数据：
`Image.open` 是惰性的，截断的 JPEG 照样通过 mode/size 检查（PIL 对 JPEG 的 `verify()`
是基类空实现，同样拦不住），而原先的整帧解码会抛 `OSError: image file is truncated`。
因此补了 end-of-image 标记校验——HDF5 存的是 `|S` 定长 bytes，NumPy 取值时剥掉尾部
NUL 填充，`bytes(...)` 得到的正是净荷，末两字节必须是 `ffd9`——并把文件头校验扩展到
`format == "JPEG"`，使该标记校验的报错归因准确。由于同样输入产出的字节已与 4.0.0 不同，
`PantheraPhoneCylinderSocketV2` 升版到 `4.1.0`；`4.0.0` 是 128 条正式集已完成的重编码
产物，记录保持不变。另外 `download_and_prepare` 对已存在的完整版本目录是 no-op，而验收
读的 `splits` 来自源 HDF5、察觉不到复用，因此两个转换入口在 `rlds.ok` 未通过时一律先把
已有版本目录可恢复归档为 `.unverified-<时间戳>` 再重建。

## 9. 实际 qpos 速度连续性门禁

2026-09-17 复核合并审计的覆盖范围时发现一处缺口：速度和加速度上界只加在**重定时后的
规划轨迹**上（`joint_retime_audit` 的 `peak_joint_velocity_radps` / `peak_joint_acceleration_radps2`），
而实际 `qpos` 只查近零速停顿。停顿检查能发现"路线停住"，发现不了"速度跳变"。schema 9
pilot 当初正是从这个缺口漏过去的：规划审计通过，实际 `qpos` 和人工视频仍然顿挫。

补充的判据用的是重定时契约自身的量纲：有界加速度在一个采样间隔内最多改变
`a_limit * dt` 的速度，因此观测到的逐关节速度跳变除以该预算，就是以契约为单位的不连续度。
窗口、内部区间定义与既有停顿检查完全一致（`continuous_motion_audit` 的 route 段、五次
几何进度 0.05–0.95），并且只在**规则间隔**的样本对上评估——录制器会在阶段边界插入诊断帧，
跨越不规则间隔做有限差分测到的是采样本身而不是轨迹。

24 集抽样的横向对比（同一套处理）：

| 数据集 | 最差跳变比 | 违规/评估对 | 判定 |
| --- | --- | --- | --- |
| 固定机位 1280（全量 1280 集复核） | **1.0673** | 860 / 896455（0.096%） | passed |
| 随机机位 1280 | 1.0394 | 19 / 16084（0.12%） | passed |
| schema 9（已否决） | **17.4060** | 235 / 7925（3.0%） | **failed** |

结论：两个 schema 10 的 1280 集在这项上等价，且残余量级属于控制器跟踪而非规划缺陷；
被否决的 schema 9 超出预算 17.4 倍，该判据能以一个数量级的间隔把它区分出来。门禁阈值
取 `1.5`：全量最差值 1.0673 之上仍有 1.4 倍余量，距 schema 9 有 11.6 倍隔离。该阈值是
经验值，依据即上表。

判据已并入 `tools/run_lab_robotwin_panthera_v2_pilot.sh` 的合并审计（与停顿检查同一个
循环，复用同一份已打开的 HDF5 与 route 窗口），新增摘要字段
`maximum_actual_velocity_jump_over_budget_ratio`、`actual_velocity_jump_gate_ratio`、
`actual_velocity_jump_violations` 和 `actual_velocity_jump_evaluated_pairs`。对**已经生成**
的数据集用独立入口补测：

```bash
python packages/panthera_sim/diagnostics/audit_actual_qpos_continuity.py \
  --source-root <数据集根> --episode-count 1280 --output <报告.json>
```

两者判据一致；独立入口在超限时以非零码退出。产物位于
`/data/lyy/panthera-vla/reports/dataset-review/qpos-continuity/`。

注意两条口径限制。其一，本判据的阈值加在**实际** `qpos` 上，不能与规划空间的
`0.601 / 2.002` 直接比较——实际关节速度受控制器响应影响天然更大。其二，被弃用的 v3
`panthera_phone_vertical_sft_v1` 早于 `continuous_motion_audit` 字段，没有 route 窗口
元数据，无法用同一定义度量，因此不进入上表。

## 10. 数据集取舍（2026-09-17）

用户决定弃用 v3 `panthera_phone_vertical_sft_v1`，正式训练只采用**固定机位 1280**
（`panthera_phone_cylinder_socket_v2_single_grasp_sft_v2_fixedcam`）。弃用理由为早期
生成规则的两个缺陷：关节速度在每个时间点不保证连续，以及物体只在固定区域生成而不是
机械臂可达范围。

对这两条的核对结果：

- **物体生成区域已修复。** `panthera_v2_sampling.py` 现按绕基座的极坐标环形采样：
  半径 `0.22–0.345 m`（`nominal_reach_radius_m 0.46 × reach_fraction 0.75`）、前半圆、
  3 径向 × 12 角度 = 36 格、格内按面积均匀（对 r² 取均匀再开方）。固定机位 1280 实测
  圆柱与槽各覆盖 36/36。需要注意它是可达范围的 **75% 子集**且有 `0.22 m` 内径，属于
  刻意保留的余量，不是全可达范围。
- **速度连续性：问题属于 schema 9 / symmetric-smooth 两个 pilot，不属于 v3。** 见第 9 节。
  v3 因缺少 route 元数据无法用同一判据度量，但它被弃用依据的是上面第一条与更早的相机
  构图否决，该决定依然成立。

随机机位 1280 已通过人工批准（`human-review-approval.json` 与 `dataset.ok` 均存在），
但按本次决定不用于正式训练，保留为证据。

固定机位 1280 的人工批准已于 2026-09-17T22:09:25+08:00 记录：审阅视频
`balanced-stratified-10min-2x-1024x768.mp4`（604.3 s、1024×768、68571586 B）在写入标记前
重新计算 SHA-256 并与 `review-video.sha256` 逐位核对一致
（`57e833f790d07de62910e8e177bc8b2a9116eac4e927becfea4ea00431b7cf27`）。
标记按既有约定由 agent 代填并记录 `recorded_by`，批准主体为用户本人。

**128 条 schema 10 线一并退役。** 其 `rlds_phone_cylinder_socket_v2_sft_v1` 的 4.0.0 产物
与 `rlds.ok` 保持冻结不动，`run_lab_panthera_v2_sft_rlds.sh` 及三个消费入口
（`run_lab_openvla_panthera_v2_sft_{10k,step_smoke,earlystop}.sh`）的版本号保留 4.0.0
以匹配磁盘上的冻结产物，并在文件头标注"不要再运行"。当前 builder 已是 4.1.0，这些入口
若被重跑会在验收处明确失败，而不是静默训练到过期数据上。`4.1.0` 现在只出现在扩充集
那条线（`expanded` 与 `expanded_fixed` 共用同一个转换入口）。

Lab 入口：

```bash
bash /data/lyy/panthera-vla/start_lab_robotwin_panthera_v2_expanded_dataset.sh
bash /data/lyy/panthera-vla/status_lab_robotwin_panthera_v2_expanded_dataset.sh
bash /data/lyy/panthera-vla/install_lab_panthera_v2_expanded_autoresume.sh
```

原始数据目标路径：

```text
/data/lyy/panthera-vla/data/place_randomized_cylinder_in_socket/
  panthera_phone_cylinder_socket_v2_single_grasp_sft_v2
```

审阅视频目标路径：

```text
/data/lyy/panthera-vla/reports/dataset-review/
  panthera_phone_cylinder_socket_v2_single_grasp_sft_v2/
  balanced-stratified-10min-2x-1024x768.mp4
```

## 11. 专家轨迹开环可复现性与场景契约（2026-09-18）

目标是"专家轨迹自己必须能复现成功完成任务"，因为它是闭环评测的上界：专家自己都复现
不了的episode，策略不可能在上面被公平评判。用 `audit_dense_execution.py` 回放数据集里
记录的 50 Hz 动作（绕开 TOPP，每个目标保持 5 个物理步），在固定机位 1280 上分层抽 20 立
姿 + 20 躺姿。

### 11.1 结论

| 配置 | 立姿 | 躺姿 |
|---|---|---|
| 未恢复场景契约 | 20/20 | **1/20** |
| 恢复场景契约后 | 20/20 | **17/20** |

躺姿几乎全废**不是**执行器、物理或数据集缺陷，而是回放时重建的场景根本不是那串动作对应
的场景。

### 11.2 根因

`_sample_task_scene` 的躺姿角度有两条来源：默认由 `sample_scene(seed)` 按
`lying_angle_bin = (seed // 2) % 8` 决定，但 `forced_lying_angle_bin`（配置或
`PANTHERA_V2_FORCED_LYING_ANGLE_BIN`）会用一个独立 RNG 在指定的 π/8 扇区里重采。

采集端 `run_lab_robotwin_panthera_v2_pilot.sh` 是**按分片**生成配置的，每片钉死
`forced_posture` 与 `forced_lying_angle_bin`（80 个立姿分片 + 8 扇区 × 10 个躺姿分片）。
所以数据集里的躺姿角度是被强制覆盖过的，与 seed 天然导出的扇区无关：

- ep700 元数据 `lying_angle_bin=0`、`cylinder_angle_rad=0.26497 rad`（15.18°）
- 用不带强制项的配置回放，seed 10870007 → `(10870007//2)%8 = 3` → 1.2478 rad（71.49°）
- 两者绕竖直轴相差 **56.31°**

立姿 `cylinder_angle_rad` 恒为 0 且绕竖直轴旋转对称，所以完全不受影响——这正是 20/20 与
1/20 分裂的来源。

失败链条（已逐段实测，不是推断）：物体位置其实完全一致（xy 精确相同，接触前漂移
0.00 mm），但躺倒方向错了 → top-down 合拢沿错误的弦压下去 → 圆柱被推滚约 1.6 cm（半径
仅 2.75 cm）→ 手指从旁边合过去，实测开口一路合到 0.30，而采集时被圆柱顶住停在 0.697
→ 全程没抓起来（最高点比初始只高 0.02 cm），后续抬升搬运插入全在空抓。

### 11.3 排除过的假设

诊断过程中先后指向过四个错误方向，都被自己的实验否掉，记录以免重走：

- **TOPP 重定时**：改用稠密执行绕开 TOPP，躺姿仍 0/2。
- **50 Hz 零阶保持太粗**：改成一阶保持（线性插值到 250 Hz）逐条比对，成功数完全不变。
- **速度前馈缺失**：`--velocity-mode finite_difference` 把手臂跟踪偏离压了 6 倍（中位
  27.7 → 4.6 mrad，抓取瞬间 34.9 → 12.9 mrad），但躺姿依旧失败。此项是真实改进，已保留
  为默认推荐，但不是根因。
- **夹爪抖动松脱**：加了棘轮（只许收紧不许张开），A/B 完全不变。事后看测试设计本身就不
  成立——专家动作没有抖动，棘轮对它是 no-op。

另有两次测量本身是错的，值得单独记：`get_left_arm_jointState()` 返回的是**驱动目标**、
`get_left_gripper_val()` 返回的是**上一条指令**，拿它们算"跟踪误差"得到的 0 是同义反复。
真实测量必须走 `_actual_robot_state()`——那正是记录器自己写进数据集的那个函数，用它两边
才同源。

### 11.4 已落地的防护

`audit_dense_execution.py` 现在做两件事：

1. `_restore_scene_contract()` 从 `scene_info.json` 取回该 episode 的 `cylinder_posture`
   与 `lying_angle_bin`，注入 `task_randomization` 后再 `setup_demo`。
2. `_verify_scene()` 在 setup 之后比对实际位姿与元数据（朝向 < 1°、位置 < 1 mm），不一致
   直接抛错中止，而不是静默跑出一个无意义的成功率。

影响范围是有界的：

- **回放已记录的轨迹**（诊断、审计、DAgger 式重放）必须恢复强制契约并校验，否则场景不对。
- **从 seed 新开评测**不受影响：默认采样的姿态是按 seed 奇偶 50/50、躺姿扇区是
  `(seed//2)%8` 八等分均匀，与训练集分布一致。即将建的 RoboTwin 评测器不需要强制扇区。

**但"分布一致"不等于"seed 标识场景"。** 采集端的 `forced_posture` /
`forced_lying_angle_bin` 覆盖了 seed 本来的决定，覆盖的结果只写进 `scene_info.json`，
不回写 seed。所以一个数据集 episode 的 seed 拿到无强制的环境里重建，得到的往往是
**另一个场景**——实测固定机位训练子集 112 条中有 57 条（50.9%）连姿态都不同，躺姿
还额外差一个 π/8 扇区。

推论，凡是要复用数据集 seed 的地方都适用：

- 不要用 seed 指代场景；场景的唯一真源是 `scene_info.json` 的 `realized_geometry`。
- 要在训练场景上做对照（如"训练集内能否成功"），要么恢复强制契约，要么只取
  "采集姿态与 seed 推导姿态恰好一致"的子集，并**逐条核验**圆柱 xy、槽 xy、朝向、
  姿态四项后才可解读结果。
- 这个隐含假设在 2026-09-18 的专家重放里造成过一次长时间误诊（躺姿 1/20，先后错误
  怀疑 TOPP、速度前馈、夹爪棘轮、接触求解器），见本节开头与 `docs/18` 第 8 节。

### 11.5 遗留

- 恢复契约后仍有 3/20 躺姿失败（ep640、864、896），属于开环重放对初始接触的真实敏感性，
  未进一步定位。专家开环上界因此是 ~92.5%，闭环采集端接受率为立姿 ~95%、躺姿 ~65%
  （由 `seed.txt` 的 seed 间隙统计得到）。
- `_eval_step_limit.yml` 仍缺 `place_randomized_cylinder_in_socket`，回放日志一直在报
  "not in step limit file, set to 1000"。专家轨迹 50 Hz 栅格步数实测：立姿中位 1076 /
  max 1283，躺姿中位 2211 / max 5101。该条目的单位取决于评测适配器每次策略调用执行多少
  栅格步，须在适配器落地时一并确定，此处只留测量值，不先填一个语义不明的数字。
- `audit_dense_execution.py` 的限速钳位一旦触发就不会追回，现已额外报告
  `max_speed_limit_lag_rad`，避免被大幅钳位的回放伪装成忠实回放。
