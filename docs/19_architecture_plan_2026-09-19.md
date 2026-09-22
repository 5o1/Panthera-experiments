# 19. 配置与目录架构规划（2026-09-19）

## 0. 为什么

2026-09-18 到 19 的排查里，几乎每一个缺陷都是同一个形状：**某处配置决定了实验结果，
而使用方不知道它在哪，也没有任何机制强制一致**。

- 评测相机默认 `pose_mode: randomized`，训练数据是单一固定机位。离线 skill 从 0.576
  塌到 0.049（`docs/18` 第 2 节）。
- `_eval_step_limit.yml` 缺本任务，`eval_mode` 静默把 `step_lim` 降到 1000，评测理论
  上限被压到 7.0%（`docs/18` 第 1 节）。
- 采集端按分片强制姿态与角度扇区，只把结果写进 `scene_info.json`；从 seed 重建场景
  在训练子集 112 条里有 57 条姿态都不同（`docs/18` 第 8 节）。
- `PANTHERA_ACTION_CHUNK` / `ROBOT_PLATFORM` / `unnorm_key` 等模型契约由评测入口的
  环境变量决定，与 checkpoint 无关联。错了不报错，只会静默预测错。

这些都不是"写错了一个值"，而是**没有"以产物为准"的约定**。

## 1. 原则

**产物自持配置，下游只读产物。** 数据集带着它的生成配置与仿真配置；checkpoint 带着
复现与评测它所需的全部契约。凡是"从别处读配置"的路径都要消除——环境变量、外部 YAML、
散落的硬编码常量。不一致时**报错中止**，而不是静默降级。

## 2. 现状实测

| 项 | 实测 |
|---|---|
| `scene_info.json` | 25.4 MB；149 个字段在 1280 条上完全相同（任务级配置重复 1280 遍），仅 35 个逐条不同 |
| 仿真环境变量 | 17 个 `PANTHERA_*`，分散在三个 env 模块的 `os.environ.get()` 里，无集中声明 |
| checkpoint | 有权重、骨干 `config.json`、`dataset_statistics.json`；**缺** action_chunk、unnorm_key、proprio 契约、图像预处理、训练数据集身份、超参 |
| 上游侵入 | RoboTwin 未跟踪 415 个文件（401 个是分片生成配置、7 个 envs、3 个审计脚本）、已修改 9 个（含 3 个就地编辑的配置）；RLinf 已修改 7 个；**两者本地提交数均为 0** |
| 补丁 | 22 个 `.patch`，约 520 行，多个改同一文件且顺序隐式 |
| 仓库 | `tools/` 170 文件 / 139 shell；122 个 lab 脚本中 **84 个从未被任何文档引用** |

## 3. 目标布局

```
panthera/
  externals/                  固定 commit 的上游，只读，git status 永远干净
    RoboTwin/  @0008ae6
    RLinf/     @a3816b5
    openvla-oft/
  packages/                   我们的可导入 Python（含审计脚本）
    panthera_sim/             仿真配置、资产 profile、数据集访问层、闭环执行器
    panthera_vla/             RLDS、训练接入、策略审计
  overlays/                   注入运行树的我们的文件
    robotwin/{envs,description,task_config,assets/profiles,patches}
    rlinf/{config,evaluations,seeds,patches}
    openvla/{patches}
  var/                        生成物，gitignore
    runtime/RoboTwin/         装配结果 = externals + overlay + 补丁
    task_config/generated/    分片生成的配置放这里，不再进上游
  pipelines/{collect,train,evaluate,ci}
  archive/experiments/        84 个历史一次性入口
  ros2_ws/  docs/
```

### 3.1 装配规则

大文件符号链接、需改的文件真拷贝 → 叠加 overlay → 按序应用补丁 → **校验
`git -C externals/<repo> status --porcelain` 为空**，不空即报错。审计脚本留在
`packages/`，通过把 `var/runtime/RoboTwin` 加入 `sys.path` 来 import，不再拷进上游。

补丁仍存在（18 个确实需要改上游行为），但集中在 `overlays/*/patches/`、顺序显式、
只作用于运行树。另外两类要消掉：装机脚本补丁（3 个）换成我们自己的安装步骤；
`_eval_step_limit.yml` 这类本该我们拥有的配置数据整份纳入 overlay，不再用补丁追加。

## 4. 数据集自持快照

```
<dataset_root>/
  dataset.json      任务级快照（单份）
  scenes.json       seed → realized_geometry，场景真源
  scene_info.json   仅逐条字段
  data/ instructions/ seed.txt
```

`dataset.json` 含契约、任务配置、仿真配置、资产 profile 引用与哈希、上游 commit、
溯源。其中 **`required_action_budget` 由专家轨迹长度实测得出**——评测读数据集声明的
预算，`docs/18` 第 1 节那个缺陷在此结构下不可能发生。

**访问层是强制的**：`panthera_sim.dataset.open_dataset(path)`。仓库内不允许任何代码
直接打开这些文件，用一个扫描源码的测试保证。这样文件内部结构可以自由演进。

## 5. checkpoint 自持快照

新增 `training.json`，含：

- `policy_contract`：action_dim / action_chunk / proprio_dim / use_proprio /
  use_l1_regression / num_images_in_input / unnorm_key / robot_platform
- `observation_contract`：图像尺寸、center_crop、resize 策略
- `dataset`：root、name、`dataset.json` 的哈希、切分
- `base_model`：路径，并注明 action head 为随机初始化
- `optimisation`、`stopping`、`merge`

评测入口不再接受这些环境变量，改为从 checkpoint 读，并校验待评测数据集的
`dataset.json` 哈希；跨数据集评测必须显式声明并记入摘要。

## 6. 资产 profile

机体、执行器、场景物体按 profile 成包，**行为代码与参数放在一起**：

```
overlays/robotwin/assets/profiles/panthera_phone/
  embodiment.yml
  provenance.json
  meshes/
  end_effector/
    gripper.yml            行程、开合范围、摩擦、驱动参数
    behaviours/ratchet.py  棘轮夹持：只许收紧不许张开
```

`GripperRatchet` 现在是 `audit_dense_execution.py` 的私有类，但它描述的是**执行器的
行为**，真机、评测、采集都可能要用。归位后"用哪个机体"即决定"夹爪有哪些可用行为"。
`dataset.json` 只需记录用了哪个 profile 的哪个版本与哈希，而非逐个抄数值。

## 7. 顺序

1. `packages/panthera_sim/config.py` —— 17 个环境变量与常量集中声明，可单测，不动数据
2. 资产 profile 归位，含 `GripperRatchet` 移出审计脚本
3. `dataset.json` + `scenes.json` + 访问层 + 禁止直接读文件的测试
4. `training.json` + 评测入口改为从 checkpoint 读
5. `externals/` + 装配器 + 目录重整 + 归档 84 个历史入口

第 3、4 步消除耦合，第 5 步动静最大但风险最低。

## 8. 与门禁的关系

`docs/18` 定的两道 CI 门禁依赖本规划：

- 门禁 1（专家轨迹 100% 自复现）需要场景与仿真配置可精确复现 → 第 1、3 步
- 门禁 2（单轨迹过拟合必须闭环成功）需要 checkpoint 契约与评测一致 → 第 4 步

门禁 1 当前基线 **1249/1280 = 97.6%**（`--velocity-mode finite_difference`、恢复场景
契约、释放接触求解器、最终静置）。已否决的改法：求解器位置迭代 20→64 在单条上能消除
振荡，但全量 1237/1280 更差（立姿 +1、躺姿 −13），不采用。

## 9. 执行结果（2026-09-19）

五步全部落地，并以"既有实验结果必须逐条复现"为验收。

### 9.1 已完成

| 步骤 | 产物 |
|---|---|
| 1 仿真配置单一声明 | `packages/panthera_sim/config.py`，17 个参数含默认值/范围/存在理由；29 项测试 |
| 2 资产 profile | `overlays/robotwin/assets/profiles/`，机体 + 执行器接触物理 + 行为；`GripperRatchet` 移入 `panthera_sim/behaviours/ratchet.py`，按上下文启用（采集时不启用）；23 项测试 |
| 3 数据集自持快照 | `dataset.json` + `scenes.json` + `panthera_sim/dataset.py` 访问层；回填工具；12 项测试，含"禁止直接读文件"的源码扫描 |
| 4 checkpoint 自持契约 | `panthera_vla/checkpoint.py` + `training.json`；13 项测试；rollout 改为从 checkpoint 读契约 |
| 5 目录重整 | `packages / overlays / pipelines / archive`；补丁按上游归位（robotwin 12、rlinf 7、openvla 3）；`pipelines/assemble_runtime.py` |

本地测试 **101 项全过**，不需要 GPU 或 Lab。

### 9.2 复现验证

重构的验收不是"能跑"，而是既有结果逐条相同。

| 实验 | 重构前 | 重构后 | 逐条 |
|---|---|---|---|
| 专家重放全量 1280 | 1249/1280（立 629 躺 620） | 1249/1280（立 629 躺 620） | **完全一致**，31 条失败的 episode 编号一模一样 |
| 策略闭环 12 条训练场景 | 0/12 | 0/12 | **完全一致**，动作数与查询数全部相同 |

过程中发现并修掉一个真实差异：`rollout.py` 的专家 shim 原先按"查询次数"前进索引，内层因成功提前跳出时会与已执行动作数错开。修正为按已执行动作数索引后，`replay.py` 与 `rollout.py --source expert` 在同样 episode 上动作数完全一致（820/797/706/876）。旧记录里的 711/875 是该 shim 的缺陷，不是科学结论的差异。

### 9.3 已关闭的复现缺口

- 机体的 URDF/SRDF/config/provenance（12 KB 文本）纳入仓库；网格仍由 `build_panthera_embodiment.py` 从固定的 ROS 2 源确定性重建
- 三个"仓库领先但未部署"的脚本已同步，其中 `run_lab_openvla_sft.sh` 的反斜杠续行 bug 在 Lab 上一直是坏的
- 历史入口归档：按"文档未引用"初筛 84 个，但脚本之间互相引用，该标准漏掉了传递依赖——
  `run_lab_openvla_sft.sh` 是仍在用的共享训练入口，却被初筛归档了。改按引用闭包迭代恢复
  （3 轮，22 个），最终 `tools/` 139 → 80，归档 55 个。归档集已验证自洽：无活跃代码引用，
  内部依赖也都在归档内。另有 7 个基础设施脚本（含 `lab_ssh.sh`）因匹配模式写成 `*lab*`
  被误伤，已恢复。

### 9.4 门禁现状

`pipelines/ci/gate_expert_replay.sh` 端到端跑通并**正确判红**：40 条分层抽样 39/40，退出码 1，逐条列出 ep896 的失败判据。它红是因为底层缺陷真实存在（全量 31/1280），不是门禁本身的问题。

`pipelines/ci/gate_single_trajectory_overfit.sh` 已改到新入口，尚未实跑。

### 9.5 收尾（同日续）

- **RoboTwin 四个全局配置纳入 overlay**：`_eval_step_limit.yml`（含本任务的
  `step_lim: 3200`）、`_embodiment_config.yml`、`_camera_config.yml`、
  `_config_template.yml`。随之删除 `robotwin_eval_step_limit_panthera.patch`——这是计划
  里"本该我们自己拥有的配置数据"那一类，不再用补丁追加。robotwin 补丁 12 → 11。
- **`externals/` 建立**：`pinned.json` 记录两个上游的 URL、分支与固定 commit
  （RoboTwin `0008ae6` / RLinf `a3816b5`，与数据集里记的 `robotwin_source_commit`
  一致），`setup.sh` 按 commit 克隆，checkout 本身不入库（约 28 GB）。
- **装配器通过实测**：对当前 Lab 上的 RoboTwin 正确拒绝，点名 **424 处本地改动**——
  正是它要防的侵入。9 项测试覆盖：干净上游可装配、大目录走符号链接、装配后上游未被
  写入、脏上游被拒、补丁只作用于运行树而源保持原值、补丁打不上时显式失败。
- **访问层债务 6 → 4**：`build_episode_subset.py` 与 `panthera_rlds.py` 已迁移。前者
  现在产出的子集**自带 `dataset.json` / `scenes.json`**，可被同一访问层打开；后者的
  `discover_episodes` 改为从快照组装元数据，并支持 `validate_on_train` 的单条切分。
- 本地测试 **110 项**（packages 101 + pipelines 9），无需 GPU 或 Lab。

### 9.6 门禁2 首次运行时修掉的四处

第一次实跑立刻暴露了重构留下的断点，都已修：

1. `run_finetune.py` 用 `parents[1]` 推 workspace，目录深一层后失效，且报错只说文件不
   存在、不说原因。改为向上查找并支持 `PANTHERA_VLA_ROOT` 覆盖。
2. `panthera_rlds.py` 的 `split_episodes` 要求 train/val 互斥非空，而过拟合门禁的定义
   就是两者相同。增加 `validate_on_train` 分支，由数据集快照自己声明。
3. schema 与 scene profile 的期望值来自环境变量，门禁没设，默认 v3 与数据集的 v10 不符。
   门禁现在显式导出，并在注释里说明这描述的是本仓库针对的数据生成、不是调用方偏好。
4. wandb 默认实体是占位名，联网会 404。门禁强制 `WANDB_MODE=offline`——门禁不该依赖账号。

### 9.7 遗留

- 访问层债务已清零：`NOT_YET_MIGRATED` 为空集合，仓库内无任何代码直接打开数据集文件。
  迁移中发现快照未携带 `continuous_motion_audit` 等逐条审计轨迹，处理方式是给访问层加
  `episode_record()` 读 `scene_info.json`，而不是把它们塞回快照——这样快照保持精简，
  而该文件仍然没有直接读者。本地测试 113 项。
- ~~`externals/` 尚未建立~~ **已关闭（同日晚）**。`externals/RoboTwin` 钉在 `0008ae6`、
  git 状态 0 处改动；16 GB 资产按顶层项判断，被跟踪的保留实体、未跟踪的符号链接，
  所以只占 12 MB。`runtime/robotwin` 由装配器生成，与实验实际跑的那棵树在 12 个关键
  文件上逐字节一致。`run_lab_robotwin_panthera_v2_pilot.sh` 已改为先装配再用 runtime，
  不再 `git apply` 进上游、不再 `install` 进上游。
- ~~四个全局配置仍以就地编辑存在~~ **已关闭**：overlay 的 `task_config/` 整份携带它们，
  装配时随 overlay 放置。
- 装配过程中发现两处此前不可见的问题，已各加一条测试：
  1. `script/collect_data.py` 被三个按功能划分的 patch 同时修改，各自的 diff 都假设
     另外两个已应用，`git apply` 到第二个就拒绝。改为**每个上游文件一个 patch**。
     这套 patch 集在仓库里已存在多日，因为从未有人从干净上游装配过，所以一直没暴露。
  2. 三个 `task_config` 文件既被 overlay 整份携带、又被 patch 覆盖。装配仍然成功
     （patch 反向应用后报"already present"），所以重复是隐形的，改一处另一处会静默分叉。
- 装配器新增 `writable` 声明：采集器会在 `assets/embodiments` 下生成 embodiment，
  而 `assets` 是整目录符号链接，写入会穿透到 16 GB 的共享资产。声明为可写的路径
  重建为实目录、子项逐个链接。
- 31 条专家不可复现的 episode 未解决：11 条是释放后在槽中持续振荡（求解器接触精度不足，提高迭代到 64 在全量上更差），20 条是位置类判据不达标
