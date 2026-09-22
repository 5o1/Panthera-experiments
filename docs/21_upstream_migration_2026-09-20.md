# 21 · 上游从 0008ae6 迁到 6dde571 (2026-09-20)

`docs/19` 的装配器落地之后，升级上游第一次成为可做的事：改动全在 patch 里，
上游检出是干净的，冲突可以在临时克隆上测出来而不必赌。

## 为什么要升

`docs/20` 记录的通道顺序问题，上游在 **2026-09-10** 的 `95479f0`
（"Route collection and HDF5 loaders through encode/decode_image_bit"）已经修掉了。
我们钉在 2026-05-19 的 `0008ae6`，落后 **223 个提交**，正好停在这条修复之前。

## Lab 取不到 GitHub

`git fetch` 300 秒超时，而 HuggingFace 下载正常——是 GitHub 这条路不通。
反向 SSH 隧道也没打通：转发端口在 Lab 上能接受连接，但没有数据回传。

可行的路径是在 WSL 本地经代理取回，打成 **9.3 MB 的 git bundle** 传过去，
Lab 侧 `git fetch /tmp/rt.bundle`。不需要 Lab 联网。

## 补丁的去留

先在临时克隆（`/tmp/rt-new`，从干净 externals 克隆并 checkout 新提交）上逐个试，
固定版本一动没动。初测 6 个补丁只有 1 个还能直接应用。逐个查清原因后：

| 补丁 | 处置 | 原因 |
|---|---|---|
| `generate_episode_instructions` | **丢弃** | 上游已原生具备 `save_path` / `scene_info_path`，实现与我们一致 |
| `vector_env` | **丢弃** | 上游删除 RLinf 支持；我们不做 RL，评测走自己的 `rollout.py`，不依赖它 |
| `planner.py` | 34 行 → **13 行** | 仅剩 SRDF→acm 一处；"保留完整 qpos" 那处上游新版已天然正确（`plan_pose` 直接用 `np.array(now_qpos)`，不再做关节索引提取） |
| `_base_task.py` | **+31** | 全局采样时钟与统一推进入口，上游仍然没有 |
| `robot.py` | **+90** | 含恢复 mplib 分派，见下 |
| `collect_data.py` | 迁到 `scripts/`，**+102/−31** | 三处冲突合并 |

补丁集 6 个 → **4 个**。

## 三件实质变更

### 上游把规划后端写死成 Curobo

`planner_backend` 这个概念在新版消失，`set_planner` 只创建 `CuroboPlanner`。
我们的 embodiment 配的是 `planner: mplib_RRT`，没有 curobo 配置，直接就报
`curobo_left.yml` 不存在。

但**消失的是分派，不是能力**：`MplibPlanner` 和场景后端 `SapienPlanningWorld`
都还在 `planner.py` 里，只是没人调用。按 `self.left_planner_type`（新版仍然保留，
默认 `mplib_RRT`）把分派加回去即可，不引入任何新行为。

选这条而不是改用 Curobo，是因为后者会让 `docs/18`、`docs/20` 里所有关于规划的
测量作废——包括专家复现 1249/1280 这个基线。技术选型该单独立项，不混进版本迁移。

### 上游新增关节限位检查

`bd56368` 加了 `planned_joints_legal()`，与我们的连续失败计数在同一段代码里冲突。
合并时让**限位不合格也计入失败预算**——否则一个只产出非法规划的采集永远不会停。

### 目录与命名变更

```
script/         → scripts/
task_config/    → env_cfg/task_config/
robotwin/       → 整个包删除（RLinf 集成）
episode1.hdf5   → episode_0000001.hdf5
新增            XPolicyLab（子模块）、code_gen、data、env_cfg
```

- 装配器的 `PLACEMENT` 与 `copy_first` 已更新
- `robotwin_env._task_config_root` 同时认两种布局，因为搭建脚本仍在用旧树
- `dataset._episode_file` 同时认两种命名：现有数据集全是旧名，从迁移后的 runtime
  采的会是新名，所以访问层解析两者，而不是让每个读取方各自猜

## 验证

```
装配        从 6dde571 成功，4 个补丁全部应用，上游 git 0 处改动
专家复现    ep2 / ep3 成功，姿态 0.0000°，位置 9.2e-07 m，相机 3.4e-08 m
            —— 与迁移前逐位一致
本地测试    155 项通过
```

gate 1 全量（1280 条）在迁移后的 runtime 上重跑，用于确认 1249/1280 基线未变。

## 入口迁移

Lab 上 13 个入口仍指向旧的、被改过的 `RoboTwin` 树。其中：

- 10 个只是**使用** RoboTwin → 改为先装配再用 `runtime/robotwin`
- `bootstrap_lab_vla.sh` 负责**搭建**上游（clone/安装）→ 保留指向检出，这是它的职责
- 另外 3 个（`bootstrap_lab_panthera_embodiment.sh`、`run_lab_panthera_policy_eval.sh`、
  `run_lab_robotwin_panthera_phone_sft_dataset.sh`）此前已迁；其中 bootstrap 里
  原有 **21 处 `git apply` 进上游**，整段塌缩成一次装配调用

## 遗留

- 旧的 `RoboTwin` 树仍在（16 GB 资产存在其中，externals 与 runtime 都符号链接到它），
  只有搭建脚本还指向它
- `overlays/rlinf/` 可以归档：上游已删 RLinf 支持，我们只从 RLinf 用两个符号
  （`get_model`、`center_crop_image`），两者在 OpenVLA-OFT 的 `experiments/robot/`
  里都有等价物
- 新命名的数据集尚未产生，兼容代码尚未在真实数据上跑过

## 8. WSL 仓库收口（2026-09-20）

Lab 迁移完成后，WSL 工作树一度仍把 `externals/pinned.json` 和若干入口固定在旧提交，
形成“文档/补丁面向 6dde571、声明文件面向 0008ae6”的不可复现状态。现已统一为完整提交
`6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755` 与 `main` 分支，并由装配器在生成 runtime 前
强制核对 pin；仅仅保持上游 git-clean 已不再足够，提交不匹配同样会中止。

RoboTwin 新目录契约也已收口：采集入口为 `scripts/collect_data.py`，任务配置位于
`env_cfg/task_config/`，CI 门禁默认读取 `runtime/robotwin`。RLinf 的三个同时修改
`requirements/install.sh` 的补丁已合成一个文件，防回归测试现在同时检查 RoboTwin 与
RLinf，禁止多个补丁拥有同一上游文件。

本节只记录代码与配置收口；没有启动 Lab GPU、仿真长任务或真机。
本地验收为仿真/VLA 侧 187 项测试、ROS 侧 74 项测试全部通过，所有仓库 Shell 入口通过
`bash -n`，overlay JSON/YAML 与 pin 均完成解析检查。
