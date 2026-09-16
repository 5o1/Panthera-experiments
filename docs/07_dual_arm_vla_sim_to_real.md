# 历史记录：双臂 VLA 圆柱入槽探索

更新日期：2026-09-12

> **状态：已被当前任务取代。** 2026-09-13 用户明确当前实验是一台 Panthera 机械臂、
> 一个夹爪将圆柱插入凹槽，不是双机械臂。本文只保留早期双臂实现、失败案例和兼容性
> 证据；其中 14 维动作、双夹持、D6 attach-on-grasp、双臂训练集和双臂真机拓扑均不得
> 当作当前需求。当前状态以
> [Lab 实验阶段报告](08_lab_experiment_progress_2026-09-13.md) 和
> [下一阶段 TODO](01_next_stage_vision_teleop_todo.md) 为准。

## 1. 目标与结论

目标是让双臂夹爪协作，把圆柱体放入指定凹槽。先在仿真中完成任务定义、控制接口、数据
采集和策略评估，再把相同 observation/action 契约迁移到 Panthera 真机。

该任务可行，但属于接触丰富的双臂装配任务。VLA 适合根据图像和指令产生短时动作块或
子目标，不应直接承担 200 Hz 电机闭环、关节限位、碰撞约束和失联停车。最终对准和插入
若间隙很小，还需要视觉伺服、力/力矩反馈，或者带倒角和装配容差的任务设计。

## 2. 设备分工

```text
实验台摄像头
  └─ SRT + 源 PTS ──> WSL 接收/解码/episode 编排
                         ├─ 图像 + 双臂状态 + 指令 ──> Lab VLA 推理
                         ├─ 动作块/子目标 <──────────── Lab
                         └─ 新鲜度、限位、状态机检查 ──> 控制盒
                                                          └─ 高频双臂闭环
```

- WSL 笔记本：Codex、任务调度、时间戳对齐、数据记录、安全网关和实验状态机。
- `scalelab01`（简称 Lab）：仿真、训练和 VLA 推理；不直接持有真机串口或绕过安全网关。
- 机械臂控制盒：机器人反馈和确定性的高频执行。
- 固定外部摄像头：任务场景观测。SRT 只负责传输，episode 数据仍需保存解码帧的 PTS、
  接收时间、机器人状态时间和实际执行动作。

## 3. 第一版任务定义

先固定桌面、摄像头、圆柱尺寸和凹槽位置，把长任务分解为：

1. 双臂进入观察位；
2. 定位圆柱和凹槽；
3. 左/右臂分别到达预抓取位；
4. 抓稳圆柱；
5. 双臂协调搬运并对准凹槽；
6. 低速插入；
7. 判断成功后释放并撤离。

必须明确双臂协作语义：两臂共同夹持同一圆柱，还是一臂持圆柱、另一臂稳定工装。两种
任务的动作空间、约束和训练数据不同，不能在实现中混用。

成功判据应由仿真状态或传感器给出，例如圆柱轴与凹槽轴夹角、横向误差、插入深度、
接触力和释放后保持时间。不能只用“画面看起来放进去了”。

## 4. 仿真实施顺序

1. 导入并校验双臂 URDF、关节方向、夹爪几何、碰撞体、惯量和限位。
2. 用脚本状态机完成一次任务，先证明环境、动作接口和成功判据正确。
3. 建立与真机一致的 observation/action 契约和 episode 记录格式。
4. 采集或生成示范，先复现 RoboTwin/RLinf 的现成 VLA 评测配方，再建立本任务基线。
5. 随机化物体位姿、摩擦、质量、相机外参、光照和纹理，但每种随机化必须有范围记录。
6. 固定测试集统计成功率，并单独统计抓取、搬运、对准、插入各阶段失败率。
7. 真机前先运行 shadow mode：Lab 预测但不下发，只与已知动作或人工示范比较。

本项目采用 **RoboTwin 2.0 + RLinf**：

- RoboTwin 负责 SAPIEN 双臂仿真、相机、物体资产、脚本专家、数据生成、domain
  randomization 和成功判定；
- RLinf 负责 VLA 的 SFT/评测与 PPO、GRPO、DAgger 等训练流程，并通过 Ray 调度模型和
  RoboTwin rollout worker；
- 第一阶段先复现 RLinf 已提供的 `place_empty_cup`：其 RoboTwin 仿真 embodiment 实际为
  `[piper, piper, 0.6]`，OpenVLA-OFT 侧使用 ALOHA 的 14 维动作/本体状态契约；之后再接入
  Panthera embodiment 和圆柱入槽任务；
- RoboTwin 的 `qpos` action 顺序是
  `[left_arm_joints, left_gripper, right_arm_joints, right_gripper]`。两条 Panthera 各六
  个臂关节加一个夹爪，天然对应 14 维状态/动作，但关节方向、范围和夹爪归一化必须逐项
  校验，不能只因维度相同就直接复用 ALOHA 权重。

RoboTwin 支持把两台单臂 embodiment 组合成
`[left_robot, right_robot, interval]`。Panthera 需要补齐 URDF/SRDF、CuRobo 左右臂配置、
碰撞球、末端 link、夹爪 mimic、home state、相机和坐标变换。圆柱入槽应建立自定义 task，
可分别借鉴 `lift_pot` 的双臂共同持物和 `place_empty_cup` 的目标放置逻辑。

### 4.1 程序化圆柱入槽 oracle 基线（2026-09-12）

第一版 `place_cylinder_in_groove` 已在固定 RoboTwin 提交的双 Piper embodiment 上运行。
任务使用程序化圆柱、两个可拆卸式抓取套环和直线 U 形槽；两臂从两个抓取站共同持物，
执行抓取、抬升、搬运、下降、打开、向两侧抽离、250 个仿真步沉降和回原位。采集按动作
完成事件与固定仿真步推进，不使用墙钟 `sleep` 截取轨迹。

RoboTwin 的 `together_move_to_pose` 仍分别规划左右轨迹，只按归一化进度同步，未求解
“两臂 + 共享刚体”的闭链。直接依赖摩擦时，规划均可返回 Success，但圆柱会滑移或被两
条轨迹挤出。当前 scripted oracle 因此在检测到双夹爪接触后建立 SAPIEN D6
attach-on-grasp 约束，约束存续时屏蔽圆柱碰撞以避免桌面/双手闭环反作用，打开夹爪后恢复
碰撞并解除约束，再用真实动力学完成落槽与稳定性验收。这个约束是当前任务生成器的明确
建模假设，不代表 Panthera 真机已有等价能力；后续可以在更好的闭链规划、软约束或真实
抓取证据充分时弃用。

四个固定种子全部通过，且每条轨迹保留 13 个阶段帧。最终最大横向误差 0.32 mm、最大
X 误差 0.57 mm、最大圆柱轴误差 0.62°、高度误差约 4.5 mm；释放后线速度和角速度均为
0，双夹爪均打开。四张 contact sheet 非空，运行日志无 Traceback/RuntimeError，结束后
GPU 与 Ray 进程均为 0。结果位于
`/data/lyy/panthera-vla/results/cylinder-oracle-smoke/`，成功标记为
`/data/lyy/panthera-vla/.cylinder-oracle-state/oracle.ok`。这只证明程序化任务、动作流程和
成功判据成立；在该双 Piper 阶段结束时，Panthera embodiment、示范数据、VLA 策略和
shadow mode 尚未完成。

### 4.2 Panthera MPLib embodiment 与 oracle（2026-09-12）

第一版 Panthera embodiment 由
`simulation/robotwin_overlay/build_panthera_embodiment.py` 从官方
`Panthera-HT_ROS2` 提交 `b08633d6c5bce89baad1821fd598243a84bc3a84` 的带夹爪 xacro
确定性生成。生成器只改写 `package://` mesh URI，并增加 RoboTwin 配置和精简 SRDF；
官方源码与 mesh 不被修改，来源提交、许可证和每个生成文件的 SHA-256 记录在
`provenance.json`。仿真模型包含 10 个 link、`joint1..joint6` 六个旋转关节及
`L_finger_joint/R_finger_joint` 两个直线关节，限位与官方模型逐项一致。夹爪归一化
`0..1` 映射到左指 `0..0.04 m`、右指 `0..-0.04 m`。仿真 homestate 采用官方 SRDF
`pose1 = [0, 1.6, 1.6, 0, 0, 0]`，不能把它称为 Host 真机 `position0`。

接入过程中发现并以可选/通用补丁处理了三个 RoboTwin 适配问题：

1. `MplibWrapperPlanner` 原来把完整八关节 qpos 裁成六轴，MPLib 随后把两个夹爪补零，
   把实际张开的夹爪错误地变成闭合碰撞状态；现在完整 qpos 会保留，规划结果仍为六轴。
2. scene-backed `SapienPlanningWorld` 不读取传入的 SRDF；wrapper 现在把全部
   `<disable_collisions>` 条目载入允许碰撞矩阵，使其与 standalone MPLib 语义一致。
3. RoboTwin 原来用关节 frame 的 `joint.global_pose` 当末端姿态。Panthera 的 joint6
   frame 与 child `link6` 在转动后不相同，位置能对上但四元数距离达到 `1.4142`；新增
   embodiment 级 `ee_pose_from_child_link` 开关，只让 Panthera 从 child link 读取姿态，
   不改变已有 Piper 行为。

`bootstrap_lab_panthera_embodiment.sh` 已完成 500 个无墙钟 sleep 的物理步、限位、夹爪
开合、14 维动作顺序和双臂七方向 IK 验收。左右臂当前位姿均返回 Success，六个非零
`±1 cm` 目标也全部成功，轨迹为 87–117 点，接触对为空。随后双 Panthera
`place_cylinder_in_groove` 在 4/4 固定种子通过，每条保留 13 个事件阶段；最终最大 X
误差 0.707 mm、横向误差 0.296 mm、轴误差 0.074°、高度误差约 4.50 mm，释放后
线速度和角速度均为 0。结果与成功标记分别位于
`/data/lyy/panthera-vla/results/panthera-cylinder-oracle-smoke/` 和
`/data/lyy/panthera-vla/.panthera-cylinder-oracle-state/oracle.ok`。

这个结果完成的是 Panthera 几何、MPLib 规划接口、14 维契约和程序化 oracle 验收。
它仍沿用 4.1 的 attach-on-grasp 假设；CuRobo 碰撞球、正式数据集、VLA 策略、无辅助
约束的双臂物理抓取和 shadow mode 尚未完成。

### 4.3 上游格式示范数据 smoke（2026-09-12）

双 Panthera oracle 已接入 RoboTwin 官方 `script/collect_data.py`，以 250 Hz 物理步进、
每 5 个仿真步采样一次，在单个无人值守流程内生成 4 个 episode。每集有 628 个头部 RGB、
左右六关节、左右夹爪和末端位姿样本，以及一条 628 帧 MP4；14 维顺序、有限值、官方
关节限位、HDF5/视频非空、缓存清理和 Ray/GPU 进程清理均通过。结果位于
`/data/lyy/panthera-vla/data/place_cylinder_in_groove/panthera_cylinder_dataset_smoke/`。
RoboTwin 在保存后销毁机器人、再调用成功判定，因此任务会在环境关闭前锁存终态指标；
这只修正对象生命周期，不改变动作或物理。

该结果只证明上游采集和回放格式可用，不能作为正式训练集：当前 4 个种子没有引入场景
变化；HDF5 不含时间戳；名为 `joint_action` 的关节量来自 drive target，而不是实际 qpos；
MP4 固定以 30 FPS 编码，不能代表 50 Hz 仿真时间；自然语言脚本还会把绝对 `save_path`
错误地再次拼到 RoboTwin 根目录。正式数据集必须先分离 action/observation、保存确定性
step/time 和完整 episode 元数据，并加入受控随机化。

随后完成的正式契约 smoke 位于
`/data/lyy/panthera-vla/data/place_cylinder_in_groove/panthera_cylinder_contract_smoke/`。
4 个 seed 使用确定性的毫米级圆柱/凹槽位置扰动，统一时钟重采后分别生成 701–710 帧。
每帧分离保存 14 维 action target 与 14 维实际 articulation qpos/夹爪开度，并用集中替换所有
`scene.step()` 的计数器保存精确 simulation step/time。直接物理沉降循环也按全局步数
采样，因此包括释放后 1 秒落槽过程在内，相邻样本最多间隔 5 个 250 Hz 物理步；成功路径
不使用墙钟 sleep。每集生成 7 条语言指令，并在 `scene_info.json` 记录两份固定源码提交、
随机化 seed/边界、实际几何、动作顺序和 attach-on-grasp 标记。旧的最大 250 步采样空洞
版本已改名为 `panthera_cylinder_contract_smoke.v1-noncontinuous/` 保留，不参与后续使用。
这些 4 集只验收数据契约；其数量和随机化范围不足以训练或证明 VLA 策略。

### 4.4 OpenVLA RLDS 适配（2026-09-12）

检查 RLinf 固定的 OpenVLA-OFT 源码后确认，`JOINT_POS_BIMANUAL` 的注释虽然仍写
“Joint Delta Position”，实际 `materialize.py` 将 14 个维度全部标为 absolute action，
轨迹末端也会重复最后一个目标，不做关节差分。因此 Panthera 的左右六关节加双夹爪
绝对目标与该接口语义一致，但不能复用 `place_empty_cup_1k` 的 ALOHA/Piper 归一化统计。

RoboTwin 的采样顺序是先设置 drive target、推进物理、再调用 `_take_picture()`；同一行
图像对应的 `joint_action` 因而是已经执行过的目标。新增
`simulation/openvla_adapter/panthera_rlds.py` 在转换时只保留全局仿真 step 可被 5 整除的
样本、按 step 保留最后一张、要求相邻网格严格为 5 个 250 Hz 物理步，并把图像/实测 qpos
配到下一网格动作。RoboTwin 两处运动循环的局部采样计数也已改为全局
`simulation_step_count`；原始 HDF5 仍可保留阶段首尾诊断帧，训练时间轴不会受其影响。

`panthera_cylinder` TFDS 1.0.0 已在 `/data/lyy/panthera-vla/rlds/` 生成，smoke 明确划分
train `[0,1,2]`、val `[3]`。OpenVLA-OFT 的真实 `make_single_dataset` 路径成功解码并缩放
主相机到 `1x224x224x3`，输出 `1x14` proprio 窗口和 `25x14` absolute-action chunk，
同时计算 Panthera 自己的 14 维 action/proprio min/max。RLinf RoboTwin 在线适配也改为
优先读取 `observation/robot_state/vector`，仅在旧任务没有实测字段时回退到
`joint_action/vector`。这完成数据接口验收，不等于策略训练或闭环成功。

正式 SFT v1 将 128 个 episode 划分为 train 0–111、val 112–127，并同时随机化圆柱、凹槽、
头部相机外参、背景和光照。`run_lab_panthera_sft_pipeline.sh` 不用固定时长轮询：它先阻塞
等待当前数据集文件锁释放，再把 train/val 样本送入 RoboTwin 真正用于策略评测的
`gen_sparse_reward_data` 动作路径回放。这个门槛不会调用 `play_once()`，也禁止创建 oracle
D6 约束；只有物理回放成功才继续 RLDS 转换和单 GPU OpenVLA 最小优化器 smoke。后续阶段
只在前一阶段写出通过标记后启动；正式采集与三阶段全部通过前，不能把它记作 SFT 基线完成。

训练后的闭环评测配置已经独立完成 Hydra 解析验收，但尚未运行策略。它使用 16 个固定的
未训练 seed、双 Panthera 14 维绝对关节动作、25 步动作块、800 步 episode 上限和
`panthera_cylinder` 自身归一化统计；四 GPU 并行四个环境、运行四轮并逐轮推进固定 seed，
从而覆盖 16 条不同轨迹。800 步不是任意放宽成功判据，
而是覆盖当前示范约 700 个 50 Hz 网格目标并保持 25 的整数倍。闭环评测仍须检查成功率、
分阶段失败、完整视频、致命日志以及 Ray/GPU 清理后才能写通过标记。

第二级无人值守流水线使用上游文件锁事件接续，不按预计时长轮询。它逐项读取
128-episode 数据、无约束回放、正式 RLDS 和优化器 smoke 的 JSON 摘要，全部通过后才以
现有 `place_empty_cup` checkpoint 作为明确标注的 transfer initialization，在四张 GPU
上进行首轮 5000 步 LoRA SFT。之后自动运行上述 16 轨迹评测；`success_once` 至少 75%、
16 条视频完整且无致命日志/残留进程时才写最终通过标记。低于门槛会保留模型和全部证据，
但停止流水线，不会自动放宽门槛或进入真机。

RLinf 当前官方 RoboTwin 训练配方标注为 8–16 GPU、1–2 节点；本 Lab 是四张 48 GB
RTX 4090（总显存约 192 GiB），因此先做单机评测、数据生成和缩小并行度的可运行性验证，
再决定如何缩减 actor/rollout worker、batch size 或模型规模。不要直接按默认分布式配置
启动完整 GRPO。

Lab 的工程根目录、源码 checkout、文档、RoboTwin 资产、模型权重、容器绑定目录、数据集
和训练结果必须放在 `/data/lyy/`。`/home/lyy` 只允许保存少量 shell/SSH 配置和入口脚本，
不能作为工程或实验数据目录。

Lab 的日常 SSH、源码管理、Miniforge/venv、Python 依赖、仿真、训练、评测和任务调度均
使用普通用户 `lyy`。管理员账户 `mole` 只用于无法由用户环境解决的驱动、系统包、重启
和文件权限修复；使用前必须说明原因，不用管理员账户运行训练或持有项目文件。

## 5. 第一版数据契约

```text
observation:
  external_rgb, optional_wrist_rgb
  left_joint_position, right_joint_position
  left_gripper_state, right_gripper_state
  optional_effort_or_force
  task_instruction

action:
  timestamped left/right joint target chunk
  left/right gripper command

episode metadata:
  camera intrinsics/extrinsics, robot/calibration version
  object/groove geometry, randomization seed
  success, failure_stage, abort_reason
```

人类遥操作输入用于产生示范动作，不应默认作为部署时 VLA 的 observation。

## 6. 真机安全边界

- VLA 输出必须带生成时间、有效期和序号；迟到、乱序或断线后保持并锁存禁用。
- WSL 安全网关检查关节名、有限值、限位、最大速度/加速度、工作空间和双臂碰撞。
- 高频命令仍由控制盒执行；Lab 网络中断不能让上一动作块无限继续。
- 仿真成功不证明真实摩擦、夹持力、标定误差或急停有效。
- 真机首次只验证只读状态和 shadow mode；任何运动仍遵守真机检查表与明确确认规则。

## 7. 当前基础设施状态与阻断项

- WSL 已在 UDP 9000 运行 SRT listener。`tools/receive_srt_camera.sh` 会生成 2 fps 的
  `/tmp/panthera-srt-camera/latest.jpg` 和约两分钟循环缓存。真实摄像头已用
  HEVC 1920×1080@30 fps + AAC 成功接入。
- WSL 使用 NAT，地址为 `192.168.30.150/20`；Windows WLAN 为 `192.168.31.30`，且
  Hyper-V 默认阻止入站。现已只允许 `192.168.31.0/24` 访问 Windows UDP 9000，并由
  `tools/windows_srt_udp_proxy.ps1` 双向转发到 WSL。
- WSL 原生 SSH 到 Lab 的 `172.17.20.31:22` 超时；Windows OpenSSH 可以登录，说明当前
  VPN/路由未透传进 WSL。短期统一使用 `tools/lab_ssh.sh` 调用 Windows OpenSSH，长期应
  修复明确的路由边界。
- Lab 已重启，内核模块、NVML 和驱动均为 595.91.07；四张 48 GB RTX 4090 全部正常，
  温度约 28–30°C。主机为双路 Xeon Gold 6330、112 逻辑 CPU、251 GiB RAM；`/data`
  共 7.3 TiB、可用 4.6 TiB，`/data/lyy` 权限正常。GPU 没有 NVLink，GPU 0/1 位于 NUMA 0、
  GPU 2/3 位于 NUMA 1；分布式 worker 应按同 NUMA GPU 对优先放置。按用户要求采用
  无 Docker 原生环境：Miniforge 26.7.2-0 已以普通用户安装到
  `/data/lyy/tools/miniforge3`，安装器 SHA-256 已与 GitHub release 元数据核对；RLinf
  使用 `/data/lyy/panthera-vla/RLinf/.venv` 的 Python 3.11 Conda prefix。系统已有
  `/usr/local/cuda-13.2`，无需再复制旧 rootless CUDA 工具链。
- RLinf 和 RoboTwin 已直接从官方 GitHub 检出到 `/data/lyy/panthera-vla/`，分别固定为
  `a3816b596478dcd8a5c69a6ec1468c9519f77b5b` 和
  `0008ae6800df9f75fc8de7098bacb01735fd8fd2`。`tools/bootstrap_lab_vla.sh` 提供可恢复、
  分阶段、有锁和日志的后台构建；`tools/activate_lab_vla.sh` 负责后续激活。环境依赖和
  RoboTwin 官方 assets 已完成下载、解压和路径配置；`environment.ok`、`assets.ok`、
  `verified.ok` 均已生成。验收结果为 PyTorch 2.11.0+cu130、CUDA runtime 13.0、四张
  48 GB RTX 4090，以及 RLinf、SAPIEN、MPLib、PyTorch3D、cuRobo 全部导入成功。
- CUDA 13.2 已移除 `compute_70`，且 CUDA 13 改变了模板符号可见性。构建工具将扩展目标
  固定为本机 Ada `sm_89`，并应用 PyTorch3D 官方建议的 CUDA 13 NVCC 兼容参数；相对
  RLinf 固定提交的最小补丁保存在 `tools/patches/rlinf_robotwin_cuda_arch.patch`。RLinf
  原安装器会取 cuRobo `HEAD`，当前 V2 API 已移除 RoboTwin 使用的 `curobo.types.math`；
  因此固定到 NVIDIA 的 V1 兼容标签 v0.7.8，并为新版 `uv` 显式提供 setuptools-scm
  wheel 版本。对应补丁为 `rlinf_robotwin_curobo_v1.patch` 和
  `rlinf_robotwin_curobo_metadata.patch`。
  Lab 到 `huggingface.co` 的直连会停在 TCP 握手，本机现有代理也无法完成 TLS；
  `hf-mirror.com` 已验证可用，因此资产和模型下载默认使用可覆盖的 `HF_ENDPOINT`，仍由
  huggingface-hub 校验并续传。
- 官方 OpenVLA-OFT `place_empty_cup` 模型固定为提交
  `04150e05ded8b915ccf6adb7ea903ff13dc3aa27`（22 个文件，15,152,598,778 bytes）。
  `tools/run_lab_robotwin_baseline.sh` 将模型续传、Hydra 配置验收、Ray 冲突检查和 4 卡
  4 环境/200 步 smoke 合并为单个无人值守流程，状态写入
  `/data/lyy/panthera-vla/.robotwin-baseline-state/`。2026-09-12 已完成一次有效复现：
  `success_once=0.75`、`success_at_end=0.75`、4 条轨迹、平均 episode 长度 200，四条非空
  MP4 齐全，日志中无 Traceback/导入/Ray worker 异常，结束后 Ray 与 GPU 进程均为 0。
  一次较早运行曾因 cuRobo V2 导入失败却返回 0 并汇总出同样指标，现已归档为失败证据；
  验收脚本现在同时要求退出码、无致命日志、完整指标和视频数量，避免再次出现假成功。
- 当前摄像头画面只确认一条机械臂；还需确认第二条机械臂、双夹爪、控制盒网络接口、
  双臂 URDF 和统一关节命名。
- 外部相机真实流已收到，方向已摆正，主工作台和黄色圆柱/底座位于画面中央。广角镜头
  边缘畸变明显；两只夹爪尚未同时以任务工作姿态进入中央操作区，仍需完成内参、外参、
  中央 ROI 和插入阶段遮挡检查。

## 8. NAS 资源审计

2026-09-12 对 Lab 挂载的 `/mnt/hulab/pro6000/data` 做了浅层只读审计。广域网 NFS 是
`hard` 挂载且吞吐较低，不应先把全部目录、虚拟环境和缓存无差别复制到 Lab。

- `RLinf/` 是最高优先级资源。它是官方 RLinf checkout（`main`，提交
  `a3816b596478dcd8a5c69a6ec1468c9519f77b5b`），且有 NAS 独有的
  `experiment_notebooks/` 三个 `place_empty_cup` walkthrough/说明文件。这些文件可用于
  理解 RoboTwin 任务、分阶段截图和报告生成，但其中启动脚本当前假定
  `RLinf/RoboTwin`、`RLinf/.venv-robotwin` 和 `RLinf/.rootless-tools` 同目录，迁移后必须
  按 `/data/lyy/` 的新布局修正，不能原样执行。`migrate_rlinf_to_mnt_data.sh` 写死旧用户
  `/home/liyuyang` 与 `/mnt/data/liyuyang`，只作历史参考，禁止运行。
- `RLinf-rootless/RoboTwin/` 是直接有用的 RoboTwin 2.0 `RLinf_support` checkout，提交为
  `0008ae6800df9f75fc8de7098bacb01735fd8fd2`，包含任务源码和资产。只保留 RoboTwin
  checkout 或从官方仓库重建同一提交；不要复制同级 `.venv-robotwin/` 和
  `.rootless-tools/`。两者分别写死旧 Python 路径并打包了一套 CUDA/CMake/Ninja
  toolchain，在当前 Lab 驱动和 `/data/lyy` 布局下不可作为可复现环境。
- `ConCon/` 是次级设计参考，而不是 Panthera 的运行依赖。它实现了 OpenArm v2 + Sharpa
  双臂在 MuJoCo 中操作游戏手柄，并接入 RLinf。可借鉴多时钟控制、latest-command
  bridge、分阶段策略、batched env adapter、物理状态 guard、固定 seed 验收、失败阶段
  metrics 和离线 recovery memory；其 OpenArm/Sharpa 模型、游戏手柄任务、策略权重和
  标定值不能直接用于 Panthera 圆柱入槽。其 origin 是 `https://github.com/5o1/ConCon`，
  NAS 快照的本地 `main` ref 不完整，所以应重新 clone 后只选择性取回 NAS 独有证据或
  checkpoint，不复制整个 `.git`、`.venv`、`external` 和 `downloads`。
- `fast-zeroshot/` 是 MRI 零样本重建/INR 加速，`traj-correction/` 是 MRI k-space 轨迹
  优化，`Lessons-in-Cast/` 是游戏语音、TTS 和配音流水线；三者与本任务无关，不应占用
  Lab 的迁移时间或 Panthera 工程空间。

迁移策略是：优先从官方 GitHub clone 并固定提交，只从 NAS 复制不可重建的自定义代码、
实验说明、少量验收记录和确实要复现的 checkpoint；所有 Python 环境在
`/data/lyy/panthera-vla/` 重新创建。权重、数据集和可再下载的第三方仓库在明确首个
evaluation 配方后再按需取回。

## 9. 下一里程碑

1. [已完成] 在 `/data/lyy/` 建立无 Docker Lab 环境并复现 RoboTwin + RLinf 官方
   evaluation。
2. [已完成] 明确第一版双臂共同夹持语义、成功公差，并让双 Piper 程序化圆柱入槽
   scripted oracle 通过 4/4 固定种子。
3. [已完成：MPLib] 收集 Panthera URDF、夹爪、关节限位和命名映射，建立并验收双
   Panthera RoboTwin embodiment 与圆柱入槽 oracle；没有沿用 Piper/ALOHA 的关节范围
   或归一化。CuRobo 配置只在后续配方需要时补充。
4. [进行中] oracle 的上游格式、4-episode 正式契约以及 OpenVLA RLDS 单批次 smoke 均已
   通过；下一步采集 128-episode SFT v1，扩大有界几何/相机/背景/光照随机化，再建立 VLA
   SFT/评测基线，并在固定测试集单独统计抓取、搬运、对准和落槽阶段失败率。
5. 固定外部相机，标定内参/外参和中央 ROI，并确认圆柱、凹槽和两只夹爪在关键阶段可见；
   决定第一版是否加入腕部相机或力/力矩传感器。
6. Panthera 真机前先跑 shadow mode；在此之前不得启动真实运动。

## 10. 官方实现入口

- [RLinf 的 RoboTwin VLA 配方](https://github.com/RLinf/RLinf/blob/main/docs/source-zh/rst_source/examples/embodied/robotwin.rst)
- [RoboTwin 2.0 配置格式](https://robotwin-platform.github.io/doc/usage/configurations.html)
- [RoboTwin 新机器人 embodiment 接入](https://robotwin-platform.github.io/doc/usage/new-embodiment.html)
- [RoboTwin qpos/末端动作接口](https://robotwin-platform.github.io/doc/usage/control-robot.html)
- [SAPIEN 3 PhysxDriveComponent API](https://sapien-sim.github.io/docs/api/sapien.pysapien.render.html#sapien.pysapien.physx.PhysxDriveComponent)
