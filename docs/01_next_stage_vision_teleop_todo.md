# Panthera 单目视觉遥操作：下一阶段 TODO

## 0. 本阶段从真机测试得到的结论

真机链路已经打通，但操作者进行了上下、前后和手臂旋转动作时，机械臂只在很小
范围内移动。这与当前安全配置一致，不是驱动漏掉了大部分动作。

当前位置映射的核心公式是：

```text
robot_offset = clip(position_gain * axis_map * human_delta,
                    -max_position_offset,
                    +max_position_offset)
```

当前参数位于 `ros2_ws/src/panthera_vision_teleop/config/teleop.yaml`：

| 人体动作 | 机器人轴 | 当前增益 | 最大偏移 | 当前效果 |
|---|---:|---:|---:|---|
| 靠近/远离摄像头 | X 前后 | 0.00 | 0 cm | 完全锁定 |
| 向人体左右移动 | Y 左右 | 0.25 | 5 cm | 人手 10 cm → 机器人约 2.5 cm |
| 向上/向下移动 | Z 上下 | 0.12 | 4 cm | 人手 10 cm → 机器人约 1.2 cm |
| 转动手臂 | 末端姿态 | — | 20°（尚未启用） | `enable_orientation=false`，完全忽略 |

位置还会经过 `position_alpha=0.25` 的 EMA。当前 mapper 以 20 Hz 生成预览目标，并使用
逐轴 6 mm/3 mm 进入—退出迟滞；真实 `/pos_cmd` 仍受“一次一个在途点”的阻塞接口限制。

真机测试还得到两项约束：

- MediaPipe 单目 world-landmark 深度能反映前后趋势，但静止及移动时的 X 噪声明显；
- 通用手势模型会把自然放松手误判成 `Open_Palm`，夹爪不能直接随分类结果动作。

2026-09-02 内置摄像头长流程表征数据位于
`bags/camera_characterization_20260902_001702`：静止/前后/左右/上下各收集 299 个有效帧，
屈伸和 roll/pitch/yaw 各 239 帧，捏合 299 帧。视觉链路平均约 15–18 Hz，WebSocket
附加传输延迟 p95 为 13.6 ms。主要结论：

- 左右动作的 camera-X 5%–95% 跨度为 13.86 cm；上下动作的 camera-Y 跨度为
  10.78 cm。经 safe 增益和 2 mm 死区后，典型 Y/Z 目标只约 3.3 cm / 1.1 cm，
  证实“机械臂移动范围小”是当前保守参数的直接结果。
- 静止时 world-depth 5%–95% 波动 2.75 cm，前后动作信号只有 4.26 cm，因此 X 继续锁定。
- pitch 阶段因可见度下降用了约 55 s 才收齐 239 个有效帧，姿态真机开放前还需要
  更强的连续性/遮挡验收。
- 刻意捏合阶段的有效比例呈明显两群，但静止和摆臂阶段也会长时落入“闭合”侧；
  仅改双阈值仍会误触，夹爪保持禁用。
- 正常打字时手腕必然离开屏幕顶部内置摄像头视野，因此该附加 case 在当前硬件
  几何下判定为不适用，从验收范围删除，不要求添购摄像头。

2026-09-02 首次 `safe` Y/Z 真机长流程位于
`bags/robot_yz_acceptance_20260902_004734`。static、左右、上下、YZ 小圆和 stop-hold 五个
阶段均完成。66 条 `/pos_cmd` 全部落在 WARMUP/RECORDING marker 内，等待阶段没有命令；
static 和 stop-hold 都没有发送位移命令，stop-hold 的末端 Y/Z 峰峰漂移仅
0.093/0.241 mm。目标范围为：左右阶段 Y 53.7 mm、上下阶段 Z 39.4 mm、YZ 小圆阶段
Y/Z 71.2/44.5 mm。safe Y/Z 因而通过首次真机验收，但这不构成开放 X、姿态或夹爪的依据。
正常退出的全零 park 服务返回成功，不过录包末帧第三关节仍有 0.035 rad 残差；扩大增益前
先补充退出位置容差审计。

## 1. P0：先建立可量化的观测工具

- [x] mapper 发布映射中间量：人体相对位移、原始机器人偏移、死区后偏移、限幅后
      偏移和滤波后偏移；后续仍需增加命令状态和自动统计摘要。
- [x] 提供一键 rosbag 录制脚本，至少记录：
      `/teleop/human_pose`、`/teleop/pose_score`、`/teleop/debug_target`、
      `/end_pose_euler`、`/arm_status` 和 `/pos_cmd`。
- [x] 在预览窗口显示 `Δhuman`、`Δtarget`、当前启用轴以及 `CLIPPED/STALE` 状态。
- [x] 编写离线绘图脚本，对比人体输入、目标和真机反馈，不再依靠肉眼猜测增益。

验收标准：一次测试结束后可以回答“操作者移动了多少、目标移动了多少、机器人实际
移动了多少、哪一段被滤波或限幅”。

已完成的通用安全加固：失跟、低置信度、断线或超时都会锁定启用状态；恢复数据本身
不会自动续动。来源帧积压超过阈值会被拒绝。由于官方驱动的点位回调是单线程阻塞
执行，mapper 还把每条不可中断的笛卡尔位移限制为 12 mm；这仍不能替代物理急停。

## 2. P0：扩大 Y/Z 有效运动范围，但保持可控

- [x] 将参数整理为 `safe`、`normal`、`extended` 三个 Y/Z profile，不直接覆盖唯一配置。
- [x] 用一次 camera envelope 长流程测得操作者输入范围：camera-X 的 5%–95% 跨度
      35.84 cm，camera-Y 为 23.23 cm；不再用逐档真机动作搜索映射参数。
- [x] normal 工作空间固定为 Y `±8 cm`、Z `±6 cm`，映射增益为 Y `0.35`、Z `0.20`。
- [x] 增加每轴 deadband、目标速度上限和目标加速度上限；EMA 只负责降噪，不承担
      安全限速职责。
- [x] 将“6 mm 最小命令距离”改成逐轴迟滞，避免某一噪声轴触发整个位姿命令。
- [ ] 停止逐级速度/增益真机试验。改用 Host 已验证的 0.6/1.0 rad/s、2.0 rad/s²边界和
      非阻塞控制架构；集成完成后只运行一次 Y/Z 端到端连续录包验收。

验收标准：人体左右或上下移动约 20 cm 时，机械臂稳定移动 5–8 cm；人体静止时，
末端目标峰峰抖动不超过 5 mm，且不会持续发送微小往返命令。

## 3. P0：重新设计单目深度 X，而不是直接恢复增益

- [x] 用 rosbag 分别采集“静止、前伸/后收、左右、上下”样本，量化
      `wrist.z - shoulder.z` 的噪声、迟滞和轴间串扰。
- [x] 协议、ROS 话题和 rosbag 已同时记录三种相对深度原始特征，已采集实际摄像头样本并
      生成分阶段 JSON/CSV/PNG。后续融合时对比：
      1. MediaPipe world-landmark 相对 Z；
      2. 以肩宽或上臂长度归一化的 wrist/shoulder 图像尺度；
      3. 两者融合后的 2.5D 深度估计。
- [x] 为 X 使用独立滤波、deadband、增益和速度限制，不复用 Y/Z 标量参数。
- [x] 增加“深度 clutch”：只有按 X 进入专门的前后控制状态时，
      X 才更新；松开后保持当前 X。
- [ ] 通过 camera dry-run 后，以 `±1 cm`、`±2 cm`、`±4 cm` 分级开放真机范围。

验收标准：静止 10 秒时 X 目标峰峰抖动不超过 5 mm；前伸/后收动作方向正确、重复
三次结果一致，不因普通左右摆动产生明显前后运动。

注意：单 RGB 摄像头无法像深度相机那样直接提供稳定的公制深度。本阶段追求可靠的
相对 2.5D 控制，不宣称毫米级绝对深度。

## 4. P1：分阶段启用末端姿态

- [ ] 用录制数据检查肩—肘—腕构造的手臂坐标系在弯臂、直臂和遮挡时是否连续。
- [x] 直臂导致平面法向量退化时保持上一姿态，并在预览中显示 `ORIENTATION HELD`。
- [x] 提供仅开放一个旋转轴和 `±5°` 的 roll/pitch/yaw 独立 dry-run profile。
- [ ] 分别完成真实摄像头方向验收 roll、pitch、yaw；禁止一次同时开放三个未经验证的轴。
- [x] 使用四元数相对旋转、Slerp、旋转向量角速度限制和角加速度限制，避免欧拉角跳变。
- [ ] 最后才将 `enable_orientation` 设置为真机 profile 的可选项，默认仍关闭。

验收标准：单轴转动手臂时，末端只沿预期轴旋转；回到标定姿势后误差小于 3°；姿态
丢失或手臂伸直时不会突然翻转。

## 5. P1：改善命令调度与运动手感

- [x] 保留 latest-only 语义，继续禁止积压历史视觉动作。
- [x] 审计官方驱动：确认 `/pos_cmd` 的阻塞来自 `arm_control_node` 使用
      `posVelMaxTorque(..., true)`；SDK 和 `PantheraHardwareInterface` 已支持非阻塞命令。
- [x] 建立纯 GenericSystem 的 100 Hz `JointTrajectoryController` 冒烟测试；六关节
      `FollowJointTrajectory` 已返回 `error_code: 0` 和 `SUCCEEDED`。
- [x] 用项目内 overlay 覆盖官方 ros2_control 的 controller 名称和 command-interface
      不一致，不修改官方仓库；deactivate stop 仍需真机验证/上游补偿：controller
      名称、position/velocity interface、限速、deactivate hold/stop 和命令过期 watchdog。
- [x] 轻量 URDF FK/Jacobian/IK + 短轨迹 backend 已在 GenericSystem 通过目标到达、stale
      cancel/hold。当前阶段弃用 Servo 默认路线：本机未安装 Servo，且尚无证据证明其依赖
      与复杂度能带来额外收益；需要碰撞场景时可重新评估。
- [ ] 在用户在场前只允许 mock；真机依次验证只读反馈、当前位置保持、极小单关节轨迹、
      action cancel/超时和故障恢复，禁止直接从视觉真机测试开始。
      - 2026-09-01：首次 hold 在 action 前因七电机断连和 SDK `999` 无效位置中止；已增加
        启动日志与初始关节双重硬阻断。盒子上电后重试，电机全在线、当前位置 action
        成功、漂移不超过 0.01 rad；随后合并 case 中 joint1 0.005 rad 往返和 action cancel /
        取消后保持/返回均通过。hold、micro、cancel 已验收，故障恢复与 deactivate stop 未验收。
- [x] 笛卡尔 mapper 已使用速度/加速度限制；连续候选 backend 使用 latest-only 短关节段。
- [x] 区分输入采样频率、控制更新频率和驱动命令频率；rosbag 分析按话题输出实际频率、
      中位/P95/最大采样间隔，并单独记录 WebSocket 额外传输延迟。
- [x] 对假的慢速 `/pos_cmd` 反馈做压力测试，验证慢动作时不会形成快速 FIFO 补发；
      真实官方驱动压力测试仍属于真机阶段。
- [x] 假机器人可选择订阅 `/pos_cmd` 并以有限速度产生反馈，用纯软件验证“一次只发
      一条、到达后再发下一条”；safe 真机 Y/Z 连续动作已验证没有等待阶段补发，扩大增益
      后仍需重新检查阻塞驱动的吞吐和停止延迟。
- [x] 连续候选 backend 对目标跟踪误差、action 超时/异常和连续 IK 失败计数；达到阈值后
      `FAULT_LOCK`，只能在无在途 goal 且存在有效反馈时调用 `/teleop/reset_backend`，并要求
      reset 后发布全新目标。该流程已由 GenericSystem 验证。
- [x] 核对 Host 实际路径：主循环 200 Hz、常用目标速度 0.6 rad/s、应用/硬件配置速度
      上限 1.0 rad/s、加速度上限 2.0 rad/s²，并以 `Joint_Pos_Vel(..., iswait=False)`
      持续发送最新目标。
- [x] 新增连续 backend：视觉目标 latest-only，50 Hz IK/关节运动生成，200 Hz
      `ros2_control` 非阻塞写入；目标或反馈过期立即保持。纯 GenericSystem 整链已通过。
- [ ] 运行一次 `robot_yz_fast_response` 作为最终连续真机验收并保存完整 bag；不再安排
      多档速度或逐轴速度搜索。

验收标准：连续摆动时运动平滑、无停顿式追赶；停止人体动作后机械臂在规定时间内
稳定，不继续执行过期目标。

## 6. P1：重新设计夹爪手势

- [x] 在第一版位置/姿态控制稳定前，保持 `publish_gripper_commands=false`。
- [ ] 收集本人实际环境下的 `open/closed/neutral` 数据，不只看模型自带类别分数。
- [x] 控制语义改为归一化捏合比例 + neutral 双阈值 + 0.6 秒保持 + 转换冷却；通用分类
      手势只作诊断，不直接驱动夹爪。
- [x] 使用归一化 thumb-index/palm-width 比例和双阈值 neutral 区；neutral 不改变夹爪。
- [x] 增加可配置稳定时间与转换冷却时间；默认稳定时间已提高到 0.6 秒，最终阈值仍由
      本人相机数据决定。
- [ ] dry-run 中连续测试 5 分钟无误触后，才允许单独开启夹爪真机开关。

验收标准：自然摆臂和打字不会触发夹爪；10 次刻意开/合手势均正确识别且每次只发
一条状态变化命令。

## 7. P1：完善启动、停机和操作界面

- [x] `preflight_panthera.sh` 自动检查摄像头、`caf1:ffff`、七个 `/dev/ttyACM*`、包/插件、
      急停确认和旧进程占用。
- [x] 对 `usbipd` 的 `Shared` 但未 `Attached` 状态给出可复制的 `usbipd attach` 恢复命令。
- [x] 将状态机、失跟/锁定、命令在途、限幅和姿态保持状态显示在预览 HUD。
- [x] 正常 Esc：禁用视觉 → 停止 pipeline → 返回 park → 关闭驱动（阻塞在途命令仍由官方驱动决定）。
- [ ] 正常 park 返回成功后继续读取关节反馈并做容差判定；当前一次实测第三关节残差约
      0.035 rad，不能只把 service 的 `success=True` 当作“精确全零”。
- [x] 异常中断停止新目标并关闭驱动；提供独立、二次确认的 `park_panthera.sh`。
- [x] 公共进程组清理函数已有 shell 集成测试，重复清理/二次信号路径幂等。

验收标准：任意阶段退出后没有残留 ROS 节点或串口占用；正常退出必定收到 park 的
成功响应，异常退出能明确告诉操作者当前位置未停放。

## 8. 推荐实施顺序

```text
诊断与 rosbag
  → Y/Z profile 与逐轴迟滞
  → 单目 X 的 2.5D 特征与 clutch
  → 单轴姿态
  → 平滑目标生成器
  → 明确的夹爪手势状态机
  → 完整交互与退出集成测试
```

不要同时提高位置增益、开放深度、开放旋转和开启夹爪。每次只引入一个新自由度，
先 dry-run 量化，再做小范围真机验收，这样 Review 时能明确知道每一层代码改变了
什么行为。

## 9. P2：单 Panthera 圆柱入槽 VLA

2026-09-13 用户明确了当前任务：**一台 Panthera 机械臂使用一个夹爪，将一个圆柱插入
凹槽**。不是双机械臂任务。当前仿真和后续真机的外部 observation/action 契约均为 7 维：
`joint1..joint6` 加一个归一化夹爪量。早期双 Piper、双 Panthera、双夹持和 14 维数据只
保留为历史探索证据，不能用于当前训练、评测或真机部署。详细阶段报告见
[单 Panthera 圆柱入槽 Lab 报告](08_lab_experiment_progress_2026-09-13.md)；旧路线文档已标记
为[历史双臂探索](07_dual_arm_vla_sim_to_real.md)。

### 9.1 当前单臂路线

- [x] 将 `place_cylinder_in_groove` 改为单台 Panthera、单个中央抓取点和普通圆柱；移除
      双抓取套环与双臂协同语义。
- [x] 把任务 schema 升级为 v3；采集文件对外保存 7 维实际状态和 7 维绝对动作。RoboTwin
      内部为兼容其 legacy left/right handle 会把同一条 7 维命令复制给同一个 articulation，
      但这不会形成第二台机器人，也不会改变对外契约。
- [x] 单 seed 程序化规划与真实 `gen_sparse_reward_data` 回放均通过；回放禁止创建 D6
      attach-on-grasp。最终 X/横向/高度误差分别约 0.036/0.504/0.091 mm，圆柱轴误差
      0.190°，夹爪打开，最大接触点 8。
- [x] 单集连续数据已保存为非空 HDF5（约 21 MB）和 MP4（约 395 KB），Lab 退出后无
      RoboTwin/Ray/GPU 残留进程。
- [x] 正式随机化配置的 4-seed 规划/物理预检通过 4/4。首轮 2/4 失败被定位为圆柱随机
      初始位置与凹槽近侧导轨过近；将初始 Y 基准从 `-0.06 m` 调整到 `-0.02 m` 后，
      同一四个 seed 全部通过。失败运行和阶段图保留在 Lab，没有进入 128 集采集。
- [x] 以新的单臂配置采集有界随机化的正式训练/验证数据；未复用旧 128 集双臂 SFT v1。
      v3 的 128 集已完成采集、合并、视频/schema 门禁和 episode 0/112 无附着回放。
- [x] 将 RLDS/OpenVLA adapter 从固定 14 维改为单臂 7 维，使用 TFDS 2.0.0、独立
      `panthera_single_cylinder` 统计和 `unnorm_key`。两集提前 smoke 已通过 OpenVLA
      真实管线，输出 `5x7` action chunk、`1x7` proprio 和 `224x224` RGB。
- [x] 建立 schema v4 竖直插入任务：竖直黄色圆柱、黄色浅槽、单 Panthera 顶部抓取。
      固定 seed 0–3 与随机化 seed 0/10000/20000/30000 的物理 oracle 均通过 4/4，不使用
      D6 attach-on-grasp。
- [x] 完成 phone-SRT 视觉构图门禁。v1 的 55°/中心机械臂相机被抽帧审计否决；其 128 集
      数据已可恢复地归档。v2 的任务区 `x=-0.30/-0.20 m` 候选分别因初始抓取 IK 和槽边
      抬升工作空间失败而弃用。v3 使用 75° FOV、中心 4:3 裁剪、`320x240` 输入，机器人
      基座 Y 为 `-0.35 m`，任务区和相机中心 X 为 `-0.25 m`；固定与随机化 oracle 各通过
      4/4，开场、搬运和插入抽帧也通过“机器人在左、任务在中”的构图审计。
- [x] 完成对齐版 128 集数据。v3 四分片采集、合并、schema/随机化/视频门禁，以及
      episode 0/112 无附着物理回放均已通过。
- [x] 生成对齐版 TFDS 3.0.0、独立归一化统计并完成 OpenVLA 最小优化器 smoke；拆分为
      112 个训练 episode 和 16 个验证 episode，动作块为 `5x7`。
- [ ] 执行正式 SFT，并在 16 个固定未训练 seed 上闭环评测；达到 75% 成功率且视频、报告、
      清理均通过后，才进入真机 shadow mode。5000 步 SFT 已完成；首次 0/16 被定位为
      RLinf 评测遗漏独立 L1 action head，修正后训练/未见 seed 双轨仍为 0/2。逐块遥测显示
      动作执行误差小，但策略首步将状态快进约 84 个专家帧，并在空抓后收敛到固定姿态。
      20000 步续训已完成，最佳/最终验证 L1 为 `0.04076/0.04802`，但新双轨仍为 0/2：
      机械臂只在复位附近微动，圆柱不动，训练 seed 最终停在专家索引 42。将初始夹爪
      预置为 0.9 的独立双轨也是 0/2，排除了单纯 reset 值不匹配。审计发现索引 42 的
      未来 5 帧仍全是静止目标，而未来 25 帧已经包含手臂离开动作，因此当前训练契约改为
      `25x7`（0.5 秒），保留 50 Hz 执行频率。RLDS `(25,7)` 门禁、一步优化器 smoke 和
      GPU1–3 上的 10000 步训练均已完成，最佳/最终验证 L1 为 `0.03362/0.03456`。25 帧
      全执行双轨仍为 0/2，但训练 seed 已不再卡在专家索引 42，而是完成接触、夹持和搬运后
      因圆柱渐进倾斜/滑移及释放对准失败。执行视野 `10/15/20` 和 `18/19/21/22` 的有界
      扫描均为 0/1，`20/25` 最接近成功，因此停止盲扫。相对下插辅助也失败。分阶段技能
      经过稳定窗口、固定工具偏置、槽外定姿和释放后退夹后，定向回归通过 4/4；技能只读取
      标定槽位、夹爪开度和末端反馈，不以仿真圆柱真值控制动作。接管条件改为可靠抓取代理
      成立且末端达到标定插入高度，避免 VLA 长程搬运累计倾斜；路径采用 30 mm 短抬升、
      槽外横移、50 mm 净空抬升、定姿、对齐、插入、100 个物理步稳定、松爪和退夹。
      10k 模型的 16-seed 开发集只有 11/16 形成可靠抓取，技能再稳定也无法达到 75%。从
      10k 以原 `5e-4` 续训到 30k 后发生策略坍缩，dev16 仅 1/16，已否决；10k 稳定路径
      回归仍为 4/4，证明退化来自模型而非执行层。随后从 10k 以 `5e-5` 续训到 15k；纯
      VLA 为 4/16，延迟到策略请求松爪后再接管为 11/16。7 mm 固定偏置回归为 1/4，
      0.82 分段松爪和高 30 mm 浅插入均为 2/4，均未增加成功 seed，现已恢复 4.5 mm 深插入
      基线并停止扫描末端几何常数。随后在 GPU1–3 从稳定 15k 模型以 `1e-5` 无损续训到
      20k；验证 Next-Actions L1 从 `0.03578` 降到 `0.02537`，但 dev16 只有 2/16，成功
      seed 为 200004、200009，且仅 4/16 触发末端接管。离线 loss 改善没有转化为闭环能力。
      门禁已阻止 final seed 300001–300016 运行，所有进程/GPU 已清理。2026-09-15 按用户
      要求在本组归档后暂停，等待人工改良；不得自动启动新的训练、调参或评测。
- [x] 生成第一版随机圆柱入槽 v2 pilot，不再复用固定物体布局。64 条成功轨迹中圆柱直立/
      平躺各 32 条，平躺方向 8 个区间各 4 条；圆柱和凹槽在任务实测有效半径 `0.46 m`
      的 75% 范围内独立采样。成功路径不使用 D6 attach、物体瞬移或墙钟 `sleep`，并通过
      单臂、7 维、有限值、唯一 seed 和 HDF5 帧数检查。圆柱位置网格覆盖 33/36，凹槽只
      覆盖 18/36；搜索 64 条成功轨迹期间另有 64 次失败，说明当前仍是经过成功筛选的
      pilot，不是最终训练分布。Lab 已生成覆盖全部平躺角度的 616.834 秒、2 倍速人工审阅
      视频；采集、Ray、编码进程和 GPU 1–3 已清理。详见
      [v2 数据集 Pilot 报告](11_panthera_v2_dataset_pilot_2026-09-16.md)。
- [x] 因第一版 `90°..175°` 采样使物体始终位于机械臂单侧而否决该数据。重做为
      `5°..175°` 左右对称前半圆，并把相机中心移到 `x=0`；平躺转正、重抓和抬升工位也
      按物体所在侧镜像。新 64 条成功轨迹中圆柱左/右为 32/32，凹槽左/右为 31/33，二者
      空间网格覆盖均为 33/36；姿态仍为直立/平躺各 32，8 个平躺角度各 4 条。修正后
      64/86 个候选成功。权威 2 倍速视频为 635.434 秒，抽样左/右各 8 条并覆盖全部角度。
      人工审阅发现抓取后搬运周期性顿挫，已否决进入训练。详见
      [v2 左右对称 Pilot 报告](12_panthera_v2_symmetric_dataset_pilot_2026-09-16.md)。
- [x] 首次尝试用拼接关节路径和单次 TOPP 消除分段停车，生成 64 条
      `v2_symmetric_smooth_pilot`。该版自动门禁只检查规划速度；随后 HDF5 实际 `qpos` 和
      人工视频仍观察到内部停顿，且平躺圆柱包含落桌后二次抓取，故已否决。详见
      [v2 连续轨迹 Pilot 报告](13_panthera_v2_smooth_dataset_pilot_2026-09-16.md)。
- [x] 完成 schema 9 单次抓取直接释放原型。完整关节路径先做四轮 Chaikin 局部凸平滑、
      逐点碰撞检查，再统一 TOPP 和一次执行。平躺圆柱沿轴向端部偏置 `44 mm` 抓取，空中
      转正并直接搬到槽口；释放时圆柱底面低于槽口顶面 `14 mm`，相对共同最低可用值
      `15 mm` 保留 `1 mm` 容错。抓取前预检两端与两种转正径向的完整路线，不再二次抓取。
      最终录制 smoke 为 2/2，实际 `qpos` 的内部近零速比例、最长停顿及 ≥80 ms 停顿均为
      0。18 组广域回归严格稳定门槛为 16/18；另两组已满足插入几何，但存在 PhysX 槽内
      接触抖动。详见
      [v2 单次抓取、连续搬运与直接释放修复](14_panthera_v2_single_grasp_direct_release_2026-09-16.md)。
- [x] 完成 schema 10 正式 single-grasp 数据集。三次样条几何路径配合全局弧长五次时间律
      消除了 schema 9 在高曲率路点的实际减速停车；正式集为 128 条（直立/平躺各 64，
      8 个平躺角度区间各 8），由 169 个候选产生，41 个失败候选未进入数据。HDF5 实际
      `arm_qpos` 在几何进度 5%–95% 内的近零速比例、最长近零速段和 ≥80 ms 停顿均为 0。
      圆柱左/右为 62/66，凹槽左/右为 63/65，两者空间网格覆盖均为 36/36。GPU1–3 上
      9 个采集器并行完成；10 分钟平衡分层视频、机器可读摘要和 `dataset.ok` 已保存。
      尚未转换 RLDS 或启动训练，必须先由人审阅视频。详见
      [v2 schema 10 正式数据集](15_panthera_v2_formal_dataset_2026-09-16.md)。
- [x] 实现有界异步整集编码/落盘任务池：按待处理 episode 数和 cache 字节双重限流，独立
      进程完成 JPEG/HDF5/MP4，partial 文件通过帧数检查后原子提交，异常回传且退出前
      drain。成功与畸形 cache 故障注入均已通过；帧序和时间对齐不变。
- [x] 生成 1280 条扩充数据集并停在人工验收门前：直立/平躺各 640，8 个平躺角度区间各
      80；160 个 8-episode 可恢复分片在 GPU0–3 上完成，1280/1280 原子提交，圆柱左右
      619/661、凹槽左右 636/644，空间覆盖 36/36，机位互异 1280。首次合并审计因 3 条
      轨迹的重定时峰值加速度越过 `2.002 rad/s²` 门禁而中止，定点重采并原位替换该 3 条
      后于 2026-09-17 04:48 通过全量审计，最大峰值加速度 2.001881，真实 `qpos` 近零速
      比例与 ≥80 ms 停顿均为 0；`automated-audit.ok`、`awaiting-human-review.txt` 和
      603.767 s 审阅视频已生成。当前等待人工审阅，未转换 RLDS、未训练。
      详见 [扩充数据集与异步落盘流水线](17_panthera_v2_expanded_dataset_pipeline_2026-09-17.md)。
- [ ] 修正重定时收敛容差的量纲不一致：规划器用 scale 空间 `scale <= 1.001`（加速度上限
      2.004002），审计门禁用值空间 2.002，落在两者之间的轨迹会被接受又被否决。改为值
      空间判据后再进行下一次大规模生成，避免重复定点重采。固定机位 1280 如预期再次
      触发（shard 101、115 定点重采），确认该缺陷会在每一次大规模生成上复发。
- [x] 生成固定机位 1280 扩充集（`..._sft_v2_fixedcam`）：2026-09-17 19:57 完成，机器审计
      通过，`unique_camera_positions = 1`（方位 90°、离桌 0.60 m、俯视 39.6°），圆柱与槽
      空间覆盖各 36/36，1694 个候选产出 1280 条。用户已人工审阅 10 分钟视频。
- [x] 补齐合并审计缺口：原判据的速度/加速度上界只加在重定时**规划**轨迹上，实际 `qpos`
      只查停顿，测不到速度跳变（schema 9 当初即由此漏过）。新增实际 `qpos` 逐关节速度
      跳变判据，以 `a_limit * dt` 为预算、只在规则间隔样本对上评估，门禁比 `1.5`。
      固定机位全量 1280 最差 1.0673、随机机位 1.0394、已否决的 schema 9 为 17.4060。
      已并入 `run_lab_robotwin_panthera_v2_pilot.sh` 合并审计，并提供
      `packages/panthera_sim/diagnostics/audit_actual_qpos_continuity.py` 给已生成数据集补测。
- [x] 数据集取舍（2026-09-17）：弃用 v3 `panthera_phone_vertical_sft_v1`，正式训练只用
      固定机位 1280。核对结论：物体生成区域缺陷已由极坐标环形采样修复（0.22–0.345 m、
      前半圆、36 格按面积均匀）；速度连续性问题属于 schema 9 / symmetric-smooth 两个
      pilot 而非 v3，但 v3 依据物体区域与更早的相机构图否决仍应弃用。弃用后
      `panthera_phone_vertical_cylinder` 的 seed 列表、评测配置与末端技能标定常数一并
      转为历史；新训练不得从 15k/20k checkpoint 续训（相机几何与任务分布均已改变）。
      128 条 schema 10 线同时退役，其 4.0.0 产物与 `rlds.ok` 冻结，相关入口保留 4.0.0
      并标注不要再运行。固定机位 1280 的人工批准已于 2026-09-17T22:09:25+08:00 记录，
      写入前重新核验了审阅视频 SHA-256。
      详见 [扩充数据集与异步落盘流水线](17_panthera_v2_expanded_dataset_pipeline_2026-09-17.md)
      第 9、10 节。
- [x] 固定机位 1280 的后续阶段已全部完成：媒体审计（2026-09-17 22:48）→ TFDS/RLDS
      `4.1.0`（23:03，1280 集、52 GB、576 分片）→ 一步优化器 smoke → round 0 SFT。
      停止判据改为墙钟预算，不使用已被证伪的验证集 L1 早停，理由见
      [OpenVLA 闭环失败分析](10_openvla_closed_loop_failure_analysis.md) 第 9 节。
- [x] round 0 SFT 与闭环评测（2026-09-18）：3 小时墙钟预算、4 卡、每卡 batch 6
      （有效 batch 24），9379 步、22.5 万样本、20.8 样本/秒，最佳 step 9000、验证 L1
      `0.018131`。超时收尾属预期，包装器合法未写 `train.ok`，改记
      `train-timebudget.ok` 并带外补做 LoRA 合并（见 `manual-merge.json`）。
      18 seed 纯 VLA 闭环评测为 **0/18**，18 条全部跑满 1600 步、reward 恒为 0。
      抽帧判读 4 条轨迹：**策略从未出现"接近—下降—闭合夹爪"的序列**，机械臂或在物体
      附近小幅移动、或直接退开，两个黄色物体多数情况下全程未被触碰。
- [x] 由 round 0 得出的关键量化结论：训练信号的有效单位不是帧数而是**每个不同场景分到
      的样本数**。50 Hz 采样使相邻帧高度冗余（相邻动作范数中位数 `0.0031 rad`），119 万
      帧的信息量接近 1280 个场景。对照：历史上闭环可靠抓取 11/16 的模型为 536 样本/场景，
      round 0 只有 195 样本/场景——总样本多 3.7 倍，单场景曝光却只有 36%。数据集扩大 10 倍
      而算力不变，等于把单场景曝光稀释 10 倍。
- [ ] 子集对照实验（2026-09-18 04:26 起，3 小时）：从固定机位 1280 均衡抽取 128 集
      （直立/平躺各 64、8 个角度区间各 8、左右各 64、36 个空间格全覆盖，符号链接实现，
      入口 `packages/panthera_vla/build_episode_subset.py`），训练集 112 集，
      预计约 2006 样本/场景，是历史可靠抓取模型的 3.7 倍。该实验用于单独检验"曝光不足"
      是否为瓶颈：若仍完全不抓取，则可排除训练量解释，应转查动作表示、观测通路或数据本身。
- [ ] DAgger 方案暂缓，前置条件未满足。原设想是定期用策略 rollout 的失败状态调用 oracle
      生成恢复轨迹并回灌训练集；oracle 确为状态条件（`_execute_single_grasp_route` 现读
      当前 `cylinder`/`ee` 位姿重规划），机制可行。但 round 0 的失败类型是"从未抓取"而非
      "抓取后偏离"，而 oracle 的恢复入口要求圆柱仍在夹爪内
      （`0.07 <= offset_norm <= 0.16`），对空爪状态只能从头重抓——产出与原始干净演示同分布，
      没有纠错信号。**DAgger 需等到策略至少能稳定抓取之后再启动。**
- [ ] 基座模型 `config.json` 的并发隐患：训练入口每次启动就地修改
      `models/openvla-oft-place-empty-cup/config.json`（已累积 15 份 `.back.<时间戳>`）。
      串行运行无碍，但并行训练会互相覆盖且可能静默加载错误配置。引入并行训练前必须改为
      每次运行使用独立副本或改由参数传递。详见
      [扩充数据集与异步落盘流水线](17_panthera_v2_expanded_dataset_pipeline_2026-09-17.md) 第 5.2 节。
- [x] 修复 `--max_steps` 被早停补丁静默失效：OpenVLA 补丁恢复硬退出分支并把
      `max_steps` 列为合法 `stop_reason`；Gate 2 默认上限 40000 step，墙钟只作挂死保护。
- [ ] 离线指标与闭环能力的关系（2026-09-17 实测）：对 25x7 的 15k 与 20k checkpoint 在
      同一 16 条验证集上做复制-proprio 基线对比，20k 在**全部**离线指标上更好
      （skill 0.554→0.729、位移幅度比 1.131→1.036、方向余弦 0.381→0.652），而其
      release-gate dev16 为 2/16 对 15k 的 11/16（Fisher 精确检验 p = 0.0032）。结论是
      离线指标并非奖励了"照抄 proprio"的捷径，而是**在专家状态分布上度量**，无法预测
      闭环；因此不得以验证集 loss 作为选模或停止判据。另注：两个模型在 chunk 第 1 帧上
      skill 均为负（−1.82 / −0.54），即执行前缀含噪声，与既往"执行视野越长越好"的扫描
      结果一致。审计脚本为 `packages/panthera_vla/audit_copycat_baseline.py`，产物在
      Lab `reports/panthera-phone-vertical-sft-v3/copycat-audit/`。
- [x] 优化 RLDS 转换并加入并行开关：逐帧 JPEG 解码 + TFDS 重新编码被确认为纯浪费，
      改为透传原始 JPEG 字节后单帧成本 0.770 → 0.043 ms，整体 0.65 → 3.2 条/秒（1280
      条约 33 分钟 → 约 7 分钟）；并行路径已实现并验证与串行逐条一致，但实测各档并行
      比串行慢 20–40%（瓶颈是 TFDS 单进程写入），因此默认串行，`PANTHERA_RLDS_WORKERS`
      可强制档位或设为 `auto` 按 CPU/内存推算。
- [x] 历史上实现过验证集 L1 早停，但该方案已被后续实验证伪并废止：专家状态分布上的
      L1 不能预测闭环成功率。正式训练恢复固定计算预算和周期 checkpoint；Gate 2 由独立
      GPU 做同场景闭环 rollout，只有实际成功才提前停止。实现与证据见
      [单轨迹过拟合门禁收口](22_single_trajectory_overfit_gate_hardening_2026-09-20.md)。
- [x] Gate 2 编排缺陷已修复：显式使用 25×7 契约，checkpoint 记录单 episode subset 与
      父数据 lineage，训练/闭环 GPU 分离，完整快照防撕裂，唯一运行目录默认保留产物。
- [x] Gate 2 的 Lab 运行环境完成收口：固定上游重新装配、OpenVLA 25×7 契约、
      curobo v0.7.8、PyTorch CUDA 13 四卡可见性和 Lab 侧 187 项测试全部通过；未启动训练。
- [ ] 在上述干净 runtime 中重新运行 Gate 2。当前尚无新的闭环成功结果；通过前不得恢复
      1280 条正式训练或真机实验。2026-09-20 已先启动 episode 2 的 24 小时纯过拟合训练：
      GPU1–3 三卡 DDP、每卡 batch 6、每 5000 step 保留独立 checkpoint，W&B 在线同步。
      GPU0 当时被 `mole` 的任务占用，所以本轮不在训练期间争抢 GPU 做 rollout；训练完成后
      必须逐 checkpoint 做同场景闭环验证，不能把训练运行本身记为 Gate 2 通过。运行目录为
      `/data/lyy/panthera-vla/ci/overfit-ep2-24h/overfit-ep2-24h-b6-20260920T140123Z/`。
- [ ] 编号 checkpoint 的 GPU0 积压评测已切换为
      `panthera-overfit-numbered-paired-eval-v2-20260921.service` 无人值守执行；旧的单版本
      `panthera-overfit-numbered-eval-v3-20260921.service` 已停止。24 小时训练关闭了
      validation，因此没有可追溯的 W&B 验证 loss；不得用训练 loss 冒充。评测会在同一
      episode 2 的全部记录帧上补算 bounds-q99 归一化 25×7 离线验证 L1，并把 checkpoint
      step 与该值同时写入视频。从 step 30000 起，每个 checkpoint 固定生成 `h20` 原始执行
      和 `h1 + coefficient=0` 的逐步查询、历史覆盖预测等权平均两个版本，并拼成同一画幅。
      此值仅用于 checkpoint 横向比较；最终验收
      仍由推理一致性通过后的同场景闭环结果决定。step 5000 已完成：1037 个唯一 50 Hz
      帧的归一化 L1 为 `0.04261852`，闭环 0/1；渲染一致性通过，但模型/管线中位误差
      12.20/10.75 mrad 未过 10 mrad 门槛，故 `gate_success=false`。单视频和第一版总拼接
      视频均已生成并目视确认标注可见，队列已自动进入 step 10000。
- [x] step 10000 双执行模式诊断：`h20` 共查询 53 次，策略调用中位 `128.77 ms/query`，
      按每块实际执行步数摊销为 `6.45 ms/action`，即 50 Hz 预算的 `0.32×`；逐控制步重新
      生成 25 帧并对覆盖当前步的历史预测等权平均时，共查询 1042 次，中位
      `128.59 ms/action`、P95 `130.73 ms/action`，为 20 ms 预算的 `6.43×/6.54×`，全部
      1042 次查询超期。该计时从已解码图像和机器人状态在内存中可用时开始，包含预处理、
      H2D、模型推理和动作 D2H，排除仿真渲染、物理、视频编码；也排除真机相机采集/传输/
      解码、ROS 和控制器，因此是现实工作台策略延迟的偏乐观下界。两版均未成功：`h20`
      到达插入阶段，等权滑窗版只到槽附近。并排视频和 JSON 摘要位于运行目录下
      `paired-preview-step-10000/`。
- [ ] step 10000 不松爪诊断：已重跑并记录 53 个完整 25×7 预测块。策略在 row 140 首次
      闭合后，后续所有块中夹爪 action 均未再次达到 0.5；后五个未执行 offset 也没有松开
      action，故已排除 20/25 执行视野截断。专家在 row 801 产生开爪转换，而模型到达插入
      阶段后仍持续输出约 0.3 的闭合值。下一步应定位模型为何没有学会/识别释放阶段，而不
      应先改执行器阈值或强制开爪。
- [ ] step 45000 的逐时间步离线 L1 排除了“模型根本没学会松爪”。在记录的专家帧上，
      专家夹爪于 row 801 越过 0.5，模型于 row 803 越过，只晚 2 个 50 Hz 帧。释放窗口
      row 777–825 的 25-step 夹爪 L1 均值为 `0.03201`，是全局 `0.01955` 的 `1.64×`；
      当前步夹爪 L1 为 `0.03398`，是全局 `0.01632` 的 `2.08×`。但 25×7 总 L1 在该窗口
      反而低于全局均值，因此只看聚合 loss 会掩盖释放困难。自主闭环中，同一 checkpoint
      的 h20 在夹爪仍张开时剧烈碰撞圆柱，根本没有形成稳定抓取：接触只发生在 row 79–113，
      此时夹爪命令仍约 `0.89–0.91`；圆柱到 row 111/120 已分别偏移 `3.68/10.02 cm`，而
      夹爪到 row 140 才首次低于 0.5。row 160 的重新张开是碰撞失效后的后续动作，不能当成
      掉落原因。h1 等权滑窗则真正夹住圆柱并到达插入阶段，却始终不松爪；后者
      row 800 后的完整原始预测夹爪范围仅 `0.301–0.310`。训练/评测相机位置已经通过
      `3.36e-8 m` 一致性检查，因此不是相机错位；但单帧 RGB+proprio 的阶段可观测性很弱：
      专家 row 800→801 的整幅图平均变化仅 `0.20/255`，状态差仅 `4.65e-5`。当前主因应按
      闭环协变量偏移与静止/释放阶段混叠处理；更合适的相机角度、腕部/侧视相机、时间历史
      或显式接触/稳定阶段状态可作为改善观测的候选，而不能把移动相机当作已证实修复。
      曲线、CSV 和摘要在 `artifacts/step-45000-temporal-loss/`，Lab 原件在该运行目录的
      `paired-preview-step-45000/standard-h20/`。
- [x] step 65000 的滑动平均版经用户视频审阅，已经能稳定、准确地把圆柱送到槽点正上方，
      但仍不松爪。现有 trace 的离线重算确认 row 750--1041 的 newest 原始夹爪预测只有
      `0.305--0.312`，ensemble 为 `0.303--0.308`，全轨迹没有任何 raw-open 被平均压掉的
      时间步，因此问题不在夹爪维 ensemble。rollout 最终圆柱高度与 schema 10 直接释放位
      只差约 `0.8 mm`，也不是尚未下降到释放位。专家 source row 812→813 的标签跨过开爪
      阈值时，RGB 平均变化仅 `0.200/255`、7 维状态 L2 仅 `4.29e-5`；模型在专家帧上能
      触发开爪，在自主闭环轻微偏移下却停在闭合固定点。结论是单帧阶段混叠/闭环分布偏移，
      不是相机位姿错配、执行视野截断或滑动平均。证据见
      [Gate 2 收口文档](22_single_trajectory_overfit_gate_hardening_2026-09-20.md) 第 8 节。
- [ ] 用最小、同口径实验验证释放阶段的可观测性修复：优先比较短时间历史或通用阶段转移
      输出；不得按槽口几何直接强制开爪，也不得把这种规则接管计作 VLA Gate 2 成功。
- [x] 修复连续动作头绕过 DDP 包装器造成的多卡梯度分叉，并用三卡启动前审计验证 L1 与
      diffusion 动作头跨卡梯度、参数和共同输入输出差异均为零。修复后的 28 维单轨迹训练
      保存 17 个模型存档；最终 34 次模式评测出现 4 次准时成功，来自 30000、45000 和
      75000 步存档，其中 75000 步在 h20 与逐步等权两种模式下都成功。80000 步离线 L1
      更低但未持续松爪，85000 步退化到抬起阶段，因此继续以闭环执行选模，不用离线 L1
      或最后一个存档自动选模。完整证据、方法和限制见
      [单轨迹过拟合实验阶段报告](panthera_single_trajectory_overfit_stage_report_zh.html)。
- [x] 用户已人工审阅并批准 schema 10 的 10 分钟平衡分层视频；批准记录连同视频 SHA-256
      保存在正式数据集状态目录。
- [ ] schema 10 无人值守训练流水线正在运行：媒体审计和 RLDS 使用有界 CPU 并行；正式
      训练为 GPU1–3 上一次连续 10,000 步，不在中途调参；训练完成后用 GPU1–3 并行跑
      18 条开发轨迹，达到 75% 才解锁另一组 18 条 final。普通用户 `@reboot` 已安装，阶段
      完成标记允许重启恢复，质量门终态不会反复重跑。详见
      [无人值守训练流水线](16_panthera_v2_unattended_training_pipeline_2026-09-16.md)。

### 9.2 历史双臂探索（仅作证据，不是当前 TODO）

- [x] 建立 WSL SRT listener、最新帧和有限循环缓存；合成流验证通过。
- [x] 接收 HEVC 1920×1080@30 fps 真实实验台画面，方向和主工作区构图已正确；仍需固定
      相机、标定广角内参/外参和中央 ROI，并检查圆柱、凹槽与双夹爪是否同时可见。
- [x] Windows 只允许 `192.168.31.0/24` 访问 UDP 9000，并双向转发到 WSL listener。
- [x] Lab 已重启，内核模块/NVML/驱动统一为 595.91.07；四张 48 GB RTX 4090 正常，
      `/data` 可用 4.6 TiB。后续工程、文档、资产、模型和数据全部放在 `/data/lyy/`。
- [x] 审计 `/mnt/hulab/pro6000/data`：优先保留 RLinf、RoboTwin 和 NAS 独有的
      `place_empty_cup` walkthrough；ConCon 仅作双臂 RLinf/物理 guard/验收设计参考；
      MRI 项目 `fast-zeroshot`、`traj-correction` 与语音项目 `Lessons-in-Cast` 不迁移。
      旧 `.venv`、rootless CUDA 工具链、缓存和可重新 clone 的第三方仓库均不整包复制。
- [x] 在 `/data/lyy/panthera-vla/` 以普通用户完成无 Docker 的 Miniforge/Python 3.11、
      RLinf、RoboTwin、官方 assets 和四卡环境验收；PyTorch 2.11.0+cu130 可见 4 张
      48 GB RTX 4090，SAPIEN、MPLib、PyTorch3D 与 cuRobo v0.7.8 的 V1 API 均可导入。
      CUDA 13、cuRobo API 与 wheel 元数据补丁已纳入 `overlays/rlinf/patches/`，不依赖一次性
      手工修改。
- [x] 第一版明确为两臂在两个抓取套环上共同夹持同一圆柱；一臂持物/一臂稳定工装保留
      为后续可替换方案，不构成硬约束。
- [x] 盘点官方 Panthera ROS 2 提交 `b08633d6...` 的带夹爪 URDF、mesh、SRDF、六轴
      限位和夹爪 mimic。RoboTwin 侧明确使用
      `[left joint1..joint6, left gripper, right joint1..joint6, right gripper]` 14 维顺序；
      这只是本项目契约，不等同于 Piper/ALOHA 的归一化。
- [x] 复现 RLinf + RoboTwin 官方 `place_empty_cup` 评测：固定模型提交 `04150e05...`，
      使用双 Piper 仿真 embodiment 与 OpenVLA-OFT 的 ALOHA 14 维契约，在 4 卡、4 环境、
      200 步 smoke 上得到 `success_once=0.75`（3/4）；四条视频齐全、日志无致命异常，
      Ray 和 GPU worker 均已清理。
- [x] 加入并验收 Panthera 双臂 MPLib embodiment 与动作/状态映射：10 个 link、六个旋转
      关节、两个直线夹爪关节均由 SAPIEN 正确加载，官方限位一致，双夹爪对称开合目标
      为 `+0.04/-0.04 m`；左右臂当前位姿和六个 `±1 cm` 方向 IK 全部成功，规划轨迹为
      87–117 点且没有自碰撞/环境碰撞。仿真 homestate 使用官方 SRDF `pose1`
      `[0, 1.6, 1.6, 0, 0, 0]`，不冒充真机 Host 的 `position0`。
- [ ] 若后续数据生成或策略评测确实需要 CuRobo，再补 Panthera 碰撞球和 CuRobo 配置并
      与 MPLib 固定 case 对照；当前没有因 Piper 默认使用 CuRobo 就把它当硬约束。
- [x] 在 RoboTwin 双 Piper 中完成程序化圆柱入槽 scripted oracle：抓取、约束确认、
      抬升、协同搬运、下降、开夹爪、横向抽离、250 个仿真步沉降和回原位合并为单次
      事件驱动 case；4/4 固定种子通过，每条 13 个阶段，最终横向误差 ≤0.32 mm、X
      误差 ≤0.57 mm、轴误差 ≤0.62°，释放后线/角速度为 0，日志无致命异常且无残留
      GPU/Ray 进程。双 Piper 独立路径规划不提供共享刚体闭链，因此当前 oracle 在确认
      双夹爪接触后使用 attach-on-grasp D6 约束，释放前恢复物体碰撞。该结果只验收任务
      几何/流程/判据，当时尚未验收 Panthera embodiment、无辅助约束物理抓取或 VLA 策略。
- [x] 在双 Panthera embodiment 上运行同一程序化圆柱入槽 oracle：4/4 固定种子通过，
      每条 13 个事件阶段，最终最大 X 误差 0.707 mm、横向误差 0.296 mm、轴误差
      0.074°、高度误差约 4.50 mm，释放后线/角速度均为 0；四张阶段图非空、日志无
      致命异常且无残留 GPU/Ray 进程。该结果仍使用 attach-on-grasp，不代表真机闭链
      抓取或 VLA 已完成。
- [x] 用 RoboTwin 官方 `collect_data.py` 完成双 Panthera 上游格式示范 smoke：4 个
      episode 各 628 个 RGB/14 维动作样本和 628 帧视频，HDF5/视频均非空，有限值、动作
      形状、官方关节限位和进程清理通过。成功路径没有墙钟 `sleep`。该数据仍是相同固定
      几何，且上游 HDF5 无时间戳、`joint_action` 保存驱动目标而非实际 qpos、视频按
      30 FPS 编码，语言生成器对绝对 `save_path` 拼接错误；仅作为采集链路 smoke。
- [x] 补齐正式 episode 契约 smoke：4 个确定性随机化 seed 均通过，统一时钟重采后分别
      生成 701–710 帧；每帧分离 14 维 action target 与实际 articulation qpos/夹爪开度，
      并保存精确 simulation step/time。相邻样本最大间隔为 5 个 250 Hz 物理步，抓取沉降和释放后
      1 秒落槽过程也连续采集；每集生成 7 条语言指令，并记录源码提交、实际几何、随机化
      参数和 attach-on-grasp 标记。旧的 250 步采样空洞版本已改名归档，没有覆盖。
- [x] 统一 RoboTwin 数据采样时钟并接入 OpenVLA RLDS：控制、夹爪和沉降都按全局仿真
      step 的 5-step 网格采样；RLDS 丢弃阶段边界额外帧并按 step 去重，同时把拍照后样本
      对齐到下一网格绝对关节目标。TFDS 1.0.0 使用 train episode 0–2、val episode 3，
      OpenVLA 官方管线已读出 `224x224` 主相机、14 维实测 proprio 和 `25x14` 动作块，
      action/proprio 统计来自 Panthera 数据。RLinf 在线观测也已优先读取实测 qpos，旧任务
      缺少该字段时才回退到 drive target。
- [x] 建立 RLinf/RoboTwin 闭环评测配置并通过纯配置验收：16 个固定测试 seed 与训练集
      seed 分离，双 Panthera 使用 14 维绝对关节目标、25 步动作块和 800 步 episode 上限；
      四个并行环境运行四轮并逐轮推进 seed，`unnorm_key=panthera_cylinder`，不会误用
      ALOHA/Piper 统计。该项没有启动策略仿真，
      不能替代训练后闭环成功率评测。
- [ ] 在正式契约上扩大有界的物体、凹槽、相机和视觉域随机化并划分训练/固定测试集，
      建立 VLA 的 SFT/RL 基线和分阶段失败指标。4-episode contract smoke 不作为训练集；
      真机先执行 shadow mode。
      当前 128-episode SFT v1 采集已经在 Lab 普通用户环境运行；事件驱动续跑脚本会在
      数据集锁释放后先把 train/val 样本通过真实 `gen_sparse_reward_data` 路径回放，禁止
      创建 oracle D6 约束，再依次完成 RLDS 转换和 OpenVLA 最小优化器 smoke。任一阶段
      失败都会停止后续阶段并保留日志。全部通过前本项保持未完成。
      第二级事件驱动流水线也已挂起：只有上述 JSON 证据全部通过，才执行四卡 5000 步
      transfer-init SFT，并在 16 个固定测试 seed 上做闭环评测；成功率门槛为 75%。未达到
      门槛会保留模型、视频和指标并停止，不会写通过标记或自动接触真机。

VLA 只输出带时间戳和有效期的短时动作块或子目标。关节限位、速度/加速度限制、碰撞、
失联锁存和高频执行仍属于确定性的安全与控制层。

## 10. P0：π0.5 候选 baseline（2026-09-21）

- [x] 固定外部单相机、7 维绝对目标、25-step chunk，不要求三台物理相机；缺少的腕部
      相机槽使用黑图和 false mask。
- [x] LeRobot 转换复用 RLDS 的 50 Hz 网格、RGB 修复和 next-action 对齐，不另写一套
      标签时序。
- [x] 实现 OpenPI 7→32 padding 前的物理维 normalization、六关节 delta/absolute
      双向变换和 32→7 输出裁剪；夹爪始终保持绝对语义。
- [x] 将 OpenVLA-OFT 与 π0.5 接到相同 RoboTwin rollout、parity 门禁和结果比较器；比较器
      拒绝数据摘要、episode、预算、执行窗口或 temporal ensemble 不一致的输入。
- [ ] 固定 OpenPI 上游提交并在 `/data/lyy` 建立独立环境；不得改写上游 checkout。
- [ ] 转换正式 schema 10 数据，核对 episode/帧数、源摘要、图像颜色和随机 action。
- [ ] 完成 normalization stats、单 batch 前后向与 episode 2 小预算 overfit smoke。
- [ ] episode 2 的记录帧/渲染帧 parity 和闭环 Gate 2 通过后，才运行 1280 条数据的正式
      π0.5 训练，并与冻结的 OpenVLA-OFT baseline 做同口径比较。

详细合同、入口和命令见
[OpenPI π0.5 迁移与同口径基线](23_openpi_pi05_migration_baseline_2026-09-21.md)。

## 11. P0：28 维动力学 proprio 对照实验（2026-09-21）

- [x] 保持 25×7 action 不变，把候选 observation 扩为
      `qpos(7)+qvel(7)+qacc(7)+effort(7)`；没有加入稳定时长、任务阶段或槽口几何。
- [x] 为仿真在线状态、RLDS 5.0.0、OpenVLA proprio projector、rollout 和 parity 增加
      7/28 维显式合同检查，维数不一致时失败，不允许静默裁剪或填充。
- [x] 在 GPU0 完成 episode 2 派生数据、RLDS 验证和两步真实 optimizer smoke。1037 个
      50 Hz 样本的重放 qpos 中位/P99 误差为 `0.0002734/0.0072121 rad`，均通过门禁。
- [x] 将 28 维正式 24 小时过拟合挂到当前 7 维任务的同一把锁后。它只在前序写出
      `TRAINING_COMPLETE` 后自动占用 GPU1–3；每卡 batch 6，每 5000 step 保存 checkpoint，
      W&B online。2026-09-21 19:20 CST 已确认队列正在等待，当前训练仍正常运行。
- [ ] 28 维训练完成后按相同 scene/episode/相机/动作预算评测所有候选 checkpoint，同时
      生成 h20、h1 滑窗视频、释放段 trace 和全 checkpoint 拼接视频；只以闭环抓取、到位、
      开爪、落槽判断是否改善。2026-09-22 已启动独立后台队列：GPU0 立即消费已有编号
      checkpoint，GPU1–3 同时继续训练；队列持续轮询新增 checkpoint，直到训练写出
      `TRAINING_COMPLETE` 且积压清空后才生成总拼接视频并结束。

实现、数据语义、Lab 路径和验收规则见
[OpenVLA 28 维动力学 proprio 过拟合实验](24_openvla_dynamics_proprioception_overfit_2026-09-21.md)。

## 12. P0：OpenVLA-OFT 多卡 action head 同步（2026-09-23）

- [x] 在实际 OpenVLA-OFT `0.0.1` / PyTorch `2.11.0+cu130` 环境复现 DDP forward 绕过：
      三卡不同局部 batch 做一步 AdamW 后，L1 action head 参数最大差异为 `0.001000002`，
      同一输入的输出最大差异为 `0.171851695`；经过 DDP forward 的对照均为 `0`。
- [x] 给 L1 与 diffusion action head 增加标准 `forward()`，并把参数化训练调用从
      `action_head.module.predict_*` 改为 `action_head(...)`。无梯度 diffusion sampling
      保持原推理接口。
- [x] 新增三卡回归和一键入口。L1/diffusion 的故障负对照仍能观察到分叉，修复后的梯度、
      参数和共同输入输出跨卡差异均为 `0`；bootstrap 首次应用和幂等复跑均通过。
- [x] 把早停/硬预算/学习率与 DDP 修改合并到同一个 `finetune.py` 补丁，满足“一个上游文件
      只由一个 patch 所有”的装配契约；同时修复 constants 补丁不可逆和状态漂移。
- [x] 在独立 OpenVLA-OFT checkout、基于 `upstream/main@e4287e9` 创建
      `fix/ddp-action-head-forward-sync`，提交最小上游修复 `bf0b44c`；两 rank CPU/Gloo
      的 L1/diffusion 回归均通过。Panthera 两个仓库中误建的同名分支已删除。
- [x] 将 OpenVLA-OFT 修复分支推到 `5o1/openvla-oft`，并向上游创建
      [PR #162](https://github.com/moojink/openvla-oft/pull/162)；PR 不携带 Panthera 专用
      常量、早停、外部停止或技术报告。创建时 PR 为 open、非 draft，暂无自动检查。
- [x] 处理 PR review 提出的测试污染：提交 `2c767d9` 在 `finally` 中恢复合成
      `prismatic` 的 `sys.modules` 状态，并增加导入隔离回归；最终为 `3 passed, 2 subtests passed`。
- [ ] 将修复前 GPU1–3 训练的 7D/28D checkpoint 标记为诊断历史，不再作为正确多卡 SFT
      基线。它们的闭环视频保留，但不能据此确认或否定 proprio、时间历史或抖动方案。
- [ ] 从共同基础模型重新运行 episode 2 三卡成对训练；保持 25×7 action、28D proprio、
      每卡 batch 6、学习率 `5e-4` 和每 5000 步 checkpoint，再以相同闭环与视频口径比较
      step 15000/40000/65000。完成前不得宣称 DDP 修复已经消除抖动或松爪失败。
- [x] 2026-09-23 已启动修复后的 28D episode 2 重训：即时三卡 DDP preflight 为 `pass`，
      L1/diffusion 的梯度、参数和共同输入输出跨 rank 差异均为 `0`；用户级 systemd 单元为
      `panthera-overfit-ddpfix-20260922T173901Z.service`，运行目录为
      `/data/lyy/panthera-vla/ci/overfit-ep2-dynamics-24h/ddpfix-dynamics-28d-ep2-24h-20260922T173901Z/`，
      W&B run 为 `assanekowww/panthera-ci-overfit-dynamics-24h/1s35hyv6`。该勾选只表示任务
      已可靠启动，不表示训练或闭环验收已经完成。
- [x] 已同时启动 `panthera-overfit-ddpfix-eval-20260922T173901Z.service`。它在 GPU0 等待
      每个 5000-step checkpoint，生成 h1/h20 双模式视频和最终总拼接视频；等待阶段不占用
      GPU0，且不与 GPU1–3 的训练争卡。
- [x] 24 小时修复后训练已正常收口：墙钟预算结束、总退出码为 `0`，保留 step
      5000–85000 共 17 个完整 checkpoint。旧自动评测处理到 step 75000 后，在 step 80000
      尚未原子写完时误入视频拼接并退出；训练和 checkpoint 未受影响。
- [x] 修正闭环预算语义：1037 只作为专家时限门禁，rollout 继续到 2074；结果明确分为
      `on_time_success`、`delayed_success`、`failure`。同时让 checkpoint 队列在权重写完前
      等待，避免再次触发 step 80000 竞态。
- [ ] 已启动全部 17 个 checkpoint 的扩展预算 h20/h1 重评。外部 `patchconvmoe` 任务结束后，
      当前服务 `panthera-ddpfix-extended-eval-20260923T220819Z.service` 使用 GPU0–3 四卡并行，
      输出到运行目录下的 `extended-budget-eval-20260923T214752Z/`。完成后检查 34 个模式
      结果、17 个双模式视频、总拼接视频和 `EVALUATION_COMPLETE`。
- [x] 修复评测 GPU 编号契约：原脚本用 `nvidia-smi` ordinal 做空闲检查，却没有固定 CUDA
      枚举顺序，第一次 GPU2/3 启动实际撞到物理 GPU0/1 并 OOM。现在强制
      `CUDA_DEVICE_ORDER=PCI_BUS_ID`。随后又发现 rollout 子进程的 `--gpus 0` 会覆盖外层
      `CUDA_VISIBLE_DEVICES`，现改为传递 worker 的真实 ordinal；checkpoint 合并阶段也绑定
      同一张卡。四个活跃合并进程已分别验证为物理 GPU0、1、2，GPU3 则已进入 rollout；
      错误服务均在产出正式结果前停止。

技术细节、证据 JSON、复现命令和历史产物影响见
[OpenVLA-OFT 多卡 action head 同步修复报告](25_openvla_action_head_ddp_sync_fix_2026-09-23.md)。

- [x] 收口 WSL/Lab 仓库边界：技术报告和分析记录只保存在 WSL `docs/`；`gpu_node`
      保留实验环境、脚本、资产、运行配置及日志/视频/checkpoint。同步入口不再传输
      `docs/` 或仓库角色文件，Lab 仓库合同会拒绝重新出现 `docs/`。
