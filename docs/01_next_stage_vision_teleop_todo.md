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
