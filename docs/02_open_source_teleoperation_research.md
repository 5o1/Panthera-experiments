# 开源视觉遥操作实现调查与 Panthera 技术路线

调查日期：2026-09-01

本文中的项目只作为设计候选、实现证据和反例，不是本项目必须遵循的架构或依赖。后续
可以按 Panthera 的实际驱动能力、单目可观测性、测试数据和安全目标修改或弃用任何方案。

## 1. 调查目标和结论

这次调查针对的是当前 demo 的完整问题，而不只是“MediaPipe 能不能识别人”：

```text
普通 RGB 摄像头
  → 人体/手部观测
  → 坐标标定和重定向
  → 逆运动学或关节映射
  → 连续运动生成
  → 机器人驱动
  → 失跟、延迟、限位和碰撞处理
```

没有找到一个可以直接替换当前 Panthera 实现的仓库。最接近普通摄像头条件的项目，
只控制两个图像平面方向或少数几个关节；能够稳定控制完整机械臂位姿的系统，普遍使用
深度相机、多相机、Vision Pro/VR，或者额外的三维手部位姿模型。这不是偶然：单张 RGB
图像不能直接观测公制深度，而肩、肘、腕三个点也不能观测手腕绕前臂轴的自转。

高质量实现的共同结构是：视觉以约 25–30 Hz 产生带时间戳的观测；重定向器根据机器
人模型、关节限位和上一时刻状态生成目标；运动生成器以 100 Hz 以上把稀疏目标变成连
续、限速的关节轨迹；最后由更高频的闭环控制器执行。当前 Panthera demo 的主要瓶颈
已经不是 MediaPipe，而是官方 `/pos_cmd` 是一个会阻塞到动作完成的离散点位接口。

## 2. 仓库横向比较

| 项目 | 输入设备 | 控制方法 | 实际解决的范围 | 对本项目最有价值的部分 |
|---|---|---|---|---|
| [Vision Robotic Arm Gesture Recognition](https://github.com/e-candeloro/Vision-Robotic-Arm-Gesture-Recognition) | 单 RGB + MediaPipe | 人体角度直接映射到部分关节 | Panda/Gazebo 中少数关节跟随 | 尺度无关的角度和夹爪开度特征 |
| [openarm_teleoperation](https://github.com/ahsanali555/openarm_teleoperation) | 单 RGB + MediaPipe | 图像位移转末端速度，交给 MoveIt Servo | 双臂 Y/Z 速度控制 | 连续速度接口、逐轴死区、命令超时 |
| [humanoid-arm-retarget](https://github.com/Hao-Starrr/humanoid-arm-retarget) | Vision Pro | 带关节约束和连续性代价的优化重定向 | 双 7-DoF 手臂 | 显式坐标标定、warm start、单帧跳变拒绝、clutch |
| [dex-retargeting](https://github.com/dexsuite/dex-retargeting) | 手部关键点 | 关键点向量优化到机器人关节 | 多种灵巧手和夹爪 | 相对向量、关节边界、时间连续性、连续捏合量 |
| [mediapipe_ros2_suite](https://github.com/PME26Elvis/mediapipe_ros2_suite) | ROS Image + MediaPipe | 感知结果发布为 ROS 自定义消息 | ROS2 感知层 | 感知/控制解耦、源时间戳、图像回放和调试标记 |
| [MIRROR](https://github.com/junhengl/mirror) | ZED 深度人体跟踪 | 30 Hz 跟踪、500 Hz 差分 IK、1 kHz 控制 | 人形机器人全身重定向 | 分频循环、安全状态机、平滑进入、跟踪 ID 锁定 |
| [AnyTeleop](https://roboticsproceedings.org/rss19/p015.pdf) | RGB/RGB-D/多相机 | 感知、优化重定向、碰撞约束运动生成 | 多种手臂和灵巧手 | 完整系统分层，以及单 RGB 深度局限的处理方式 |

仓库的星数和 README 不是本调查的主要依据。下面的判断来自实际执行路径中的代码；有
些项目的 README 描述比当前可执行代码更先进，两者不能混为一谈。

## 3. 各项目究竟解决了什么

### 3.1 单目 MediaPipe 直接映射关节

`Vision-Robotic-Arm-Gesture-Recognition` 是输入条件最接近当前 demo 的实现。代码使用
肩、肘、腕三个 world landmark，通过两个向量的点积计算肘角；再把左右肘角、肩角
和手掌开度线性映射到 Panda 的 joint 1、2、4、6，其余关节保持固定。相关实现见
[主控制循环](https://github.com/e-candeloro/Vision-Robotic-Arm-Gesture-Recognition/blob/63c15610848c9107c60a043318c204fca0a3a59a/Demo_2_Panda_Control/ROS/panda_control/src/vision_arm_control/scripts/main.py)、
[人体角度计算](https://github.com/e-candeloro/Vision-Robotic-Arm-Gesture-Recognition/blob/63c15610848c9107c60a043318c204fca0a3a59a/Demo_2_Panda_Control/ROS/panda_control/src/vision_arm_control/scripts/Detector_Modules/PoseDetectorModule.py)
和[手掌开度计算](https://github.com/e-candeloro/Vision-Robotic-Arm-Gesture-Recognition/blob/63c15610848c9107c60a043318c204fca0a3a59a/Demo_2_Panda_Control/ROS/panda_control/src/vision_arm_control/scripts/Detector_Modules/HandDetectorModule.py)。

它解决的是“不同身高、不同离摄像头距离时，动作仍然有相似含义”。人体关节角不依赖
手臂在图像中的绝对长度；手掌开度则用指尖中心到掌根中心的距离除以掌长，消除了手离
摄像头远近造成的整体缩放。代价是它没有控制任意末端位置：人的肩角只对应指定机器人
关节，人体和机器人形态不同所产生的误差也没有通过 IK 处理。

这条路线适合“手臂弯曲多少就让某个关节转多少”的动作模仿，不适合当前“让 Panthera
末端跟随人的手在三维空间移动”的目标。可以直接借鉴的只有两个特征设计：用人体角度
作为不受尺度影响的辅助输入，以及把夹爪改成经过掌长归一化的连续开度。

该实现主要运行于 Gazebo，没有完备的失跟锁定、来源帧超时、机器人关节限位、自碰撞
和真机急停路径，因此不能照搬其发布循环。

### 3.2 单目位移转速度，再由 MoveIt Servo 求关节运动

`openarm_teleoperation` 读取左右腕的归一化图像坐标，连续检测 15 帧后记录参考点。
腕相对参考点的位移经过逐轴死区、工作框裁剪和 EMA 后，转换成 `TwistStamped` 的 Y/Z
速度；X 和三个角速度实际为零。MoveIt Servo 再根据机器人模型把末端速度转换成关节
运动。可执行版本位于
[pose_teleop_node.py](https://github.com/ahsanali555/openarm_teleoperation/blob/668bb6d85b9e15596789f1a06bed55751583d27b/openarm_teleop/openarm_teleop/pose_teleop_node.py)。

这里解决了当前 demo 最明显的“一个离散点走完，再走下一个点”的停顿。视觉节点不再
请求一次完整动作，而是持续表达“现在希望末端沿哪个方向、以多快速度移动”。Servo
根据每个最新命令计算下一小段关节状态。新命令停止后，命令时间戳超时并平滑停止，不
会继续处理一列积压的历史目标。

MoveIt Servo 本身还提供奇异位形减速/停止、关节位置和速度边界、可选自碰撞/环境碰撞
检查及平滑插件。它可以接收 `JointJog`、`TwistStamped` 或 `PoseStamped`，输出
`JointTrajectory` 或关节数组。官方说明也明确给出了接入新机器人所需的最低条件：有
效的 URDF/SRDF、能接收关节位置或速度的控制器，以及快速准确的关节反馈，见
[MoveIt Servo 官方文档](https://moveit.picknik.ai/humble/doc/examples/realtime_servo/realtime_servo_tutorial.html)
和[参数定义](https://github.com/moveit/moveit2/blob/main/moveit_ros/moveit_servo/config/servo_parameters.yaml)。

当前 demo 不能直接接入它：现用的 `arm_control_node` 只把视觉目标接到会阻塞执行的
`/pos_cmd`。后续驱动审计确认官方仓库还包含非阻塞 SDK 调用、`ros2_control` hardware
plugin 和 `JointTrajectoryController` 配置，因此不必从零增加连续关节接口；但这些配置
存在 controller 名称、command interface 和停机行为不一致，当前机器也尚未安装
`moveit_servo`。仅仅在现有 `/pos_cmd` 上游加 Servo 仍会造成阻塞点位积压。连续路线必须
改走并先验收 `ros2_control` hardware plugin，详见 `03_reference_adoption_matrix.md`。

该仓库本身也有明显原型痕迹：一个 Python 文件内保留了多代控制器，有些新版本只是三
引号字符串中的代码，README 与当前真正执行的类并不完全一致；当前活动版本在失跟或出
框后自动调用回零动作，并在重新检测 15 帧后自动恢复控制。对真机而言，“失跟后自主移
动到 home”和“看见人后自动恢复”均不应复制。其 Servo 配置还关闭了碰撞检查。

### 3.3 用优化器处理人体与机器人形态差异

`humanoid-arm-retarget` 的输入是 Vision Pro 给出的完整腕部和前臂 4×4 位姿，而不是
普通 RGB 关键点。它先用 YAML 中的 `left_base/right_base` 把传感器世界坐标转换到机器
人左右臂基座坐标，再用 `left_wrist/right_wrist` 对齐“人腕坐标系”和“机器人末端工具
坐标系”。这是两个不同问题：前者解决摄像头与机器人放置方向不同，后者解决传感器定义
的手掌朝向与机器人夹爪坐标轴不同。实现见
[arms_retarget.py](https://github.com/Hao-Starrr/humanoid-arm-retarget/blob/5aff12cebf1dbfd6155f4c9202cecc1ec087d77e/arms_retarget.py)
和[机器人配置示例](https://github.com/Hao-Starrr/humanoid-arm-retarget/blob/5aff12cebf1dbfd6155f4c9202cecc1ec087d77e/config_fftai_gr1.yaml)。

它用 SLSQP 优化关节角，目标函数同时包含：末端位置误差、末端姿态误差，以及新关节角
与上一帧关节角的差。关节上下限作为显式约束，上一帧结果作为本帧初值（warm start）。
位置/姿态项让机器人完成动作，连续性项阻止 IK 在多个可行解分支之间跳动，warm start
则减少求解时间并倾向保持同一分支。最后两个腕关节另行解析计算并裁剪。

运行循环约 90 Hz，并使用每帧最大关节变化阈值拒绝突然跳变；捏指手势在 Engage 和
Detach 间切换，并带冷却时间，属于明确的 clutch，而不是检测到人就自动接管。实现见
[teleop.py](https://github.com/Hao-Starrr/humanoid-arm-retarget/blob/5aff12cebf1dbfd6155f4c9202cecc1ec087d77e/teleop.py)
和[gesture.py](https://github.com/Hao-Starrr/humanoid-arm-retarget/blob/5aff12cebf1dbfd6155f4c9202cecc1ec087d77e/gesture.py)。

值得借鉴的是优化目标和标定结构，不是 Vision Pro 数据格式。当前单目肩—肘—腕可以
构造一个手臂平面，但不能观测真实手腕 roll；手臂接近伸直时，平面法向量也不再稳定。
因此不能用更复杂的优化器“算出”输入中不存在的自由度。另一个需要修正后才能借鉴的点
是：该项目在跟踪无效时可无限保留最后数据；真机必须在有限超时后锁定，而当前 demo
已经采用了更安全的做法。

### 3.4 用关键点向量控制夹爪和灵巧手

`dex-retargeting` 不要求人的手指关节与机器人关节一一对应。`VectorOptimizer` 比较的
是“两个关键点之间的向量”和“两个机器人 link 之间的向量”；优化变量是机器人关节角，
约束来自 URDF 关节范围，代价中另有与上一帧关节角差的平方项。`PositionOptimizer` 使用
Smooth L1/Huber 类损失降低单个异常关键点的影响。顺序重定向器保存上一帧关节解作为下
一帧初值，并可增加低通滤波。代码见
[optimizer.py](https://github.com/dexsuite/dex-retargeting/blob/3f56141bc8bd2760d5e452e382937269554ebb21/src/dex_retargeting/optimizer.py)
和[seq_retarget.py](https://github.com/dexsuite/dex-retargeting/blob/3f56141bc8bd2760d5e452e382937269554ebb21/src/dex_retargeting/seq_retarget.py)。

这种相对向量设计解决了人的手和机器人手大小不同、整体平移不同的问题。Panda 夹爪示
例直接用 MediaPipe 的拇指尖 4 到食指尖 8 的向量驱动两根夹爪手指，配置见
[panda_gripper.yml](https://github.com/dexsuite/dex-retargeting/blob/3f56141bc8bd2760d5e452e382937269554ebb21/src/dex_retargeting/configs/teleop/panda_gripper.yml)。

Panthera 只有开/合夹爪，不需要整个优化器，但应采用同一类观测：计算拇指尖—食指尖
距离，再除以掌宽或掌长得到连续 `pinch_ratio`。随后用两个不同阈值形成迟滞：低于关闭
阈值才合拢，高于打开阈值才张开；两阈值之间保持原状态，并增加保持时间和冷却时间。
它比通用 `Open_Palm` 分类直接触发更能解决当前自然手误触问题。

该仓库还暴露了一个容易忽略的工程问题：不同 URDF 库的关节数组顺序可能不同。任何后
续关节级接口都应按 joint name 建立映射并验证，不能假定数组下标天然一致。

### 3.5 把 MediaPipe 封装为 ROS2 感知节点

`mediapipe_ros2_suite` 将 `sensor_msgs/Image` 作为输入，采用 MediaPipe Tasks 的
`LIVE_STREAM` 异步模式，将手、手势或人体关键点转换成自定义 ROS 消息，并提供调试图像
和 MarkerArray。图像订阅使用 BEST_EFFORT、KEEP_LAST，适合实时视觉中“宁可丢旧帧，
不要等待旧帧”。实现见
[mp_node.py](https://github.com/PME26Elvis/mediapipe_ros2_suite/blob/d669d5e6679c9974ada7fd8891f02bf724ea3782/src/mediapipe_ros2_py/mediapipe_ros2_py/mp_node.py)。

它解决的是软件边界：相机驱动、MediaPipe 感知和机器人控制之间只交换稳定消息；可以
录制原始图像和感知输出，再脱离摄像头回放测试。来源图像时间戳一路保留，也有利于判断
结果是不是已经过期。

本项目已经用 Windows/WSL WebSocket 隔离了 MediaPipe 与 ROS Python/NumPy 环境冲突，
当前没必要为了形式统一而整体换成该仓库。但是可以借鉴三点：保留原始采集时间而不是只
记接收时间；把原始关键点作为独立消息留给后续算法；录制相机输入，使回归测试可重复。
需注意该项目的 pose 消息主要是归一化/像素关键点，并不是机器人 `map` 坐标系中的公制
三维点；frame_id 不能代替真实的坐标变换。异步推理还必须绑定“结果对应的那一帧”，不
能简单读取回调发生时最新的图像缓存。

### 3.6 低频视觉、差分 IK、高频控制和安全状态机

`MIRROR` 是结构上最完整的参考：ZED 深度人体跟踪约 30 Hz，重定向器约 500 Hz，控制
器约 1 kHz。重定向循环在视觉帧之间使用零阶保持，即在新观测到来前继续使用同一个目标，
而不是让 30 Hz 的相机直接决定电机更新频率。项目架构见
[README](https://github.com/junhengl/mirror/tree/c3fb080feec4041885dcb37a5a89b4ec0325aafe)。

跟踪层锁定一个 ZED person ID，避免画面中第二个人突然接管；目标人物丢失后冻结，五秒
后只有一个人可见时才允许换 ID，并重置位置滤波器。`PositionFilter` 对正常样本执行
EMA，对超过阈值的跳变只吸收 5%，而不是立即跟随。实现见
[body_tracking_node.py](https://github.com/junhengl/mirror/blob/c3fb080feec4041885dcb37a5a89b4ec0325aafe/real_time_sim/nodes/body_tracking_node.py)。

重定向层同时给手和肘建立任务空间目标，利用机器人雅可比矩阵做差分 IK/QP，关节状态
由反馈 warm start；输出再做关节增量裁剪、EMA、角度环绕和 NaN/Inf 检查。这样不仅要
让腕到达目标，还用肘目标选择冗余机械臂的姿态。当前代码事实上把手姿态 RPY 乘零，说明
即便有深度三维肘腕位置，可靠姿态仍被开发者有意暂时关闭。实现见
[retargeting_node.py](https://github.com/junhengl/mirror/blob/c3fb080feec4041885dcb37a5a89b4ec0325aafe/real_time_sim/nodes/retargeting_node.py)。

状态机具有 INIT、IDLE、TRACKING、SAFETY_STOP、SHUTDOWN。INIT 从当前反馈用三次 Hermite
平滑插值进入默认姿态；进入 TRACKING 时也先做平滑混合；1 秒没有有效跟踪则退出跟踪；
反馈关节速度超过阈值则在当前位置进入安全保持。实现见
[fsm.py](https://github.com/junhengl/mirror/blob/c3fb080feec4041885dcb37a5a89b4ec0325aafe/real_time_sim/control/fsm.py)。

它对本项目的核心启示是状态与连续运动生成必须独立于视觉回调。不过不能原样复制：跟踪
层和重定向层都有“保留最后有效样本”的逻辑，必须由状态机超时才能最终退出；若任何一
层时间戳处理错误，就可能无限沿用旧目标。其安全检查目前主要只有关节速度，关节限位、
跟踪误差和 COM 等还写在待补注释中；大跳变被低权吸收而非拒绝，也可能造成目标缓慢漂向
错误值。当前 Panthera 的失跟锁定策略比这里更严格，应保留。

### 3.7 AnyTeleop 的完整系统设计

AnyTeleop 是论文系统而不是可直接部署到 Panthera 的单一 ROS 包。它把全局腕位姿估计、
局部手指重定向和机器人运动生成分开。RGB-D 路线利用深度、相机内参和 PnP 得到三维腕
位置；仅 RGB 路线必须额外训练弱透视尺度网络来近似全局三维位置，论文明确说明该位置
不如 RGB-D 准确。多相机路线优先使用相对运动，因为普通 RGB 的绝对位置误差更大，并
基于手形参数偏差选置信度更高的视角。

手部重定向最小化缩放后的人手关键点向量与机器人 link 向量之差，同时惩罚相邻时刻关
节变化，并带关节范围约束。运动生成器把约 25 Hz 的末端目标变成约 120 Hz、满足关节限
位和碰撞约束的关节轨迹。论文中的相机对比也显示单 RGB 可以完成任务，但完成时间和错
误率均弱于 RGB-D/多相机。详见
[AnyTeleop 论文](https://roboticsproceedings.org/rss19/p015.pdf)；其中独立出来并持续维护
的手部重定向实现就是上面的 `dex-retargeting`。

这说明当前 X 深度噪声不是再加一个普通 EMA 就能彻底解决的问题。可行选择只有明确承认
2.5D 相对控制并为 X 设置 clutch，增加深度传感器，增加第二视角，或者引入专门训练的三
维全局位姿模型。它们解决的问题和成本不同，不能称为同一种“滤波优化”。

## 4. 对当前 Panthera demo 的具体判断

### 4.1 已经做对、应当保留的部分

- 失跟、低置信度、连接中断和来源帧过期后锁定，恢复画面不会自动恢复运动。这个策略比
  多个开源原型的无限保持或自动恢复更适合真机。
- 使用相对标定姿势，而不是把 MediaPipe 的绝对坐标直接当机器人坐标。
- latest-only、小 QoS 深度、250 ms 来源积压检查，以及 `/pos_cmd` 一次只允许一个在途
  命令，均在限制旧视觉动作排队。
- X 深度和末端姿态默认关闭，是与单目可观测性相符的阶段性设计，不是缺少一行开关。
- 每条不可中断点位最多 12 mm，能限制当前驱动接口下单次失控后果；它不能代替真正可
  抢占控制器和物理急停。

### 4.2 当前真正缺失的层

```text
现状：视觉目标 ──5 Hz──> 阻塞式 /pos_cmd ──动作完成──> 下一个目标

目标：视觉观测 ──25~30 Hz──> 最新期望状态
                         ↓
                100 Hz 受限目标生成器
                         ↓
              可抢占的关节位置/速度接口
                         ↓
                  高频关节反馈闭环
```

EMA 只能降低测量噪声；它没有显式限制速度、加速度或 jerk，也不能使阻塞动作可中断。
因此下一阶段不能只提高增益和范围。驱动审计已经确认 SDK 和官方
`PantheraHardwareInterface` 存在非阻塞连续命令基础，纯 GenericSystem 的标准
`FollowJointTrajectory` 通路也已通过测试。下一步应先修复/覆盖官方 ros2_control 配置，
并验证 hardware plugin 的反馈周期、当前位置保持、取消、命令超时和停止能力，再比较
MoveIt Servo、轻量差分 IK或自行实现的目标生成器。若这些真机验收失败，仍可弃用该路线
并改造直接 SDK 节点。

### 4.3 不应从开源仓库复制的行为

- 失跟后自动回 home/park：失跟本身意味着环境观测变差，此时发起较长自主运动并不安全。
- 重新看见操作者就自动继续：应维持当前显式重新 enable 的锁存机制。
- 无限保持最后视觉样本：短暂零阶保持可跨越相机帧间隔，但必须由源时间戳截止。
- 关闭碰撞检查却把 Servo 称为完整安全控制：限位、奇异位形、碰撞和急停是不同约束。
- 从腕在图像中的位移宣称得到完整 6D 位姿：单 RGB 和三个手臂点不提供这些观测。
- 按数组下标连接不同机器人库：必须按关节名映射并检查反馈顺序。

## 5. 建议实施顺序

调查后，建议把现有 TODO 的优先级调整为以下顺序：

1. 完成 rosbag 离线绘图，分别量化人体输入、映射目标、真实反馈、延迟和限幅；否则任何
   增益调整仍靠肉眼判断。
2. 审计 Panthera SDK/官方驱动，确认是否存在非阻塞关节位置、关节速度、抢占和立即停止
   API。这一步决定整个运动生成架构。
3. 把安全状态整理成显式 FSM：DISABLED、CALIBRATING、ARMED、TRACKING、STALE_LOCK、
   FAULT、PARKING。状态改变和机器人动作分开，所有恢复都要求显式确认。
4. 在可用的底层接口之上实现固定频率目标生成器，显式限制末端/关节速度、加速度和每步
   变化；视觉只更新最新目标，不直接发起完整动作。
5. 先扩大 Y/Z 并做量化验收。X 采用独立相对深度特征和 clutch；不要把单目结果解释为
   精确公制深度。
6. 夹爪改用归一化 `pinch_ratio + 双阈值迟滞 + 保持时间 + 冷却时间`，替代通用
   `Open_Palm/Closed_Fist` 直接触发。
7. 姿态先开放可观测且稳定的单轴；若需要可靠腕 roll，应增加手部关键点坐标系或更好的
   传感器，而不是从肩—肘—腕平面猜测。
8. 最后接入 MoveIt 的奇异位形、关节限位和碰撞能力，或在不采用 MoveIt 时为这些约束
   提供等价实现和测试。

短期最值得直接实现的不是复制某个仓库，而是组合其经过验证的原则：保留当前严格失跟
锁定；采用 MIRROR 的分频循环和状态机；采用 MoveIt Servo 的新鲜命令/连续受限运动语
义；采用 dex-retargeting 的归一化捏合特征和时间连续性；采用 humanoid-arm-retarget
的显式坐标标定和 clutch。底层驱动能力审计完成前，不应把 MoveIt Servo 写进真机启动
链路。
