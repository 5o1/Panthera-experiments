# Panthera-HT 单目视觉手臂与手势遥操作实验计划

安装前先看配套的[依赖列表](<Panthera-HT 单目视觉遥操作依赖列表.md>)。

## 1. 实验定位

这是一个面向具身智能和 ROS 2 入门的教学 Demo，不是工业遥操作系统。

最终只验收以下能力：

1. 笔记本单目摄像头能够识别单个操作者的右肩、右肘、右腕和右手。
2. 操作者右臂的位置和朝向变化能够控制 Panthera-HT 末端的位置与姿态。
3. `Closed_Fist/Open_Palm` 或自定义的 `snake_closed/snake_open` 视觉手势能够控制夹爪关闭和打开。
4. 整条链路通过 ROS 2 节点和话题连接，并能用 ROS 2 工具观察、调试和录制。

不纳入本次范围：多人跟踪、双臂、云端推理、语义理解、视觉语言模型、障碍物识别、自动抓取、MoveIt 轨迹规划、生产级急停和精确的毫米级遥操作。

## 2. 为什么这适合作为具身智能入门项目

本实验可以压缩成最小的感知—决策—执行闭环：

```text
环境中的人
  ↓ RGB 图像（Observation）
视觉模型
  ↓ 人体关键点、手部关键点、手势类别（State / Feature）
ROS 2 重定向节点
  ↓ 末端位姿、夹爪状态（最小 Policy）
Panthera ROS 2 驱动
  ↓ 电机动作（Action）
机械臂与环境
```

当前的“策略”只是确定性的坐标映射。以后学习模仿学习、VLA 或强化学习时，可以保留摄像头节点、ROS 2 话题和机器人驱动，只把重定向节点换成学习策略。因此本实验重点不是追求复杂算法，而是理解具身系统中观察、表示、策略、动作、反馈和数据记录之间的边界。

完成实验时应掌握：

- 视觉模型推理：图像如何变成结构化关键点和类别。
- ROS 2：package、node、topic、message、parameter、launch、rosbag。
- 机器人表示：笛卡尔位置、旋转矩阵、四元数、欧拉角和关节状态。
- Motion Retargeting：人的运动怎样映射到不同结构的机器人。
- 数据闭环：怎样记录 observation-action 轨迹，为以后训练策略准备数据。

## 3. 最终采用的技术路线

### 3.1 视觉模型

视觉侧使用两个 MediaPipe Tasks 模型：

| 模型 | 输入 | 本实验使用的输出 |
|---|---|---|
| Pose Landmarker / BlazePose | 单目 RGB 帧 | 右肩 12、右肘 14、右腕 16 的 3D world landmarks |
| Gesture Recognizer | 单目 RGB 帧 | 右手 21 个 landmarks、手势类别、类别分数 |

两者都是视觉神经网络，不需要自己训练人体姿态模型。BlazePose 的原始论文描述了单 RGB 图像实时输出 33 个人体关键点；MediaPipe Hands 论文描述了手掌检测和 21 点手部骨架模型。

第一阶段直接使用 Gesture Recognizer 自带的 `Closed_Fist` 和 `Open_Palm`。第二阶段采集自己的图片，通过 MediaPipe Model Maker 训练：

```text
snake_open
snake_closed
none
```

这样既能尽快打通机械臂，又能完整练习一次小型视觉模型的数据采集、训练、评估和部署。

### 3.2 ROS 2 与机器人控制

使用 Panthera 官方 ROS 2 Humble 仓库中的 `panthera_arm_control`，不经过 MoveIt。官方节点已经提供：

```text
/pos_cmd           panthera_interfaces/msg/PosCmd
/gripper_cmd       example_interfaces/msg/Bool
/end_pose_euler    panthera_interfaces/msg/EndPoseEuler
/joint_states_single
/arm_status
/stop_srv
```

本实验只需发布 `/pos_cmd` 和 `/gripper_cmd`，并订阅 `/end_pose_euler` 取得激活时的机器人末端零点。

### 3.3 WSL 内的进程分工

```text
WSL 视觉虚拟环境
└── windows_vision/vision_sender.py（目录名是历史遗留，程序同时支持 Linux）
    ├── OpenCV 读取笔记本摄像头
    ├── Pose Landmarker
    ├── Gesture Recognizer
    └── WebSocket JSON

WSL ROS 2 Humble 环境
├── vision_bridge_node.py
│   └── WebSocket JSON → ROS 2 topics
├── teleop_mapper_node.py
│   └── 人体位姿 → Panthera 位姿/夹爪命令
└── panthera_arm_control
    └── ROS 2 命令 → Panthera 真机
```

摄像头和机械臂 USB 都通过 `usbipd-win` 交给 WSL。视觉程序运行在独立 venv 中但不直接控制机器人；它只向本机 WebSocket 发送观测，所有机器人命令都在 ROS 2 图中可见。

## 4. 建议的工程结构

本项目保持为独立 GitHub 仓库和独立 ROS 2 overlay workspace，不把实验代码复制进 Panthera 官方仓库。在本仓库的 `ros2_ws/src` 下建立 Python package：

```bash
cd ~/panthera/ros2_ws/src
ros2 pkg create --build-type ament_python panthera_vision_teleop \
  --dependencies rclpy geometry_msgs std_msgs example_interfaces sensor_msgs panthera_interfaces
```

```text
panthera/
├── ros2_ws/
│   └── src/
│       └── panthera_vision_teleop/
│           ├── package.xml
│           ├── setup.py
│           ├── setup.cfg
│           ├── resource/
│           ├── launch/
│           │   └── vision_teleop.launch.py
│           ├── config/
│           │   └── teleop.yaml
│           └── panthera_vision_teleop/
│               ├── vision_bridge_node.py
│               ├── teleop_mapper_node.py
│               ├── rotation_utils.py
│               └── debug_pose_publisher.py
├── windows_vision/                 # 历史目录名；当前由 WSL 视觉 venv 运行
│   ├── vision_sender.py
│   ├── capture_gesture_dataset.py
│   ├── train_gesture_model.ipynb
│   └── models/
│       ├── pose_landmarker.task
│       ├── gesture_recognizer.task
│       └── custom_gesture_recognizer.task
└── bags/
```

Panthera 官方工作区作为 underlay 独立存在。构建本项目之前按以下顺序 source：

```bash
source /opt/ros/humble/setup.bash
source ~/Panthera_HT_ROS2/install/setup.bash
cd ~/panthera/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

`debug_pose_publisher.py` 用固定的测试数据模拟视觉输入。这样可以在摄像头或视觉模型尚未完成时单独学习和调试 ROS 2 链路。

## 5. ROS 2 图与消息约定

### 5.1 实验内部话题

| Topic | 类型 | 发布者 | 订阅者 | 内容 |
|---|---|---|---|---|
| `/teleop/human_pose` | `geometry_msgs/PoseStamped` | bridge | mapper | position 为肩到腕向量；orientation 为人体右臂坐标系 |
| `/teleop/gesture` | `std_msgs/String` | bridge | mapper | `Closed_Fist`、`Open_Palm`、`snake_open` 等 |
| `/teleop/gesture_score` | `std_msgs/Float32` | bridge | mapper | 当前手势置信分数 |
| `/teleop/enabled` | `std_msgs/Bool` | bridge | mapper | 是否允许发送机器人目标 |
| `/teleop/recalibrate` | `std_msgs/Bool` | bridge | mapper | 单次重新记录人和机器人零点 |
| `/teleop/debug_target` | `geometry_msgs/PoseStamped` | mapper | 调试工具 | 映射后的机器人目标 |

### 5.2 Panthera 官方话题

| Topic | 类型 | 方向 | 用途 |
|---|---|---|---|
| `/end_pose_euler` | `panthera_interfaces/EndPoseEuler` | driver → mapper | 当前末端 XYZ+RPY |
| `/pos_cmd` | `panthera_interfaces/PosCmd` | mapper → driver | 目标末端 XYZ+RPY |
| `/gripper_cmd` | `example_interfaces/Bool` | mapper → driver | `true=打开`，`false=关闭` |
| `/joint_states_single` | `sensor_msgs/JointState` | driver → tools | 记录关节反馈 |
| `/arm_status` | `panthera_interfaces/ArmStatus` | driver → tools | 观察使能和故障 |

实验开始时先运行 `ros2 interface show` 查看实际字段，不要只根据本文猜测：

```bash
ros2 interface show panthera_interfaces/msg/PosCmd
ros2 interface show panthera_interfaces/msg/EndPoseEuler
```

### 5.3 WSL 视觉 venv 到 ROS bridge 的 JSON

每次视觉推理只发送一个最新状态：

```json
{
  "v": 1,
  "seq": 1208,
  "source_time_ms": 32813311,
  "enabled": true,
  "recalibrate": false,
  "pose_valid": true,
  "hand_valid": true,
  "shoulder": [0.11, -0.24, -0.05],
  "elbow": [0.28, -0.31, -0.14],
  "wrist": [0.43, -0.39, -0.21],
  "pose_score": 0.91,
  "gesture": "Open_Palm",
  "gesture_score": 0.86
}
```

`source_time_ms` 只用于日志和检查递增。bridge 的数据超时必须使用收包时记录的本地单调时钟，不能拿发送进程的 monotonic 与接收进程时钟直接相减。

`pose_score` 取右肩、右肘、右腕三个 landmark 的 `visibility` 最小值。Gesture Recognizer 根据 `handedness` 只选择右手；输入模型的图像保持原方向，需要镜像时只镜像显示窗口。按 `R` 时仅让一个数据包的 `recalibrate=true`，mapper 收到后重新记录零点。

## 6. 从人体关键点构造手臂位姿

### 6.1 位置

用右腕相对右肩的向量作为人的末端位置表示：

\[
\mathbf h_p = \mathbf p_{wrist} - \mathbf p_{shoulder}
\]

这样操作者身体整体平移时影响较小。激活控制时保存：

```text
human_position_zero = h_p0
robot_position_zero = p_r0
```

之后映射为：

\[
\mathbf p_{target}
=
\mathbf p_{r0}
+
\mathbf S_p \mathbf P_p
(\mathbf h_p-\mathbf h_{p0})
\]

其中 `P_p` 负责轴交换和正负号，`S_p` 是位置缩放。第一轮建议：

```yaml
position_gain: [0.25, 0.25, 0.12]
max_position_offset: [0.05, 0.05, 0.04]
```

先在 dry-run 中做“手向右、向上、向前”三个动作，确定轴映射。不要镜像后再送入模型；需要镜像时只镜像显示窗口。

### 6.2 朝向

用肩、肘、腕建立右臂局部坐标系。设：

\[
\mathbf z_h = \operatorname{normalize}(\mathbf p_{wrist}-\mathbf p_{elbow})
\]

\[
\mathbf v_h = \operatorname{normalize}(\mathbf p_{elbow}-\mathbf p_{shoulder})
\]

\[
\mathbf x_h = \operatorname{normalize}(\mathbf v_h \times \mathbf z_h)
\]

\[
\mathbf y_h = \mathbf z_h \times \mathbf x_h
\]

形成旋转矩阵：

\[
\mathbf R_h=[\mathbf x_h\;\mathbf y_h\;\mathbf z_h]
\]

当手臂完全伸直时，两个方向接近平行，叉积会趋近零；此时保持上一帧有效朝向，不更新 orientation。

激活时保存 `R_h0` 和当前机器人 `R_r0`。人体相对旋转为：

\[
\Delta \mathbf R_h = \mathbf R_h \mathbf R_{h0}^{T}
\]

用一个行列式为 `+1` 的正交矩阵 `P_r` 将旋转从人体/相机轴映射到机器人基座轴：

\[
\Delta \mathbf R_r
=
\mathbf P_r \Delta \mathbf R_h \mathbf P_r^T
\]

\[
\mathbf R_{target}=\Delta \mathbf R_r \mathbf R_{r0}
\]

最后使用 SciPy `Rotation` 转成 Panthera `/pos_cmd` 需要的 roll、pitch、yaw。第一轮把相对旋转限制在约 `±20°`；如果姿态过于抖动，先只启用 XYZ，确认位置闭环后再打开 orientation。

### 6.3 滤波

位置先用最简单的指数滑动平均：

\[
\mathbf x_f(t)=\alpha \mathbf x(t)+(1-\alpha)\mathbf x_f(t-1)
\]

初始使用 `alpha=0.35`。姿态使用旋转插值而不是直接平均欧拉角；可用 SciPy `Slerp` 或“上一姿态到当前姿态只走固定比例”的方式。

## 7. 手势模型与夹爪状态机

### 7.1 第一阶段：预训练视觉模型

MediaPipe 官方 Gesture Recognizer 的预训练模型已经包含：

```text
Closed_Fist
Open_Palm
Pointing_Up
Thumb_Down
Thumb_Up
Victory
ILoveYou
```

第一版直接定义：

```text
Open_Palm  → gripper open
Closed_Fist → gripper close
其他/Unknown → 保持上一个夹爪状态
```

接受一个手势前满足：

```text
gesture_score >= 0.70
并且连续 5 帧类别相同
```

夹爪只在状态变化时发布一次，不要每帧重复开合。

### 7.2 第二阶段：自定义“蛇头开合”模型

训练集目录遵守 MediaPipe Model Maker 约定：

```text
gesture_dataset/
├── none/
├── snake_open/
└── snake_closed/
```

数据采集建议：

1. 每类先采集 300～500 张图片。
2. `snake_open` 包含不同开口大小、左右轻微旋转和远近变化。
3. `snake_closed` 包含手指并拢、拇指与其余手指靠拢的变化。
4. `none` 放入普通张手、握拳、指向、半开和容易混淆的姿势。
5. 至少在 3 种背景和 2 种光照下采集。
6. 按“录制批次”切分训练集和验证集，避免同一段视频的相邻帧同时进入训练集和验证集。

训练流程：

```text
图片目录
→ mediapipe-model-maker Dataset.from_folder
→ train / validation / test split
→ GestureRecognizer.create(...)
→ model.evaluate(...)
→ export gesture_recognizer.task
→ WSL 视觉程序替换模型文件
```

Model Maker 官方要求数据集中必须有一个名为 `none` 的类别。导出的 `.task` 文件可以直接交给 Gesture Recognizer。Model Maker 当前处于“仍可用但不再积极维护”状态，因此建议在独立 Python 环境或 Google Colab 中训练，不要与 ROS 2 环境混装。

自定义模型映射：

```text
snake_open   → gripper open
snake_closed → gripper close
none/低分    → 保持
```

本 Demo 只做二值开合，不做连续夹爪开度回归。

## 8. 参数文件建议

`config/teleop.yaml` 初始内容：

```yaml
vision_bridge_node:
  ros__parameters:
    websocket_host: "0.0.0.0"
    websocket_port: 8765
    receive_timeout_sec: 0.5

teleop_mapper_node:
  ros__parameters:
    command_rate_hz: 10.0
    pose_score_min: 0.6
    gesture_score_min: 0.7
    gesture_stable_frames: 5
    position_alpha: 0.35
    rotation_alpha: 0.25
    position_gain: [0.25, 0.25, 0.12]
    max_position_offset: [0.05, 0.05, 0.04]
    max_rotation_deg: 20.0
    open_labels: ["Open_Palm", "snake_open"]
    close_labels: ["Closed_Fist", "snake_closed"]
    publish_robot_commands: false
```

`publish_robot_commands=false` 是 dry-run。确认 `/teleop/debug_target` 正确后再改成 `true`。

## 9. 分阶段实验步骤

### 阶段 0：先认识 Panthera ROS 2 接口

目标：不接视觉，独立证明 ROS 2 能控制机器人。

1. 启动 `panthera_arm_control`。
2. 用 `ros2 topic list`、`ros2 node list`、`ros2 interface show` 查看图。
3. 用官方示例命令读取 `/end_pose_euler`。
4. 手工发布一次很小的 `/pos_cmd`。
5. 手工发布一次 `/gripper_cmd` 开、关。
6. 用 `rqt_graph` 观察节点连接。

完成条件：命令行能够让末端小幅移动，夹爪能够开合。

### 阶段 1：视觉模型离线验证

目标：WSL 视觉 venv 只显示结果，不连接 ROS 2。

1. OpenCV 打开笔记本摄像头，分辨率先用 640×480。
2. Pose Landmarker 使用 `VIDEO` 模式，同步调用 `detect_for_video()`。
3. Gesture Recognizer 使用同一帧和同一递增时间戳调用 `recognize_for_video()`。
4. 在窗口画出肩、肘、腕和手部关键点。
5. 显示当前 gesture label、score、FPS。
6. 只处理一人和一只右手。

完成条件：正常光照下能够连续看到右肩、右肘、右腕；张手和握拳能够显示正确类别。

### 阶段 2：ROS 2 视觉桥

目标：视觉数据进入 ROS 2，但不控制机器人。

1. WSL 视觉 venv 中的程序通过本机 WebSocket 发送 JSON。
2. `vision_bridge_node` 收到 JSON 后发布 `/teleop/*`。
3. 用以下命令观察：

   ```bash
   ros2 topic echo /teleop/human_pose
   ros2 topic echo /teleop/gesture
   ros2 topic hz /teleop/human_pose
   ```

4. 断开视觉发送程序，确认 bridge 在 0.5 秒后发布 `enabled=false`。

完成条件：ROS 2 中能稳定看到约 10～15 Hz 的人体位姿和手势。

### 阶段 3：重定向 dry-run

目标：验证人的运动如何变成机器人目标，不发送真机命令。

1. `teleop_mapper_node` 订阅人体位姿和 `/end_pose_euler`。
2. WSL 摄像头预览窗口按空格切换 enable；enable 上升沿记录人和机器人零点。
3. 发布映射结果到 `/teleop/debug_target`。
4. 分别做右、上、前三个动作，修改轴映射直到符合直觉。
5. 弯曲手肘、旋转前臂，观察目标 RPY。
6. 验证手臂近乎伸直时朝向保持而不是产生 NaN。
7. 验证超过位置/角度范围时被截到小工作区。

完成条件：打印出的 XYZ/RPY 连续、方向正确、无明显大跳。

### 阶段 4：真机位置控制

目标：先完成最容易稳定的 XYZ 跟随。

1. 保持 `max_position_offset` 为 3～5 cm。
2. 暂时固定机器人 orientation 为激活瞬间姿态。
3. 设置 `publish_robot_commands=true`。
4. 以 10 Hz 发布 `/pos_cmd`。
5. enable=false、视觉失踪或数据超时时停止发布新目标。

完成条件：右腕相对肩部移动时，机械臂末端能按比例跟随三个方向。

### 阶段 5：加入姿态控制

目标：人体手臂平面控制机械臂末端方向。

1. 打开旋转映射。
2. 从最大 `±10°` 开始，再增大到 `±20°`。
3. 观察机器人是否出现欧拉角跳变；内部始终使用旋转矩阵或四元数，只在发布 `/pos_cmd` 前转换成 RPY。
4. 如果单目 Z 抖动明显，减小 Z gain，不必追求精确复现。

完成条件：至少两个方向的手臂旋转能引起可辨认的末端姿态变化。

### 阶段 6：夹爪视觉手势

目标：视觉模型类别控制夹爪。

1. 先使用 `Open_Palm/Closed_Fist`。
2. 验证稳定帧和分数门限。
3. 再采集蛇头手势数据并训练自定义模型。
4. 替换 `.task` 文件，映射 `snake_open/snake_closed`。

完成条件：连续做 5 次打开和关闭动作，至少 4 次能正确改变夹爪状态；无法识别时夹爪保持原状态。

### 阶段 7：记录一段 observation-action 数据

目标：把 Demo 与后续具身学习连接起来。

记录：

```bash
ros2 bag record \
  /teleop/human_pose \
  /teleop/gesture \
  /teleop/gesture_score \
  /teleop/enabled \
  /teleop/debug_target \
  /end_pose_euler \
  /joint_states_single \
  /arm_status
```

回放时保持真机命令关闭：

```bash
ros2 bag play <bag-directory>
```

理解每个时间步中的：

```text
observation = 人体位姿 + 手势 + 机器人当前状态
action      = 机器人目标位姿 + 夹爪状态
```

这就是最简单的具身轨迹数据。以后可以用相同数据接口替换或训练策略。

## 10. 最终启动顺序

1. 启动 WSL，确认机械臂 USB 已 attach。
2. 启动 Panthera 官方驱动：

   ```bash
   source /opt/ros/humble/setup.bash
   source ~/Panthera_HT_ROS2/install/setup.bash
   ros2 launch panthera_arm_control arm_control.launch.py
   ```

3. 启动视觉桥和重定向节点：

   ```bash
   source /opt/ros/humble/setup.bash
   source ~/Panthera_HT_ROS2/install/setup.bash
   source ~/panthera/ros2_ws/install/setup.bash
   ros2 launch panthera_vision_teleop vision_teleop.launch.py
   ```

4. 在 WSL 视觉 venv 启动 `vision_sender.py`。
5. 观察预览，确认只有右侧操作者和右手被识别。
6. 按空格启用/停止，按 `R` 请求重新建立零点，按 `Esc` 退出。
7. 调试时另开终端观察 `/arm_status`、`/end_pose_euler` 和 `rqt_graph`。

## 11. 简化的验收清单

- [ ] 摄像头预览能画出右肩、右肘、右腕和右手骨架。
- [ ] 视觉程序能输出模型手势类别和置信分数。
- [ ] ROS 2 能看到 `/teleop/human_pose` 和 `/teleop/gesture`。
- [ ] ROS 2 图中能看到 vision bridge、mapper 和 Panthera driver。
- [ ] 手向右、向上、向前时机械臂末端发生对应移动。
- [ ] 手臂转动时机械臂末端姿态有可辨认变化。
- [ ] 张手/蛇头张开能够打开夹爪。
- [ ] 握拳/蛇头闭合能够关闭夹爪。
- [ ] 人离开画面或关闭 enable 后不再发送新运动目标。
- [ ] 能录制并回放一次 rosbag（回放时关闭真机命令）。

达到以上条件即可认为项目通过，不要求位置精度、姿态精度或手势识别率达到论文指标。

## 12. 常见失败和最短处理路径

| 现象 | 优先检查 |
|---|---|
| 摄像头打不开 | usbipd 是否 attach、WSL 是否有 `/dev/video*`、video 组权限、`VideoCapture(0/1)`、是否被其他程序占用 |
| 手部经常消失 | 手距离摄像头过远、运动模糊、逆光；先降低动作速度 |
| 左右手识别反了 | 是否在送模型前镜像图像；应只镜像预览 |
| 机械臂方向反了 | 修改 `P_p/P_r` 轴映射，不改视觉模型 |
| 姿态突然跳变 | 手臂过直导致叉积退化；保持上一有效姿态，减小角度范围 |
| 夹爪来回抖 | 提高分数门限、增加稳定帧数、只在状态变化时发布 |
| ROS 2 看不到数据 | 每个终端是否 source ROS 与工作空间；检查节点和 topic 名称 |
| 找不到机械臂串口 | `usbipd list`、重新 attach、WSL 中查看 `/dev/ttyACM*` |
| 编译 Panthera 仓库失败 | 确认 Ubuntu 22.04 + Humble；工作区路径不要含中文；按官方 README 补依赖 |

## 13. 论文与技术档案

以下资料足以覆盖本计划中的模型、接口和数学工具。实现时优先看“官方实现文档”，论文用于理解模型原理。

这些资料已经下载到仓库，离线文件、校验值和失败记录见 [`docs/references/README.md`](references/README.md)。

### 视觉模型

1. [BlazePose: On-device Real-time Body Pose Tracking](https://arxiv.org/abs/2006.10204) — 单目人体 33 点姿态模型论文。
2. [MediaPipe Hands: On-device Real-time Hand Tracking](https://arxiv.org/abs/2006.10214) — 单目手掌检测和 21 点手部模型论文。
3. [Pose Landmarker Python 官方指南](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) — Python 模型加载、VIDEO/LIVE_STREAM API 和结果格式。
4. [Gesture Recognizer 技术说明](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer) — 模型组成、预训练类别、关键点输出和自定义模型入口。
5. [Gesture Recognizer Python 官方指南](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python) — Python 推理代码和 `.task` 模型加载。
6. [自定义 Gesture Recognizer 官方训练教程](https://developers.google.com/edge/mediapipe/solutions/customization/gesture_recognizer) — 数据目录、Model Maker、评估和 `.task` 导出。
7. [MediaPipe Holistic 技术说明](https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/) — 理解身体与手部模型组合方式；本实验不必运行完整面部模型。

### ROS 2 与 Panthera

1. [Panthera-HT ROS 2 官方仓库](https://github.com/HighTorque-Robotics/Panthera-HT_ROS2) — Ubuntu/ROS 版本、构建、`panthera_arm_control`、话题和服务。
2. [ROS 2 Humble Ubuntu 安装](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) — 官方安装入口。
3. [ROS 2 Python Publisher/Subscriber 教程](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html) — rclpy 节点和话题。
4. [创建 ROS 2 Package](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html) — ament_python package 结构。
5. [ROS 2 参数教程](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-Python.html) — 把 gain、阈值和映射放入参数文件。
6. [ROS 2 rosbag2 教程](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html) — 记录和回放 observation-action 数据。

### 数学、通信与系统

1. [SciPy Rotation 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) — 旋转矩阵、四元数和欧拉角互转。
2. [SciPy Slerp 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html) — 姿态平滑插值。
3. [OpenCV VideoCapture 文档](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) — 摄像头读取。
4. [WebSocket RFC 6455](https://www.rfc-editor.org/rfc/rfc6455) — 视觉 venv/ROS 状态传输协议。
5. [Python websockets 文档](https://websockets.readthedocs.io/) — Python 客户端和服务端实现。
6. [Microsoft WSL USB 连接文档](https://learn.microsoft.com/en-us/windows/wsl/connect-usb) — `usbipd-win` 和 USB attach。
