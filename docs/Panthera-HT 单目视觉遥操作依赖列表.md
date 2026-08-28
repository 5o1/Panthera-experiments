# Panthera-HT 单目视觉遥操作依赖列表

这份清单只回答“需要准备和安装什么”。具体安装过程优先阅读每一项后面的官方文档。建议把视觉、ROS 2 和手势训练分成三个环境，不要把所有 Python 包塞进同一个环境。

完整实验步骤见[实验计划](<Panthera-HT 单目视觉手臂与手势遥操作实验计划.md>)。

## 1. 推荐环境布局

```text
Windows 主系统
├── 笔记本摄像头
├── Python 视觉环境
└── usbipd-win

WSL 2：Ubuntu 22.04
├── ROS 2 Humble
├── Panthera-HT_ROS2
└── panthera_vision_teleop ROS package

独立训练环境（任选）
├── Google Colab
或
└── 单独的 Python/conda 环境 + mediapipe-model-maker
```

Panthera 官方 ROS 2 仓库目前明确以 Ubuntu 22.04 + ROS 2 Humble 为目标。即使电脑里已有其他版本的 Ubuntu/WSL，也建议单独创建 Ubuntu 22.04 WSL distribution，不要为了本 Demo 改坏现有环境。

## 2. Windows 侧

### 必需软件

| 软件 | 用途 | 备注/官方资料 |
|---|---|---|
| Windows 10/11 | 摄像头和 WSL 宿主 | WSL 2 与 usbipd-win 所需 |
| WSL 2 | 运行 Ubuntu 和 ROS 2 | [Microsoft WSL 文档](https://learn.microsoft.com/windows/wsl/install) |
| Ubuntu 22.04 WSL | Panthera ROS 2 Humble 环境 | 与 Panthera 官方仓库一致 |
| usbipd-win | 把 Panthera USB/CAN 设备交给 WSL | [Microsoft USB 连接文档](https://learn.microsoft.com/en-us/windows/wsl/connect-usb) |
| Git | 获取项目和模型示例 | [Git 官网](https://git-scm.com/) |
| Python 3.10 或 3.11 x64 | 运行摄像头和视觉模型 | 建议独立 virtualenv/conda 环境 |

### Windows Python 包

```text
mediapipe
opencv-python
numpy
scipy
websockets
```

用途：

- `mediapipe`：Pose Landmarker 和 Gesture Recognizer。
- `opencv-python`：读取摄像头、画关键点和调试文字。
- `numpy`：向量、矩阵和叉积。
- `scipy`：旋转矩阵、四元数、RPY 和 Slerp。
- `websockets`：向 WSL 发送视觉结果。

对应文档：

- [MediaPipe Python Setup](https://developers.google.com/edge/mediapipe/solutions/setup_python)
- [Pose Landmarker Python](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [Gesture Recognizer Python](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python)
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [SciPy Rotation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
- [websockets](https://websockets.readthedocs.io/)

### 视觉模型文件

需要下载并保存在 `windows_vision/models/`：

```text
pose_landmarker.task
gesture_recognizer.task
```

模型下载入口位于对应 MediaPipe 官方任务页面的 Models 区域：

- [Pose Landmarker 模型与说明](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
- [Gesture Recognizer 模型与说明](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer)

完成自定义蛇头手势训练后再增加：

```text
custom_gesture_recognizer.task
```

## 3. WSL Ubuntu 22.04 侧

### ROS 2 基础

| 软件/包 | 用途 |
|---|---|
| ROS 2 Humble Desktop | ROS 2 命令、rclpy、常用消息、RViz2 |
| `ros-dev-tools` | colcon、rosdep 等开发工具 |
| `rqt` / `rqt_graph` | 查看 ROS 图和调试 |
| `rosbag2` | 记录视觉与机器人数据 |

官方入口：

- [ROS 2 Humble Ubuntu Deb 安装](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [ROS 2 Humble 教程](https://docs.ros.org/en/humble/Tutorials.html)

不要用 conda Python 运行 Humble 的 rclpy 节点；优先使用 Ubuntu 22.04 自带的系统 Python 3.10，避免 ROS 2 的二进制 Python 包与 conda 冲突。

### Panthera 官方 ROS 2 工作空间

需要获取：

- [HighTorque-Robotics/Panthera-HT_ROS2](https://github.com/HighTorque-Robotics/Panthera-HT_ROS2)

本 Demo 实际使用其中：

```text
hightorque_robot
panthera_interfaces
panthera_arm_control
```

不要求在运行链路中使用 MoveIt、Gazebo、Pinocchio 阻抗控制示例，但完整编译官方仓库时仍可能需要仓库 README 列出的相关依赖。

### Panthera 仓库列出的系统库

```text
libyaml-cpp-dev
libserialport-dev
libeigen3-dev
libboost-all-dev
liburdfdom-dev
```

如果完整编译 Panthera ROS 2 工作空间，还按官方 README 准备：

```text
ros-humble-moveit
ros-humble-ros2-control
ros-humble-ros2-controllers
ros-humble-controller-manager
ros-humble-robot-state-publisher
ros-humble-rviz2
ros-humble-xacro
ros-humble-joint-state-broadcaster
ros-humble-joint-trajectory-controller
ros-humble-gazebo-ros-pkgs
```

如果只编译直接驱动链路，可先尝试：

```text
panthera_interfaces
hightorque_robot
panthera_arm_control
```

然后用 `rosdep` 根据这些 package 的 `package.xml` 补齐实际依赖。是否能选择性编译以当前官方仓库为准。

### 自己的 ROS 2 Python package 依赖

`panthera_vision_teleop/package.xml` 至少会用到：

```text
rclpy
geometry_msgs
std_msgs
example_interfaces
sensor_msgs
panthera_interfaces
```

WSL Python 还需要：

```text
python3-numpy
python3-scipy
websockets（Ubuntu 包或 pip，选一种）
```

不需要在 WSL 安装 MediaPipe，也不需要把笔记本摄像头 attach 给 WSL。

## 4. 自定义手势模型训练环境

### 推荐：Google Colab

优点是 TensorFlow 和驱动问题较少，训练完成后只下载一个 `.task` 文件到 Windows。

需要：

```text
mediapipe-model-maker
tensorflow 2.x
matplotlib
```

官方教程：

- [MediaPipe 自定义 Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/customization/gesture_recognizer)

### 本地训练

如果选择本地训练，单独创建一个 Python 环境，不与 Windows 实时推理环境或 WSL ROS 环境共用。Model Maker 官方已经标记为“不再积极维护但仍可使用”，所以遇到 TensorFlow 版本冲突时优先使用官方 Colab，而不是反复修改 ROS 环境。

## 5. 硬件与外设

| 项目 | 是否必需 | 说明 |
|---|---|---|
| Panthera-HT + 控制盒 + 24 V 电源 | 必需 | 已有设备 |
| 笔记本内置 RGB 摄像头 | 必需 | 本实验唯一视觉传感器 |
| Panthera USB/CAN 连接 | 必需 | 通过 usbipd-win 交给 WSL |
| 可触及的停止/断使能手段 | 必需 | Demo 也应在手边保留 |
| 独立显卡 | 不必需 | MediaPipe 首版可用 CPU |
| 深度相机 | 不需要 | 明确保持单目方案 |
| 云服务器 | 不需要 | 所有推理本地运行 |
| 脚踏开关 | 可选 | 以后可替换键盘 enable |

## 6. 安装完成后的最小自检

Windows：

```text
Python 能 import mediapipe、cv2、numpy、scipy、websockets
OpenCV 能打开摄像头
两个 .task 模型能够加载
```

WSL：

```text
ros2 --help 正常
colcon --help 正常
ros2 pkg list 能找到 panthera_arm_control
ros2 interface show panthera_interfaces/msg/PosCmd 正常
/dev/ttyACM* 能看到 Panthera 设备
```

ROS 2 工作空间：

```text
colcon build 成功
source install/setup.bash 后能找到 panthera_vision_teleop
ros2 launch panthera_arm_control arm_control.launch.py 能启动
```

通过这些自检后再进入实验计划中的阶段 0。
