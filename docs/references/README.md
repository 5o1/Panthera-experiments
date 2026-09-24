# Panthera-HT 实验资料本地索引

下载日期：2026-08-28

本目录保存《Panthera-HT 单目视觉手臂与手势遥操作实验计划》直接依赖的论文、官方技术文档、模型和源码归档。共尝试 27 项，成功 26 项，失败 1 项；每个文件的原始 URL、字节数和 SHA-256 见 [download-log.csv](download-log.csv)，失败原因见 [FAILED_DOWNLOADS.md](FAILED_DOWNLOADS.md)。

HTML 文件是单页源码快照，正文和代码示例可离线阅读；页面样式、图片或站内跳转仍可能访问在线资源。

## 论文

- [BlazePose: On-device Real-time Body Pose Tracking](papers/BlazePose_2006.10204.pdf)
- [MediaPipe Hands: On-device Real-time Hand Tracking](papers/MediaPipe_Hands_2006.10214.pdf)

### 层级技能与状态转移（2026-09-24）

为分析 OpenVLA 闭环到槽上方后不松爪的问题，新增 7 篇关于层级 VLA、技能分解、技能终止
和物理状态验证的原始论文。文件、来源和 SHA-256 见
[层级控制论文索引](hierarchical_control_papers.md)，项目适用性分析见
[层级技能转移文献调查](../26_hierarchical_skill_transition_literature_review_2026-09-24.md)。

### 搬运—释放差异与接触切换（2026-09-24）

为区分条件稳定的持物搬运、抓取前抖动和末端释放失败，新增 6 篇关于模仿学习分布偏移、
接触不连续性、关键帧、pick/place 动作抽象、动作分块和对象中心子任务的原始论文。文件、
来源和 SHA-256 见[搬运—释放论文索引](release_transition_papers.md)。

### 强化学习释放切换与奖励设计（2026-09-24）

为检验模型能否从物理结果奖励中学习保持—释放切换，新增 12 篇关于势函数塑形、稀疏奖励、
反向课程、示范重置、奖励机、延迟信用分配、残差 RL 和 VLA 强化微调的原始论文。文件、
来源和 SHA-256 见[强化学习奖励设计论文索引](rl_reward_design_papers.md)，项目方案见
[OpenVLA 释放切换强化学习奖励设计](../27_openvla_rl_release_reward_design_2026-09-24.md)。

### 动作 chunk 历史条件与时序一致性（2026-09-24）

为核对“加入上一个 chunk 并增加 consistency loss”能否解决高频抖动，新增 RTC、Soft RTC、
SEAM 和 ChunkFlow 四篇原始论文。它们分别覆盖推理时旧 chunk 约束、训练时 action prior、
旧 chunk 尾部引导，以及历史条件、边界损失和连续性约束。文件、适用边界和 SHA-256 见
[动作 chunk 一致性论文索引](chunk_consistency_papers.md)。

### 连续可接受状态与多解动作分布（2026-09-24）

为检验“释放不应只在单个专家状态触发、同一状态也可对应多条正确轨迹”的假设，新增
16 篇关于条件密度、集合值回归、概率校准、可生存状态—动作集合、机器人可行性学习和
生成式策略的原始论文。文件、来源和 SHA-256 见
[连续可行域论文索引](continuous_feasible_distribution_papers.md)，数学对象、可识别性限制和
Panthera 监督后训练对照见
[连续可接受状态与多解动作分布研究](../28_continuous_feasible_state_action_distribution_research_2026-09-24.html)。

## 可直接运行的视觉模型

- [Pose Landmarker Full](models/pose_landmarker_full.task)
- [Gesture Recognizer](models/gesture_recognizer.task)

实现时可把它们复制到 `windows_vision/models/`，并分别命名为 `pose_landmarker.task` 和 `gesture_recognizer.task`。自定义的 `custom_gesture_recognizer.task` 必须用自己的蛇头手势数据训练，因此无法预先下载。

## MediaPipe 文档

- [Pose Landmarker Python](docs/mediapipe_pose_landmarker_python.html)
- [Pose Landmarker 概览](docs/mediapipe_pose_landmarker_overview.html)
- [Gesture Recognizer Python](docs/mediapipe_gesture_recognizer_python.html)
- [Gesture Recognizer 概览](docs/mediapipe_gesture_recognizer_overview.html)
- [自定义 Gesture Recognizer](docs/mediapipe_custom_gesture_recognizer.html)
- [MediaPipe Python 环境](docs/mediapipe_python_setup.html)
- [MediaPipe Holistic 技术文章](docs/mediapipe_holistic_blog.html)

## ROS 2 与 Panthera

- [ROS 2 Humble：Ubuntu 安装](docs/ros2_humble_install_ubuntu.html)
- [ROS 2 Humble：教程目录](docs/ros2_humble_tutorials.html)
- [Python Publisher/Subscriber](docs/ros2_python_pub_sub.html)
- [创建 ROS 2 Package](docs/ros2_create_package.html)
- [Python 参数](docs/ros2_python_parameters.html)
- [rosbag2 录制与回放](docs/ros2_rosbag2.html)
- [Panthera-HT ROS 2 Humble 源码归档](source/Panthera-HT_ROS2-humble.zip)

## 数学、摄像头、通信与系统

- [SciPy Rotation](docs/scipy_rotation.html)
- [SciPy Slerp](docs/scipy_slerp.html)
- [OpenCV 官方 videoio.hpp](source/opencv_videoio.hpp)（`VideoCapture` API 文档的可用替代资料）
- [RFC 6455：WebSocket](standards/RFC6455_WebSocket.txt)
- [Python websockets](docs/python_websockets.html)
- [WSL 安装](docs/microsoft_wsl_install.html)
- [WSL 连接 USB](docs/microsoft_wsl_usb.html)
- [Git 官网快照](docs/git_homepage.html)

## 重新下载

在仓库根目录执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\download_references.ps1
```

脚本保留已经存在的非空文件，只重试缺失项；下载顺序为直连优先，失败后使用 `http://127.0.0.1:7897`。
