# Panthera-HT 实验资料本地索引

下载日期：2026-08-28

本目录保存《Panthera-HT 单目视觉手臂与手势遥操作实验计划》直接依赖的论文、官方技术文档、模型和源码归档。共尝试 27 项，成功 26 项，失败 1 项；每个文件的原始 URL、字节数和 SHA-256 见 [download-log.csv](download-log.csv)，失败原因见 [FAILED_DOWNLOADS.md](FAILED_DOWNLOADS.md)。

HTML 文件是单页源码快照，正文和代码示例可离线阅读；页面样式、图片或站内跳转仍可能访问在线资源。

## 论文

- [BlazePose: On-device Real-time Body Pose Tracking](papers/BlazePose_2006.10204.pdf)
- [MediaPipe Hands: On-device Real-time Hand Tracking](papers/MediaPipe_Hands_2006.10214.pdf)

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
