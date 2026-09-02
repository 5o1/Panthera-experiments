# WSL 单目视觉进程

目录名 `windows_vision` 是早期 Windows/WSL 分工留下的；当前 `vision_sender.py` 在 WSL 的 `.venv-wsl-vision` 中运行。它只完成：

```text
/dev/video* RGB 图像
→ Pose Landmarker（右肩 12、右肘 14、右腕 16）
→ Gesture Recognizer（右手 Open_Palm/Closed_Fist）
→ WebSocket JSON
```

它不导入 ROS、不打开串口，也不能直接控制 Panthera。

单独查看摄像头推理：

```bash
cd ~/panthera
.venv-wsl-vision/bin/python windows_vision/vision_sender.py
```

bridge 不在线时程序会继续显示预览并每秒尝试重连。Space 切换 enable，R 让下一包携带 recalibrate，Esc 退出。模型默认从 `docs/references/models/` 读取。

WSL/usbipd 下默认强制使用 `MJPG`；这台集成摄像头用 YUYV 会返回绿色坏帧。若画面接近全黑，先检查笔记本摄像头的物理隐私滑块，而不是调映射参数。

不使用摄像头测试协议：

```bash
.venv-wsl-vision/bin/python windows_vision/vision_sender.py \
  --synthetic --duration 10
```

`capture_gesture_dataset.py` 仍是第二阶段自定义 `snake_open/snake_closed` 数据采集练习，不影响预训练手势 Demo。
