"""蛇头开合手势图像采集工具（待实现）。

这个文件应该做什么：
1. 用 OpenCV 读取摄像头并显示当前类别、已保存数量和操作按键。
2. 将图像分别保存到 gesture_dataset/none、snake_open、snake_closed。
3. 支持单张采集和低频连续采集，避免保存大量几乎相同的相邻帧。
4. 文件名应包含类别、录制批次和递增编号，方便按“录制批次”切分数据集。
5. 不在这里训练模型，也不发送 ROS 2 或机械臂命令。

数据要求：
- 第一轮每类 300-500 张。
- 至少覆盖 3 种背景、2 种光照、不同距离和轻微手腕旋转。
- none 必须包含普通张手、握拳、指向、半开等容易混淆的负样本。
- 训练/验证/测试应按录制批次拆分，不能随机打散同一段视频的相邻帧。

本地参考资料：
- docs/Panthera-HT 单目视觉手臂与手势遥操作实验计划.md：第 7.2 节。
- docs/references/docs/mediapipe_custom_gesture_recognizer.html
- docs/references/source/opencv_videoio.hpp

完成标准：生成符合 MediaPipe Model Maker 目录约定、可追溯录制批次的数据集。
请自行设计按键、参数解析、目录检查、图像质量提示和 main()。
"""
