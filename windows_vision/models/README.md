# MediaPipe 模型目录

编码前把本地已下载模型复制到本目录：

```text
docs/references/models/pose_landmarker_full.task
  -> windows_vision/models/pose_landmarker.task

docs/references/models/gesture_recognizer.task
  -> windows_vision/models/gesture_recognizer.task
```

完成蛇头手势训练后再放入：

```text
windows_vision/models/custom_gesture_recognizer.task
```

模型文件体积较大，已由根目录 `.gitignore` 忽略。本地模型来源、校验值和下载 URL 见：

```text
docs/references/README.md
docs/references/download-log.csv
```
