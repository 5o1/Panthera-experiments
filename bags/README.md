# rosbag2 输出目录

这个目录用于保存 observation-action 实验轨迹，不保存源码。推荐记录的话题和安全回放要求见：

```text
docs/Panthera-HT 单目视觉手臂与手势遥操作实验计划.md（阶段 7）
docs/references/docs/ros2_rosbag2.html
```

每次录制使用独立子目录，并在名称中写日期和实验条件。回放前必须保持真机命令发布关闭，避免历史目标重新驱动机械臂。实际 bag 子目录已被 `.gitignore` 忽略。
