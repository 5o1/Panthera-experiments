# Panthera ROS 2 话题清单

## 1. 当前最短控制链路（阶段 0）

本项目当前最小可运行链路是：

```text
panthera_arm_control
  -> /end_pose_euler
  -> /arm_status

manual_target_node
  -> /teleop/manual_target_preview
  -> /pos_cmd（仅在 send_robot_command=true 且安全检查通过时发布）
```

### 1.1 官方/驱动端发布

- `/end_pose_euler`
  - 类型：`panthera_interfaces/msg/EndPoseEuler`
  - 含义：末端当前位姿反馈（当前机器人位置）
  - 由：官方机械臂控制节点发布

- `/arm_status`
  - 类型：`panthera_interfaces/msg/ArmStatus`
  - 含义：机械臂使能、空闲/忙碌、故障、限位状态等
  - 由：官方机械臂控制节点发布

### 1.2 本地节点发布

- `/teleop/manual_target_preview`
  - 类型：`panthera_interfaces/msg/PosCmd`
  - 含义：用于 dry-run/预览，检查目标是否正确，但不会控制机械臂
  - 由：`manual_target_node` 发布

- `/pos_cmd`
  - 类型：`panthera_interfaces/msg/PosCmd`
  - 含义：控制机械臂移动到某个目标位置
  - 由：`manual_target_node` 条件性发布
  - 只有在 `send_robot_command=true` 且状态安全检查通过后才会发布

---

## 2. 本地视觉遥操作链路（后续阶段）

在后续阶段，视觉节点会发布这些 topic：

- `/teleop/human_pose`
  - 类型：`geometry_msgs/msg/PoseStamped`
  - 含义：人体关键点形成的位姿
  - 由：`vision_bridge_node` 发布

- `/teleop/pose_score`
  - 类型：`std_msgs/msg/Float32`
  - 含义：人体姿态质量分数
  - 由：`vision_bridge_node` 发布

- `/teleop/gesture`
  - 类型：`std_msgs/msg/String`
  - 含义：识别出的手势名称
  - 由：`vision_bridge_node` 发布

- `/teleop/gesture_score`
  - 类型：`std_msgs/msg/Float32`
  - 含义：手势识别置信度
  - 由：`vision_bridge_node` 发布

- `/teleop/enabled`
  - 类型：`std_msgs/msg/Bool`
  - 含义：视觉遥操作是否有效启用
  - 由：`vision_bridge_node` 发布

- `/teleop/recalibrate`
  - 类型：`std_msgs/msg/Bool`
  - 含义：是否需要重新标定
  - 由：`vision_bridge_node` 发布

---

## 3. 代码中对应位置

### 本地声明/订阅的 topic

见：`ros2_ws/src/panthera_vision_teleop/panthera_vision_teleop/manual_target_node.py`

```python
self.declare_parameter("preview_topic", "/teleop/manual_target_preview")
self.declare_parameter("robot_command_topic", "/pos_cmd")
self.declare_parameter("end_pose_topic", "/end_pose_euler")
self.declare_parameter("arm_status_topic", "/arm_status")
```

以及：

```python
self.end_pose_subscription = self.create_subscription(
    EndPoseEuler, self.end_pose_topic, self.end_pose_callback, 10
)
self.arm_status_subscription = self.create_subscription(
    ArmStatus, self.arm_status_topic, self.arm_status_callback, 10
)
```

这表示：
- 这个节点订阅 `/end_pose_euler` 和 `/arm_status`
- 它自己不负责发布这两个 topic

### 本地发布的 topic

```python
self.preview_publisher.publish(target)
```

对应：
- `/teleop/manual_target_preview`

```python
self.robot_command_publisher.publish(target)
```

对应：
- `/pos_cmd`

---

## 4. 结论

在当前阶段最核心的官方 topic 是：

- `/end_pose_euler`
- `/arm_status`

在当前阶段最核心的本地发布 topic 是：

- `/teleop/manual_target_preview`
- `/pos_cmd`

如果你要做调试，最直接的查看方法是：

```bash
ros2 topic list
ros2 topic echo /end_pose_euler --once
ros2 topic echo /arm_status --once
ros2 topic echo /teleop/manual_target_preview --once
ros2 topic echo /pos_cmd --once
```
