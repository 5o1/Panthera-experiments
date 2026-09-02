# 无需真机的软件验收

截至 2026-09-01，下面命令构成当前软件基线：

```bash
./tools/preflight_panthera.sh mock

source /opt/ros/humble/setup.bash
source ~/Panthera_HT_ROS2/install/setup.bash
cd ~/panthera/ros2_ws
colcon build --packages-select panthera_vision_teleop --symlink-install
colcon test --packages-select panthera_vision_teleop
colcon test-result --verbose

cd ~/panthera
./tools/test_process_cleanup.sh
./tools/test_vision_pipeline_mock.sh
./tools/test_blocking_scheduler_mock.sh
./tools/test_continuous_control_mock.sh
./tools/test_continuous_streaming_mock.sh
```

覆盖范围：协议畸形输入和 backlog、显式 re-arm 状态机、位置/旋转速度和加速度、逐轴迟滞、
深度 clutch、夹爪 neutral/保持/冷却、URDF FK/Jacobian/IK、关节顺序/上下限/速率/stale
guard、GenericSystem 标准轨迹成功/部分目标拒绝/action cancel/保持、轻量笛卡尔 backend
到达和 stale 保持、连续 IK 失败锁定、`/teleop/reset_backend` 显式恢复、WebSocket 到 HUD
的完整链路、dry-run 无 `/pos_cmd` publisher，以及进程组清理幂等性。action 超时、拒绝、
异常状态和关节跟踪误差共用同一失败锁；stale 导致的预期 cancel 不计为故障。

软件 PASS 不能证明摄像头噪声、机械臂方向、串口周期、真实制动或物理急停有效。所有这些
仍必须按 [真机实验检查表](04_real_hardware_experiment_checklist.md) 逐级验证。
