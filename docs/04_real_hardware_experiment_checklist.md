# Panthera 真机实验检查表

本文件定义实验顺序并记录已完成的真机证据。每次只运行一个驱动：`arm_control_node`
与 `ros2_control_node` 绝对不能同时存在。

## 1. 每次上电后的共同检查

```bash
cd ~/panthera
./tools/preflight_panthera.sh robot       # 视觉真机实验
./tools/preflight_panthera.sh hardware    # 不使用摄像头的控制器实验
```

必须全部 PASS，并人工确认：机械臂固定可靠、工作空间无人无障碍、线缆余量足够、物理急停
触手可及、操作者不需要双手同时打字。预检是只读的，不会打开串口。

出现串口缺失时，在 Windows 管理员 PowerShell 运行 `usbipd list`，确认通信板为
Attached；Shared 但未 Attached 时执行 `usbipd attach --wsl --busid BUSID`。不得用
`chmod 777` 代替 dialout/video 组权限。

## 2. 视觉链路顺序

1. `./run_vision_teleop_demo.sh synthetic safe`，确认 HUD、stale 锁和退出无残留。
2. `./run_vision_teleop_demo.sh camera safe`，录制静止、左右、上下各 10 秒。
3. 分别运行 camera `normal`、`depth_world`、`orientation_roll`、`orientation_pitch`、
   `orientation_yaw`；这些档位被脚本禁止用于 robot。
4. 用 `./tools/record_vision_teleop.sh LABEL` 录制，再用
   `./tools/analyze_vision_teleop.py bags/BAG_DIR` 生成 JSON/CSV/PNG。
5. 只有 safe 的方向、峰峰噪声、限幅和 stale 行为满足 TODO 验收值后，才运行
   `./run_vision_teleop_demo.sh robot safe`。夹爪始终保持关闭。

实际采集优先使用长流程单例，不要逐阶段反复运行上面的底层命令：

```bash
./tools/run_data_collection_case.sh camera_characterization
./tools/run_data_collection_case.sh robot_yz_acceptance
```

它们各自只启动一个 bag，并按有效视觉帧和 phase marker 推进，详见
[连续数据采集单例 Case 设计](06_data_collection_cases.md)。

2026-09-02，`robot_yz_acceptance` 的五个阶段已用 `safe` profile 完成。66 条真实
`/pos_cmd` 全部位于 `WARMUP/RECORDING` marker 内，等待阶段没有命令；static 与 stop-hold
阶段均为 0 条命令，stop-hold 的 Y/Z 反馈峰峰值仅 0.093/0.241 mm。lateral、vertical、
YZ combined 的目标范围与方向均符合预期。正常退出调用全零 positionpark 并返回成功，
退出后无残留进程。完整记录：
[`bags/robot_yz_acceptance_20260902_004734/experiment_status.md`](../bags/robot_yz_acceptance_20260902_004734/experiment_status.md)。
录包末帧第三关节仍有约 0.035 rad 残差，因此扩大增益前还要给退出路径增加反馈容差审计；
X、姿态和夹爪仍保持关闭。

## 3. 连续 ros2_control 候选后端

先确认 `./tools/test_continuous_control_mock.sh` PASS。真机阶段必须严格逐次执行，每次结束
后审查日志，不能在一次会话中跳级：

```bash
./tools/run_continuous_hardware_experiment.sh hold
./tools/run_continuous_hardware_experiment.sh micro
./tools/run_continuous_hardware_experiment.sh cancel
./tools/run_continuous_hardware_experiment.sh suite   # 三阶段同一驱动/同一 bag，推荐
./tools/run_continuous_hardware_experiment.sh remaining # hold 已通过后，合并 micro + cancel
```

- `hold`：发送当前六关节位置，允许误差不超过 0.01 rad。
- `micro`：joint1 慢速往返 0.005 rad，其余关节保持。
- `cancel`：joint1 规划 0.015 rad/6 s，0.5 s 后取消，取消后 0.5 s 漂移不超过 0.003 rad，
  再返回起点。

launch 使用项目内 position-only controller overlay，并要求独立确认词。它不会启动视觉。
`suite`/`remaining` 只要求一次总确认；后续阶段仅在前一阶段 PASS 后自动继续，任一失败立即
终止并保留同一个 bag。
官方 hardware plugin 的 `on_deactivate()` 尚未显式 stop，这是当前真机否决点之一；任何
阶段出现继续运动、串口错误、关节顺序/方向错误或取消不保持，立即急停并放弃该路线。

2026-09-01 首次 `hold` 尝试在 action 发送前中止：CANboard 报告 7 个电机全部断连，SDK
以 `999` 作为无效位置，但官方 hardware plugin 仍返回 configure/activate 成功并进入写
循环；SDK 关节限位拒绝了这些写入。项目入口现会识别电机断连、`[999,...]`、串口错误，
探针也会验证初始六关节反馈全部位于审计限位内。确认电机电源/CAN 连接恢复前，禁止重试
`hold`，更不能进入 `micro`、`cancel` 或 `suite`。

同日盒子上电后重新验证：7 个电机连接正常（v4.7.3），初始六关节反馈通过限位检查，
`arm_controller` 接受当前位置目标并在约 1 秒后返回成功；探针验证最大关节漂移不超过
0.01 rad。PASS bag：`bags/hardware_hold_20260901_235340`，launch 日志：
`bags/hardware_hold_20260901_235340.launch.log`。`hold` 已通过，但 `micro`、`cancel` 和
deactivate 主动 stop 仍未验证。

同日合并 `remaining` case 完成：micro 的 joint1 `0.005 rad` 三秒偏移与三秒返回均成功；
cancel 的 `0.015 rad / 6 s` 目标在约 0.5 秒收到取消，取消状态与 0.5 秒保持检查通过，随后
三秒返回起点成功。四个 START/PASS marker、1248 条 `/joint_states` 与 457 条 controller
state 保存在 `bags/hardware_remaining_20260901_235632`。`hold/micro/cancel` 至此均通过；
尚未验证的是故障注入、长时间稳定性及 hardware plugin 的 deactivate 主动 stop。

上述底层验收完成后，不再安排逐档速度实验。视觉连续控制只运行一次最终长流程：

```bash
./tools/run_data_collection_case.sh robot_yz_fast_response
```

该 case 使用 Host 基线（0.6 rad/s 常用速度、1.0 rad/s 硬件上限、2.0 rad/s² 加速度、
200 Hz 非阻塞硬件循环），一次确认后连续录制自然速度 Y/Z 动作和停止保持。

## 4. 故障与回退

- 正常视觉 Esc 会尝试返回 positionpark；Ctrl+C/异常退出不会追加未知环境中的停放轨迹。
- 环境确认安全后，可单独执行 `./tools/park_panthera.sh`，它有设备预检和确认词。
- SDK/串口崩溃后先在 Windows 重新 attach USB，再确认没有残留驱动；不要同时反复启动。
- 真机出现任何与预期不一致的行为时，保留 `log/`、`~/.ros/log/` 和 rosbag，不要为了
  “继续测试”放宽限位或绕过确认。

## 5. 每次实验记录模板

```text
日期/操作者：
Git commit 或 git diff 摘要：
实验入口与 profile：
USB BUSID、/dev/ttyACM 数量：
上电初始关节/末端位姿：
目标变化量、持续时间：
实际峰值/最终误差/停止延迟：
是否触发 clipped、rate-limited、stale、fault：
退出方式与 park 结果：
日志/rosbag 路径：
PASS/FAIL 及下一步：
```
