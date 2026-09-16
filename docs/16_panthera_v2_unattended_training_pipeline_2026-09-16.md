# Panthera v2 无人值守训练流水线

日期：2026-09-16

状态：用户已审阅 schema 10 的 10 分钟视频并批准进入训练；无人值守流水线已经在 Lab
启动。启动时 PID 为 `1577249`，运行日志位于
`/data/lyy/panthera-vla/.panthera-v2-schema10-unattended-state/`。

## 1. 设计目标

该流水线不以 coding agent 或操作者在线为前提。一个 supervisor 从已完成标记恢复，连续
执行完整阶段；它只在数据/质量门不通过时停止，不在训练途中观察短期曲线、修改参数或做
超参数搜索。

正式训练是单次连续 10,000 步任务，不拆成若干需要人工确认的小段。普通用户 crontab 已
安装 `@reboot` 恢复入口。机器重启后会跳过已有成功标记并继续后续阶段；被中断的 TFDS
或训练输出会先以时间戳可恢复归档，再从该完整阶段重新执行。开发集或最终集低于质量门后
属于终态，重启不会反复重跑实验。

## 2. 自动阶段

1. 8 个有界 CPU worker 并行逐集验证 128 个 HDF5、指令和 MP4；检查视频帧数与 HDF5
   帧数一致。
2. 转换为独立数据集 `panthera_phone_cylinder_socket_v2`、TFDS/RLDS `4.0.0`。验证集固定
   为 16 条：8 条直立，8 个平躺角度区间各 1 条，同时贪心平衡左右侧与空间格。
3. 用真实 OpenVLA 数据管线验证 `(25, 7)` 动作窗口、`(1, 7)` proprio 与
   `(1, 224, 224, 3)` RGB。
4. 静态解析三卡闭环评测配置，不启动仿真。
5. 在 GPU1 运行一步优化器 smoke。
6. 从基础模型重新建立任务动作头，在 GPU1–3 进行一次连续 10,000 步 DDP SFT。固定配置
   为 25×7 动作窗口、L1 目标、每卡 batch 1、学习率 `5e-5`、500 步 warmup、8000 步后
   衰减。不会继承旧竖直任务中已经观察到固定点的动作头。
7. 完整训练结束后，在 GPU1–3 并行运行 18 条开发轨迹，每卡 6 条；执行视野固定 20，
   episode 上限 1600 个动作步，不启用末端辅助。成功率至少 75% 才进入下一阶段。
8. 仅在开发门通过后，才用另一组 18 个固定 seed 做一次最终闭环评测。

GPU0 不属于本实验。媒体审计和 RLDS 转换是 CPU/磁盘任务，刻意不占 GPU；训练与闭环
评测才使用 GPU1–3。GPU 暂时被其他任务占用时，supervisor 最长等待 24 小时，而不是失败
后等待人工重启。

## 3. 脚本入口

```text
主 supervisor：
/data/lyy/panthera-vla/run_lab_panthera_v2_unattended_pipeline.sh

后台启动器：
/data/lyy/panthera-vla/start_lab_panthera_v2_unattended_pipeline.sh

重启恢复安装器：
/data/lyy/panthera-vla/install_lab_panthera_v2_autoresume.sh

可选只读状态入口：
/data/lyy/panthera-vla/status_lab_panthera_v2_unattended_pipeline.sh
```

阶段脚本分别为：

- `run_lab_panthera_v2_media_audit.sh`
- `run_lab_panthera_v2_sft_rlds.sh`
- `run_lab_panthera_v2_eval_config_smoke.sh`
- `run_lab_openvla_panthera_v2_sft_step_smoke.sh`
- `run_lab_openvla_panthera_v2_sft_10k.sh`
- `run_lab_panthera_v2_policy_eval.sh dev|final`

所有阶段使用独立状态目录、锁、日志和原子 JSON 摘要。`pipeline-summary.json` 的终态可能为
`passed`、`stopped_below_dev_threshold`、`stopped_below_final_threshold` 或 `failed`。

## 4. 安全边界

流水线只运行 Lab 仿真、数据转换和离线训练。它不会共享 WSL USB 设备、启动 Panthera
官方驱动、发布真机命令或越过 sim-to-real 门禁。开发集未通过时不会运行 final；final
通过也只代表仿真门通过，仍需重新安装并标定真实相机、恢复实验台和完成 shadow mode。
