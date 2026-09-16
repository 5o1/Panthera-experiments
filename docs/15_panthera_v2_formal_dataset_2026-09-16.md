# Panthera v2 单次抓取正式数据集

日期：2026-09-16

状态：schema 10 正式集已完成并通过自动门禁；未转换 RLDS，未启动训练。

## 1. 数据集结论

正式数据集 `panthera_phone_cylinder_socket_v2_single_grasp_sft_v1` 包含 128 条成功轨迹。
采集器共检查 169 个候选，拒绝 41 个未完成完整任务或规划失败的候选；没有把失败轨迹
混入正式集，也没有用事后筛掉“运动不够平滑”的方式改变成功分布。

分布如下：

| 项目 | 结果 |
| --- | ---: |
| 直立 / 平躺 | 64 / 64 |
| 平躺角度 8 个区间 | 每区间 8 |
| 圆柱左 / 右 | 62 / 66 |
| 凹槽左 / 右 | 63 / 65 |
| 圆柱 / 凹槽空间网格覆盖 | 36 / 36 |
| 单集原始帧数范围 | 947–4415 |

每条轨迹都是单 Panthera、单夹爪、7 维状态/绝对动作、单次抓取。平躺圆柱在空中转正，
然后在槽口直接释放，不放回桌面进行第二次抓取。成功路径不使用 D6 附着、物体瞬移或
墙钟 `sleep`。

## 2. 连续运动修正与验收

schema 9 的全局 TOPP 仍会在高曲率路点附近明显减速。schema 10 改为：先对 Chaikin
几何路径建立三次样条，再以全局单调五次时间律沿弧长执行。直立和平躺路径分别使用
`0.120 rad` 和 `0.200 rad` 的样条控制点分辨率；最终配置名为
`chaikin_arc_length_quintic_single_grasp_v3`。

连续性主门禁读取 HDF5 中的实际 `arm_qpos` 和仿真时间。为了不把全局五次时间律设计中
必然存在的起停减速误报为内部停车，只检查几何进度 5%–95% 的区间。在该定义下，128 条
轨迹的实际近零速比例最大值为 0，最长近零速段为 0 秒，达到 80 ms 的内部停顿为 0。

轨迹重定时的最大关节速度为 `0.500095 rad/s`。名义加速度上限为 `2.0 rad/s²`，审计最大
值为 `2.001620 rad/s²`；这是有限差分与收敛容差造成的 0.081% 数值超差，门禁明确记录
`0.002 rad/s²` 容差，而不是宣称所有样本严格不超过 2.0。

## 3. 采集并行与异步写盘结论

本次正式采集使用物理 GPU1–3，每张卡运行 3 个隔离采集器，共 9 个并行进程。GPU0 未被
本实验使用。采集结束后没有残留的 RoboTwin、采集或编码进程。

后续可以把单集的视频编码和 HDF5 落盘改为异步任务池，但这不是本次正式集的必要条件，
因此没有在已通过的数据上继续更改 I/O 实现。建议实现遵守以下契约：

1. 仿真线程只产生一个完整 episode 的不可变快照和元数据，提交给有界任务池；队列未满
   时立即继续下一集，队列满时才施加背压。
2. 队列同时限制任务数和待写字节数，避免少数长轨迹耗尽内存。编码属于 CPU 密集任务，
   使用独立进程池；不要让多个线程共享同一个 HDF5 句柄或 FFmpeg stdin。
3. 每个任务先写入 `.partial` 临时文件，完成 `flush/fsync`、可重新打开检查和帧数校验后，
   再用同文件系统原子 rename 提交。只有 HDF5、视频、指令和元数据全部成功才计入 episode。
4. worker 异常必须传回主流程并停止生成新任务；正常退出时 drain 队列，异常退出时保留
   可诊断的 partial 清单，不能提前写 `dataset.ok`。
5. 为每个 episode 保存仿真帧序号和时间戳。异步编码可以改变完成顺序，但不能改变 episode
   编号、状态/动作/图像对齐或随机种子映射。

这种设计不会改变训练数据语义，主要风险是内存膨胀、静默丢失 worker 错误、跨 episode
错配和程序退出时的半文件；上述有界背压、不可变快照、原子提交和最终 drain 可分别处理。

## 4. Lab 产物

```text
正式数据集（4.8 GiB，128 HDF5 + 128 MP4）：
/data/lyy/panthera-vla/data/place_randomized_cylinder_in_socket/
  panthera_phone_cylinder_socket_v2_single_grasp_sft_v1/

机器可读门禁与完成标记：
/data/lyy/panthera-vla/.panthera-v2-single-grasp-formal-state/
  dataset-summary.json
  geometric-interior-qpos-audit.json
  dataset.ok

平衡分层抽样的人工审阅视频：
/data/lyy/panthera-vla/reports/dataset-review/
  panthera_phone_cylinder_socket_v2_single_grasp_sft_v1/
  balanced-stratified-10min-2x-1024x768.mp4
```

审阅视频为 H.264、1024×768、30 FPS、608.567 秒、45,022,812 字节；SHA-256 为
`6c4187689405c17ef584d6c36c891acfb444c2a18eaaedc7cfa23378f49494ff`。

被正式门禁否决的 schema 9 数据、状态和分片以 `schema9-rejected-20260916` 后缀保留，
没有删除，也没有混入 schema 10 正式集。

## 5. 当前边界

该数据集完成的是 oracle 示范数据生成，不代表 VLA 已训练或闭环通过。下一步必须先由人
审阅 10 分钟视频，再单独决定是否转换 RLDS 和开始新的训练；既有 OpenVLA 失败分析与
真机门禁仍然有效。
