# Panthera 随机圆柱入槽 v2 左右对称 Pilot 报告

日期：2026-09-16

状态：**已否决**。空间分布合格，但人工审阅发现抓取后搬运存在周期性停顿；本数据不得进入
训练。替代结果见
[v2 连续轨迹 Pilot 报告](13_panthera_v2_smooth_dataset_pilot_2026-09-16.md)。

否决原因不是场景采样，而是 oracle 把长距离搬运拆成 20–35 mm 的多个独立 `move()`；每段
轨迹都从零速起步并在零速结束，导致夹住圆柱后反复减速、停止、再加速。2 倍速审查视频会
放大观感，但不是根因。

## 1. 重做原因与修正

旧 pilot 的极角范围是 `90°..175°`。按
`x = base_x + radius * cos(angle)` 计算时，所有样本必然满足 `x <= base_x`，所以物体
总在机械臂单侧。旧数据已明确否决并保留作审计，不进入本轮数据。

重做后采用关于机械臂正前方对称的 `5°..175°` 半圆扇区。圆柱和凹槽分别独立分层采样，
位置半径仍为任务实测有效半径 `0.46 m` 的 75%，即 `0.22..0.345 m`。相机中心从
`x=-0.25 m` 移到 `x=0 m`，使左右工作区具有相同构图条件。平躺圆柱使用的转正、重抓和
高位抬升工位也根据圆柱所在侧镜像，侧向腕部姿态在左侧使用 180°、右侧使用 0°。

## 2. 数据契约与分布

| 项目 | 结果 |
| --- | --- |
| 成功 episode | 64 |
| 圆柱左右分布 | 左 32 / 右 32 |
| 凹槽左右分布 | 左 31 / 右 33 |
| 初始姿态 | 直立 32 / 平躺 32 |
| 平躺方向 | 8 个角度区间各 4 条 |
| 圆柱位置网格覆盖 | 33 / 36 |
| 凹槽位置网格覆盖 | 33 / 36 |
| 单集原始帧数 | 975–3047 |
| observation/action | 单臂、有限值、均为 7 维 |
| D6 附着或物体瞬移 | 未使用 |
| 成功路径墙钟 `sleep` | 未使用 |
| 训练是否启动 | 否 |

修正后的采集日志记录了 22 次失败尝试，各分片依次为
`[2, 0, 0, 0, 0, 2, 0, 3, 1, 1, 4, 9]`。64 条成功轨迹来自 86 次候选执行，候选成功率
约为 `74.4%`。失败轨迹没有混入数据集，但正式扩大数据时仍应保留失败类型统计。

## 3. 权威人工审阅视频

```text
/data/lyy/panthera-vla/reports/dataset-review/
  panthera_phone_cylinder_socket_v2_symmetric_pilot/
  balanced-stratified-10min-2x-1024x768.mp4
```

视频由 16 条完整轨迹拼接，以 2 倍速播放，实际时长 `635.434 s`，分辨率
`1024x768`，大小 `37,247,211 B`。抽样左右各 8 条，包含直立 4 条、平躺 12 条，并覆盖
全部 8 个平躺角度区间。SHA-256：

```text
141530f82a19b4a50b952131f670a9dafcd9471ec8baec408c585ade57de6c65
```

同目录下没有 `balanced-` 前缀的视频来自较早的随机补足抽样，虽然使用同一合格数据集，
但抽样为左 12 / 右 7，不作为本轮权威验收视频。

## 4. Lab 产物

```text
数据集：
/data/lyy/panthera-vla/data/place_randomized_cylinder_in_socket/
  panthera_phone_cylinder_socket_v2_symmetric_pilot/

机器可读摘要与日志：
/data/lyy/panthera-vla/.panthera-v2-symmetric-pilot-state/

左右平衡视频抽样清单：
/data/lyy/panthera-vla/.panthera-v2-symmetric-pilot-state/
  review-selection-balanced-stratified.json
```

数据目录约 `2.8 GiB`。采集、Ray 和编码进程均已退出，GPU1–3 已释放。当前停在人工审阅
门禁；用户确认轨迹行为前，不转换 RLDS、不扩大正式数据，也不启动训练。
