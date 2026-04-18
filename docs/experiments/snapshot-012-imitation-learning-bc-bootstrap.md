# SNAPSHOT-012: Imitation Learning / Behavior Cloning Bootstrap

- **日期**: 2026-04-13
- **负责人**:
- **状态**: 第一阶段已完成；第二阶段转入 [SNAPSHOT-015](snapshot-015-behavior-cloning-team-bootstrap.md)

## 1. 动机

当前 shaping 主线的主要问题不是“完全学不会”，而是：

- PPO from scratch 能学到一些合理行为
- 但上限很难继续抬高
- 继续单靠 reward shaping 调参，收益已经递减

同时，仓库里存在一个最值得利用但此前一直未系统利用的资源：

- [ceia_baseline_agent](../../ceia_baseline_agent)

它既是我们当前的主要评分对手，也是一个可无限查询的 teacher policy。

## 2. 假设

如果我们先从 baseline teacher 采集大量 `(obs, action)` 监督样本，再做 behavior cloning 预训练，那么：

1. 可以更快得到一个至少接近 baseline 行为分布的初始策略；
2. 然后再接 PPO fine-tune 时，训练目标会比纯 scratch 更对齐；
3. 这条线比继续只改 shaping 更有希望把上限拉高。

## 3. 第一阶段范围

本 snapshot 先只做第一阶段：

1. 搭建 baseline 轨迹采集脚手架；
2. 明确数据格式；
3. 完成一次小规模 smoke 采样，确认数据可用。

本轮**不**在这里直接落 BC trainer，本 snapshot 先把 “teacher data pipeline” 做稳。
BC trainer、deployment wrapper 与 batch 的实际落地，已转入 [SNAPSHOT-015](snapshot-015-behavior-cloning-team-bootstrap.md)。

## 4. 数据设计

### 4.1 主要采样模式

优先采：

- `baseline vs baseline`

作为 teacher 数据主来源。

原因：

- 两边都由 teacher 决策
- 样本最稳定
- 不会掺进 random 的低质量动作

### 4.2 数据粒度

默认以 `team-level` 为主：

- team observation = 我方两个球员 observation 拼接
- team action = 我方两个球员 action 拼接

原因：

- 当前主线最稳定的 shaping PPO 也是 team-level
- 后续最容易直接对接当前 `trained_team_ray_agent` 风格

同时保留 `player-level` 采样选项，作为后续对比或单玩家 imitation 的备用。

### 4.3 当前交付物

脚手架脚本：

- [collect_baseline_trajectories.py](../../scripts/tools/collect_baseline_trajectories.py)

预期输出：

- shard 化的 `.npz`
- `manifest.json`
- 明确记录：
  - obs 维度
  - action 维度
  - 采样模式（team / player）
  - episodes / steps / saved samples

当前已落地：

- [collect_baseline_trajectories.py](../../scripts/tools/collect_baseline_trajectories.py)
- baseline self-play team 模式 smoke 产物：
  - [smoke_baseline_selfplay_team](artifacts/imitation/smoke_baseline_selfplay_team)
  - [manifest.json](artifacts/imitation/smoke_baseline_selfplay_team/manifest.json)
- baseline self-play team 正式 teacher dataset：
  - [baseline_selfplay_team_20260413_022138](artifacts/imitation/baseline_selfplay_team_20260413_022138)
  - [manifest.json](artifacts/imitation/baseline_selfplay_team_20260413_022138/manifest.json)
- 标准批处理入口：
  - [collect_baseline_selfplay_team.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_collect_baseline_selfplay_team.batch)

工程修正：

- 采集脚本现在会周期性打印进度，并增量写出 `manifest.partial.json`
- 这样长时间 baseline self-play 收集任务不再表现为“只有 shard 在默默增长、收尾前完全不可见”

## 5. 第一阶段 smoke 命令

示例 smoke：

```bash
python scripts/tools/collect_baseline_trajectories.py \
  --team0-module ceia_baseline_agent \
  --team1-module ceia_baseline_agent \
  --mode team \
  --episodes 10 \
  --max-steps 400 \
  --base-port 61205 \
  --save-dir docs/experiments/artifacts/imitation/smoke_baseline_selfplay_team
```

## 6. 成功判据

第一阶段只看数据链路，不看最终胜率。

通过标准：

1. 能稳定采出 team-level 轨迹；
2. `obs/action` shape 一致；
3. manifest 与 shard 数量、样本数能对上；
4. 能确认数据能直接喂给后续 BC 训练脚本。

## 7. 后续阶段

第二阶段才进入：

- BC trainer
- BC checkpoint 导出
- baseline 500 评估
- PPO fine-tune

当前状态更新：

- 第一阶段（teacher data pipeline）已完成
- 第二阶段（BC trainer / wrapper）见 [SNAPSHOT-015](snapshot-015-behavior-cloning-team-bootstrap.md)

## 8. 当前判断

这条线的关键价值不是“替代 RL”，而是：

- 用 teacher policy 给出一个真正和目标对齐的强初始化

如果这步打通，它会比继续做 scratch-only PPO 更有信息量。
