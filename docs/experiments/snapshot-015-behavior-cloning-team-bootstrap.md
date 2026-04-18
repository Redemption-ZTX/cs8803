# SNAPSHOT-015: Behavior Cloning Team Bootstrap

- **日期**: 2026-04-13
- **负责人**:
- **状态**: 已完成（首轮 formal BC train）

## 1. 背景

[SNAPSHOT-012](snapshot-012-imitation-learning-bc-bootstrap.md) 完成了 baseline self-play teacher 数据采集链路，但当时还没有真正落下可训练、可部署的 BC trainer。

到当前为止：

- teacher dataset 已完整采集
- team-level 数据格式已确认稳定
- 下一步不该再停在数据层，而应真正把 `BC team policy` 训练闭环打通

## 2. 本轮目标

本轮只做最小版 BC 闭环：

1. 读取 baseline self-play team-level dataset
2. 训练一个 team-level supervised MLP policy
3. 输出可直接部署的 BC checkpoint
4. 提供一个 team-level deployment wrapper，供官方 evaluator 使用

本轮先**不**做：

- BC -> PPO / MAPPO fine-tune
- MARWIL
- dataset augmentation
- ensemble

## 3. 数据来源

当前默认 teacher dataset：

- [baseline_selfplay_team_20260413_022138](artifacts/imitation/baseline_selfplay_team_20260413_022138)
- [manifest.json](artifacts/imitation/baseline_selfplay_team_20260413_022138/manifest.json)

关键规模：

- `episodes = 5000`
- `total_env_steps = 286839`
- `team total_samples = 573678`
- `team shards = 12`
- 单样本 shape：
  - `obs = (672,)`
  - `action = (6,)`

## 4. 最小实现

### 4.1 模型

- [imitation_bc.py](../../cs8803drl/branches/imitation_bc.py)

定义：

- team obs MLP backbone
- 六个动作分支各自一个分类 head
- loss = 六个分支交叉熵平均

### 4.2 训练入口

- [train_bc_team_policy.py](../../cs8803drl/training/train_bc_team_policy.py)

功能：

- 读取 shard 化 `.npz`
- 按 `episode + side` 分组切 train / val
- 输出：
  - `progress.csv`
  - `training_curve.png`
  - `checkpoint_XXXXXX/`
  - `training_summary.json`

### 4.3 部署 wrapper

- [trained_bc_team_agent.py](../../cs8803drl/deployment/trained_bc_team_agent.py)

约定：

- 优先读取 `TRAINED_RAY_CHECKPOINT`
- 也兼容 `TRAINED_BC_CHECKPOINT`
- 因此可以直接接现有 `evaluate_official_suite.py`

### 4.4 标准 batch

- [soccerstwos_h100_cpu32_bc_team_baseline_selfplay_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_bc_team_baseline_selfplay_512x512.batch)

## 5. 成功判据

第一轮通过标准：

1. BC 训练能稳定读完整个 dataset
2. run 目录能正常生成 checkpoint / curve / summary
3. deployment wrapper 能正常加载 BC checkpoint
4. 能接上官方 evaluator 做后续 baseline 复核

当前验证状态：

- `py_compile` 已通过
- batch `bash -n` 已通过
- smoke train 已通过：
  - [BC_smoke_20260413_032513](../../ray_results/BC_smoke_20260413_032513)
  - [training_summary.json](../../ray_results/BC_smoke_20260413_032513/training_summary.json)
- smoke checkpoint reload 已通过
- formal team-BC train 已完成：
  - [BC_team_baseline_selfplay_512x512_20260413_033033](../../ray_results/BC_team_baseline_selfplay_512x512_20260413_033033)
  - [training_summary.json](../../ray_results/BC_team_baseline_selfplay_512x512_20260413_033033/training_summary.json)
  - [training_curve.png](../../ray_results/BC_team_baseline_selfplay_512x512_20260413_033033/training_curve.png)
  - [checkpoint_000030](../../ray_results/BC_team_baseline_selfplay_512x512_20260413_033033/checkpoint_000030)
- formal run 关键读法：
  - `best_epoch = 30`
  - `best_val_exact_match = 0.1242`
  - `best/final checkpoint = checkpoint_000030`

这意味着 `015` 已经不再只是 smoke-only。到当前为止，team-level BC trainer、checkpoint 产物和 deployment wrapper 都已经有了正式可复用资产；后续 `BC -> PPO` 的主线推进转入 [SNAPSHOT-028](snapshot-028-team-level-bc-to-ppo-bootstrap.md)。

## 6. 与总路线的关系

这条线对应 [PLAN-002](../plan/plan-002-il-mappo-dual-mainline.md) 的 Mainline-A：

- teacher data pipeline 已打通
- 本轮开始正式进入 `BC bootstrap`

后续下一阶段才进入：

- `BC -> PPO`
- `BC -> MAPPO`
- 与 scratch PPO / MAPPO 的公平比较
