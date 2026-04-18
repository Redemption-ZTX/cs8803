# SNAPSHOT-008: Starter-Aligned Base Model Lane

- **日期**: 2026-04-11
- **负责人**:
- **目标**:
  - 停止混用 base model 训练与下游 adaptation 训练
  - 回到 starter 语义，建立 scratch-only 的基础模型训练入口
  - 用干净对照回答“哪一个 starter-aligned base model 最强”
- **配置**:
  - Base-A 训练入口: [train_ray_base_team_vs_random.py](../../cs8803drl/training/train_ray_base_team_vs_random.py)
  - Base-B 训练入口: [train_ray_base_team_vs_random.py](../../cs8803drl/training/train_ray_base_team_vs_random.py)
  - Base-C 训练入口: [train_ray_base_ma_teams.py](../../cs8803drl/training/train_ray_base_ma_teams.py)
  - Base-A batch: [soccerstwos_h100_cpu32_base_team_vs_random_512x512.batch](../../scripts/batch/base/soccerstwos_h100_cpu32_base_team_vs_random_512x512.batch)
  - Base-B batch: [soccerstwos_h100_cpu32_base_team_vs_random_512x256.batch](../../scripts/batch/base/soccerstwos_h100_cpu32_base_team_vs_random_512x256.batch)
  - Base-C batch: [soccerstwos_h100_cpu32_base_ma_teams.batch](../../scripts/batch/base/soccerstwos_h100_cpu32_base_ma_teams.batch)
  - 训练内筛选口径: `50` 局
  - 最终确认口径: 官方 `200` 局

## 1. 决策背景

此前多轮实验混入了以下问题：

- 把 base checkpoint 构建与 downstream adaptation 混在一起
- 同时改模型宽度、初始化方式和队友分布，导致实验不可解释
- 试图用 reward/loss 曲线去解释一个与最终对战胜率明显脱钩的训练目标

本轮重置后的原则是：

- base model 必须是 `scratch`
- base lane 不允许 `WARMSTART_CHECKPOINT` 或 `RESTORE_CHECKPOINT`
- 先用 starter 对齐的基础任务找最强 base checkpoint
- 然后才允许进入 reward / observation / architecture modification lane

## 2. Base Lane 设计

### Base-A

- 任务语义对齐 [example_ray_team_vs_random.py](../../examples/example_ray_team_vs_random.py)
- `variation = team_vs_policy`
- `multiagent = False`
- `scratch`
- `fcnet_hiddens = [512, 512]`

### Base-B

- 与 Base-A 完全相同
- 唯一差别是 `fcnet_hiddens = [512, 256]`

### Base-C

- 任务语义对齐 [example_ray_ma_teams.py](../../examples/example_ray_ma_teams.py)
- `variation = multiagent_team`
- shared policy
- `scratch`
- 默认保持 starter 风格，不强制指定隐藏层

## 3. 结构化放置

这些基础模型脚本被明确收纳到：

- [scripts/batch/base/](../../scripts/batch/base)

而不是：

- adaptation lane
- experiments lane

这样目录语义就和实验语义一致：

- `base/` 只做基础模型
- `adaptation/` 只做基于已有 base checkpoint 的微调
- `experiments/` 保留探索性、混杂性或已归档路线

## 4. 当前判断

- 这是对 SSOT/starter 的一次真正回正
- 之后如果某条 base lane 跑出更强 checkpoint，它才有资格成为 warm-start 来源
- 在此之前，不再拿 adaptation 结果去反推基础模型结论

## 下一步

1. 运行 Base-A 与 Base-B
2. 用官方 evaluator 做 `200` 局确认
3. 如果两者都不够强，再运行 Base-C
4. 选出最强 base checkpoint 后，再进入 modification lane

## 相关

- [snapshot-007-base-lane-reset-and-directory-reorg.md](snapshot-007-base-lane-reset-and-directory-reorg.md)
- [directory-governance.md](../management/directory-governance.md)
- [scripts/README.md](../../scripts/README.md)
