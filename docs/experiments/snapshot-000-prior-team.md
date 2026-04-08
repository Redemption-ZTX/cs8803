# SNAPSHOT-000: 前任团队实验结果分析

- **日期**: 2026-04-07（分析日期）
- **训练时间范围**: 2026-03-30 ~ 2026-04-01
- **训练环境**: PACE 集群
- **分析人**: —
- **相关**: [code-audit-000.md](../architecture/code-audit-000.md)、[overview.md](../architecture/overview.md)

---

## 训练配置速查

来源: [code-audit-000 § 2.1](../architecture/code-audit-000.md#21-整体设计)、[code-audit-000 § 3.1](../architecture/code-audit-000.md#31-整体设计)

| 参数 | team_vs_random_shaping | selfplay | curriculum |
|------|----------------------|----------|------------|
| 网络 | [512, 512] | [256, 256] | [256, 256] |
| lr | 3e-4 | — | — |
| gamma | 0.99 | — | — |
| clip_param | 0.2 | — | — |
| entropy_coeff | 0.0 | — | — |
| train_batch_size | 4000 | — | — |
| rollout_fragment | 1000 | 5000 | 5000 |
| batch_mode | truncate | complete | complete |
| 停止条件 | 15M steps / 2h | 15M steps / 2h | 15M steps / 2h / reward 1.9 |
| reward shaping | 开启 | 开启 | 无 |
| 对手 | 90% baseline + 10% random | 70% baseline + 30% selfplay pool | 静止 → 随机 |

---

## 总览

`ray_results/` 下共 7 个实验目录，33 个 trial。其中大量 trial 的 progress.csv 为空（0 行），说明启动失败或很快崩溃。只有 5 个 trial 有有效训练数据：

| 实验 | 有效 trial | 总 trial | 说明 |
|------|-----------|----------|------|
| PPO_team_vs_random_shaping | 3 | 6 | 初始训练，reward shaping 开启 |
| PPO_team_vs_mix_baseline90_random10_cont | 1 | 4 | 从上面的 checkpoint 继续 |
| PPO_team_vs_mix_cont_eval_cont | 1 | 1 | 继续训练 + 评估回调 |
| PPO_team_vs_mix_cont_eval_cont2 | 1 | 2 | 最终延长训练 |
| PPO_selfplay_rec | 2 | 5 | 自博弈训练 |
| PPO_team_vs_random_shaping_ablation | 0 | 4 | 消融实验，全部失败 |

## 训练轨迹

这些实验实际上是一条**连续的训练链**（team_vs_policy 路线），中间多次中断续训：

```
PPO_team_vs_random_shaping (2h, 484K steps)
    → reward: -0.97 → +0.09
    → 从零开始，reward shaping 开启，对手为 baseline 混合
    └── PPO_team_vs_mix_baseline90_random10_cont (4.7h, 1.19M steps)
            → reward: 0.00 → +1.49
            → 从上一个 checkpoint 续训
            └── PPO_team_vs_mix_cont_eval_cont (5.8h, 1.87M steps)
                    → reward: +1.59 → +1.80
                    → 继续训练，加入评估回调
                    └── PPO_team_vs_mix_cont_eval_cont2 (7.7h, 2.85M steps)
                            → reward: +1.85 → peak +1.91 → end +1.76
                            → 最终延长，reward 已收敛并开始震荡

PPO_selfplay_rec (2h, 486K steps)
    → reward: -0.31 → -0.46（持续下降）
    → 独立实验，selfplay 模式，完全失败
```

## 关键数据点

| 阶段 | 总 steps | 训练时间 | 起始 reward | 峰值 reward | 结束 reward |
|------|---------|---------|------------|------------|------------|
| 初始训练 | 484K | 2.0h | -0.969 | +0.090 | +0.090 |
| 续训 1 | 1.19M | 4.7h | 0.000 | +1.494 | +1.494 |
| 续训 2 | 1.87M | 5.8h | +1.593 | +1.806 | +1.801 |
| 续训 3 | 2.85M | 7.7h | +1.853 | **+1.912** | +1.758 |
| selfplay | 486K | 2.0h | -0.310 | -0.264 | -0.456 |

## 配置对比

| 参数 | team_vs_policy 链 | selfplay |
|------|------------------|----------|
| 网络 | [512, 512] | [256, 256] |
| lr | 3e-4 | 未记录 |
| entropy_coeff | 0.0 | 未记录 |
| num_gpus | 4 → 1 | 1 |
| num_workers | 4 | 8 |
| rollout_fragment | 1000 | 5000 |
| reward shaping | 开启 | 开启 |
| 对手 | 90% baseline + 10% random | 70% baseline + 30% selfplay pool |

## 可用 Checkpoints

| 实验 | 最新 checkpoint | 对应 reward |
|------|----------------|------------|
| team_vs_random_shaping | checkpoint_000121 | ~+0.09 |
| cont_eval_cont | checkpoint_000460 | ~+1.80 |
| cont_eval_cont2 | checkpoint_000712 | ~+1.76 |
| selfplay_rec | checkpoint_000105 | ~-0.46 |

最佳 checkpoint: **cont_eval_cont2/checkpoint_000712**（reward 峰值 +1.91 附近）

## 分析与结论

**team_vs_policy 路线（成功）**:
- reward 从 -0.97 稳步上升到 +1.91，说明 reward shaping + baseline 混合对手的方案有效
- 但 reward 在 +1.8~1.9 区间开始震荡，说明可能遇到瓶颈
- 总共训练了约 2.85M steps / 7.7h 才达到这个水平
- **没有 baseline 对战胜率数据**（评估回调中没找到 win_vs_baseline 指标），所以无法确认是否满足 [作业要求](../references/Final%20Project%20Instructions%20Document.md#policy-performance-50-points)（打赢 baseline 9/10）
- 相关代码分析见 [code-audit-000 § 2](../architecture/code-audit-000.md#2-train_ray_team_vs_random_shapingpy--主力训练脚本)

**selfplay 路线（失败）**:
- reward 始终为负且持续下降，训练完全没有收敛
- 可能原因：网络太小（[256,256] vs [512,512]）、对手太强（70% baseline）、没有足够的探索（entropy=0）
- 相关代码分析见 [code-audit-000 § 3](../architecture/code-audit-000.md#3-train_ray_selfplaypy--自博弈训练)

**消融实验（全部失败）**:
- `PPO_team_vs_random_shaping_ablation` 的 4 个 trial 全部 progress.csv 为空
- 目录名带 `reward_shaping=True/False`，说明想对比有无 reward shaping 的效果，但环境启动就失败了
- 这个消融对报告很有价值，值得重做

## 待验证

- [ ] checkpoint_000712 对 random agent 的实际胜率
- [ ] checkpoint_000712 对 baseline agent 的实际胜率
- [ ] reward 值 +1.9 对应什么级别的比赛表现
