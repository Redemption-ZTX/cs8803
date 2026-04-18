# SNAPSHOT-009: Team-vs-Baseline Scratch Base Lane

- **日期**: 2026-04-11
- **负责人**:
- **目标**:
  - 停止把 `random` 当作基础模型主训练对手
  - 在 starter 的 `team_vs_policy` 语义上，直接对准课程评分中的 `baseline`
  - 用干净的 scratch 对照回答“更大的基础模型在 baseline 目标上是否更有效”
- **配置**:
  - 训练入口: [train_ray_base_team_vs_baseline.py](../../cs8803drl/training/train_ray_base_team_vs_baseline.py)
  - Base-D batch: [soccerstwos_h100_cpu32_base_team_vs_baseline_512x512.batch](../../scripts/batch/base/soccerstwos_h100_cpu32_base_team_vs_baseline_512x512.batch)
  - Base-E batch: [soccerstwos_h100_cpu32_base_team_vs_baseline_512x256.batch](../../scripts/batch/base/soccerstwos_h100_cpu32_base_team_vs_baseline_512x256.batch)
  - 训练语义: `variation = team_vs_policy`, `multiagent = False`, `scratch`
  - 对手设置: `baseline_prob = 1.0`
  - 训练内筛选口径: `50` 局
  - 最终确认口径: 官方 `200` 局

## 1. 背景

截至本轮之前：

- Base-A / Base-B 是 starter-faithful 的 `team_vs_random`
- Base-C 是 shared-policy `multiagent_team` self-play
- 三者都没有直接对准课程评分里最关键的 `baseline` 对手

因此，即使 loss/entropy/KL 看起来“更健康”，也不能说明它们在真正的评分目标上走对了方向。

## 2. 本轮决策

在不混入 warm-start、fixed teammate、reward shaping 或 adaptation 的前提下，
新增一条更贴近评分目标的 `base model lane`：

- 仍沿用 starter 的 `team_vs_policy` 语义
- 但训练对手不再是默认 random
- 直接使用 `baseline` 作为训练对手

这样 Base-D / Base-E 回答的问题就很干净：

> 当基础任务直接对准 baseline 时，`512x512` 和 `512x256` 哪个更适合作为 base checkpoint 来源？

## 3. 和旧 Base Lane 的关系

- [snapshot-008](snapshot-008-starter-aligned-base-model-lane.md) 保留作为“starter-faithful scratch base lane”的记录
- 本 snapshot 是在其基础上的一次任务定义修正
- Base-C 可继续作为 `self-play` 参考线，但不再作为当前主线判断依据

## 下一步

1. 运行 Base-D 与 Base-E
2. 观察 `checkpoint_eval.csv`
3. 用官方 evaluator 做 `200` 局确认
4. 只有在此基础上选出的 checkpoint，才有资格成为后续 modification lane 的 base model

## 相关

- [snapshot-008-starter-aligned-base-model-lane.md](snapshot-008-starter-aligned-base-model-lane.md)
- [directory-governance.md](../management/directory-governance.md)
- [scripts/README.md](../../scripts/README.md)

## 4. 实际运行

### Base-D: `team_vs_baseline 512x512 scratch`

- run_dir: [PPO_base_team_vs_baseline_512x512_20260411_192140](../../ray_results/PPO_base_team_vs_baseline_512x512_20260411_192140)
- trial_dir: [BaseTeamVsBaselinePPOTrainer_Soccer_3e75a_00000_0_2026-04-11_19-22-01](../../ray_results/PPO_base_team_vs_baseline_512x512_20260411_192140/BaseTeamVsBaselinePPOTrainer_Soccer_3e75a_00000_0_2026-04-11_19-22-01)
- loss 曲线: [training_loss_curve.svg](../../ray_results/PPO_base_team_vs_baseline_512x512_20260411_192140/training_loss_curve.svg)
- 训练内评估: [checkpoint_eval.csv](../../ray_results/PPO_base_team_vs_baseline_512x512_20260411_192140/checkpoint_eval.csv)

### Base-E: `team_vs_baseline 512x256 scratch`

- run_dir: [PPO_base_team_vs_baseline_512x256_20260411_192339](../../ray_results/PPO_base_team_vs_baseline_512x256_20260411_192339)
- trial_dir: [BaseTeamVsBaselinePPOTrainer_Soccer_85e93_00000_0_2026-04-11_19-24-01](../../ray_results/PPO_base_team_vs_baseline_512x256_20260411_192339/BaseTeamVsBaselinePPOTrainer_Soccer_85e93_00000_0_2026-04-11_19-24-01)
- loss 曲线: [training_loss_curve.svg](../../ray_results/PPO_base_team_vs_baseline_512x256_20260411_192339/training_loss_curve.svg)
- 训练内评估: [checkpoint_eval.csv](../../ray_results/PPO_base_team_vs_baseline_512x256_20260411_192339/checkpoint_eval.csv)

## 5. 结果摘要

### Base-D (`512x512`)

- 训练步数: `500 iterations / 10M+` 量级训练内 checkpoint 评估
- `episode_reward_mean`: `-0.0968 -> 1.8945`
- `entropy`: `6.5888 -> 1.7006`
- `kl`: `0.0028 -> 0.0119`
- 最好 baseline checkpoint: `checkpoint-410`, `41W-9L = 0.82`
- 最后 baseline checkpoint: `checkpoint-500`, `34W-16L = 0.68`
- 最好 random checkpoint: `checkpoint-90`, `50W-0L = 1.00`
- 最后 random checkpoint: `checkpoint-500`, `49W-1L = 0.98`

### Base-E (`512x256`)

- 训练步数: `500 iterations / 10M+` 量级训练内 checkpoint 评估
- `episode_reward_mean`: `-0.3121 -> 1.8952`
- `entropy`: `6.5882 -> 1.5282`
- `kl`: `0.0035 -> 0.0166`
- 最好 baseline checkpoint: `checkpoint-320`, `33W-17L = 0.66`
- 最后 baseline checkpoint: `checkpoint-500`, `32W-18L = 0.64`
- 最好 random checkpoint: `checkpoint-160`, `50W-0L = 1.00`
- 最后 random checkpoint: `checkpoint-500`, `50W-0L = 1.00`

## 6. 解读

### 哪条更值得保留

- `512x512` 明显优于 `512x256`
- `512x256` 可以降级，不再作为优先 base 候选
- 当前更值得继续做官方大样本复核的是 `512x512 / checkpoint-410`

### 为什么 `reward_mean` 会接近 `2.0`

这不是我们手动把 reward 裁到了 `[-2, 2]`，而是 starter 的 team wrapper 本身就把一队两名球员的 reward 相加：

- [TeamVsPolicyWrapper.step](../../../.conda/envs/soccertwos/lib/python3.8/site-packages/soccer_twos/wrappers.py)
  - 返回 `reward[0] + reward[1]`
- [MultiAgentUnityWrapper.step](../../../.conda/envs/soccertwos/lib/python3.8/site-packages/soccer_twos/wrappers.py)
  - 单个球员 reward 是 `info.reward[i] + info.group_reward[i]`

所以 team-level 的单步 reward 天然就是两名球员 reward 的和，量级接近 `[-2, 2]` 是环境设计带来的结果。

更关键的是，RLlib 训练里看的 `episode_reward_mean` 是整局累计回报；官方 evaluator 的 `win_rate` 则是按每局最终 reward 的正负号计胜负，见 [evaluate.py](../../../.conda/envs/soccertwos/lib/python3.8/site-packages/soccer_twos/evaluate.py)。  
因此：

- `episode_reward_mean ≈ 2.0`
- `baseline win_rate 只有 0.82`

这两者可以同时成立。它反映的不是“数值爆炸”，而是“训练目标和最终评估目标仍然没有完全对齐”。

## 7. 本轮踩到的工程问题

### 1. 训练内 checkpoint-eval 端口溢出

长跑时训练内评估原先会把 `EVAL_BASE_PORT` 单调递增，最终超过 `65535`，触发：

- `OverflowError: bind(): port must be 0-65535`

该问题已修复：

- [train_ray_team_vs_random_shaping.py](../../cs8803drl/training/train_ray_team_vs_random_shaping.py)
- [backfill_run_eval.py](../../scripts/eval/backfill_run_eval.py)

现在端口会在安全区间内循环复用。

### 2. 训练结束后的 summary 打印 bug

Base-D / Base-E 跑完后曾报：

- `_print_progress() got an unexpected keyword argument 'iteration'`

这是训练收尾阶段的 summary 调用参数写错，不影响 checkpoint 产出，但会让 run 看起来像“异常结束”。  
该问题已修复于：

- [train_ray_base_team_vs_baseline.py](../../cs8803drl/training/train_ray_base_team_vs_baseline.py)

并已为两条 run 补生成 `training_loss_curve.svg`。

## 8. 结论

- `team_vs_baseline scratch` 比之前的 `team_vs_random scratch` 更接近真正任务目标
- 在这条更干净的 base lane 上，`512x512` 明显优于 `512x256`
- 但 `512x512` 的最佳训练内 baseline 也只有 `0.82`，仍未接近课程目标 `0.90`
- 因此，本轮最合理的后续不是继续扩大 `512x256`，而是：
  1. 仅对 `512x512 / checkpoint-410` 做官方 `200` 局复核
  2. 再决定下一轮是否修改 reward / observation / task semantics
