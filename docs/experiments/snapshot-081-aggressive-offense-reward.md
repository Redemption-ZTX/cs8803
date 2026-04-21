# SNAPSHOT-081: Aggressive-Offense Reward Scratch — Orthogonal Reward Family

- **日期**: 2026-04-21 (19:10 EDT)
- **状态**: 预注册 + 启动
- **前置**: [snapshot-075 strategic synthesis §3](snapshot-075-strategic-synthesis-toward-0.93.md) (identifies reward-family diversity gap) / user directive 2026-04-21 — "PBRS 是某种程度上的 v2 like,需要一个全新 reward,从 scratch 建立完全不同攻击策略的 agent"

## 0. 背景 — reward axis 诊断

project 所有 lanes 的 reward signal 都来自 v2 shape 或 v2-derivative:
- **v2 shape** (055/055v2/055lr3e4/056/054): time_penalty + ball_progress + opp_progress_penalty + possession_bonus + deep_zone penalties。强调**平衡的 possession-based play**
- **PBRS** (053D/068): calibrated outcome predictor ΔV。但 predictor 的训练数据来自 v2-shape lanes 的 trajectories → **仍带 v2 偏向** (v2-policy-distribution 下 optimal outcome)
- **Learned reward** (036/051): neural reward model 也是从 v2-policy data 训的 → derivative
- **Curriculum no-shape** (062/058): SPARSE only (无 shaping) → closer to orthogonal but reward-trivial, not "different strategy" just "no strategy"

**所有 members 训出来的 policy 都倾向于:保护球 → 慢推进 → 防守回收**。074 family 失败的根本原因就是 members 虽然训练 recipe 不同,但 policy 行为模式同质。

## 1. 核心假设

### H_081

> **Reward 完全 asymmetric 偏向进攻**,从 scratch 训练出来的 agent 会产生 **qualitatively different failure modes**:
> - 大量时间在对方半场 → baseline 没法 exploit 我方防守(因为我方也不防)
> - 疯狂射门 → noise 伤对方更多?
> - 失球多(因为不防守),但进球也多(因为疯狂进攻)
>
> **Peak baseline 1000ep WR 预期 0.75-0.85** (NOT SOTA tier — 故意的,为 ensemble diversity 牺牲 single strength)
>
> **Ensemble value**: 与 055v2_extend@1750 配对,失败模式正交 → 真 +1-3pp deploy-time lift 可能性 > 074 family(因 074 members 全 v2 family 重叠)

### H_081-a (alt)

> 如果 WR > 0.88,那我们无意中发现了"进攻即最佳防守"的 principle,也是有价值的 science result

### H_081-b (anti)

> 如果 WR < 0.60,reward 太 asymmetric 导致 policy collapse,lane dead

## 2. Reward 配置

对比 v2 / PBRS,**每一项都改**:

| Item | v2 default | **081 aggressive** | 动机 |
|---|---|---|---|
| `time_penalty` | 0.001 | **0.003** | 3× stronger time pressure,no slow buildup |
| `ball_progress_scale` | 0.01 | **0.05** | **5×** reward for moving ball toward opp goal |
| `goal_proximity_scale` | 0.0 OFF | **0.015** ON | new — reward for being near opp goal |
| `goal_proximity_gamma` | 0.99 | 0.99 | keep |
| `opponent_progress_penalty_scale` | 0.01 | **0.0** OFF | **no defense signal — don't care** |
| `possession_bonus` | 0.002 | **0.0** OFF | **don't hold ball — shoot!** |
| `progress_requires_possession` | 0 | 0 | progress count regardless of possession |
| `deep_zone_outer_penalty` | 0.003 | **0.0** OFF | no position restriction |
| `deep_zone_inner_penalty` | 0.003 | **0.0** OFF | no position restriction |
| `defensive_survival_bonus` | 0 | 0 | keep off |
| `fast_loss_penalty_per_step` | 0 | 0 | don't punish quick losses |
| **`event_shot_reward`** | 0.0 OFF | **0.10 STRONG** | **main reward signal — every shot attempt** |
| `event_tackle_reward` | 0.0 | 0.0 | not defensive |
| `event_clearance_reward` | 0.0 | 0.0 | not defensive |
| `event_cooldown_steps` | 10 | 5 | more frequent events |

**Net effect**: reward dominated by (1) ball_progress toward opponent goal, (2) shots on goal, (3) goal_proximity. **Zero incentive for defense / possession / positional discipline**.

## 3. 预注册 verdict

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 stretch: WR ≥ 0.88 | unexpectedly strong | "aggression ≈ best defense" discovered |
| §3.2 main: WR ∈ [0.75, 0.88) | designed range | aggressive policy works, orthogonal failure modes verified |
| §3.3 marginal: WR ∈ [0.60, 0.75) | weak but usable | as ensemble member still valuable |
| §3.4 fail: WR < 0.60 | policy collapse | reward too asymmetric, drop |

## 4. 执行

- Student: 031B Siamese + cross-attn (same as 055 for fair comparison)
- **SCRATCH** (no warm), 1250 iter
- LR=1e-4 (no need for higher — simple reward)
- Baseline opponent 100% (BASELINE_PROB=1.0)
- Budget: 50M steps, 12h

## 5. 后续

若 §3.2 HIT:
- Stage 1 post-eval on all peaks
- **ENSEMBLE TEST** with 055v2_extend@1750: 074G = {1750 (weight 0.5) + 081_peak (weight 0.3) + 055 (weight 0.2)}
- Pool E idea: 4-teacher distill with 1750 + 081 + 055 → maximum reward-orthogonal teacher pool

若 §3.3 or §3.4:
- Lane 关 but agent 本身作 "reward modification" submission role (满足 rubric 40pts)

## 6. 代码改动

None — all reward change via env vars。Reuse `train_ray_team_vs_baseline_shaping.py` + `_launch_081_aggressive_offense_scratch.sh`.

## 7. Verdict

_Pending — Stage 1 post-eval ~10h 后_
