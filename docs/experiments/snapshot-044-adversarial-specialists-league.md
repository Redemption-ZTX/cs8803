# SNAPSHOT-044: Adversarial Specialists League Self-Play

- **日期**: 2026-04-19
- **负责人**:
- **状态**: 预注册 / 三 phase 设计就绪 / specialist mode infra 待实施
- **依赖**:
  - [SNAPSHOT-043](snapshot-043-frontier-selfplay-pool.md) 的 `FrozenTeamCheckpointPolicy` adapter（共享）
  - 当前 frontier checkpoints (`029B@190` / `025b@080` / `028A@1060` / `036D@150` / `031A@1040`)
  - 训练 / 评估 reward + metric infra 扩展（`SPECIALIST_MODE` env var）

## 0. 动机与设计灵感

[SNAPSHOT-043](snapshot-043-frontier-selfplay-pool.md) 把 main agent 放进 `baseline + frontier-mix` 的 opponent pool 里训。但 frontier 都是 **balanced strong policies**——风格相似（v2 shaping、warmstart 链类似），main agent 见到的 strategy distribution 偏窄。

DeepMind AlphaStar 的 League 训练揭示一个关键洞察：**只跟 main agent 同质的 opponents 训练，policy 学不到鲁棒性。需要风格 *显式异质* 的 exploiter agents 作为 sparring partners**：
- **Main exploiter**: 极端进攻型，专找 main agent 的防守漏洞
- **League exploiter**: 极端防守型，专找 main agent 的进攻效率漏洞

我们这里：
- **矛 (spear)** = main exploiter——只奖励速胜，policy 学到极端进攻风格
- **盾 (shield)** = league exploiter——tie 等价 win，policy 学到极端防守风格
- **Main agent** = trainee，必须在 **frontier + 矛 + 盾** 这个 maximally diverse pool 上 robust

## 1. 核心假设

> **031A@1040 当前 0.860 vs baseline 的瓶颈，不是 policy class 容量，而是 opponent diversity 不足——见过的 opponents 都是"balanced 强对手"，没见过 extreme attack / defense 风格。把 main agent 放进包含 矛 (极端进攻) + 盾 (极端防守) 的 league 里训，强迫学 general 应对策略，预期 vs baseline 1000ep ≥ 0.88 (+2pp)。**

子假设：

- **H1 (矛 trainable)**: 用 `success = wins_under_100_steps / total` 做 train reward + eval metric，能训出 fast_win_rate ≥ 0.85 的 specialist
- **H2 (盾 trainable)**: 用 `success = (wins + ties) / total` 做 train reward + eval metric，能训出 non_loss_rate ≥ 0.85 的 specialist
- **H3 (league boosts main)**: main agent (031A 起点) 在 league pool 里训完，**vs baseline 1000ep ≥ 0.88**，且 H2H vs frontier 全面非退化

如果 H3 突破到 ≥ 0.90，**项目首次直接达到 9/10 阈值**。

## 2. Specialist 设计的核心简化

之前的设计想用复杂 reward shaping (e.g., "+5 for tie, -10 for loss") 去 encode specialist 目标。**但这个本质是 eval metric 没对齐**——一旦 eval metric 是 specialist 目标本身，training reward 就只是它的 sparse 镜像，没必要复杂化。

### 2.1 矛 (spear) — 速胜 specialist

| Aspect | Value |
|---|---|
| **Eval metric** | `fast_win_rate = wins_under_FAST_WIN_THRESHOLD_steps / total_episodes` |
| **Training reward (per episode)** | `R(W if step ≤ N) = +1, R(W if step > N) = 0, R(L) = -1, R(T) = 0` |
| **`FAST_WIN_THRESHOLD` (N)** | 100 (baseline 平均 episode 一半时长) |
| **Best ckpt selection** | argmax fast_win_rate |
| **辅助 dense shaping** | 可选保留 `SHAPING_BALL_PROGRESS=0.01` `SHAPING_OPP_PROGRESS_PENALTY=0.01`；移除 `SHAPING_DEEP_ZONE_*` (鼓励压前) |

**预期 policy 行为**: 开局狂攻、追求快速进球；防守随便（被进了也无所谓，反正下一 episode 重置）。

### 2.2 盾 (shield) — 不输 specialist

| Aspect | Value |
|---|---|
| **Eval metric** | `non_loss_rate = (wins + ties) / total_episodes` |
| **Training reward (per episode)** | `R(W) = +1, R(T) = +1, R(L) = -1` (T 和 W 同价) |
| **Best ckpt selection** | argmax non_loss_rate |
| **辅助 dense shaping** | 可选保留 `SHAPING_OPP_PROGRESS_PENALTY=0.02` (加重)；移除 `SHAPING_BALL_PROGRESS` (不奖励主动推进) |

**预期 policy 行为**: 风险规避、不主动冒进、守住中场就好；偶尔机会很好才进攻。

### 2.3 关键 infra 需求

需要扩展 ~80 行代码：

1. **`cs8803drl/core/utils.py` `RewardShapingWrapper`** 加 `SPECIALIST_MODE={none|spear|shield}` 分支，覆盖 `match_reward`：
   ```python
   if SPECIALIST_MODE == "shield":
       if outcome == TIE: match_reward = +1.0
   elif SPECIALIST_MODE == "spear":
       if outcome == WIN and episode_step > FAST_WIN_THRESHOLD: match_reward = 0.0
       elif outcome == TIE: match_reward = 0.0
   ```

2. **`scripts/eval/evaluate_official_suite.py` / `evaluate_matches.py`** 加 `--success-metric {win_rate|non_loss_rate|fast_win_rate}`，输出对应 metric 作主排序键

3. **Training 脚本** `best_eval_baseline` 的判断 follow `EVAL_SUCCESS_METRIC` env var

4. **Episode step 注入**: `RewardShapingWrapper` 必须能拿到当前 step（`info["step_count"]` 或自己 counter）才能判 fast win

## 3. 三 Phase 设计

### Phase 1: 044A (矛) 训练
- **Base**: scratch 或 from 028A_1060 base (team-level, 短训)
- **Reward**: `SPECIALIST_MODE=spear, FAST_WIN_THRESHOLD=100`
- **Budget**: 200 iter (~3h on H100)
- **Useful 判据**: 1000ep `fast_win_rate ≥ 0.85` AND `mean episode L ≤ 80` 步
- **Gating**: 不达标 → 044A 弃用，044C league pool 不含矛

### Phase 2: 044B (盾) 训练
- **Base**: scratch 或 from 028A_1060 base
- **Reward**: `SPECIALIST_MODE=shield`
- **Budget**: 200 iter (~3h)
- **Useful 判据**: 1000ep `non_loss_rate ≥ 0.85` AND `loss_rate ≤ 30%`
- **Gating**: 不达标 → 044B 弃用，044C league pool 不含盾

### Phase 3: 044C (Main agent League train)
- **Trainable**: `031A@1040` (current SOTA)
- **Architecture**: SiameseTeamModel (与 031A 同款)
- **Opponent pool** (动态根据 Phase 1/2 gating 调整):
  | Opponent | Share (full league) | Share (no spear) | Share (no shield) | Share (specialists fail) |
  |---|---:|---:|---:|---:|
  | baseline | 30% | 35% | 35% | 40% |
  | 矛 044A | 20% | — | 25% | — |
  | 盾 044B | 20% | 25% | — | — |
  | 029B@190 | 10% | 15% | 15% | 15% |
  | 025b@080 | 10% | 15% | 15% | 15% |
  | 028A@1060 | 5% | 5% | 5% | 15% |
  | 036D@150 | 5% | 5% | 5% | 15% |
- **Reward**: 标准 v2 shaping (no specialist mode for main agent)
- **Budget**: 500 iter (~12h with FrozenPolicy inference overhead)
- **Eval**: `EVAL_SUCCESS_METRIC=win_rate` (回到正常 grading metric)

## 4. 与 SNAPSHOT-043 的关系

044 和 043 是**两个并行 league 实验**，不是替代：

| 维度 | SNAPSHOT-043 | **SNAPSHOT-044 (本)** |
|---|---|---|
| Pool 组成 | baseline + frontier-only | baseline + frontier + **specialists (矛/盾)** |
| 设计哲学 | "见多识广 strong opponents" | "见极端 + 见 strong" |
| 实施成本 | ~200 行 + 12h GPU | ~280 行 (= 043 + specialist infra) + 18h (specialist + main) |
| 风险 | mode collapse to specific frontier | specialist 训不出来 → 退回 043 |
| 主张 | "frontier diversity 够用" | "需要 specialist diversity" |

**对照价值**: 跑完 043 + 044C 后能直接对比 "frontier-only league" vs "specialist-augmented league"，验证 hypothesis "extreme exploiter 是否真有用"。

## 5. 训练超参（044C league train）

```bash
# warmstart from 031A@1040
WARMSTART_CHECKPOINT=ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/.../checkpoint_001040/checkpoint-1040

# Architecture
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128

# League opponents (paths set after Phase 1/2)
OPPONENT_POOL_BASELINE_PROB=0.30
OPPONENT_POOL=spear:0.20,shield:0.20,029B:0.10,025b:0.10,028A:0.05,036D:0.05
OPPONENT_FROZEN_CKPT_spear=ray_results/044A_spear_<ts>/.../checkpoint_<best>/checkpoint-<best>
OPPONENT_FROZEN_CKPT_shield=ray_results/044B_shield_<ts>/.../checkpoint_<best>/checkpoint-<best>
OPPONENT_FROZEN_CKPT_029B=...
... (others same as 043)

# Reward (main agent uses standard v2)
SPECIALIST_MODE=none
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001
SHAPING_BALL_PROGRESS=0.01
... (standard v2 config)

# PPO
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048

# Budget
MAX_ITERATIONS=500
EVAL_INTERVAL=10
EVAL_EPISODES=50
EVAL_SUCCESS_METRIC=win_rate    # main agent 回到 grading metric
EVAL_OPPONENTS=baseline,random  # 50ep eval 只看 baseline + random (作业判据)
```

## 6. 预注册判据

### 6.1 Phase 1 / 2 specialist usefulness gate

| Specialist | useful 判据 | 不达标动作 |
|---|---|---|
| 矛 044A | `fast_win_rate ≥ 0.85 AND mean L ≤ 80` | 不进 league pool, share 转给 frontier |
| 盾 044B | `non_loss_rate ≥ 0.85 AND loss_rate ≤ 30%` | 不进 league pool, share 转给 frontier |

### 6.2 Phase 3 (main agent) 主判据

| 项 | 阈值 | 逻辑 |
|---|---|---|
| **vs baseline 1000ep** | **≥ 0.86** | 不退化于 031A 起点（保 grading 底） |
| H2H vs 031A@1040 (orig) | ≥ 0.51 | League train 后比 frozen-self 略强 |
| H2H vs frontier mix (avg) | ≥ 原始 031A 对应 H2H | 不退化于 frontier H2H 优势 |

### 6.3 Phase 3 突破判据（强成立）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| **vs baseline 1000ep peak** | **≥ 0.88** | League diversity 让 main 学到额外 +2pp |
| **vs baseline 1000ep peak** | **≥ 0.90** | 🎯 **首次直接达到 9/10 grading 阈值** |
| H2H vs 矛 044A | ≥ 0.55 | main 能扛极端进攻 |
| H2H vs 盾 044B | ≥ 0.55 | main 能破极端防守 |
| 044 vs 043 main 1000ep | 044 更高 | 验证 specialist diversity 比 frontier-only 更有效 |

### 6.4 Phase 3 失败判据

| 条件 | 解读 |
|---|---|
| vs baseline 1000ep < 0.80 | League pool 太硬 (specialist 太强), main agent 整体退化 |
| H2H vs 矛 < 0.45 OR vs 盾 < 0.45 | main over-fit 到 frontier，被 specialist 打爆 |
| 训练 reward 不收敛 | pool 配比不当, 起点 031A 不够强 |

## 7. 风险

### R1 — Specialist 训不出来

矛和盾都是新尝试。盾尤其难：env 不天然支持 "indefinite stalemate"，baseline 持续攻势下 ties 可能很难维持。

**缓解**:
- 先做 Phase 1 (矛) 单独验证 (3h GPU 投入很低)
- 矛过关后才做 Phase 2 (盾)
- 任何一个 specialist 不达标，044C league 自动调整 pool 组成 (§5 表格)
- 最坏情况: 两 specialist 都失败 → 044C fallback 到 baseline + frontier (等于 043)

### R2 — Main agent over-fit to specialists

如果 league pool 里 specialists 占 40% (20% 矛 + 20% 盾)，main 可能 over-fit 到这两种极端，反而对 baseline 这种 "中庸 baseline 风格" 表现下降。

**缓解**:
- baseline 留 30% (vs 043 的 40%)，仍是最大单 share
- 监控每 50 iter 的 vs baseline 50ep WR：如果 ≤ 0.80 提前 abort
- 如果发生，可以增加 baseline 占比 rerun

### R3 — FrozenPolicy infra 复杂度 + sample throughput

044C 的 league pool 包含 7 个 opponents (vs 043 的 5)，FrozenPolicy inference overhead 更大。

**缓解**:
- `NUM_ENVS_PER_WORKER` 从 5 降到 3
- 接受训练时间 12h → 14h
- 如果实在 throughput 太低，drop 028A/036D (它们 share 只 5% 影响小)

### R4 — Specialist mode infra 普适性

`SPECIALIST_MODE` env var 引入后，可能影响其他 lane 的训练（如果没正确 default 到 `none`）。

**缓解**:
- env var default 必须是 `none`
- 所有 specialist-related shaping 在 `RewardShapingWrapper` 里 wrap 在 `if SPECIALIST_MODE != "none"` 之内
- smoke test: 跑现有 lane (e.g., 040B) 验证 default 行为不变

## 8. 不做的事

- 不做 cyclic / dynamic specialist updates (训练中加新 specialist) — 复杂度爆炸
- 不在 044 内迭代 specialist reward 设计 (e.g., 不试 N=50 vs N=100 vs N=200) — 每条 specialist 跑一次定型
- 不混 Stage 2 reward shaping (PBRS handoff)：保留 main agent reward 是 v2 唯一变量
- **不交矛或盾作为 grading agent**——它们是 sparring partners only

## 9. 执行清单

### Phase 0: Specialist mode infra (~1 day)
1. 扩展 `RewardShapingWrapper` 支持 `SPECIALIST_MODE={none|spear|shield}` + `FAST_WIN_THRESHOLD`
2. 扩展 `evaluate_matches.py` / `evaluate_official_suite.py` 支持 `--success-metric`
3. 扩展 training 脚本的 `best_eval_baseline` selection logic 跟 `EVAL_SUCCESS_METRIC`
4. unit / smoke test: 三种 mode 各跑 1 iter 验证 reward 公式正确

### Phase 1: 044A 矛 (~3-4h)
5. 写 batch `soccerstwos_h100_cpu32_team_level_044A_spear_scratch_512x512.batch`
6. 200 iter scratch (or warmstart from 028A 短训以加速)
7. 1000ep eval with `--success-metric fast_win_rate`
8. failure capture: confirm policy 真的在追快速进球
9. **Gate**: fast_win_rate ≥ 0.85 ?

### Phase 2: 044B 盾 (~3-4h)
10. 写 batch `soccerstwos_h100_cpu32_team_level_044B_shield_scratch_512x512.batch`
11. 200 iter scratch
12. 1000ep eval with `--success-metric non_loss_rate`
13. failure capture: confirm policy 真的在守住，不是 turtle 被秒
14. **Gate**: non_loss_rate ≥ 0.85 AND loss_rate ≤ 30% ?

### Phase 3: 044C Main agent league train (~12-14h)
15. 根据 Phase 1/2 gate 结果决定 pool 组成
16. 写 batch `soccerstwos_h100_cpu32_team_level_044C_league_main_on_031A1040_512x512.batch`
17. 500 iter
18. eval pipeline + H2H vs 矛/盾/frontier 全套
19. failure capture vs baseline + vs 矛 + vs 盾
20. verdict: §10 入库

## 10. 相关

- [SNAPSHOT-023](snapshot-023-frozenteamcheckpointpolicy-opponent-adapter.md) — adapter (本 snapshot Phase 3 共享)
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031A 来源 (Phase 3 trainable 起点)
- [SNAPSHOT-043](snapshot-043-frontier-selfplay-pool.md) — frontier-only league (本 snapshot 的对照实验)
- [rank.md](rank.md) — frontier ckpts 数据

---

## 11. 首轮结果（044A 矛 / 044B 盾，2026-04-19）

200 iter 训练完成（trial dirs 列在 §0），post-train 评估按 [SKILL.md post-train-eval](../../.claude/skills/post-train-eval/SKILL.md) 执行 Stage 1 (1000ep baseline) + Stage 2 (specialist-objective 500ep capture)。**两条 specialist 均未达 §6.1 useful gate，044C league pool 需调整。**

### 11.1 044A 矛 — Stage 1 baseline 1000ep

Trial: `TeamVsBaselineShapingPPOTrainer_Soccer_e32dd_00000_0_2026-04-19_00-47-36`
Selected ckpts (top 5% + ties + ±1 by `fast_win_rate`, 50ep training-internal): **160, 170, 180**

| ckpt | baseline 1000ep `win_rate` | NW-NL-NT |
|---:|---:|---|
| 160 | 0.724 | 724W-276L-0T |
| 170 | 0.753 | 753W-247L-0T |
| **180** | **0.766** | **766W-234L-0T** |

**Raw recap** (`docs/experiments/artifacts/official-evals/044A_baseline1000.log`):
```
=== Official Suite Recap (parallel) ===
.../checkpoint_000160/checkpoint-160 vs baseline: win_rate=0.724 (724W-276L-0T)
.../checkpoint_000170/checkpoint-170 vs baseline: win_rate=0.753 (753W-247L-0T)
.../checkpoint_000180/checkpoint-180 vs baseline: win_rate=0.766 (766W-234L-0T)
[suite-parallel] total_elapsed=254.0s tasks=3 parallel=3
```

50ep peak `fast_win_rate` was 0.760 @ ckpt 170 (`checkpoint_eval.csv`)；1000ep `win_rate` peak 0.766 提供了 fast-win 的上界（fast wins ≤ all wins）。

### 11.2 044A 矛 — Stage 2 fast-win capture (500ep, ckpt 180)

`--save-mode fast_wins --fast-win-threshold 100`，shaping flags 镜像 044A 训练 batch (BALL_PROGRESS=0.01, OPP_PROGRESS_PENALTY=0.0)。

```
---- Summary ----
team0_module: cs8803drl.deployment.trained_team_ray_agent
team1_module: ceia_baseline_agent
episodes: 500
team0_wins: 385
team1_wins: 115
ties: 0
team0_win_rate: 0.770
team0_non_loss_rate: 0.770
team0_fast_wins: 358
team0_fast_win_threshold: 100
team0_fast_win_rate: 0.716
episode_steps_all: mean=44.9 median=35.0 p75=60.0 min=8 max=263
episode_steps_team0_win: mean=48.0 median=40.0 p75=62.0 min=8 max=263
episode_steps_team1_win: mean=34.6 median=25.0 p75=38.0 min=8 max=178
saved_episode_count: 358
```

Saved-episodes 路径：`docs/experiments/artifacts/failure-cases/044A_checkpoint180_baseline_500/`
Capture log：`docs/experiments/artifacts/official-evals/failure-capture-logs/044A_checkpoint180.log`

**Fast-win step 分布** (n=358, 计算自保存的 episode JSON `steps` 字段)：

| 步数桶 | 计数 | 占比 |
|---|---:|---:|
| ≤10 | 3 | 0.8% |
| 11–20 | 62 | 17.3% |
| 21–30 | 79 | 22.1% |
| 31–40 | 46 | 12.8% |
| 41–50 | 41 | 11.5% |
| 51–60 | 44 | 12.3% |
| 61–70 | 32 | 8.9% |
| 71–80 | 21 | 5.9% |
| 81–90 | 17 | 4.7% |
| 91–100 | 12 | 3.4% |
| >100 (溢出) | 1 | 0.3% |

中位 fast-win 在 **step 37**，~40% 的 fast-win 在 30 步以内完成。

**Qualitative readout** (从 8 个抽样 ep + 全 358 ep aggregate `metrics` 字段)：

- **Possession-agnostic**: 在 358 个 fast-win episode 上 `team0_possession_ratio` mean=0.497 / median=0.500，**控球率不是预测胜利的强信号**——即在控球较少时也能频繁速胜。
- **Counter-attack 占四分之一**: 23.5% 的 fast-win (84/358) 控球 < 30%——这是经典的"baseline 推上来失误，team0 抢断后远射或快速反击"模式。Episode 0001 是例子：team1 控球 84.6%、推到自己 prog=11.7m，但 14 步后 ball 走到 (-14.9, -3.8) 落入 team1 球门，team0 step reward +1.98。
- **Direct attack 主导**: 当 team0 持球 (剩余 76.5% 的 fast-win)，平均推 4–7m 朝对方球门方向；典型路径是 1–2 次中场推进后直接射门，没有耗时的盘带。

### 11.3 044A 矛 — Specialist gate verdict

| 指标 | 阈值 (§6.1) | 实测 (1000ep / 500ep capture) | 通过？ |
|---|---:|---:|:---:|
| `fast_win_rate` (500ep capture, ckpt 180) | ≥ 0.85 | **0.716** | ❌ −13.4pp |
| `mean episode L` (500ep capture, all eps) | ≤ 80 步 | **44.9** | ✅ |
| (info) `win_rate` 1000ep peak | — | 0.766 | n/a |

注：fast_win_rate 上界 = 1000ep `win_rate` = 0.766，仍不足 0.85；500ep 测得 0.716 在 ±0.022 SE 内与 0.766 上界一致。**矛 fail useful gate**：训不到 ≥0.85 的 fast-win 比例；但已学到"会进球（快胜）"这一基础能力。

### 11.4 044B 盾 — Stage 1 baseline 1000ep

Trial: `TeamVsBaselineShapingPPOTrainer_Soccer_7a066_00000_0_2026-04-19_00-51-49`
Selected ckpts (top 5% + ties + ±1 by `non_loss_rate`, 50ep training-internal): **180, 190, 200**

| ckpt | baseline 1000ep `win_rate` | `non_loss_rate` | NW-NL-NT |
|---:|---:|---:|---|
| **180** | **0.790** | **0.790** | **790W-210L-0T** |
| 190 | 0.744 | 0.744 | 744W-256L-0T |
| 200 | 0.765 | 0.765 | 765W-235L-0T |

T=0 across all 3 ckpts，所以 `non_loss_rate = win_rate`。

**Raw recap** (`docs/experiments/artifacts/official-evals/044B_baseline1000.log`):
```
=== Official Suite Recap (parallel) ===
.../checkpoint_000180/checkpoint-180 vs baseline: win_rate=0.790 (790W-210L-0T)
.../checkpoint_000190/checkpoint-190 vs baseline: win_rate=0.744 (744W-256L-0T)
.../checkpoint_000200/checkpoint-200 vs baseline: win_rate=0.765 (765W-235L-0T)
[suite-parallel] total_elapsed=246.4s tasks=3 parallel=3
```

注：50ep training-internal 上 `non_loss_rate` peak 是 0.90 @ ckpt 190；1000ep 上 ckpt 180 反而是峰值 (0.790)，190 掉到 0.744。这种 peak 重排在 SE±0.016 内是正常采样波动（180/190/200 三个值都落在 0.74–0.79 区间）。

### 11.5 044B 盾 — Stage 2 ties capture (500ep, ckpt 180)

`--save-mode ties`，shaping flags 镜像 044B 训练 batch (BALL_PROGRESS=0.0, OPP_PROGRESS_PENALTY=0.02, +deep zone penalty)。

```
---- Summary ----
team0_module: cs8803drl.deployment.trained_team_ray_agent
team1_module: ceia_baseline_agent
episodes: 500
team0_wins: 391
team1_wins: 109
ties: 0
team0_win_rate: 0.782
team0_non_loss_rate: 0.782
team0_fast_wins: 366
team0_fast_win_threshold: 100
team0_fast_win_rate: 0.732
episode_steps_all: mean=42.3 median=33.0 p75=55.0 min=6 max=187
episode_steps_team0_win: mean=44.1 median=36.0 p75=55.0 min=7 max=187
episode_steps_team1_win: mean=35.9 median=26.0 p75=42.0 min=6 max=138
saved_episode_count: 0
```

Saved-episodes 路径：`docs/experiments/artifacts/failure-cases/044B_checkpoint180_baseline_500/` (该目录因 0 ties 未创建)
Capture log：`docs/experiments/artifacts/official-evals/failure-capture-logs/044B_checkpoint180.log`

**0 ties at 500ep** 是关键 finding——shield reward 设计 (`R(W)=R(T)=+1, R(L)=-1`) 期望策略学会"实在赢不了就守平"，但实测 policy 完全不走 tie 路径，全部以 win/loss 结束。EP 长度 mean=42 步同样验证：根本没有进入"长时间僵持→tie"的 dynamic。

**Supplemental win capture** (500ep, ckpt 180, `--save-mode wins`，便于行为对比)：

```
---- Summary ----
episodes: 500  team0_wins: 384  team1_wins: 116  ties: 0
team0_win_rate: 0.768  team0_fast_win_rate: 0.728
episode_steps_team0_win: mean=42.9 median=35.0 p75=55.0
saved_episode_count: 384
```

Path: `docs/experiments/artifacts/failure-cases/044B_checkpoint180_wins_baseline_500/`

**Qualitative readout** (n=384 win episodes)：

- **Behavior 与矛几乎相同**: `team0_possession_ratio` mean=0.490 / median=0.500，control 25.7%(< 30% poss)；`team0_progress=7.39m` ~ `team1_progress=7.51m`——双方推进对称，没有明显的"shield 控球退后"模式。
- **Win 步数分布与 044A 高度重叠**: median=35 步 (vs 044A 中位 37 步)，几乎重叠。如果是真正"防守专家"，应该是 mean L 远高于 100 步、win 罕见 fast-win 占比应该崩塌；实测 fast_win_rate=0.728 反而接近 044A 的 0.716。
- **Reward 设计未实现行为差异化**: shield reward (`R(T)=+1`) 没有让 policy 偏离 spear 风格——对手是 baseline，速胜比僵持更易拿+1。Reward optimum 退化成"会进球"，与 044A spear 殊途同归。

### 11.6 044B 盾 — Specialist gate verdict

| 指标 | 阈值 (§6.1) | 实测 (1000ep) | 通过？ |
|---|---:|---:|:---:|
| `non_loss_rate` (1000ep, ckpt 180) | ≥ 0.85 | **0.790** | ❌ −6.0pp |
| `loss_rate` (1000ep, ckpt 180) | ≤ 0.30 | **0.210** | ✅ |

**盾 fail useful gate**：non_loss_rate 差 6pp；行为也未学到守平倾向（0/500 ties, mean L 42 步）。

### 11.7 对 044C league train 的影响

按 §6.1 规则："不达标 → 不进 league pool, share 转给 frontier"。两条 specialist 都失败 → 落入 §5 表格的最后一列（"specialists fail"）：

| Opponent | Share |
|---|---:|
| baseline | 40% |
| 029B@190 | 15% |
| 025b@080 | 15% |
| 028A@1060 | 15% |
| 036D@150 | 15% |

这与 [SNAPSHOT-043](snapshot-043-frontier-selfplay-pool.md) 的 frontier-only pool 几乎等价（仅 baseline 比例略调整）。**预期效应**: 044C 退化为 043 重做 + 微调，不再是 "specialist diversity vs frontier diversity" 的对照实验。

**建议**:

1. 若仍要跑 044C，应直接合并到 043 lane（不重复 GPU 投入）。
2. 若坚持 specialist exploration，需要重新设计 reward——本轮失败提示：
   - **盾**: `R(T)=+1` 对 baseline 太弱激励，policy 优先学速胜（容易拿到 reward）。需要要么加强 tie reward (`R(T)=+2`)，要么用 PBRS-style shaping 鼓励 ball 停留中场，或者把 baseline 换成更强对手让"赢"变难，逼出 tie 策略。
   - **矛**: fast_win_rate 上界 = win_rate ≈ 0.77，意味着 **win_rate 本身才是瓶颈**——不是 policy 学不会"快攻"，而是 200 iter 训练没把整体 win_rate 推到 0.85 以上。需要更长训练 (500+ iter) 或更强基础架构 (e.g., 复用 031A SOTA base 而非 scratch 起点)。
3. 若放弃 specialist 路径，044C 全停，资源转回 [SNAPSHOT-043](snapshot-043-frontier-selfplay-pool.md) 主体路径。

### 11.8 训练 caveats

- **200 iter 可能太短**: §3 说"~3h 投入很低"是为快速验证；如果 specialist 真要训出来，200 iter 的 v2-style budget 在 team-level Siamese 架构下偏紧。031A scratch 跑 1250 iter 才达 0.860，044A/B 200 iter scratch 卡在 0.77 区间符合"早期 plateau"预期。
- **Reward signal magnitude**: `R(W)=±1, R(T)=±1` 的 sparse reward 加上 dense shaping 0.001–0.02 量级，dense:sparse 比例~1:50–1:100；shield 的 `OPP_PROGRESS_PENALTY=0.02` (双倍) + deep_zone_penalty=0.005 加起来 dense 信号 ~ 0.03/step，但 1500 步 max episode 下 dense 累计可达 ±45，远超 sparse ±1，**shaping signal 实际主导 gradient**。这能解释为什么 shield 反而长出"主动赢"——deep_zone_penalty 鼓励远离自家底线，等价于鼓励推前。
- **0 ties 一致性**: 044A baseline 1000ep × 3 ckpt = 3000 ep, ties=0；044B 同样 3000 ep + 1000 ep capture, ties=0。**baseline 对手在 EP cap 1500 步内基本不会和**——这意味着"shield 学守平"是 environment-level infeasible 的 task，与策略好坏无关。这是 §7 R1 风险预言的具体体现。

### 11.9 Pipeline 副产物

Stage 0 infra 修改（atomic edit）：

- `cs8803drl/evaluation/evaluate_matches.py`: `--save-mode` 增加 `wins / fast_wins / ties` 三个 choice
- `scripts/eval/capture_failure_cases.py`: 同上 + 新增 `--fast-win-threshold` CLI flag (forward 给 evaluate_matches.py)
- `scripts/eval/pick_top_ckpts.py`: 新增 `--metric {win_rate|non_loss_rate|fast_win_rate}` flag，支持 specialist lane 选模

修改触发：之前 specialist save-mode 不存在；CSV backfill 中曾因 `dict contains fields not in fieldnames: 'fast_wins', 'fast_win_threshold', 'fast_win_rate', 'non_loss_rate'` (训练 tmux capture 中可见) 暴露了 atomic-edit 不到位的痛点，故本轮严格保证 evaluate_matches.py + capture_failure_cases.py 同时 patch。
