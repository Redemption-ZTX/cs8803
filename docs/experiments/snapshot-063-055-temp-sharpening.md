## SNAPSHOT-063: 055-temp — Temperature Sharpening Distill (Tier 3a)

**Date**: 2026-04-20
**Status**: 实现完成 / launcher + batch ready

### §0 背景

- 055 Project SOTA combined 2000ep = 0.907 (baseline vs 031B 0.880 = +0.025pp, z=2.08 `*` border sig)
- 055 使用 `TEAM_DISTILL_TEMPERATURE=1.0`（无软化）——"aggressive" distillation，把 teacher 的 argmax 当作硬目标
- Hinton 2015 原始 KD 论文经验性发现 T=2-4 对多数任务最优
- 对 policy distillation 特别地（Rusu 2016 on Atari），temperature softening 传递的是 exploration behavior，不仅仅是 argmax
- 当前 T=1 很可能在桌上留了信号：student 学到了 teacher 的 peak，却没学到 uncertainty distribution

### §1 假设

**H_063**: 以 T=2.0 做 distillation → combined 2000ep peak ≥ 0.915（相对 055 的 0.907 提升 +0.008pp，跨过可检测改进的 2σ SE 阈值）

Sub-hypotheses:
- **H_063-a**: softer teacher distribution 保留 "second-best" action 信息 → 更丰富的梯度
- **H_063-b**: RL distillation 比 supervised distillation 更受益于 T（exploration matters）
- **H_063-c**: 对 Multi-Discrete 动作空间（6 factors × 3 classes），factor-level softening 会按 per-factor 信息增益复合

### §2 设计

- **Isolation principle**：只改 temperature（T=1.0 → T=2.0）。其他一切与 055 保持一致：
  - 3-teacher ensemble（031B@1220 + 045A@180 + 051A@130）
  - LR=1e-4（不是 3e-4，为了把 T 和 LR 隔离）
  - Alpha schedule: init 0.05 → final 0.0, decay 8000 updates
  - 031B student arch（Siamese + cross-attn）
  - v2 shaping
  - 1250 iter scratch
- **Why T=2.0 specifically**：Hinton 论文中经验性最优区间 2-4 的中点；3.0/4.0 作为后续 follow-up，如果 2.0 work 的话

### §3 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: peak ≥ 0.911 | +0.004 vs 055 | T has measurable effect |
| §3.2 main: peak ≥ 0.915 | +0.008 | T detectable at 2σ SE |
| §3.3 breakthrough: peak ≥ 0.925 | +0.018 | T unlocks major gain |
| §3.4 持平: peak ∈ [0.895, 0.911) | Sub-marginal | T=1 was fine, no gain |
| §3.5 regression: peak < 0.890 | T=2 too soft | Flattened teacher signal hurt |

### §4 简化 + 风险 + 降级

- **简化**：single T value (2.0)。Full sweep (T ∈ {1.5, 2.0, 2.5, 3.0}) 作为 L2 follow-up
- **风险**：T=2.0 未必对本 action space 最优；peak 可能在 T=1.5 或 T=3.0
- **降级**：若 063 在 §3.1 边缘（persistent 但 sub-marginal），在不同 lane 并行 sweep 4 个 T 值（~48h GPU）
- **Retrograde sequence §4.4**：
  - Step 0: T=2.0 single lane（本实验）
  - Step 1（if peak < 0.911）: T=3.0 variant
  - Step 2（if both fail）: T=1.5 + T=4.0 full sweep
  - Step 3（if all fail）: Tier 3 path #2 关闭，转 #4 progressive distill

### §5 不做的事

- 不与 LR=3e-4 同时改（要 isolate T from LR confound）
- 不改 teacher pool（保持 3-teacher 034e）
- 不改 alpha schedule（same decay）

### §6 执行清单

- [x] 1. Snapshot 起草（本文件）
- [x] 2. Launcher [_launch_055temp_distill_T2.sh](../../scripts/eval/_launch_055temp_distill_T2.sh)（拷贝 055 launcher，仅改 T）
- [x] 3. Batch [soccerstwos_h100_cpu32_team_level_063_temp_distill_on_034e_ensemble_031B_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_063_temp_distill_on_034e_ensemble_031B_512x512.batch)
- [x] 4. Launched 2026-04-20 15:52 EDT（PORT_SEED=63, pre-crash trial 78f4d）
- [x] 5. Cluster mass failure @ 00:58 → resume 2026-04-21 03:06 EDT from ckpt_1030 (trial 92dce, PORT_SEED=85, BASE_PORT=60605+...)
- [x] 6. 训练完成 2026-04-21 04:58 EDT（ckpt_1250 + tune_metadata 完整，process exited）
- [ ] 7. Verdict per §3 评估 — Stage 1 post-eval in progress

### §6.1 合并训练状态（user directive: merged global view + resume-boundary awareness）

- **Pre-crash trial** (78f4d, 2026-04-20): ckpts 010-1030 (103 ckpts, inline 200ep baseline eval 每 10 iter)
- **Resume trial** (92dce, 2026-04-21): ckpts 1040-1250 (22 ckpts, 1250 no inline eval)
- **Merge artifact**: [artifacts/merged/055temp/](../../docs/experiments/artifacts/merged/055temp/) — progress.csv 1250 rows, training_loss_curve_full.png
- **Resume-boundary check**: iter 1030 (pre-crash 0.920) → iter 1040 (resume, not in inline top set) — 无 KL/reward drop 明显迹象,但 200ep 噪声 ±0.035 cover 不住

### §6.2 Stage 1 post-eval（launched 2026-04-21 05:03 EDT）

**Ckpt selection (top 5% + ties + ±1 window, 25) + ckpt 1250 (terminal, no inline eval)**:
- Pre-crash: 610/620/630, 730/740/750, 840/850/860/870/880/890, 1020/1030 (14)
- Resume: 1040, 1090/1100/1110, 1150/1160/1170, 1210/1220/1230/1240, **1250** (12)
- **Total**: 26 ckpts

**Top inline 200ep WR**: iter 1230=0.960, 850=0.950, 740=0.940, 620=0.930, 1220=0.925, 880/1030/1100/1160=0.920

**Expected**:
- §3.1 marginal ≥ 0.911: likely met (055@1150 = 0.907 combined, 055temp 稍高可能性 > 50%)
- §3.2 main ≥ 0.915: 边界；200ep 0.960 @1230 likely downshift to ~0.90-0.92 @ 1000ep 复测
- §3.3 breakthrough ≥ 0.925: 小概率

**Launch**: on jobid 5028919 (H100, fresh 16h), BASE_PORT 62205
- Log: `docs/experiments/artifacts/official-evals/055temp_baseline1000.log`

### §7 Verdict (2026-04-21 05:55 EDT)

**Outcome B — §3.4 TIED 055** (T=1 was fine, no gain from T=2.0 sharpening)

#### 7.1 Stage 1 baseline 1000ep 全量结果

Trial：pre-crash `78f4d` (iter 010-1030) + resume `92dce` (iter 1040-1250)
Ckpts 选：26 (top 5% + ties + ±1 on merged + terminal 1250)
Eval 节点：5028919 (H100), BASE_PORT 62205, -j 7 parallel, ~21 min total

| iter | 1000ep WR | (W-L-T) | notes |
|---:|---:|---:|---|
| 610 | 0.857 | 857-143-0 | 早中段 |
| 620 | 0.871 | 871-129-0 | 200ep 0.930 → -0.059 (noise) |
| 630 | 0.871 | 871-129-0 | |
| 730 | 0.885 | 885-115-0 | |
| 740 | 0.887 | 887-113-0 | 200ep 0.940 → -0.053 (noise) |
| 750 | 0.869 | 869-131-0 | |
| 840 | 0.891 | 891-109-0 | |
| 850 | 0.877 | 877-123-0 | 200ep 0.950 → -0.073 (最大 noise) |
| 860 | 0.899 | 899-101-0 | |
| 870 | 0.897 | 897-103-0 | |
| 880 | 0.885 | 885-115-0 | |
| 890 | 0.876 | 876-124-0 | |
| 1020 | 0.885 | 885-115-0 | |
| **1030** | **0.904** | **904-96-0** | **★ PEAK** (pre-crash 末端) |
| 1040 | 0.890 | 890-110-0 | resume 第一 ckpt |
| 1090 | 0.888 | 888-112-0 | |
| 1100 | 0.869 | 869-131-0 | 200ep 0.920 → -0.051 |
| 1110 | 0.886 | 886-114-0 | |
| 1150 | 0.893 | 893-107-0 | |
| 1160 | 0.884 | 884-116-0 | |
| 1170 | 0.890 | 890-110-0 | |
| 1210 | 0.901 | 901-99-0 | 次 peak |
| 1220 | 0.895 | 895-105-0 | |
| 1230 | 0.901 | 901-99-0 | 次 peak (200ep 0.960 → -0.059 noise) |
| 1240 | 0.890 | 890-110-0 | |
| 1250 | 0.899 | 899-101-0 | terminal (无 inline 参考) |

**Raw log**：[055temp_baseline1000.log](../../docs/experiments/artifacts/official-evals/055temp_baseline1000.log)

#### 7.2 判据对照

| 阈值 | peak 0.904 | 判定 |
|---|---|---|
| §3.1 marginal ≥ 0.911 | 0.904 | ✗ FAIL |
| §3.2 main ≥ 0.915 | 0.904 | ✗ FAIL |
| §3.3 breakthrough ≥ 0.925 | 0.904 | ✗ FAIL |
| **§3.4 持平 [0.895, 0.911)** | **0.904** | **✅ YES** |
| §3.5 regression < 0.890 | 0.904 | ✗ NO |

vs 055@1150 combined 2000ep SOTA **0.907**：Δ = -0.003，SE ±0.016 内 → **statistically tied**

#### 7.3 Resume-boundary 效应分析（用户指示的全局视角）

- **iter 1030 (pre-crash 末端)** 0.904 → **iter 1040 (resume 首)** 0.890 → Δ = -0.014
- 不达 2×SE 显著 regression 阈值 (-0.032)，但**方向明确下行**
- Resume 早段 (1040-1110) 0.869-0.890，普遍比 pre-crash peak 低
- Resume 尾段 (1210-1250) 0.890-0.901 才恢复到 pre-crash peak 附近
- **归因**：optimizer momentum / advantage norm reset 导致 200 iter 的 warm-up 窗口。若 resume 段继续训练到 ~1500+，可能再超 0.904
- **这是"resume 节点本身带来的变化"的实证**：训练曲线不该拼接前后两段独立判，而要作为一条有 boundary effect 的全曲线读

#### 7.4 Extend 决策 — **延后，不关闭**

- **不立刻 extend 的理由**：
  - Peak argmax iter 1030 **在 pre-crash 末端**，resume 段没有持续爬升到新高
  - Late window (1200-1250) peak 0.901 < pre-crash peak 0.904 < 055 SOTA 0.907
  - 没有"曲线仍在爬"的明确证据
- **保留 extend 可能性的理由**：
  - Resume 段仅 220 iter (1030→1250)，boundary-warmup 占了前 ~100 iter
  - 真正"稳定的 resume 训练"只有 iter 1150-1250 = 100 iter 窗口，其中 1210/1230 @ 0.901 已接近 pre-crash peak
  - 若 extend 到 1750-2000，resume 段能有 600-700 iter 完整发育，有望突破 0.904
- **ROI 判断**：ROI 较低（相比 055v2@peak + 068 PBRS + 066 progressive distill 等），故 **延后而非放弃**
- **触发 extend 条件**：
  - 066A/066B progressive distill + 068 PBRS 给出明确 verdict (tied or -) 且 节点空闲
  - 或其他 lane (055v2_extend, 054M_extend, etc.) 全部判完后，还有富余时间冲刺 055temp→2000

#### 7.5 科学结论

- **T=2.0 对本 action space 不产生可观增益**（T=1 已接近 reward signal 上限）
- 跟 Hinton 2015 在 image classification 上的 T=2-4 最优区间**不适用**于本 RL distillation：
  - 可能原因：6 factor × 3 class 的 Multi-Discrete 动作空间 entropy 已足够高，再 soften 反而稀释 teacher 的 argmax 信号
  - 或：034e 3-teacher ensemble 本身已足够多样化，第二轮 softening 冗余
- **对未来 distill 路径的启示**：
  - 不再 sweep T ∈ {1.5, 2.5, 3.0}（除非 extend 后出意外）
  - 转向 **Tier 3b #4 progressive distill (066A/B)** 已在进行
  - 也可考虑 **logit-level distill** 作为替代软化机制 (backlog)

#### 7.6a Stage 2 capture (2026-04-21 06:10 EDT) — peak ckpt 1030 失败分析

**`055temp@1030` vs baseline n=500 @ BASE_PORT 63205**:

```
---- Summary ----
team0_wins: 450
team1_wins: 50
ties: 0
team0_win_rate: 0.900
team0_non_loss_rate: 0.900
team0_fast_wins: 437
team0_fast_win_rate: 0.874  (< 100 steps)
episode_steps_all: mean=38.5 median=32.0 p75=51.0 min=7 max=151
episode_steps_team0_win: mean=40.1 median=34.0 p75=54.0 min=7 max=151
episode_steps_team1_win: mean=25.0 median=21.0 p75=26.0 min=9 max=82
```

- 500ep WR 0.900 跟 Stage 1 1000ep 0.904 一致 (SE 0.010 内,验证 Stage 1 真实性)
- **No turtle behavior** (max 151 steps 明显低于 shaping 问题 lane 常见的 300+ 长 episode)
- fast_win_rate 0.874 — 87% 胜利在 100 steps 内完成,baseline exploit 效率高

**Loss bucket tally (50 losses 分类):**

| Bucket | 数量 | 比例 | 机制解读 |
|---|---:|---:|---|
| late_defensive_collapse | 28 | 56% | 控球 → 被断 → 快速反攻 failed (防守不及) |
| low_possession | 12 | 24% | 整局控球不足,被对手主导 |
| unclear_loss | 5 | 10% | 机制不明(短局 random loss) |
| poor_conversion | 5 | 10% | 有进攻机会但射门/传球失误 |

**vs 055@1150 对照** (参考 [rank.md §3.3 055 entry](../rank.md#33-official-baseline-1000frontier--active-points-only)):
- 055 distill 的主要 loss pattern 是 `wasted_possession + possession_stolen` (持球但丢球)
- 055temp 的主要是 `late_defensive_collapse` (56%) — 防守端漏洞比 055 显著
- 说明 T=2.0 softer teacher 信号**轻微劣化了防守端学习**,而进攻端保持稳定
- 但整体 WR tied 说明 **defensive 劣化被 offensive 稳定抵消了** — zero-sum 交换,非 actionable advantage

[full log](../../docs/experiments/artifacts/official-evals/failure-capture-logs/055temp_checkpoint1030.log) / [saved episodes (50)](../../docs/experiments/artifacts/failure-cases/055temp_checkpoint1030_baseline_500/)

#### 7.6 Stage 3 H2H (2026-04-21 06:15 EDT) — 直接确认 tied

**`055temp@1030 vs 055@1150` n=500 @ BASE_PORT 64205**:

| matchup | sample | 055temp wins | 055 wins | 055temp rate | z | p | sig |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 055temp@1030 vs 055@1150 | 500 | 253 | 247 | **0.506** | 0.27 | 0.79 | `--` NOT sig |

Side split (side luck check): blue 0.488 / orange 0.524 — gap -0.036 within noise，无结构性侧别差异

**Raw H2H Recap**:
```
---- H2H Recap ----
team0_module: cs8803drl.deployment.trained_team_ray_agent  (055temp@1030)
team1_module: cs8803drl.deployment.trained_team_ray_opponent_agent  (055@1150)
episodes: 500
team0_overall_record: 253W-247L-0T
team0_overall_win_rate: 0.506
team0_edge_vs_even: +0.006
team0_net_wins_minus_losses: +6
team0_blue_win_rate: 0.488
team0_orange_win_rate: 0.524
```

**Interpretation**:
- H2H 0.506 = **pretty much random-coin-flip result**, essentially equivalent policies
- 跟 Stage 1 baseline axis (Δ-0.003) 完全一致 — peer axis 也 tied
- 证明 **T=2.0 softening 不改变 policy 的竞争面貌**,不是"学到 niche 强项但 baseline 下降"那种 trade-off，而是**完全无信号**
- 对 T-sweep 的否决加强 — 不必测 T=1.5 / T=2.5 / T=3.0

[full log](../../docs/experiments/artifacts/official-evals/headtohead/055temp_1030_vs_055_1150.log)

---

#### 7.7 T=4.0 (063_T40) 4-sample combined 3000ep correction — tied 055, T sweep REMAINS CLOSED (2026-04-21 14:55 EDT, append-only)

**Context**: snapshot-067 5-point T sweep (T ∈ {1.0, 1.5, 2.0, 3.0, 4.0}) includes T=4.0 as the upper bound. 063_T40 retry (held-srun 5028753, resume from iter 370 → 1250) completed 2026-04-21 上午; Stage 1 post-eval (25 ckpts × 1000ep) found ckpt 1060 = **0.923 (923W-77L)** — initially looked like a §3.1/§3.2 marginal HIT, potential "marginal SOTA" above 055 0.907。User + subagent ran 3 reruns on ckpt 1060 to verify:

| Sample | n | WR | (W-L) | 备注 |
|---:|---:|---:|---:|---|
| Stage 1 baseline1000 | 1000 | **0.923** | 923-77 | initial peak, small-sample upward bias |
| rerun500 | 500 | 0.930 | 465-35 | 2nd (further positive fluctuation) |
| rerun500v3 | 500 | 0.904 | 452-48 | 3rd (regression toward mean) |
| rerun1000v4 | 1000 | **0.897** | 897-103 | 4th, largest sample, pulls combined down |
| **COMBINED** | **3000** | **0.9123** | **2737-263** | **SE 0.0051, 95% CI [0.902, 0.923]** |

**vs 055@1150 combined 2000ep 0.907**: **Δ = +0.005**, within combined SE ±0.005 → **statistically tied 055** (not "marginal SOTA" as first 3 samples suggested with mean 0.919)

**Stage 3 H2H** `063_T40@1060 vs 055@1150` n=500: **0.508 (254W-246L, z=0.36, p=0.72 NOT sig)**; side split blue 0.528 / orange 0.488 gap +0.040 within noise。

Raw H2H log：[063T40_1060_vs_055_1150.log](../../docs/experiments/artifacts/official-evals/headtohead/063T40_1060_vs_055_1150.log)

**判据 re-check (combined 3000ep)**:

| 阈值 | peak 0.912 | 判定 |
|---|---|---|
| §3.1 marginal ≥ 0.911 | 0.912 | just at threshold, tied within SE |
| §3.2 main ≥ 0.915 | 0.912 | ✗ MISS |
| §3.3 breakthrough ≥ 0.925 | 0.912 | ✗ MISS |
| §3.4 持平 [0.895, 0.911) | 0.912 borderline | essentially tied 055 |
| §3.5 regression < 0.890 | 0.912 | ✗ NO |

**Sweep meta-verdict** (snapshot-067 anchored here since T=4.0 is the upper edge):
- T=1.0 (055 anchor) combined 2000ep = 0.907
- T=2.0 (063 / 055temp @1030) single-shot 1000ep = 0.904 (tied, §7.2)
- T=4.0 (063_T40 @1060) combined 3000ep = **0.912 (tied within SE)**
- Both T ∈ {2.0, 4.0} give NO actionable uplift over T=1.0
- **T path remains CLOSED** — T sweep was supposed to prove a peaked pattern; actual pattern is **flat across [1.0, 4.0]** → snapshot-067 Pattern C confirmed (T not informative)

**Lesson — small-sample upward bias (4-sample rerun saga)**:
- First 3 samples (n=1000+500+500=2000) mean 0.919 → initial "marginal SOTA" claim
- 4th sample (n=1000) 0.897 pulled combined toward true mean ~0.912
- SE on n=2000 alone was 0.0062; SE on n=3000 is 0.0051 — tightening CI by 18% revealed tied status
- Reinforces MEMORY `feedback_inline_eval_noise.md`: **even 2000ep single-peak verification can be +0.012pp optimistic** if first 2 samples are both positive outliers
- **Rule tightening**: for "marginal SOTA" candidates (within 1.5×SE of 055), require ≥3000ep combined before claiming shift

**Raw logs referenced**:
- [063_T40_baseline1000.log](../../docs/experiments/artifacts/official-evals/063_T40_baseline1000.log) — Stage 1 sample 1 (1000ep)
- [063_T40_rerun500.log](../../docs/experiments/artifacts/official-evals/063_T40_rerun500.log) — sample 2 (500ep)
- [063_T40_rerun500v3.log](../../docs/experiments/artifacts/official-evals/063_T40_rerun500v3.log) — sample 3 (500ep)
- [063_T40_rerun1000v4.log](../../docs/experiments/artifacts/official-evals/063_T40_rerun1000v4.log) — sample 4 (1000ep)
- [headtohead/063T40_1060_vs_055_1150.log](../../docs/experiments/artifacts/official-evals/headtohead/063T40_1060_vs_055_1150.log) — Stage 3 H2H

**Decision**: T-sweep closed for good. T ∈ {1.5, 3.0} pending verdicts will not re-open path even if marginal (already 4 points show no trend). Resources redirect to progressive distill (066A/066B) / distill pools (071/072/073) / arch axis (076 wide student / 077 per-agent).

### §8 后续路径

- **Outcome A**（breakthrough ≥ 0.915）: sweep T ∈ {1.5, 2.5, 3.0}，确认 optimum
- **Outcome B**（tied 055）: T=1 was fine，转 Tier 3b #4 progressive distill
- **Outcome C**（regression）: T=2 too soft，尝试 T=1.5（intermediate）或放弃 T path

### §9 相关

- [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) — current SOTA，T=1.0 baseline
- [snapshot-059](snapshot-059-055lr3e4-combo.md) — LR=3e-4 variant（parallel Tier 1a）
- [snapshot-061](snapshot-061-055v2-recursive-distill.md) — 5-teacher variant（parallel Tier 2）
- [BACKLOG.md](BACKLOG.md) — Tier 3b/3c follow-ups
- **Theoretical**: Hinton 2015 "Distilling the Knowledge in a Neural Network" (§3 temperature section); Rusu 2016 "Policy Distillation" (Atari); Allen-Zhu & Li 2020 on ensemble distillation theory
- Launcher: [scripts/eval/_launch_055temp_distill_T2.sh](../../scripts/eval/_launch_055temp_distill_T2.sh)
- Batch: [scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_063_temp_distill_on_034e_ensemble_031B_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_063_temp_distill_on_034e_ensemble_031B_512x512.batch)
