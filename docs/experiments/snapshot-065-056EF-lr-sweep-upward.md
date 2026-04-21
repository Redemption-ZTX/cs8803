## SNAPSHOT-065: 056E + 056F — LR Sweep Upward (beyond 3e-4)

**Date**: 2026-04-20
**Status**: 预注册 / launching same day

### §0 背景

- 056 简化 LR sweep 结果：
  - 056A lr=3e-5 → peak 0.78
  - 056B lr=7e-5 → peak 0.86
  - 056C lr=1.5e-4 → peak 0.88
  - 056D lr=3e-4 → peak 0.891（combined 2000ep）
- 056extend 在 056D 基础上续跑 +260 iter，撞墙于 iter 1510，peak 0.896（与 056D 基本持平，增益 marginal）
- **单调趋势观察**：3e-5 → 7e-5 → 1.5e-4 → 3e-4 区间内 peak 随 LR 单调上升
- **核心问题**：3e-4 是否为 LR 全局最优，还是趋势仍可向上延伸（5e-4、7e-4）？
- 056-PBT-full 因 GPU 成本过高（60h）+ 收益不确定暂停；本 snapshot 的定向双值 sweep 是同一比较下的 A-tier 替代方案（合计 24h GPU）

### §1 假设

**H_065**（primary）：
- **H_065-a**：056E lr=5e-4 持平或超过 056D 0.891（若单调趋势继续）
- **H_065-b**：056F lr=7e-4 或达峰或回归（有用 LR 范围的上边界）
- **H_065-c**：最优 LR 落在 [3e-4, 7e-4] 区间，定义本 HP landscape 上沿

### §2 设计

- 两条并行独立 lane（不做参数交换，纯 HP 探针）
- **056E**：LR=5e-4（056D 的 1.67×）
- **056F**：LR=7e-4（056D 的 2.33×）
- 其余所有配置与 056D 完全一致：team-level 031B arch、v2 shaping、BASELINE_PROB=1.0、1250 iter scratch、相同 batch/workers
- PORT_SEED：056E=67，056F=69（均空闲，不与已运行 lane 冲突）

### §3 预注册判据

（相对 056D 0.891 baseline）

| 判据 | 阈值 per lane | verdict |
|---|---|---|
| §3.1 marginal: peak ≥ 0.895 | +0.004 vs 056D | LR 趋势仍在上行 |
| §3.2 main: peak ≥ 0.905 | +0.014 | sweep 找到更高上限，2σ SE 可检测 |
| §3.3 breakthrough: peak ≥ 0.915 | 接近 055 | sweep 带来重大增益 |
| §3.4 持平 [0.882, 0.895) | sub-marginal | LR=3e-4 即为最优 |
| §3.5 regression: peak < 0.870 | LR 过高，训练发散 | 单调趋势断裂 |

### §4 简化 + 风险 + 降级

- **4.1** 仅两个 LR 值（5e-4、7e-4）—— 廉价探针，非完整 sweep
- **4.2** 不做其他 HP 搜索 —— 严格隔离到 LR 轴
- **4.3** 风险：高 LR 可能发散（PPO 在 5e-4 以上通常不稳定）。若 056F 在 iter < 500 即发散，kill 该 run 释放节点
- **4.4** Retrograde：若 056E 已回归，056F 大概率更差（趋势已断），可提前 kill 056F

### §5 不做的事

- 不同时改 clip / batch（保持 056D 配置）
- 不做 1e-3 / 2e-3（先看 5e-4 / 7e-4 再决定）
- 不做 launch 前 smoke test（LR 改动简单，0 工程风险）

### §6 执行清单

- [ ] 创建启动脚本 `scripts/eval/_launch_056E_lr5e4.sh` + `scripts/eval/_launch_056F_lr7e4.sh`（复制 056 sweep 脚本，设置 LR）
- [ ] 寻找 2 个空闲节点（每个 ≥ 12h 剩余）
- [ ] 通过 run_with_flags.sh wrapper launch
- [ ] Post-eval 后按 §3 给出 verdict

### §7 Verdict

### §7.1 (2026-04-21 ~07:30 EDT) — 056F retry Stage 1 baseline 1000ep verdict (append-only) — sub-SOTA CONFIRMED, lane CLOSED

#### §7.1.1 Data (25 ckpts × 1000ep, merged pre-crash trial + retry trial)

Full 1250 iter 覆盖由两段 trial 拼接:
- **Pre-crash trial** (run dir `056F_pbt_lr0.00070_scratch_20260420_164850`): ckpts 740/750/760/790/800/810/920/930/940 (9 ckpts)
- **Retry trial** (run dir `056F_pbt_lr0.00070_scratch_20260421_053930`, held-srun after Ray init hang kill): ckpts 1010-1250 (16 ckpts)
- 合计 25 ckpts × 1000ep 并行 eval (total_elapsed=1079.9s, parallel=7, base_port=63505)

| ckpt | 1000ep WR | W-L | 备注 |
|---:|---:|---:|---|
| 740 | 0.822 | 822-178 | pre-crash early |
| 750 | 0.834 | 834-166 | — |
| 760 | 0.855 | 855-145 | — |
| 790 | 0.857 | 857-143 | — |
| 800 | 0.860 | 860-140 | — |
| 810 | 0.854 | 854-146 | — |
| 920 | 0.859 | 859-141 | pre-crash mid |
| 930 | 0.853 | 853-147 | — |
| 940 | 0.852 | 852-148 | pre-crash last (boundary) |
| 1010 | 0.864 | 864-136 | retry first ckpt (post-boundary) |
| 1020 | 0.841 | 841-159 | — |
| 1030 | 0.862 | 862-138 | — |
| 1080 | 0.858 | 858-142 | — |
| 1090 | 0.861 | 861-139 | — |
| 1100 | 0.856 | 856-144 | — |
| 1110 | 0.861 | 861-139 | — |
| 1140 | 0.863 | 863-137 | — |
| 1150 | 0.855 | 855-145 | — |
| 1160 | 0.843 | 843-157 | local dip |
| 1170 | 0.858 | 858-142 | — |
| 1180 | 0.854 | 854-146 | — |
| 1200 | 0.862 | 862-138 | — |
| 1210 | 0.867 | 867-133 | near-peak |
| 1220 | 0.857 | 857-143 | — |
| **1250** | **0.868 (peak)** | **868-132** | **terminal, peak** |

Plateau iter 1100-1250 (16 ckpts): WR range [0.843, 0.868], mean ≈ **0.858**。

Full log: [`artifacts/official-evals/056F_baseline1000.log`](../../docs/experiments/artifacts/official-evals/056F_baseline1000.log)

#### §7.1.2 预注册判据对照 (§3)

| 判据 | 阈值 per lane | 056F peak 0.868 | verdict |
|---|---|---|---|
| §3.1 marginal | ≥ 0.895 | -0.027 | ✗ **MISS** |
| §3.2 main | ≥ 0.905 | -0.037 | ✗ MISS |
| §3.3 breakthrough | ≥ 0.915 | -0.047 | ✗ MISS |
| §3.4 持平 [0.882, 0.895) | sub-marginal | 0.868 below range | ✗ **MISS below** |
| §3.5 regression | < 0.870 | 0.868 | ✅ **HIT** (just at threshold — mild regression) |

#### §7.1.3 关键对比

| 对照 | Δ vs 056F 0.868 | verdict |
|---|---:|---|
| 056D (lr=3e-4) 0.891 | **-0.023pp** | 056F 明显 sub-056D |
| 031B (lr=1e-4 baseline) 0.880 | **-0.012pp** | 056F slightly worse than 1e-4 baseline |
| 056C (lr=1.5e-4) 0.883 | -0.015 | 056F < 056C |
| 056B (lr=7e-5) 0.86 | +0.008 | tied |
| 055 SOTA 0.907 | -0.039 | far below SOTA |

**LR 单调趋势 in sweep 方向**:
- lr=3e-5 → peak 0.78
- lr=7e-5 → peak 0.86
- lr=1.5e-4 → peak 0.883
- lr=3e-4 → peak 0.891 (**plateau max**)
- lr=5e-4 (056E) → pending
- lr=7e-4 (056F) → **0.868 (下降)**

3e-4 仍是 peak, 7e-4 已明显越过 optimum, **单调趋势断裂 at lr ≥ 5e-4 — §8 Outcome B 分支命中**。

#### §7.1.4 Resume-boundary check — 无 regression spike

pre-crash last ckpt 940 = 0.852 → retry first ckpt 1010 = 0.864 (+0.012 small bump), 在 1000ep SE ±0.016 内 → **no regression spike** across resume boundary。Retry trial 的 optimizer momentum / advantage norm 正常 warm-up，boundary effect 比预期温和 (对比 055temp resume 段早期 -0.01 to -0.03pp 的 pattern)。

#### §7.1.5 机制解读

- LR=7e-4 = 056D (lr=3e-4) 的 2.33×, 是 Schulman 2017 PPO default 3e-4 的 2.33× — 已知这是 aggressive 区间, PPO 对 LR 在 > 5e-4 普遍不稳
- 056F 没有发散 (peak 0.868 仍 > 0.85, 不是 catastrophic failure), 但 learning rate 太高 → **gradient update 步长过大, policy 在 optimum 周围振荡** → plateau 稳定在 sub-031B 水平
- **LR upward sweep past 3e-4 actively harmful** — confirmed 056D @ lr=3e-4 是 narrow LR flat optimum, 不是 lower bound of optimum basin

#### §7.1.6 后续决策 — Lane CLOSED

- **056F lr=7e-4 lane CLOSED** — sub-SOTA confirmed, not worth any extended training / follow-up
- **056E (lr=5e-4) 仍在跑**, pending verdict; 按 §8 Outcome B, 预期也在 [0.882, 0.895) tied 区或略 regress, 不会超越 056D
- 056 LR sweep 以 lr=3e-4 为 peak **基本可以封闭** (pending 056E confirmation)
- 不再 launch 056G (lr=1e-3); §8 Outcome A 条件未命中, trend 已 reverse
- **资源转向 distill axis (Pool A/B/C/D / 076 / 077 / 079) 和 DIR-A wide-student**

#### §7.1.7 Raw recap (official evaluator parallel)

```
=== Official Suite Recap (parallel, 25 ckpts) ===
ckpt 740  vs baseline: win_rate=0.822 (822W-178L-0T)
ckpt 750  vs baseline: win_rate=0.834 (834W-166L-0T)
ckpt 760  vs baseline: win_rate=0.855 (855W-145L-0T)
ckpt 790  vs baseline: win_rate=0.857 (857W-143L-0T)
ckpt 800  vs baseline: win_rate=0.860 (860W-140L-0T)
ckpt 810  vs baseline: win_rate=0.854 (854W-146L-0T)
ckpt 920  vs baseline: win_rate=0.859 (859W-141L-0T)
ckpt 930  vs baseline: win_rate=0.853 (853W-147L-0T)
ckpt 940  vs baseline: win_rate=0.852 (852W-148L-0T)   [pre-crash last]
ckpt 1010 vs baseline: win_rate=0.864 (864W-136L-0T)   [retry first]
ckpt 1020 vs baseline: win_rate=0.841 (841W-159L-0T)
ckpt 1030 vs baseline: win_rate=0.862 (862W-138L-0T)
ckpt 1080 vs baseline: win_rate=0.858 (858W-142L-0T)
ckpt 1090 vs baseline: win_rate=0.861 (861W-139L-0T)
ckpt 1100 vs baseline: win_rate=0.856 (856W-144L-0T)
ckpt 1110 vs baseline: win_rate=0.861 (861W-139L-0T)
ckpt 1140 vs baseline: win_rate=0.863 (863W-137L-0T)
ckpt 1150 vs baseline: win_rate=0.855 (855W-145L-0T)
ckpt 1160 vs baseline: win_rate=0.843 (843W-157L-0T)
ckpt 1170 vs baseline: win_rate=0.858 (858W-142L-0T)
ckpt 1180 vs baseline: win_rate=0.854 (854W-146L-0T)
ckpt 1200 vs baseline: win_rate=0.862 (862W-138L-0T)
ckpt 1210 vs baseline: win_rate=0.867 (867W-133L-0T)
ckpt 1220 vs baseline: win_rate=0.857 (857W-143L-0T)
ckpt 1250 vs baseline: win_rate=0.868 (868W-132L-0T)   [peak / terminal]
[suite-parallel] total_elapsed=1079.9s tasks=25 parallel=7
```

### §7.2 (2026-04-21 15:30 EDT) — 056E Stage 1 baseline 1000ep verdict (append-only) — sub-SOTA CONFIRMED, lane CLOSED, LR optimum at 3e-4 strict

**Setup**: `056E` = PBT-simplified LR sweep, `lr=5e-4` (5× baseline 1e-4, 1.67× the 056D peak 3e-4). Merged pre-crash + resume run dirs, same evaluator parallel 1000ep protocol as 056F.

#### §7.2.1 Raw Recap top-8 (single-shot 1000ep each)

```
ckpt 700  vs baseline: win_rate=0.878   [peak 1]
ckpt 1190 vs baseline: win_rate=0.878   [peak 2]
ckpt 1180 vs baseline: win_rate=0.877
ckpt 1160 vs baseline: win_rate=0.875
ckpt 930  vs baseline: win_rate=0.874
ckpt 1110 vs baseline: win_rate=0.874
ckpt 1130 vs baseline: win_rate=0.873
ckpt 1050 vs baseline: win_rate=0.869
```

Log: `docs/experiments/artifacts/official-evals/056E_baseline1000.log`

Plateau 0.85-0.88 (dual peak 700 + 1190 both @ 0.878).

#### §7.2.2 预注册判据 (§3 / §8) 对照

| Outcome | 阈值 | single-shot @700=@1190 | verdict |
|---|---:|---:|---|
| Outcome A (breakthrough ≥ 0.905) | LR landscape 还有余量 | 0.878 | **MISS** (-0.027) |
| Outcome B (056E marginal, 056F regression) | 3e-4 到 5e-4 为最优带 | 0.878 | **PARTIAL** — 056E is not marginal gain, it regresses |
| Outcome C (both regress) | 056D @ 3e-4 is the peak | 056E 0.878, 056F 0.868 | **HIT** — both regress |

#### §7.2.3 关键对比

- vs **055 SOTA combined 2000ep 0.907**: Δ = **-0.029 sub-SOTA** (~3× SE 0.010)
- vs **056D (LR=3e-4) 0.891**: Δ = **-0.013 regression** (borderline significant)
- vs **056F (LR=7e-4) 0.868**: Δ = +0.010 — ordering confirms monotone regression as LR increases past 3e-4
- vs **031B baseline 0.880**: Δ = -0.002 **tied 031B** (no net gain from 5× LR scaling)

#### §7.2.4 Complete LR sweep summary (056 series)

| Lane | LR | peak 1000ep WR | note |
|---|---:|---:|---|
| 031B (baseline) | 1e-4 | 0.880 | combined 2000ep |
| 056C | 1.5e-4 | 0.883 | [snapshot-056 §7](snapshot-056-simplified-pbt-lr-sweep.md) |
| **056D** | **3e-4** | **0.891** | **✓ optimum** (snapshot-056 §7.1) |
| 056E | 5e-4 | 0.878 | **regression** (this verdict) |
| 056F | 7e-4 | 0.868 | **larger regression** (§7.1) |

**LR optimum at 3e-4 strict** — both upward steps (5e-4 and 7e-4) monotonically regress. No 056G (1e-3) launched (§5 already pre-registered).

#### §7.2.5 Verdict & closure

- **056E lane CLOSED** — sub-SOTA confirmed on single-shot (WR 0.878 decisively below 055 0.907 + below 056D 0.891); not worth combined rerun
- **056 LR sweep full CLOSED**: 3e-4 is peak LR; going up regresses monotonically, going down was already tested (056C 1.5e-4) with modest gain only
- **LR axis saturated for this student setup** — remaining distance to SOTA must come from non-LR axes (distillation, architecture, reward path, training horizon)
- Resources fully redirected to distillation axis (Pool A/B/C/D + 076/077/078/079) + 055v2_extend rerun verification

### §8 后续路径

- **Outcome A**（breakthrough ≥ 0.905）：LR landscape 还有余量，延伸 056G lr=1e-3
- **Outcome B**（056E marginal，056F regression）：3e-4 到 5e-4 为最优带，sweep 结束
- **Outcome C**（两者均回归）：056D @ 3e-4 即为最优 LR；转向其他轴（distill、arch）

### §9 相关

- [snapshot-056](snapshot-056-simplified-pbt-lr-sweep.md) —— 前置实验（056A-D + 056extend）
- [snapshot-064](snapshot-064-056-pbt-full.md) —— PAUSED 替代方案（PBT-full），本 lane 是其替代
- [BACKLOG.md](BACKLOG.md) —— 056E/F 原列于 056 系列 backlog 下
- **理论依据**：
  - Schulman 2017 PPO paper —— 多数任务 baseline LR 为 3e-4
  - Andrychowicz 2020 "What Matters in On-Policy RL" —— LR 为 top-3 影响力 HP
- **Code targets**：
  - `scripts/eval/_launch_056E_lr5e4.sh`
  - `scripts/eval/_launch_056F_lr7e4.sh`
