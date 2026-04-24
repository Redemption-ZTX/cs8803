# SNAPSHOT-106: Stone-pipeline methodology corrections (Wave 3 root-cause + Layer 2 design)

- **日期**: 2026-04-22 ~07:30 EDT
- **状态**: 预注册 + queued items (3 P0/P1 fixes from Stone-paper deep-dive)
- **前置**: [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) (strategic synthesis) / [snapshot-100 §7B](snapshot-100-dir-A-heuristic-selector.md) (Wave 2 regression -0.110) / [snapshot-103](snapshot-103-dir-A-wave3-sub-task-specialists.md) (sub-task lanes)

---

## 0. 触发原因

DIR-A/G/E Wave 2 三框架同 regress (0.875→0.765 / 0.900→0.645 / 0.895→0.655)。Initial diagnosis = "specialist 太弱 drag down ensemble"。但**重读 Stone 4 paper** 揭示 **methodology 本身有 3 个根本错误**:

## 1. 方法论错误 #1 — 训练 distribution mismatch

103 sub-task 全用 `BASELINE_PROB=0.0` (vs RANDOM only),但 eval 用 ceia_baseline_agent。Specialist **从没见过 baseline opponent type**。

证据:
- 101A 也是 vs random 训 → baseline WR 0.851 (因 ball-control skill 普适)
- 103A/B/C 是 vs random 训 + scenario init → baseline WR 0.20-0.55 (skill 不普适到 baseline opponent)
- BASELINE_PROB=0.0 适合 "学独立 skill",**不适合 ensemble 用途**

**Fix (P2 queue)**: 所有 103-v2 lanes 必须 BASELINE_PROB ≥ 0.3。Snapshot-103 §6 execution checklist 加这条 rule。

## 2. 方法论错误 #2 — Stone Layered Learning bootstrap 跳过

Stone & Veloso 2000 *Layered Learning* 核心:

> "Each layer's learned behavior is used as **input** to the next layer.  
>  Bootstrapping: layer N training data comes from layer N-1 frozen policy."

我们的 Layer N 训练**没用** Layer N-1 frozen — 直接 from-scratch + scenario init。结果: scenario-specialist 学到的 skill 不能在 full-game team 协调下使用。

**正确 Layered Learning**:
- Layer 1 = 101A (ball-control,0.851) ✅ 我们有
- Layer 2 应该 = 101A frozen teammate + 训 partner 学 "pass coordination" ❌ 没做
- Layer 3 应该 = Layer 1+2 frozen + 训 team strategy ❌ 没做

**Fix (P0 queue)**: **103A-refined** lane —
- Warm-start 103A@500 (保留 INTERCEPTOR scenario knowledge)
- 续训 500 iter: BASELINE_PROB=0.7 + 标准 v2 reward + INTERCEPTOR aux reward + frozen 1750 teammate (Stone bootstrap)
- 期望 baseline WR 0.70-0.80 (vs 当前 0.548)
- 跑通后 apply 相同 pattern 给 103B/103C → 103B-refined / 103C-refined

## 3. 方法论错误 #3 — Wrong metric

Wang/Stone/Hanna ICRA 2025 (SPL 2024 winner) 用的不是 full-game WR,是 **task-success in controlled scenario**:

> "NEAR-GOAL policy scores in 9/10 1v2 scoring situations"

我们的 100/104/105 metric 全是 full-game WR (specialist solo ensemble)。这测的是 specialist 的 "team-game 能力",**不是 specialist 的 sub-task skill**。

**Fix (P1 queue)**: **`scripts/research/eval_subtask_success.py`** —
- 103A success criterion: ball recovered within 30 steps of duel init
- 103B success: opp_progress blocked for 100+ steps after defender init
- 103C success: ball advanced > 10m without losing possession
- 081 success: shots-on-target within 20 steps of near-goal init
- Output: success_rate per specialist (cleaner view of skill value)

## 4. Wave 3 中检测到的 (在跑) Wave 2 fundamental issue

DIR-G/E Wave 2 同样 regress (-0.255 / -0.240) — uniform/random routing 在 8-expert pool 中实际就是 weighted average:

数学验证:
- 8 expert 平均 standalone WR = (0.9155 + 0.907 + 0.86 + 0.826 + 0.851 + 0.548 + 0.205 + 0.220) / 8 = **0.667**
- 实测 DIR-G uniform Wave 2 = **0.645** ← 几乎完全 match (差 0.022 是 routing 随机噪声)

**结论**: 没 trained router 的情况下,加入弱 specialist 必然 degrade ensemble。需要:
- (a) **trained REINFORCE router** (Wave 3 of DIR-G plan, in BACKLOG)
- (b) **prune marginal experts** (drop 103B/C from pool until v2 fix)
- (c) **103A/B/C-refined** (lift specialists 到 0.7+ standalone before pool inclusion)

## 5. 任务队列 P0 总结 (新加)

| Item | 触发条件 | ETA | 价值 |
|---|---|---|---|
| **103A-refined** | 自由 node | 4h | break "specialist trained-eval distribution mismatch" 根因 |
| Sub-task success metric harness | 自由 node | 1.5h | clean metric for specialist skill (vs ensemble noise) |
| (后续) 103B-refined / 103C-refined | 等 103A-refined verify | 4h each | parallel after pattern proves |
| (后续) Layer 2 pass-decision specialist | 等 PIPELINE V1 | 4h | 真 Stone Layered Learning Phase 2 |

## 6. Wave 3 in flight 不阻塞此 snapshot

Wave 3 (14 evals: 5 ablation + 6 scenario-replay + 3 control) 仍在跑。它会告诉我们 **当前** specialists 在 ablation 框架下的 phase-conditional contribution。但即使 Wave 3 全 negative (specialists hurt every slot),**那也只是 v1 specialist 的 verdict,不是 specialist paradigm 的 verdict**。Wave 3 verdict 会写到 100/103/104/105 但不否定本 snapshot 提的 P0 fixes。

## 7. 与 snapshot 99-105 关系

- snapshot-099 strategic — 这里加 §5D "methodology corrections (snapshot-106)" 引用
- snapshot-100/104/105 — Wave 2 verdict + Wave 3 verdict 等 Wave 3 完成再 append; 但 §7B 已注 metric 局限
- snapshot-103 — §6 execution checklist 加 BASELINE_PROB ≥ 0.3 rule

## 8. Verdict (snapshot-106 self)

_Pre-registration only. Verdict = 103A-refined / sub-task harness 出结果时再 append._

预期:
- 103A-refined: baseline WR ≥ 0.70 → confirm methodology fix is real
- Sub-task harness: 103A success ≥ 0.80, 103B success ≥ 0.70, 103C success ≥ 0.80 → confirm specialists ARE good at their job, just full-game eval misleading

## 9. 相关

- [Stone & Veloso 2000 Layered Learning book](https://www.cs.utexas.edu/~pstone/book/) — bootstrap principle
- [Wang/Stone/Hanna ICRA 2025 arXiv 2412.09417](https://arxiv.org/abs/2412.09417) — task-success metric
- [snapshot-103](snapshot-103-dir-A-wave3-sub-task-specialists.md) — 103 sub-task lanes (need v2 fix per §1+§2)
- [snapshot-100 §7B](snapshot-100-dir-A-heuristic-selector.md) — Wave 2 regression evidence
- [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) — overarching strategic doc
- [task-queue.md](task-queue.md) — P0/P1 items added per this snapshot
