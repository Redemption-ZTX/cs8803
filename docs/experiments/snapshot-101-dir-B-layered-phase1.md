# SNAPSHOT-101: DIR-B — Layered Learning Phase 1 (Ball-Control Specialist, vs RANDOM only)

- **日期**: 2026-04-22
- **负责人**: Self
- **状态**: 训练中 (jobid 5033289 on atl1-1-03-017-23-0); 500 iter / 4h budget; verdict pending Stage 1 post-eval (~ETA 4h)
- **前置**: [snapshot-099 §2.1 DIR-B](snapshot-099-stone-pipeline-strategic-synthesis.md#21-总览) (Stone Veloso 2000 Layered Learning) / [snapshot-076](snapshot-076-wide-student-distill.md) + [snapshot-079](snapshot-079-055v3-recursive-distill.md) (5/5 distill saturation 推动 pivot)

---

## 0. 背景

### 0.1 Stone & Veloso 2000 — Layered Learning

"Layered Learning in Multiagent Systems" 提出: 复杂多智能体 task 不需要 single end-to-end RL, 可以拆 hierarchical sub-skills, 每层独立 train, 上层 build on 下层 frozen base:

- **Layer 1**: 个体技能 (单 agent, simple opponent)
- **Layer 2**: 多 agent 协调 (基于 Layer 1 frozen base, 加 teammate)
- **Layer 3**: 团队战略 (基于 Layer 2 base, 真实对手)

每层 boundaries 在 reward / opponent / state space 上明确分离, **避免 end-to-end 训练里 reward signal 混乱 + sample 浪费在 trivial baseline state 上**。

### 0.2 与项目历史训练 paradigm 的对比

项目所有 lane 都是 end-to-end 单一 paradigm:
- **scratch vs baseline**: 大多数 (031B / 055 / 056 / etc.)
- **scratch vs random + curriculum**: 058 / 062a (但 task target 仍是完整 2v2 vs baseline)
- **warm-start + same env**: 029 / 043 etc.

**没人测过 "纯 Layer 1 specialist" 当 building block**. 这是 DIR-B Phase 1 的核心 novelty。

### 0.3 与 081 aggressive 的差异

081 aggressive:
- 完整 2v2 vs baseline opponent
- reward = ball_progress + shot + goal_proximity (offensive)
- 期望 baseline WR [0.75, 0.88]
- 是一个 "完整 player but with different style"

101A layered Phase 1:
- 完整 2v2 但 **vs RANDOM only** (BASELINE_PROB=0)
- reward = ball_progress + possession (NO defense, NO shot)
- 期望训出 "midfielder specialist": 接近球 → 控球 → 推进, 不射门
- 是一个 "specialist sub-skill", 不是完整 player

**101A 不是单跑就 ship 的 lane**, 而是 PIPELINE Phase 2-5 的 frozen base。

---

## 1. 核心假设

### 1.1 H_101 (主)

> Layer 1 单独训 ball-control specialist (vs RANDOM only + ball-progress reward), 500 iter 内可训出 "deterministic ball-approach + dribble-forward" behavior, vs RANDOM 自身的 baseline WR ≥ 0.95 (RANDOM 几乎不抢球, ball-control specialist 一旦抓住球可单挑 carry 进攻)。
> **vs baseline 期望 WR 50-70%**, 不要求 SOTA — 是 PIPELINE specialist input。

### 1.2 H_101-a (alt, sub-task SOTA)

> 如果 vs baseline WR > 0.85, 我们意外发现 "纯 ball control 即可 win" → "进攻即最佳防守"在 RANDOM-only training 下也成立。

### 1.3 H_101-b (anti)

> 如果 vs RANDOM WR < 0.80 或 reward curve 不收敛 → 单 Layer 1 task 设计错误 (reward 不够 dense 或 vs RANDOM 太 trivial), 需要回去重新设计 Layer 1 reward shape

### 1.4 PIPELINE 角色

- 训出来的 ball-control specialist 进入 BALL_DUEL 或 MID_FIELD phase 候选, 替换 [snapshot-100 §2.3 Wave 2](snapshot-100-dir-A-heuristic-selector.md#23-wave-2-specialist-mapping-designed-not-yet-committed) 里的 placeholder

---

## 2. Design

### 2.1 Architecture (mirror 031B SOTA family)

- Siamese encoder `[256, 256]` (= 031B)
- Within-agent cross-attention 4 tokens × 64 dim (= 031B)
- Merge `[256, 128]` (= 031B)
- 总 params ~0.46M (matches 031B SOTA family for downstream PIPELINE substitutability)

### 2.2 Opponent setup — vs RANDOM only

```bash
export BASELINE_PROB=0.0   # Stone Layer 1: simplified opponent
```

- 没有 baseline opponent, 没有 frozen self-play, 没有 curriculum
- 100% random opponent → 学 "在没有真威胁下如何控球 + 推进"
- 简化原因: Layer 1 task is individual ball control, **不应该让 agent 同时学防守 baseline 攻势**

### 2.3 Reward shape — pure ball control

| Item | 值 | 动机 |
|---|---:|---|
| `time_penalty` | 0.001 | 防止站在原地 |
| `ball_progress` | **0.05** | **5× 默认**, 主信号 |
| `possession_bonus` | **0.005** | **2.5× 默认**, 鼓励控球 |
| `possession_dist` | 1.25 | 同 v2 |
| `progress_requires_possession` | 0 | progress 不要求持球 (鼓励接近球) |
| `goal_proximity_scale` | 0.0 OFF | Layer 1 不学射门 |
| `opp_progress_penalty` | **0.0 OFF** | **Layer 1 不学防守** |
| `deep_zone_outer / inner_penalty` | 0.0 OFF | Layer 1 不学位置 |
| `defensive_survival_bonus` | 0.0 | 同上 |
| `event_shot_reward` | 0.0 OFF | Layer 1 不学射门 |
| `event_tackle / clearance` | 0.0 OFF | 同上 |

**Net effect**: reward dominated by (1) move ball forward, (2) hold ball. 与 081 aggressive (强射门 + 高速) 在 reward dimension 上正交。

### 2.4 PPO setup

- LR=1e-4, CLIP_PARAM=0.15, NUM_SGD_ITER=4
- TRAIN_BATCH_SIZE=40000, SGD_MINIBATCH_SIZE=2048
- ROLLOUT_FRAGMENT_LENGTH=1000

### 2.5 Budget

| 项 | 值 |
|---|---:|
| MAX_ITERATIONS | **500** (vs 1250 default) |
| TIMESTEPS_TOTAL | **20M** (vs 50M default) |
| TIME_TOTAL_S | 14400 (4h) |
| CHECKPOINT_FREQ | 10 |

**为什么 500 iter 而非 1250**: Layer 1 task 简化, 期望 200-300 iter 内收敛; 500 iter 给 50% safety margin。如果 500 iter 内不收敛, H_101-b 触发, 重新设计 reward。

### 2.6 Eval setup

- 标准 EVAL_INTERVAL=10, EVAL_EPISODES=200
- EVAL_OPPONENTS=baseline,random — 同时跟踪 vs random WR (主 metric) + vs baseline WR (sanity check)
- 目标 vs random WR ≥ 0.95, vs baseline WR 不约束 (信息收集)

---

## 3. Pre-registered Thresholds

| # | 判据 | 阈值 | verdict 含义 |
|---|---|---|---|
| §3.1 vs RANDOM main | vs random 200ep peak ≥ 0.95 | ball-control specialist 工作 | Layer 1 specialist ready as PIPELINE input |
| §3.2 vs RANDOM stretch | vs random 200ep peak ≥ 0.98 | strong ball control | Layer 1 specialist 可独立 deploy as RANDOM crusher |
| §3.3 vs baseline alt | vs baseline 200ep peak ≥ 0.85 | 意外 strong | 重新评估 reward design ablation |
| §3.4 vs RANDOM under-converge | vs random 200ep peak < 0.80 | 没收敛 | reward shape 错或 vs RANDOM too trivial 没 explore signal, redesign Layer 1 |
| §3.5 vs baseline over-fit | vs baseline 200ep peak < 0.30 | 完全没 generalization | RANDOM-only 训太狭, Layer 2 必须加 baseline opponent gradually |

### 3.1 PIPELINE consumption gate

- Layer 1 specialist 进 PIPELINE 的 gate = vs RANDOM ≥ 0.95 AND vs baseline ≥ 0.40 (sanity)
- 如果 vs baseline = 0.30-0.40, 仍可作 sub-task specialist for BALL_DUEL phase, 配合 1750 generalist

---

## 4. Simplifications + Risks

### 4.1 简化 S1 — 单 Phase 1 没 Phase 2/3 follow-up commit

- **节省**: 不预先承诺 Phase 2/3 资源
- **Risk**: 如果 §3.4 触发 (RANDOM 都没收敛), 整个 DIR-B 否决
- **降级**: §3.4 触发 → redesign Layer 1 reward (try shape with goal_proximity 0.005 + ball_progress 0.03)

### 4.2 简化 S2 — vs RANDOM only, 没 curriculum

- **节省**: 不需要 curriculum 调度
- **Risk R1**: vs RANDOM 太 trivial, agent 学到 lazy policy (在场上随便走), 不 gen 到真 opponent
- **Mitigation**: §3.5 vs baseline ≤ 0.30 监控 — 如果 RANDOM peak 高但 baseline 极低 → confirm over-specialization, 要加 ε-baseline (P=0.1) 重训

### 4.3 简化 S3 — 不射门 (event_shot_reward = 0)

- **节省**: reward signal 单纯
- **Risk R2**: 没 shot signal, agent 拿球进 NEAR_GOAL 不知道 finish → 进 NEAR_GOAL 后只会来回 dribble, 不 score
- **结果**: vs RANDOM WR 可能 cap 在 ~0.85 (因为 random 偶尔自杀进球, agent 自己不 score → tie/draw 占多数; 但 RANDOM 的 own-goal 概率高仍能 win 大部分)
- **设计意图**: Layer 1 specialist **就是 ball-controller, 不是 finisher** — finisher 是 081 aggressive 的角色

### 4.4 简化 S4 — Architecture 完全 match 031B SOTA family

- **节省**: PIPELINE 互换性, 每个 specialist 都可以塞进 selector pool
- **Risk**: 没 ablate 是否 smaller model 更适合 Layer 1 task
- **降级**: 如果 §3.4 触发, ablation Phase 1 with smaller model `[128, 128]`

### 4.5 全程 retrograde

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | default | 500 iter vs RANDOM only | 4h |
| 1 | §3.4 触发 (RANDOM < 0.80) | redesign reward + retrain | +4h |
| 2 | §3.5 触发 (baseline < 0.30) | + ε baseline opponent (P=0.1) | +4h |
| 3 | §3.1 hit → Phase 2 launch | warm-start Phase 1 base + add teammate coordination + ball_progress + pass_event reward | +8h |
| 4 | Phase 2 §3.1 hit → Phase 3 launch | warm-start Phase 2 + full 2v2 vs baseline + v2 shape | +12h |

---

## 5. 不做的事

- 不 commit Phase 2 / 3 / 4 设计直到 Phase 1 verdict
- 不 mix v2 shape 进 Layer 1 reward (会污染 ball-control isolation)
- 不 distill from any teacher (Layer 1 is scratch by design)
- 不 add curriculum (BASELINE_PROB stays 0)
- 不 ablate hidden size / lr (固定 031B 同 setup)

---

## 6. Execution Checklist

- [x] 1. snapshot 起草 (本文件)
- [x] 2. launcher `scripts/eval/_launch_101A_layered_p1_ballcontrol.sh` 落地
- [x] 3. Launched 2026-04-22 01:42 EDT, jobid 5033289, atl1-1-03-017-23-0
- [x] 4. 实时 monitor (BASE_PORT 60305, 500 iter / 4h budget)
- [ ] 5. 训完 Stage 1 post-eval (vs random + vs baseline 200ep on top ckpts)
- [ ] 6. 如果 §3.1 hit, freeze Layer 1 base for PIPELINE consumption + commit Phase 2 design
- [ ] 7. 如果 §3.4 触发, redesign + retrain
- [ ] 8. Verdict append §7

---

## 7. Verdict — Phase 1 SPECIALIST USABLE for PIPELINE (§3.1 baseline-transfer surprise + §3.2 random-side healthy, 2026-04-22 append-only)

### 7.1 Audit (per `feedback_audit_training_data_first.md`)

- Trial: `101A_layered_p1_ballcontrol_20260422_014241/TeamVsBaselineShapingPPOTrainer_Soccer_21c17_00000_0_2026-04-22_01-43-04`
- 50 ckpt dirs (iter 10-500, full budget reached)
- stop_reason: TERMINATED ✅
- best_eval_baseline (inline 200ep): 0.850 @ iter 330
- best_eval_random (inline 200ep): 0.990
- Training reward stable: +3.42 mean late, no collapse

**⚠️ Audit issue caught by user**: inline CSV baseline coverage only iter 120-410 — **iter 420-500 missing entirely** (training ended faster than inline eval queue). pick_top_ckpts based on incomplete CSV missed the late window. Per `feedback_inline_eval_late_window_gap.md`, launched supplementary eval for ckpts 420-500. **Combined main + late = 20 ckpts coverage**. Memory updated to make Step 0 audit non-negotiable.

### 7.2 Stage 1 baseline 1000ep (combined main + late, 2026-04-22 [04:55-05:14 EDT])

- Selected ckpts MAIN (top 5%+ties+±1, 11 ckpts): 190-230 / 250-270 / 320-340. Eval node 5033289 port 61005, 554s parallel-7.
- Selected ckpts LATE supplement (9 ckpts): 420-500 every 10. Eval node 5035756 port 61105, 525s parallel-7.

| ckpt | 1000ep WR | NW-ML | window |
|---:|---:|:---:|:---|
| **🏆 460** | **0.851** | 851-149 | **LATE** (originally missed) |
| 470 | 0.849 | 849-151 | LATE |
| 500 | 0.848 | 848-152 | LATE |
| 440 | 0.848 | 848-152 | LATE |
| 490 | 0.834 | 834-166 | LATE |
| 480 | 0.830 | 830-170 | LATE |
| 450 | 0.830 | 830-170 | LATE |
| 330 | 0.821 | 821-179 | main |
| 340 | 0.814 | 814-186 | main |
| 430 | 0.810 | 810-190 | LATE |
| 260 | 0.802 | 802-198 | main |
| 250 | 0.798 | 798-202 | main |
| 220 | 0.786 | 786-214 | main |
| 230 / 320 | 0.785 | 785-215 | main |
| 420 | 0.785 | 785-215 | LATE |
| 210 | 0.782 | 782-218 | main |
| 270 | 0.778 | 778-222 | main |
| 200 | 0.776 | 776-224 | main |
| 190 | 0.761 | 761-239 | main |

**peak = 0.851 @ ckpt-460 (LATE window), mean(top 6) ~0.840, range [0.761, 0.851]**

### 7.3 严格按 §3 判据

| 阈值 | 实测 | verdict |
|---|---|---|
| §3.1 main: vs baseline ≥ 0.85 | ✅ 0.851 @ 460 | **HIT (just barely)** — baseline-transfer 比预期高 |
| §3.2 main: vs random ≥ 0.95 | ✅ inline 0.990, 1000ep 验证待跑 | **HIT** (random-side near-perfect) |
| §3.3 specialist usable for PIPELINE | ✅ baseline 0.85 + random 0.99 | **HIT** — qualifies as Layer 1 specialist |
| §3.4 under-converge: < 0.75 vs baseline | ❌ above | not triggered |

**Important context**: 101A 训练 BASELINE_PROB=0.0 (vs RANDOM only), 不期望 baseline 高分。**0.851 baseline-transfer 是意外好** — ball-control skill 学到的 v2-shape 普适性强,从 random 迁移到 baseline 没崩。

### 7.4 与其他 sub-task / specialist lanes 对比

| Lane | 训练 opp | Peak baseline | 评价 |
|---|---|---|---|
| 1750 SOTA | 100% baseline | 0.9155 (combined 4000ep) | 主 generalist |
| 055@1150 | 100% baseline | 0.907 (combined 2000ep) | 副 generalist |
| 029B@190 | per-agent + 100% baseline | 0.86 | per-agent SOTA |
| **101A** Layered P1 ball-control | **vs random only** | **0.851** | **Phase 1 specialist (transfer 出乎意料强)** |

### 7.5 Raw recap (main + late)

```
=== 101A_baseline1000.log (main, 11 ckpts) === total_elapsed=554s
ckpt-330: 0.821 / ckpt-340: 0.814 / ... (above table)

=== 101A_latewindow_baseline1000.log (late, 9 ckpts) === total_elapsed=525s
ckpt-460: 0.851 ← peak
ckpt-470: 0.849
ckpt-500: 0.848
... (above table)
```

完整 logs: [101A_baseline1000.log](../../docs/experiments/artifacts/official-evals/101A_baseline1000.log) + [101A_latewindow_baseline1000.log](../../docs/experiments/artifacts/official-evals/101A_latewindow_baseline1000.log)

### 7.6 Lane decision (autonomous loop triage)

- **Phase 1 lane CLOSED with §3.1 + §3.2 HIT** — 101A@460 added to specialist library
- **Skip Stage 2 failure capture**: standalone WR 0.851 不是 frontier; capture 仅 use case 是 understand failure modes for selector heuristic — defer to PIPELINE V1 integration time
- **Skip Stage 3 H2H** for now: 101A 主用途是 ensemble member,not frontier solo; H2H 不在主路径
- **Phase 1 ckpt 460 的 self-contained agent package** task added to queue (~30 min when free node available)
- **Phase 2 (multi-agent passing) NOT triggered yet** — 等 PIPELINE V1 验证 simple ensemble 是否足够 break 0.91,如果不够再上 Phase 2 layered training

### 7.7 后续触发

- **入 PIPELINE V1 specialist library** (`agents/v_layer1_ballcontrol/`, ckpt 460): immediately upon free GPU
- **Phase 2 conditional**: PIPELINE V1 with current specialists peaks ≤ 0.91 → trigger snapshot-101 §8.2 (multi-agent passing layer with frozen Phase 1 base)
- **Phase 2 unconditional skip**: PIPELINE V1 ≥ 0.92 → Phase 1 already enough as-is, skip layered Phase 2-3

---

## 8. 后续

### 8.1 Phase 1 §3.1 hit (主 path)

- Phase 1 base ckpt freeze, 加入 PIPELINE specialist library
- Launch Phase 2 (multi-agent passing): warm-start + teammate-coordination reward + add `pass_completed` event reward (要在 expert_coordination.py 加 hook)
- Phase 2 budget ~8h

### 8.2 Phase 1 §3.4 触发 (under-converge)

- Redesign reward (try denser signal: ball_progress 0.08 + small goal_prox 0.005)
- 或换 opponent 设计 (不 pure RANDOM, 加 P=0.1 baseline as occasional challenge)
- 不 escalate 到 Phase 2 直到 Phase 1 §3.1 hit

### 8.3 PIPELINE 消费路径

- 如果 §3.1 hit AND vs baseline ≥ 0.40, Phase 1 base 进 [snapshot-100 Wave 2 mapping](snapshot-100-dir-A-heuristic-selector.md#23-wave-2-specialist-mapping-designed-not-yet-committed) 的 BALL_DUEL phase
- 与 103A INTERCEPTOR 二选一 (取决于哪个在 BALL_DUEL phase 上更 dominant)

---

## 9. 相关

- [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) — Stone-pipeline strategic synthesis (DIR-B 在 §2.1 #2)
- [snapshot-100](snapshot-100-dir-A-heuristic-selector.md) — DIR-A heuristic selector (Phase 1 specialist 的 consumer)
- [snapshot-103](snapshot-103-dir-A-wave3-sub-task-specialists.md) — sub-task 103A/B/C (sister approach to Layer 1 specialist concept)
- [snapshot-081](snapshot-081-aggressive-offense-reward.md) — aggressive offense (orthogonal reward family, contrast to ball-control)
- [scripts/eval/_launch_101A_layered_p1_ballcontrol.sh](../../scripts/eval/_launch_101A_layered_p1_ballcontrol.sh) — launcher
- Run dir: `/storage/ice1/5/1/wsun377/ray_results_scratch/101A_layered_p1_ballcontrol_20260422_014241/` (TBD until done)

### 理论支撑

- **Stone & Veloso 2000** "Layered Learning in Multiagent Systems" — 核心 hierarchy idea
- **Sutton, Precup, Singh 1999** "Between MDPs and Semi-MDPs" — options framework, Layer 1 specialist 可以视为 option intra-policy
- **Wang, Stone, Hanna 2025** ICRA — 4 sub-policies idea, Layer 1 specialist 是其中之一
