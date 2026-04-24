# SNAPSHOT-107: Stone Layered Learning Phase 2 — Pass-decision specialist

- **日期**: 2026-04-22
- **负责人**: Self
- **状态**: 预注册 (pre-registered, design phase complete; implementation pending launch)
- **前置**:
  - [snapshot-101](snapshot-101-dir-B-layered-phase1.md) — Phase 1 ball-control (101A@460 = 0.851 baseline)
  - [snapshot-103 §7E/§7F](snapshot-103-dir-A-wave3-sub-task-specialists.md) — Stone Layered L2 lessons (warm-source competence, KL anchor)
  - Memory `feedback_stone_layered_warm_source.md` — Layer N warm must be ≥0.85 baseline-competent
  - Memory `feedback_opponent_pool_no_resample.md` — BUG-1 fix (per-ep opp resample)
  - Memory `feedback_curriculum_bug1_dead.md` — set_pool_weights doesn't trigger reset

## 0. 背景 — 为什么做这个

### 0.1 Peter Stone 全线遗漏 (用户 2026-04-22 提出)

Stone & Veloso 2000 "Layered Learning in Multiagent Systems" 原文 RoboCup-99 CMUnited-99 用 3 层：
- **Layer 1**: ball interception (~95% success)
- **Layer 2**: pass-decision (when to pass vs dribble)
- **Layer 3**: team coordination (positional play)

我们项目至今只完成 Layer 1 (101A@460 0.851)。Phase 2 / Phase 3 **未做**。这是 Stone 全线最显著的实现 gap。

### 0.2 为什么 Phase 2 现在最有价值 (今日 verdict 整合)

- Stone L2 Phase 1 from competent warm (101A): paradigm validated
- 103A-warm-distill (Stone L2 from 1750 SOTA + INTERCEPTOR scenario): 0.914 combined ABOVE 1750 真值 (corrected to 0.9066)
- Stone Layered L2 paradigm 已 validated, 但 **only on INTERCEPTOR sub-task**. Pass-decision 是 stone 原文的 layer 2, 同 paradigm 应同样有效甚至更好（因为 PASS 是真 team-coord 技能, 比 INTERCEPTOR 更"团队层"）
- Phase 2 from competent Layer 1 warm = 真正的 Stone Layered (不是 "1750 + scenario perturbation")

### 0.3 与 103A-warm-distill 的关系

103A-warm-distill 是 "**1750 SOTA + INTERCEPTOR scenario**" — paradigm 相同但 warm source 不是 Layer 1。
本 lane 是 "**Layer 1 (101A) + Pass scenario**" — true Stone Layered Phase 2。

如果两者都成功，证明 Stone Layered 是 generalizable 的，**not just 1750 perturbation**。

---

## 1. Hypothesis

### 1.1 主假设 H_104A

> 从 101A@460 (Layer 1 specialist, baseline 0.851) warm-start + KL distill anchor (α 0.05→0.005 over 39000 updates) + `SCENARIO_RESET=pass_subtask` (NEW) + BASELINE_PROB=0.7 + light pass_event_reward (0.05 per pass), 训练 500 iter / 20M steps, **combined 2000ep peak ≥ 0.92** (= +0.014 over 1750 真值 0.9066, AND ≥ 103A-warm-distill@400 0.914 = stone layered structurally generalizable beyond INTERCEPTOR)。

### 1.2 子假设

- **H_104A-a**: pass_completed event 是真正的 team-coord 技能 — 训出来的 policy H2H vs 1750 应该 ≥ 0.55 (因为 1750 缺乏 explicit pass training, 即使 baseline 上 1750 强, 在 team-coord 能力上应有可探索空间)
- **H_104A-b**: pass scenario 的 state distribution 与 baseline play 有 overlap (不像 103A INTERCEPTOR 的 BALL_DUEL 是 narrow subset), 所以应 transfer 到 baseline well — combined ≥ 0.92 reachable
- **H_104A-c**: 101A@460 baseline 0.851 < 1750 0.9066, 但 Layer 1 specialist 经 Phase 2 训练 后应 catch up 并 surpass 101A baseline (≥ 0.90)

### 1.3 Anti-hypothesis

- **H_104A-null**: pass scenario init 与真 game 偏太多, 训出的 policy 在 baseline play 上 sub-frontier (< 0.88)
- **H_104A-bad**: pass event reward 噪音过大 / 误检率高 (滚球被识别为 pass), 引导 policy 学到 wrong behavior
- **H_104A-warm-collapse**: 101A 0.851 capacity 不够 — 经 Phase 2 训练后既忘掉 ball-control, 又没学好 pass, 退化到 < 0.851 baseline

---

## 2. Design

### 2.1 Pass scenario init (NEW: pass_subtask mode)

```python
# In cs8803drl/branches/expert_coordination.py
PASS_SUBTASK_SCENARIO = "pass_subtask"

# When mode == PASS_SUBTASK_SCENARIO:
# Variable distance 6-12m between teammates, ball with one of them,
# opp close 70% / opp neutral 30%
def _sample_pass_scenario():
    # Ball-holder agent (player0)
    player0 = _sample_position((-12.0, -2.0), (-3.0, 3.0))
    
    # Teammate (player1) at variable distance + angle
    distance = _sample_uniform((6.0, 12.0))
    angle = _sample_uniform((-1.5, 1.5))  # radians, mostly forward-spread
    player1 = [
        player0[0] + distance * np.cos(angle),
        player0[1] + distance * np.sin(angle),
    ]
    # Clamp to field
    player1[0] = max(-15.0, min(10.0, player1[0]))
    player1[1] = max(-3.5, min(3.5, player1[1]))
    
    # Ball at player0 (close to body)
    ball = [
        player0[0] + _sample_uniform((-0.6, 0.6)),
        player0[1] + _sample_uniform((-0.4, 0.4)),
    ]
    
    # Opp positioning: 70% close to ball-holder (force pass), 30% neutral (allow dribble)
    if random.random() < 0.7:
        # Opp2 close to ball-holder
        opp2 = [
            player0[0] + _sample_uniform((1.0, 3.0)),
            player0[1] + _sample_uniform((-1.5, 1.5)),
        ]
        # Opp3 between teammates (covering pass lane partially)
        opp3 = [
            (player0[0] + player1[0]) / 2 + _sample_uniform((-2.0, 2.0)),
            (player0[1] + player1[1]) / 2 + _sample_uniform((-2.0, 2.0)),
        ]
    else:
        # Neutral spacing
        opp2 = _sample_position((2.0, 8.0), (-3.0, 3.0))
        opp3 = _sample_position((4.0, 10.0), (-3.0, 3.0))
    
    return {
        "players_states": {
            0: {"position": player0, "rotation_y": 0.0},
            1: {"position": player1, "rotation_y": 0.0},
            2: {"position": opp2, "rotation_y": 180.0},
            3: {"position": opp3, "rotation_y": 180.0},
        },
        "ball_state": {"position": ball, "velocity": [0.0, 0.0]},
    }
```

### 2.2 Pass event detection (NEW: SHAPING_EVENT_PASS_REWARD)

Pass event = "ball possession transitions FROM agent A TO agent B WHERE both agents are on team0":

```python
# In cs8803drl/core/soccer_info.py compute_event_shaping(): add
def compute_event_shaping(
    ...,
    possessing_agent: Optional[int],  # NEW: which AGENT (0/1/2/3) has ball
    prev_possessing_agent: Optional[int],  # NEW: previous step
    event_pass_reward: float = 0.0,
    pass_min_speed: float = 1.0,  # ball velocity threshold
    pass_max_steps_to_receive: int = 30,
    ...
):
    ...
    # NEW: pass detection
    if (
        possession_confirmed
        and possessing_agent is not None and prev_possessing_agent is not None
        and possessing_agent != prev_possessing_agent
    ):
        # Both must be on same team
        same_team = (possessing_agent in TEAM0 and prev_possessing_agent in TEAM0) or \
                    (possessing_agent in TEAM1 and prev_possessing_agent in TEAM1)
        if same_team:
            team_id = 0 if possessing_agent in TEAM0 else 1
            # Ball had to actually MOVE (not just possession ambiguity)
            if ball_dx is not None and abs(ball_dx) >= pass_min_speed:
                _trigger_team("event_pass", team_id, event_pass_reward)
```

**Edge cases / mitigations**:
- 球员 A 抢断 → 球员 B 解围: NOT pass (ball jumps long-distance with no intent). Mitigation: also require `pass_max_steps_to_receive` < N steps between A losing and B gaining (if took >30 steps it's not a pass, it's loose-ball recovery).
- 滚球被双方 player 经过: 第一次 trigger 后冷却 cooldown_steps (复用现有机制) 避免重复 trigger.

### 2.3 Recipe (per memory: Stone L2 lessons + bug fixes)

```bash
# Warm + KL anchor from 101A@460 (Layer 1 specialist, 0.851 baseline-competent)
WARMSTART_CHECKPOINT=<101A@460 path>
TEAM_DISTILL_ENSEMBLE_KL=1
TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS=<101A@460 path>
TEAM_DISTILL_ALPHA_INIT=0.05
TEAM_DISTILL_ALPHA_FINAL=0.005       # residual anchor (BUG-2 fix lesson)
TEAM_DISTILL_DECAY_UPDATES=39000     # covers 500 iter (BUG-2 fix)

# Scenario init
SCENARIO_RESET=pass_subtask          # NEW

# Distribution match (BUG-1 fix in core code automatic now)
BASELINE_PROB=0.7

# Reward
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001
SHAPING_BALL_PROGRESS=0.01           # v2 base
SHAPING_OPP_PROGRESS_PENALTY=0.02    # mild defense
SHAPING_POSSESSION_BONUS=0.002       # v2 base
SHAPING_EVENT_PASS_REWARD=0.05       # NEW — main specialty signal
SHAPING_EVENT_PASS_MIN_SPEED=1.0
SHAPING_EVENT_PASS_MAX_STEPS=30
SHAPING_EVENT_COOLDOWN_STEPS=10
# 其他 v2 default

# Architecture: same as 101A (031B Siamese + cross-attn) — match warm ckpt
TEAM_SIAMESE_ENCODER=1 TEAM_SIAMESE_ENCODER_HIDDENS=256,256 TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1 TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

# Budget per memory feedback_slurm_wall_budget.md
TIMESTEPS_TOTAL=20000000 MAX_ITERATIONS=500 TIME_TOTAL_S=14400
```

### 2.4 Implementation 清单

| # | File | LOC | Status |
|---|---|---|---|
| 1 | `cs8803drl/branches/expert_coordination.py` — add PASS_SUBTASK_SCENARIO + sample branch | ~50 | TODO |
| 2 | `cs8803drl/core/soccer_info.py` — extend compute_event_shaping with possessing_agent + pass detection | ~70 | TODO |
| 3 | `cs8803drl/core/utils.py` — RewardShapingWrapper passthrough event_pass_reward | ~30 | TODO |
| 4 | `cs8803drl/training/train_ray_team_vs_baseline_shaping.py` — env var SHAPING_EVENT_PASS_REWARD passthrough | ~15 | TODO |
| 5 | `scripts/eval/_launch_104A_layered_p2_passdecision.sh` — launcher | ~150 | TODO |
| 6 | smoke test — verify pass_subtask scenario init + pass detection works | ~60 | TODO |

**Total ~375 LOC + this snapshot doc.**

---

## 3. 预注册判据

| § | 阈值 | verdict |
|---|---|---|
| §3.1 main: combined 2000ep peak ≥ 0.92 | +0.014 over 1750 真值 0.9066 | **Stone Layered Phase 2 paradigm validated as ABOVE INTERCEPTOR variant** |
| §3.2 SOTA-tied: combined 2000ep peak ∈ [0.910, 0.92) | tied with 103A-warm-distill@400 | paradigm 同 INTERCEPTOR variant 同强 |
| §3.3 Layer 1 lift: combined 2000ep peak ≥ 0.88 | +0.029 over 101A 0.851 | Phase 2 实现 over Phase 1 lift |
| §3.4 sub-frontier: combined 2000ep peak ∈ [0.85, 0.88) | tied 101A | Phase 2 持平 Phase 1, paradigm 弱 |
| §3.5 regression: combined < 0.85 | below 101A | Phase 2 失败 — 忘 Phase 1 + 没学 pass |

**Peer axis**:
| § | 阈值 | verdict |
|---|---|---|
| §3.6 H2H vs 1750 ≥ 0.55 z>2.24 | beat SOTA in direct play | **NEW SOTA via peer-axis** (与 baseline 0.91+ 一致 → solid SOTA shift) |
| §3.7 H2H vs 1750 ∈ [0.47, 0.53] | tied | paradigm 与 1750 same-strength |
| §3.8 H2H vs 1750 < 0.47 | LOSE | baseline-axis HIT 但 peer-axis fail (类似 083 per-ray) |

---

## 4. 风险

### R1 — Pass detection false positive
滚球 / 两个球员竞争中互相经过 → 误识别 pass。
**缓解**: cooldown 10 步 + min ball speed 1.0 + max steps to receive 30。

### R2 — 101A 0.851 base capacity 不够
101A 训练 500 iter, capacity ~031B. Phase 2 加 KL anchor + new reward, 可能 over-fit pass scenario, baseline 退化。
**缓解**: KL anchor α residual 0.005 keep Phase 1 ball-control 知识; BASELINE_PROB=0.7 maintain baseline distribution.

### R3 — Pass scenario state distribution 与真 game 偏太远
真 baseline play 中, "球员 A 持球 + 球员 B 6-12m 远 + opp 接近" 是 narrow subset (大部分 baseline 状态是 transition / contested ball)。
**缓解**: BASELINE_PROB=0.7 中 30% 不用 scenario init (走标准 reset), 保持 game-distribution exposure.

### R4 — Frozen teammate 缺失
真 Stone L2 = Layer N 训练时 Layer N-1 frozen as teammate. 我们 team-level 架构两 agent 共享 policy, 双 agent 都被训, 不是 "frozen 教学". 这是 Layered Lite, 不是 full Stone.
**缓解**: warm-from-101A 让两 agent 都从 Phase 1 起步, KL anchor 拉回 101A; 这接近 "self-teach Phase 2 with Layer 1 prior", 不完美但可行。如果 Lite 成功, 后续做 Layered Full (per-agent + frozen teammate) 作 §8.

### R5 — Pass reward 0.05 太小 / 太大
小则 specialty signal 不足 → 退化为 "warm + KL only"; 大则 dominate 其他 reward → over-pass.
**缓解**: 0.05 vs ball_progress 0.01 比例 5x, 但 pass 是 sparse event (一局可能 5-15 次), 总贡献 ~ 0.25-0.75 / episode, 与 v2 shaping 同量级. 如 over-pass, 后续 ablate.

---

## 5. 不做的事 (out of scope)

- **不做 frozen teammate (Layered Full)** — §R4 mitigation 用 Lite 路径; Full 留 §8 backlog
- **不做 Phase 3 team coordination** — 等 Phase 2 verdict 后决定
- **不改 v2 base shaping** — 只 ADD event_pass_reward, 其他保持 v2 standard
- **不做 multi-scenario mix** — 只用 pass_subtask (avoid INTERCEPTOR + pass + dribble 混合 scenario noise)
- **不在 1750 上做 pass-distill** — 那是另一种 path (类似 103A-warm-distill on 1750), 不是 Layered

---

## 6. 执行清单

1. **Implementation** (engineering):
   - [ ] expert_coordination.py: add PASS_SUBTASK_SCENARIO
   - [ ] soccer_info.py: pass event detection
   - [ ] utils.py: reward wrapper passthrough
   - [ ] train script: env var passthrough
   - [ ] launcher script
   - [ ] smoke test (env reset + 100 step trace + pass event count check)

2. **Launch & training** (~4h GPU on free node)

3. **Stage 1 baseline 1000ep eval** (post-train)

4. **Stage 2 rerun** if peak ≥ 0.91 single-shot

5. **Stage 3 H2H vs 1750** if combined 2000ep ≥ 0.91

6. **Verdict write** — append to this snapshot

---

## 7. Verdict

_Pending — 104A/B training in flight (2026-04-23 ~01:30 EDT, ETA 03:30/04:00 EDT)._

### 7.0 PRE-VERDICT REVISION 2026-04-23 01:36 (post-103A-wd v2 5000ep walk-back)

After 103A-wd v2 ckpt 400 combined 5000ep eval gave **TIED with 1750 真值 0.9066** (NOT above), the §3 verdict thresholds for 104A/B should be reinterpreted given the established 0.91 ceiling:

| Original threshold | Revised reading (post 0.91 ceiling discovery) |
|---|---|
| §3.1 main ≥ 0.92 → "paradigm-generalize beyond INTERCEPTOR" | UNREACHABLE under current architecture (031B Siamese capped at 0.91); HIT would be REVOLUTIONARY |
| §3.2 SOTA-tied ∈ [0.91, 0.92) | **REVISED MAIN TARGET**: tied at 0.91 ceiling = paradigm-generalize at SOTA-tier (consistent with 103A-wd) |
| §3.3 Layer 1 lift ≥ 0.88 | Easy floor — should HIT trivially if not buggy |
| §3.4 sub-frontier ∈ [0.85, 0.88) | Sub-Phase-1 lift, paradigm partially fails |
| §3.5 regression < 0.85 | Catastrophic, would suggest pass/defender scenario doesn't transfer |

**Revised expected outcome (most likely 60% prob)**: 104A combined ≈ 0.90-0.91, 104B combined ≈ 0.88-0.91. Both TIED at 0.91 ceiling, paradigm-generalize HIT but no breakthrough.

**Less likely (25% prob)**: 104A/B sub-frontier (0.85-0.88) — pass/defender scenario init too narrow → policy overfits scenario, fails to transfer.

**Unlikely (10% prob)**: 104A or 104B breaks 0.92 → would force re-examination of "0.91 ceiling" hypothesis. Possible if pass/defender Phase 2 unlocks state coverage that INTERCEPTOR didn't.

**Catastrophic (5% prob)**: regression < 0.85 — implementation bug (likely pass event detection error or scenario init mismatch).

### 7.1 (Pending Stage 1 verdict, will append when 104A/B training completes ~03:30/04:00 EDT)

---

## 8. 后续 (8.1-8.4 由 verdict 触发)

### 8.1 §3.1 main HIT (≥ 0.92 combined)

- 直接 declare new project SOTA candidate
- Stage 3 H2H vs 1750 SOTA (n=500)
- Trigger Phase 3 (team coordination layer): warm from Phase 2 + new "team_coord_subtask" scenario + positional reward

### 8.2 §3.2 SOTA-tied HIT (∈ [0.91, 0.92))

- Tied with 103A-warm-distill paradigm
- Stage 3 H2H vs 1750 + 103A-warm-distill@400 to compare paradigms
- Phase 3 conditional on Stone Layered being "actionable SOTA path" (need at least one HIT result)

### 8.3 §3.3-3.4 sub-frontier (∈ [0.85, 0.91))

- Phase 2 not surpassing INTERCEPTOR variant — paradigm not generalizable beyond BALL_DUEL
- Skip Phase 3, declare "Stone Layered = INTERCEPTOR-only worked" 
- Update report narrative to reflect partial validation

### 8.4 §3.5 regression (< 0.85)

- Phase 2 paradigm broken — 101A capacity insufficient OR pass detection has bugs
- Audit smoke test + retry with adjusted reward (lower α residual? higher pass reward? different scenario?)
- If retry fails, abandon Stone Layered Phase 2 path

---

## 9. 相关

- [snapshot-099 §1.2](snapshot-099-stone-pipeline-strategic-synthesis.md#12-为什么-paper-1-stone-veloso-2000-layered-进-dir-b) — DIR-B Layered Phase 1/2/3 plan
- [snapshot-101](snapshot-101-dir-B-layered-phase1.md) — Phase 1 verdict + 101A@460 specialist
- [snapshot-103 §7E + §7F](snapshot-103-dir-A-wave3-sub-task-specialists.md) — Stone L2 lessons (warm-source competence, KL anchor design, recipe components)
- [snapshot-106](snapshot-106-stone-methodology-corrections.md) — methodology doctrine (peer-axis H2H, distribution match)
- [Stone & Veloso 2000](https://www.cs.cmu.edu/~mmv/papers/00mlj-llms.pdf) — original Layered Learning paper

### 理论支撑

- **Stone & Veloso 2000** "Layered Learning in Multiagent Systems" — Phase 2 = pass-decision specialist on top of Phase 1 ball-control
- **CMUnited-99** RoboCup-99 实测 — pass-decision Layer 提供 Layer 1 ball-control 之上的 +X% 团队胜率 (相对原始 baseline 的 incremental gain)
- **101A@460 (snapshot-101 §7)** — our Phase 1 implementation, 0.851 baseline = competence threshold ≥0.85 met

### Code refs

- `cs8803drl/branches/expert_coordination.py` — sample_expert_scenario (NEW: pass_subtask)
- `cs8803drl/core/soccer_info.py` — compute_event_shaping (NEW: event_pass detection)
- `cs8803drl/core/utils.py` — RewardShapingWrapper (NEW: pass through event_pass_reward)
- `cs8803drl/training/train_ray_team_vs_baseline_shaping.py` — env var passthrough
- `scripts/eval/_launch_104A_layered_p2_passdecision.sh` — launcher (NEW)
