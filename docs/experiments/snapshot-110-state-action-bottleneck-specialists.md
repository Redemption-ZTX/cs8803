# SNAPSHOT-110: State/Action Bottleneck Specialists (Wang/Stone/Hanna 2025 paper-faithful)

- **日期**: 2026-04-23
- **状态**: 设计 + 实现 (engineering 进行中, 训练 launchers 待 prep)
- **前置**:
  - [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) — Stone 6-DIR 战略 (DIR-A heuristic selector based on Wang 2025)
  - [snapshot-103 §7F.5d/e](snapshot-103-dir-A-wave3-sub-task-specialists.md) — 0.91 ceiling 决定性证据 (10000ep aggregate)
  - [snapshot-108 §0](snapshot-108-stone-layered-l2-improvement-paths.md) — path EVs revised: state/action bottleneck = 唯一 fundamental ceiling-breaking path
  - [snapshot-109 §1 paper 4 + §2 Stage 1](snapshot-109-stone-pipeline-completeness-audit.md) — gap audit identified missing feature
  - Wang/Stone/Hanna ICRA 2025 arXiv 2412.09417 Table I — per-specialist obs/action specs

## 0. 为什么做 (motivation)

### 0.1 Today's discovery — 0.91 ceiling decisively confirmed

Combined 10000ep aggregate (5000 from 1750 fresh + 5000 from 103A-wd v2) gave **0.9069 ≈ 1750's 0.9066** with Δ < 0.001. Two completely different recipes (recursive distill vs Stone Layered L2) hit EXACTLY the same plateau. This is **architecture-imposed ceiling**, not recipe limitation.

Multiple paradigms exhausted at this ceiling:
- Recursive distill (055/055v2/1750): 0.907-0.911
- Stone Layered L2 from SOTA warm (103A-wd): 0.9069 (10000ep)
- Distill family 6/6 saturation cluster (071-080): 0.903-0.909
- Architecture variants (082/083/052/077): 0.880-0.909

### 0.2 Wang/Stone/Hanna 2025 unimplemented paper feature

Per snapshot-109 Q1 audit, paper 4 Table I specifies that each of 4 sub-policies has:
- **OWN state space** (e.g., MID-FIELD only sees ball + goalposts; NEAR-GOAL trained in 1v0 high-fidelity sim)
- **OWN action space** (e.g., MID-FIELD outputs only 1-D ΔΘ kick angle)

Our project's 5 specialists (101A / 081 / 103A v1+resume+wd / 103B / 103C v1+v2) all share:
- Full 336-d ray-cast obs (no masking)
- Full MultiDiscrete([3,3,3]) action space (rotate + forward + kick)

**State/action bottleneck = single largest unimplemented Stone-paper feature in our project**.

### 0.3 Why bottleneck might break the ceiling

Hypothesized mechanism (Wang 2025 attribution):
- **Reduced obs**: encoder learns to specialize on relevant feature subset → no capacity-budget conflict between sub-tasks
- **Reduced action**: policy learns finer-grained control on the few relevant action dims → higher precision
- **Train in sub-task-specific scenario**: each specialist only ever sees its task → specialist achieves "expert" level on that sub-task

If our 0.91 ceiling is encoder-capacity-bound (limited by 0.46M params trying to cover all states), per-specialist bottleneck could free capacity for narrow expertise → ensemble of bottleneck specialists could exceed flat-architecture full-distribution policy.

---

## 1. Hypothesis

### 1.1 主假设 H_110

> 训 4 个 bottleneck specialists per Wang 2025 spec (each with own obs/action subset + scenario init) + heuristic selector (v_selector_phase4 framework with new specialists) on full 2v2 baseline eval, **combined 2000ep peak ≥ 0.93** (= +0.023 over 0.91 ceiling, decisive)。

### 1.2 子假设

- **H_110-a (per-specialist competence)**: each bottleneck specialist trained in its scenario achieves **sub-task success rate ≥ 90%** (per Wang Table I CMUnited-99 numbers). E.g., MID-FIELD-DRIBBLE: ball-advanced ≥10m without losing possession ≥90% of episodes; NEAR-GOAL-STRIKER: shot-on-target within 20 steps of NEAR-GOAL init ≥90%.
- **H_110-b (deploy-time ensemble)**: heuristic selector routing 4 bottleneck specialists on full 2v2 baseline play achieves WR ≥ 0.93. Tests if "specialist-on-narrow-task" >> "generalist-on-everything" thesis.
- **H_110-c (vs current ceiling)**: 4-bottleneck ensemble decisively beats 1750 SOTA (H2H ≥ 0.55 z>2.24 sig) — confirms ceiling-breaking, not just tied.

### 1.3 Anti-hypothesis

- **H_110-null**: bottleneck specialists trained narrowly fail to compose at deploy time — selector routing imperfect; specialist出错 outside training distribution. Combined ≤ 0.91 (no breakthrough).
- **H_110-bad**: action bottleneck (e.g., MID-FIELD with kick-only) makes specialist UNABLE to handle state transitions where motion needed → catastrophic failure rate in ensemble.

---

## 2. Design

### 2.1 Per-specialist bottleneck spec (paper 4 Table I + adaptation to our env)

Our env: 14 rays per frame × 3 frames × 8 features = 336-d obs; MultiDiscrete([3,3,3]) action (forward 0/1/2, rotate 0/1/2, kick 0/1/2).

Wang Table I has 4 sub-policies (Mid-field / Near-goal / Ball-Duel / Defender). We map to our 4 phase categories.

| Specialist | Scenario | Obs bottleneck | Action bottleneck | Sub-task success metric |
|---|---|---|---|---|
| **110A NEAR-GOAL-STRIKER** | scenario where ball near opp goal, my agent has ball with shooting opportunity | mask all rays except: ball (type 0), opp_goal (type 4), opp_keeper-ish (type 2 nearest) | full kick (dim 2 free), forward fixed=2, rotate fixed=1 (face goal) | shot-on-target within 20 steps ≥ 0.90 |
| **110B MID-FIELD-DRIBBLE** | scenario neutral mid-field, my agent has ball | mask all except: ball (type 0), forward rays (front 7 of 14), opp_goal (type 4) | only kick (dim 2 free for kick angle); forward fixed=2 (always forward); rotate fixed=1 | ball-advanced ≥10m without losing possession ≥ 0.85 |
| **110C BALL-DUEL-INTERCEPTOR** | scenario opp near ball, my agent close to challenge | mask all except: ball, nearest opp (type 2 closest ray), own_position-ish | rotate (dim 1 free) + forward (dim 0 free); kick fixed=0 (no kick — pure intercept) | ball-recovered within 30 steps of duel init ≥ 0.85 |
| **110D POSITIONING-DEFENDER** | scenario teammate has ball own half, opp transitioning | mask all except: teammate (type 1), own_goal (type 5), nearest opp | rotate + forward (kick fixed=0 — pure positioning) | opp_progress blocked for ≥100 steps ≥ 0.80 |

### 2.2 Implementation mechanism — wrappers

**Approach**: gym wrappers that mask obs / constrain actions but keep underlying env unchanged.

#### 2.2.1 ObsBottleneckWrapper

```python
# cs8803drl/branches/bottleneck_wrappers.py
class ObsBottleneckWrapper(gym.ObservationWrapper):
    """Mask non-relevant ray channels in 336-d obs.

    Args:
        ray_type_mask: List[int] of ray-type indices to KEEP (others zeroed)
                       e.g., [0, 4] means keep ball (0) + opp_goal (4) only
        ray_index_mask: Optional List[int] of ray INDICES to keep (front/back filtering)
                        e.g., list(range(0, 7)) keeps only forward 7 rays
    """
    def __init__(self, env, *, ray_type_mask: List[int], ray_index_mask: Optional[List[int]] = None):
        super().__init__(env)
        self._type_mask = set(int(t) for t in ray_type_mask)
        self._index_mask = set(int(i) for i in ray_index_mask) if ray_index_mask else None
        # observation_space unchanged (still 336-d)
    
    def observation(self, obs):
        # obs is dict {agent_id: 336-d array} for multi-agent env
        if isinstance(obs, dict):
            return {aid: self._mask_one(o) for aid, o in obs.items()}
        return self._mask_one(obs)
    
    def _mask_one(self, arr):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1).copy()
        # Reshape: 336 = 3 frames * 14 rays * 8 features
        # Each ray block: 7 type one-hot + 1 distance
        n_blocks = arr.size // RAY_BLOCK_SIZE  # 42 blocks total
        blocks = arr.reshape(n_blocks, RAY_BLOCK_SIZE)
        # Determine ray index per block (modulo 14 within frame)
        rays_per_frame = 14
        for i in range(n_blocks):
            ray_idx = i % rays_per_frame
            type_scores = blocks[i, :RAY_TYPE_DIM]
            kept_types = type_scores[list(self._type_mask)] if self._type_mask else type_scores
            # If no relevant type detected by this ray OR ray outside index_mask → zero out distance
            type_relevant = bool(np.any(kept_types > 0.5))
            index_relevant = (self._index_mask is None) or (ray_idx in self._index_mask)
            if not (type_relevant and index_relevant):
                blocks[i, RAY_TYPE_DIM] = 1.0  # set distance to "far" (max)
                blocks[i, :RAY_TYPE_DIM] = 0.0  # clear type one-hot
        return blocks.reshape(-1)
```

#### 2.2.2 ActionBottleneckWrapper

```python
class ActionBottleneckWrapper(gym.Wrapper):
    """Force specific action dims to fixed values (specialist constraint).

    Args:
        free_dims: List[int] of MultiDiscrete dim indices the policy controls
                   e.g., [2] means policy outputs all 3 dims but only dim 2 (kick) varies
        fixed_values: Dict[int, int] for fixed dims (e.g., {0: 2, 1: 1} = forward=2, rotate=1)
    """
    def __init__(self, env, *, free_dims: List[int], fixed_values: Dict[int, int]):
        super().__init__(env)
        self._free_dims = set(int(d) for d in free_dims)
        self._fixed_values = {int(k): int(v) for k, v in fixed_values.items()}
        # action_space unchanged (policy still outputs MultiDiscrete([3,3,3]))
    
    def step(self, action):
        if isinstance(action, dict):
            constrained = {aid: self._constrain(act) for aid, act in action.items()}
        else:
            constrained = self._constrain(action)
        return self.env.step(constrained)
    
    def _constrain(self, act):
        act = np.asarray(act, dtype=np.int64).copy()
        for dim, val in self._fixed_values.items():
            if dim < act.size:
                act[dim] = val
        return act
```

### 2.3 Wrapper composition order

In the env factory `create_rllib_env()`:
```
RLLibWrapper(MultiAgentEnv)
  ↓
RewardShapingWrapper       # reward shape with sub-task aux reward
  ↓
ScenarioResetWrapper       # scenario init (existing)
  ↓
ObsBottleneckWrapper       # NEW: mask obs (snapshot-110)
  ↓
ActionBottleneckWrapper    # NEW: constrain action (snapshot-110)
  ↓
soccer_twos.make()         # base Unity env
```

ObsBottleneck applied AFTER scenario reset (state correctly initialized) but BEFORE policy sees obs (so policy gets bottlenecked). ActionBottleneck applied at step time (between policy output and env input).

### 2.4 Training recipe per specialist (e.g., 110B MID-FIELD-DRIBBLE)

```bash
# Warm + KL anchor (Stone Layered methodology, even though specialist is narrow)
WARMSTART_CHECKPOINT=$CKPT_101A   # Layer 1 ball-control (competent warm)
TEAM_DISTILL_ENSEMBLE_KL=1
TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS=$CKPT_101A
TEAM_DISTILL_ALPHA_INIT=0.05
TEAM_DISTILL_ALPHA_FINAL=0.005
TEAM_DISTILL_DECAY_UPDATES=39000

# Scenario init
SCENARIO_RESET=dribble_subtask    # existing 103C scenario

# Obs bottleneck (NEW)
OBS_BOTTLENECK_RAY_TYPES=0,4      # ball (0) + opp_goal (4)
OBS_BOTTLENECK_RAY_INDICES=0,1,2,3,4,5,6,7,8,9,10  # forward 11 of 14 rays

# Action bottleneck (NEW)
ACTION_BOTTLENECK_FREE_DIMS=2     # only kick free
ACTION_BOTTLENECK_FIXED=0:2,1:1   # forward fixed=2, rotate fixed=1

# Reward (per-specialist auxiliary)
USE_REWARD_SHAPING=1
SHAPING_BALL_PROGRESS=0.05         # main: dribble forward
SHAPING_POSSESSION_BONUS=0.005     # hold ball
SHAPING_EVENT_SHOT_REWARD=0.02     # allow occasional finish

BASELINE_PROB=0.7
TIMESTEPS_TOTAL=20000000 MAX_ITERATIONS=500 TIME_TOTAL_S=14400
```

### 2.5 PIPELINE V1 deploy-time ensemble (after 4 specialists trained)

Update `agents/v_selector_phase4` with new bottleneck-specialist preset:
```python
"pipeline_v1_bottleneck": {
    NEAR_GOAL: _110A_NEARGOAL,
    BALL_DUEL: _110C_BALLDUEL,
    POSITIONING: _110D_POSITIONING,
    MID_FIELD: _110B_MIDFIELD,
}
```

Eval: standard 1000ep vs baseline. **If WR ≥ 0.93 = decisive ceiling break (per H_110)**.

---

## 3. 实现清单

| # | File | LOC | Status |
|---|---|---|---|
| 1 | `cs8803drl/branches/bottleneck_wrappers.py` — ObsBottleneck + ActionBottleneck classes | ~200 | TODO |
| 2 | `cs8803drl/core/utils.py` — env factory adds bottleneck cfg passthrough | ~50 | TODO |
| 3 | `cs8803drl/training/train_ray_team_vs_random_shaping.py` — env var passthrough OBS_BOTTLENECK_* / ACTION_BOTTLENECK_* | ~30 | TODO |
| 4 | `scripts/eval/_launch_110A_bottleneck_neargoal.sh` (and B/C/D) | ~150 each = 600 | TODO |
| 5 | smoke test — verify mask masks correct rays + action constrain works | ~80 | TODO |
| 6 | `agents/v_selector_phase4/agent.py` — add `pipeline_v1_bottleneck` preset (after specialists trained) | ~30 | LATER |

**Total ~1000 LOC + 4 × 4h GPU per specialist + ensemble eval.**

Engineering effort upfront: ~600 LOC core + ~250 LOC launchers/smoke = ~6h work.
GPU budget: 4 × 4h sequential = 16h, OR 4 × 4h parallel on 4 nodes = 4h.

---

## 4. 预注册判据

| § | 阈值 | verdict |
|---|---|---|
| §3.1 main: 4-bottleneck ensemble combined 2000ep ≥ 0.93 | +0.023 over 0.91 ceiling | **DECISIVE ceiling break — Wang 2025 paradigm validated** |
| §3.2 SOTA-tied: combined 2000ep ∈ [0.91, 0.93) | tied with current ceiling | bottleneck doesn't break ceiling but doesn't regress; saturate same as flat-architecture |
| §3.3 partial: combined 2000ep ∈ [0.85, 0.91) | sub-ceiling | bottleneck under-fitted — possibly reward / scenario issue |
| §3.4 regression: combined < 0.85 | catastrophic | implementation bug OR action bottleneck too restrictive (specialist unable to handle env transitions) |

**Per-specialist sub-task success thresholds** (sub-§ verdicts):
- 110A NEAR-GOAL: shot-on-target ≥ 0.90 in NEAR-GOAL scenario
- 110B MID-FIELD: ball-advanced ≥10m no-loss-possession ≥ 0.85 in DRIBBLE scenario
- 110C BALL-DUEL: ball-recovered ≤30 steps of duel ≥ 0.85 in INTERCEPTOR scenario
- 110D POSITIONING: opp-progress blocked ≥100 steps ≥ 0.80 in DEFENDER scenario

---

## 5. 风险

### R1 — Action bottleneck 太严格
若 MID-FIELD 强制 forward=2 + rotate=1, agent 无法处理 ball 在 side / behind 的 state. **缓解**: free `forward` dim (allow forward/back) + only fix rotate.

### R2 — Obs bottleneck 误屏蔽关键 ray type
若 ball type 探测错误 (snapshot-082/083 论证 ray type 0 = ball 但未 100% 验证), masking 误删可见 ball ray → policy fails. **缓解**: smoke test 验证 ray type 0 == ball using known scenario; if uncertain, use union of types {0, 1} (ball + teammate as fallback).

### R3 — Encoder still uses full 336-d, mask 退化为加 noise
我们 mask 后的 obs 仍 336-d, encoder 看到的是 "informative ray + zeroed ray" 而不是真正减小输入. 若 zeroed pattern 跟 "ball not visible" 状态混淆, policy 可能学到 wrong correlation. **缓解**: 用特殊值 (e.g., -1.0 distance) 标记 "masked" vs "not visible" (1.0 = far max). 或干脆 truncate obs space to N-d (需改 model arch — 大改).

### R4 — Selector misroutes outside scenario
PIPELINE deploy 中 selector 只看 obs 选 phase, 不知 specialist 是 bottleneck-trained. 当 selector 误判 phase, 错误的 bottleneck specialist 见到 OOD state → 可能 catastrophic. **缓解**: bottleneck specialists 必须 robust on OOD via KL anchor to non-bottleneck Layer 1 (101A).

### R5 — Engineering complexity & schedule
~1000 LOC + 4 specialists × 4h training + ensemble compose = ~30+ hours engineering+compute. 大投入. **缓解**: 先做 1 个 specialist (e.g., 110B MID-FIELD-DRIBBLE) 测试 paradigm, HIT 后再扩 4 个.

---

## 6. 不做的事

- **不改 model architecture** (encoder 还是 031B Siamese 256+256+cross-attn) — 用 wrapper 实现 bottleneck, 不增工程复杂度
- **不做 truly reduced obs space** (不减 336-d → N-d) — 需 model 改造, 留 V2 if V1 plateau 后续考虑
- **不做 frozen teammate** (Layered Lite 沿用 103A-wd) — 留 Phase 3
- **不做 4 specialists 同时 fire** — 先 110B (MID-FIELD-DRIBBLE) 单 specialist verify paradigm; HIT 后扩到 110A/C/D

---

## 7. Verdict

_Pending implementation + 110B MID-FIELD-DRIBBLE single-specialist test launch._

---

## 8. 后续 (8.1-8.4 由 verdict 触发)

### 8.1 110B HIT (sub-task ≥ 0.85)

- Implement 110A/C/D 同 paradigm
- 4-specialist ensemble compose via v_selector_phase4 `pipeline_v1_bottleneck` preset
- Eval combined 2000ep
- Per §3.1: ≥ 0.93 = decisive ceiling break

### 8.2 110B sub-task ≥ 0.85 BUT ensemble < 0.93

- Bottleneck specialists work individually but selector routing imperfect
- Iterate selector design (per snapshot-100 narrow-trigger lessons)
- OR train selector via REINFORCE (DIR-G W2, currently P2)

### 8.3 110B sub-task < 0.85

- Bottleneck (action OR obs) too restrictive for sub-task
- Loosen constraint (more action dims free, broader obs mask)
- OR redefine sub-task (perhaps "MID-FIELD-DRIBBLE" wrong granularity)

### 8.4 110B regression < 0.5

- Catastrophic — wrapper bug OR fundamental incompatibility
- Audit smoke tests, retry with simpler bottleneck (e.g., obs-only, no action constraint)

---

## 9. 相关

- [snapshot-099](snapshot-099-stone-pipeline-strategic-synthesis.md) — DIR plan
- [snapshot-103 §7F](snapshot-103-dir-A-wave3-sub-task-specialists.md) — 0.91 ceiling evidence
- [snapshot-108](snapshot-108-stone-layered-l2-improvement-paths.md) — path EVs (this is new top P0)
- [snapshot-109](snapshot-109-stone-pipeline-completeness-audit.md) — gap audit identifying bottleneck as biggest gap
- Wang/Stone/Hanna ICRA 2025 arXiv 2412.09417 Table I — paper-faithful spec source

### Code refs

- `cs8803drl/branches/bottleneck_wrappers.py` — NEW (this snapshot)
- `cs8803drl/branches/obs_summary.py` — RAY_BLOCK_SIZE / RAY_TYPE_DIM constants
- `cs8803drl/branches/expert_coordination.py` — sample_expert_scenario (existing, reused)
- `cs8803drl/core/utils.py` — env factory passthrough additions
- `agents/v_selector_phase4/agent.py` — `pipeline_v1_bottleneck` preset (after specialists trained)
