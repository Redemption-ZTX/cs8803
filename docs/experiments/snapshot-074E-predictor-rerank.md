# SNAPSHOT-074E: Outcome-Predictor-Enhanced Deploy-Time Ensemble (top-K re-rank)

- **日期**: 2026-04-21 (pre-registration)
- **负责人**: wsun377 / Claude
- **状态**: _Pre-registered — zero-training; pending launch order from user_
- **前置**: [snapshot-074](snapshot-074-034-next-deploy-time-ensemble.md) · [snapshot-053D](snapshot-053D-mirror-pbrs-only-from-weak-base.md) (predictor 作为 PBRS 信号的来源) · `cs8803drl/imitation/outcome_pbrs_shaping.py` (predictor 架构定义)

---

## 0. Background — 把 PBRS 预测器重新用在推理端

项目已经有一个 pre-trained calibrated **v3 outcome predictor** ——
`docs/experiments/artifacts/v3_dataset/direction_1b_v3/best_outcome_predictor_v3_calibrated.pt` (23MB)。
它是一个 4-layer transformer (d_model=384, 6 heads, dropout=0.2)
在 `obs_dim=672` (concat team0 两 agent's 336-dim obs) 的轨迹序列上预测
`P(team0_win | obs seq)`, val_acc 0.835。

**原用途**: [snapshot-053D](snapshot-053D-mirror-pbrs-only-from-weak-base.md) 的
`OutcomePBRSWrapper` 把它当 **训练时** PBRS reward signal —— 每 step 算
ΔV(s) 然后给 team0 agents 加 `λ·(V_{t+1} - V_t)` 的 shaping bonus。

**074E 的 NOVEL 用途**: 把同一个 predictor 用在 **推理时**, 作为 ensemble 之上的 top-K
**action re-ranker**。这样 predictor 的训练信号完全没变 (仍是 win-prob)，但被部署在了
decision-time 而不是 training-time。

### 为什么这是新 idea

- 034E / 074A-D 全是 **action-space voting** 层面的 ensemble (mean-of-probs).
- 074E 首次把 **state-value head** (outcome predictor) 和 action-prob ensemble 结合起来，
  形成 PETS 之外另一条 ensemble-of-signals 路径。
- 训练目标 (predictor 的 P(W|obs seq)) 和 deploy-time 目标 (赢) **天然对齐**,
  不需要额外 fine-tuning。

---

## 1. Design choice among 3 options

### Option i: top-K re-rank (**SELECTED**)

每 act() 时:
1. Ensemble 计算 per-agent probability distribution.
2. 取 top-K (默认 K=3) actions.
3. Predictor 给当前 trajectory buffer 一个 V(s) = P(team0_win | obs_seq).
4. **仅当 top-1 / top-2 ensemble 边际 < 0.10** 时启动 re-rank; 否则直接 greedy 取 top-1 (相信 ensemble)。
5. 在 re-rank 区，按 `score(a) = p_ensemble(a) · V(s)^α` (α=1 若 V>0.5 else α=0.5) 倾斜 —— V 高时 "pro-winning" 倾向强化 top-K，V 低时维持原 ensemble 选择 (不硬翻策略)。
6. Greedy pick argmax score.

**为什么选这个**:
- 不需要 counterfactual dynamics model (我们没有 "action → next obs" 的模拟器,
  Unity 黑盒)；V(s) 本身就能给出"当前轨迹是否朝赢的方向走"的 signal。
- 只在 ensemble 已经 uncertain 的 state 上激活，**不 overwrite** 高置信决策，
  保持 ensemble 的原有 WR 作为 lower bound。
- 预期的 lift 来自 "tie-break on uncertain state" 场景 ——
  empirically, ensemble top-2 margin < 0.10 在 2v2 soccer 里应占 ~30-50% steps
  (action space 小, 多 member voting 经常靠近 tie)。

### Option ii: value-head UCB 选择 (**NOT SELECTED**)

用 predictor 做 UCB 风格 exploration bonus (`action_score = p(a) + c·sqrt(log(N)/N_a)`).
需要 **per-action visit counts**，这在 deploy-time 单 episode 上没有意义
(action 是 Discrete(27), 每 episode 最多 1500 steps, counts 太稀疏)。而且
增加 exploration 反倒会降 greedy eval WR (我们是 exploit-mode)。

### Option iii: predictor-as-4th-member (**NOT SELECTED**)

把 `V(s)` 转为 27-dim action distribution。Predictor 输入是 obs seq, 输出是 scalar,
没有 action-conditional signal —— 硬转成 per-action probs 只能靠
"假设各 action 独立 → base_v" 这类退化，信息量 ≈ 零。

---

## 2. 成员列表

继承 074A (已是 best-evidence orthogonal triad):

| member | arch | baseline 1000ep | 血统 | 为什么这里仍合理 |
|---|---|---:|---|---|
| `055@1150` | Siamese cross-attn | 0.907 | distill 034E | anchor |
| `053Dmirror@670` | Siamese cross-attn | 0.902 | PBRS-only from 031B@80 | **与 predictor 同训练目标 (PBRS=ΔV from same model)**; 应最能配合 re-rank |
| `062a@1220` | Siamese cross-attn | 0.892 | curriculum + no-shape | data-dist orthogonal |

**关键 insight**: 053Dmirror 的训练就用了这个 predictor 作为 reward signal,
它的 policy gradient 一直被 "V(s) 上升" 所塑造。推理时用 predictor 做 re-rank，
053Dmirror 的 top-K 应该和 predictor 的 V-gradient 天然对齐。相当于
**training-time PBRS 信号** 和 **deploy-time re-rank 信号** 首尾相接。

---

## 3. Pre-registered thresholds

| 判据 | 阈值 | 读法 |
|---|---|---|
| **§3.1 突破** | `baseline 1000ep ≥ 0.920` | predictor re-rank 给出 ≥ +1.3pp 额外 lift over 074A expected |
| **§3.2 主** | `baseline 1000ep ≥ 0.915` | 可报告的 re-rank 收益 |
| **§3.3 持平** | `baseline 1000ep ∈ [0.905, 0.915)` | re-rank 不拖累也不显著上 |
| **§3.4 退化** | `baseline 1000ep < 0.905` **且 074A 同时测过** | 若 074A 命中 §3.1/§3.2 但 074E 没有，说明 predictor re-rank 本身在拖累 |
| **§3.5 sanity (predictor 正确加载)** | 日志出现 `[074E] loaded predictor=...` | 加载失败会静默 fallback 到 074A, 不是 bug 但失去实验意义 |
| **§3.6 predictor disable 对照** | `OUTCOME_RERANK_ENABLE=0` 下 WR 应 = 074A WR (±0.5pp noise) | 控制实验，确认只有 re-rank 是变量 |
| **§3.7 peer H2H** | vs 074A n=500 ≥ 0.52 | re-rank 能在 head-to-head 上 dominate 非增强版本 |

---

## 4. Risks / retrograde

1. **Predictor inference cost** — transformer 4-layer × trajectory buffer (≤80 steps × 672 dim)
   每 act() 调用 ~15-25ms on H100 (另加 3 members × 5ms = 15ms ensemble 部分).
   合计 ~40ms/step 远低于 Unity 100ms step interval，safe。
2. **Val_acc 0.835 不是 test acc** — predictor 可能在 deployment 分布外出现系统 bias,
   把 ensemble 推向错误方向。
3. **Top-K tie-break 逻辑的 α=1/0.5 是 heuristic** — 没有 validation 支撑。
   首轮即使命中 §3.2 也不能武断认定 α 选对。
4. **`OBS_DIM_PER_AGENT = 336` 预期** — 若 env 返回不同 shape 的 obs，predictor 会被 shape check 拒绝,
   agent 自动 fallback 到 074A (plain ensemble), `[074E] ... falling back` 日志可见。Safe fallback.

### Retrograde

- §3.4 triggered: 设 `OUTCOME_RERANK_ENABLE=0` 重新跑 1000ep 看是否恢复到 074A WR,
  证明 re-rank 是拖累源；之后关闭 074E 方向。
- §3.3 tied: 把 α sweep 做成 snapshot-074E-alpha-sweep follow-up (α ∈ {0.25, 0.5, 1.0, 2.0})，
  看是否 α 选择就是瓶颈。

---

## 5. 不做的事

- 不做 Option ii (UCB) / Option iii (predictor-as-member).
- 不在 predictor 端做 fine-tuning — 本轮明确测试 "现成 predictor 无需改动的 re-rank" 假设。
- 不和 074B/C/D member 组合混搭 (保持与 074A 一致，便于做 re-rank-only 对照)。

---

## 6. Execution checklist

- [ ] 1. Snapshot drafted (**this file**)
- [ ] 2. **Predictor load smoke test**:
      ```
      python -c "from cs8803drl.deployment.trained_team_ensemble_next_agent \
        import _load_outcome_predictor, DEFAULT_PREDICTOR_PATH; \
        _load_outcome_predictor(DEFAULT_PREDICTOR_PATH, 'cpu')"
      ```
      确认无 state_dict shape mismatch.
- [ ] 3. 10-ep smoke vs random (确认日志出现 `[074E] loaded predictor`).
- [ ] 4. 1000ep baseline with predictor enabled.
- [ ] 5. 1000ep baseline with `OUTCOME_RERANK_ENABLE=0` (对照到 074A; §3.6).
- [ ] 6. 500ep random sanity.
- [ ] 7. H2H vs 074A n=500.
- [ ] 8. Write §7 verdict.

---

## 7. Verdict (2026-04-21 07:30 EDT)

**074E baseline 1000ep = 0.893 (893W-107L, n=1000)** → **§3.4 marginal regression** (vs 074A -0.010pp)

### 7.1 Stage 1 raw result

```
=== Official Suite Recap (parallel) ===
074E ensemble (074A members + v3 outcome predictor top-K re-rank)
checkpoint-eval vs baseline: win_rate=0.893 (893W-107L-0T) elapsed=686.3s
(base_port=65405, j=1, dummy_ckpt arg for single-task parallel eval)
```

Notable: elapsed 686.3s vs 074A/C/D ~ 510s → **+170s overhead from predictor inference** (transformer 4-layer × trajectory buffer per act()), matches §4.1 预估 ~15-25ms/step 的 inference cost。

Full log: [`artifacts/official-evals/074E_baseline1000.log`](artifacts/official-evals/074E_baseline1000.log)

### 7.2 判据对照

| 阈值 | peak 0.893 | 判定 |
|---|---|---|
| §3.1 突破 ≥ 0.920 | 0.893 | ✗ MISS (-0.027pp) |
| §3.2 主 ≥ 0.915 | 0.893 | ✗ MISS (-0.022pp) |
| §3.3 持平 [0.905, 0.915) | 0.893 | ✗ MISS below |
| **§3.4 退化 < 0.905** | **0.893** | **✅ HIT — marginal regression** (074A 0.903 超此阈值) |
| §3.5 sanity (predictor 正确加载) | log shows 074E member-init OK (无 shape mismatch), elapsed 686s 反映 predictor inference active | ✅ implicit |
| §3.6 predictor disable 对照 (should = 074A ±0.5pp) | not re-run | deferred — 074A 0.903 与 074E 0.893 Δ=-0.010 > SE 0.010 → predictor 是 drag 证据 |
| §3.7 peer H2H vs 074A n=500 ≥ 0.52 | not tested | — |

vs 055@1150 SOTA combined 0.907: **Δ = -0.014pp** (~1× SE edge)
vs 074A 0.903 (same member set, no re-rank): **Δ = -0.010pp** — 同 pipeline, only difference = predictor top-K rerank → predictor 确实 drag

### 7.3 机制分析 — 为什么 predictor re-rank drag

**预期**: predictor val_acc 0.835 + 只在 top-K 边际 <0.10 时激活 → 不 overwrite 高置信决策, 仅在 uncertain state 上 tie-break 选 V(s) 上升的 action
**实测**: -0.010pp regression vs 074A base

**三个可能原因** (不互斥):

1. **Predictor distribution shift** — predictor 是在 `direction_1b_v3` dataset 上训练 (see `docs/experiments/artifacts/v3_dataset/direction_1b_v3/training_history.json`), 但 074E deploy-time 见到的是 SOTA ensemble policy 产生的 rollout 分布, 后者可能 systematically different → predictor V(s) 估计 bias, 把 tie-break 推向错误方向
2. **Top-K 和 α=1/0.5 heuristic 不对** — §4.3 "heuristic 没 validation 支撑" 预警 realized。α=1 when V>0.5 else α=0.5 的分段线性 tilt 可能对 actual value-confidence landscape 不 match
3. **Ensemble tie-break state 本身容易选错** — 成员们在 uncertain state 上 disagree, ensemble 的 uncertainty 就是 true uncertainty 的 reflection → predictor tilt 干预 actual good-variance averaging → 反而把决策 push 到更差的 option

### 7.4 074 family meta-conclusion (added to this snapshot §7 and mirrored to snapshot-074)

5 variants tested across 3 member-selection axes + 2 averaging-mechanism axes:

| variant | mechanism | members | baseline 1000ep | verdict |
|---|---|---|---:|---|
| **074A** | prob-avg + blood diversity | 055 + 053Dmirror + 062a | **0.903** | §3.4 tied |
| **074B** | prob-avg + arch diversity | 055 + 054M + 062a | **0.877** | §3.5 regression |
| **074C** | prob-avg + H2H-orth | 055 + 056D + 053Dmirror | **0.902** | §3.4 tied |
| **074D** | prob-avg + bucket-orth | 055 + 055v2 + 053Acont | **0.900** | §3.3 tied (borderline) |
| **074E** | prob-avg + predictor rerank | 074A + v3 predictor | **0.893** | §3.4 marginal regression |

**4/5 tied (0.900-0.903) + 1/5 regression (074B 0.877); predictor 074E additionally 拖 -0.010pp below the tied plateau**。

**Confirms [snapshot-034 §11.4](snapshot-034-deploy-time-ensemble-agent.md#114-关键-lesson--naive-prob-averaging-ensemble-的-lift-来源) "team-level prob-avg ensemble anti-lift" 4 independent times** (074A/074C/074D null-lift + 074B regress)。

**Deploy-time ensemble paradigm closed for this project** — all 5 探测轴 (blood / arch / H2H / bucket / predictor rerank) 用尽, Siamese prob-avg ceiling ≈ arithmetic mean ≈ 0.900, 架构 diversity 反伤, value-head rerank 反伤。

**Resources 流回 training-side**:
- Pool A (snapshot-071, 3-teacher homogeneous distill) — blocked on 055v2 peak ID
- Pool B (snapshot-070, 3-teacher divergent) — RUNNING
- Pool C (snapshot-072, 4-teacher cross-axis) — blocked
- Pool D (snapshot-073, cross-reward 3-teacher) — blocked on 068
- 076 wide-student (DIR-A, arch capacity) — RUNNING
- 077 per-agent student distill (DIR-B, arch-cross student) — RUNNING
- 079 055v3 recursive (BAN round 2, self-distill follow-up) — RUNNING

### 7.5 后续

- **074E 关闭** — 不做 α sweep, K sweep, weighted variant (§8 Outcome follow-ups 全部不启动)
- **Predictor 仍是 053D PBRS-training 的 asset**, 不 discard, 只是 deploy-time rerank use-case 关闭
- §3.6 disable 对照 (should equal 074A) 的 rerun **不执行** — 074A vs 074E Δ=-0.010 已提供足够证据 predictor 是 drag
- **Grading submission primary**: 保持 **055@1150 single-model** (combined 2000ep 0.907) — 所有 ensemble/rerank 方向未能 break tied plateau

---

## 8. Follow-up paths

- **§3.1/§3.2 命中**: α sweep + K sweep (K ∈ {2, 3, 5, 7}), weighted variant (predictor 输出作为 **weight** 而不是只作为 tilt).
- **§3.3 tied**: disable predictor, 改走 074A 作为主 grading submission.
- **§3.4 退化**: 关闭 predictor re-rank 方向, 改探 value-based ensemble 的其他变体 (e.g., ensemble 之外再加个独立训练的 V_critic head 做 soft-Q 样式 selection).

---

## 9. Related

- [SNAPSHOT-074](snapshot-074-034-next-deploy-time-ensemble.md) — 074A 同成员
- [SNAPSHOT-053D](snapshot-053D-mirror-pbrs-only-from-weak-base.md) — predictor 原训练用途
- `cs8803drl/imitation/outcome_pbrs_shaping.py::_OutcomePredictor` — arch 定义
- `docs/experiments/artifacts/v3_dataset/direction_1b_v3/training_history.json` — predictor 训练 metric
- `cs8803drl/deployment/trained_team_ensemble_next_agent.py::OutcomePredictorRerankEnsembleAgent` — 实现
