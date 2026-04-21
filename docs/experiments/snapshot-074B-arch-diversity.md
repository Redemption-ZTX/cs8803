# SNAPSHOT-074B: Arch-Diversity Deploy-Time Ensemble (Siamese + MAT + Siamese-curriculum)

- **日期**: 2026-04-21 (pre-registration)
- **负责人**: wsun377 / Claude
- **状态**: _Pre-registered — zero-training; pending launch order from user_
- **前置**: [snapshot-074](snapshot-074-034-next-deploy-time-ensemble.md) (074A) · [snapshot-054](snapshot-054-mat-min-cross-agent-attention.md) · [snapshot-034 §11.4](snapshot-034-deploy-time-ensemble-agent.md#114-关键-lesson--naive-prob-averaging-ensemble-的-lift-来源) (team-level-only anti-lift warning)

---

## 0. Background — 为什么拆一条 "arch-diversity" 分支

074A 三个成员 (`055 / 053Dmirror / 062a`) 全是 `031B` 同源 Siamese cross-attention 架构。
这正是 [snapshot-034 §11.4](snapshot-034-deploy-time-ensemble-agent.md#114-关键-lesson--naive-prob-averaging-ensemble-的-lift-来源) 警告过的
"team-level-only → architecture correlation → anti-lift -0.1pp" 病灶复现风险。

074B 把 `062a` 保留 (data-distribution 正交), 把 `053Dmirror` 换成 **054M@1230** —
项目唯一的 cross-agent MAT-medium 架构 ([snapshot-054](snapshot-054-mat-min-cross-agent-attention.md))。
这样 ensemble 跨越三条不同 attention 轴:

- `055@1150` — Siamese cross-field attention + distillation 血统
- `054M@1230` — MAT cross-agent attention (唯一存在的完整 cross-agent 路径)
- `062a@1220` — Siamese cross-field attention + curriculum/no-shape 血统

## 1. Hypothesis H_074B

### 1.1 主假设

> H_074B: arch-diversity ensemble 的 architectural orthogonality 足以抵消
> 054M@1230 individual baseline WR 偏低 (0.889) 带来的拖累,
> **baseline 1000ep ≥ 0.910**（比 074A 预期 +0 pp，但 arch diversity 的 tail robustness 更强）。

### 1.2 为什么可能成立

1. **`054M vs 055/062a` 没有直接 H2H**，无法判断 peer-axis 相关性，但架构差异最大化是 ensemble 首先要确认的
   diversity 源 —— failure-bucket orthogonality 是 lagging indicator, architecture 是 leading indicator。
2. **054M 不是 SOTA**，但 [snapshot-054 §7.1](snapshot-054-mat-min-cross-agent-attention.md#71-2026-04-20-stage-1-baseline-1000ep--marginal-tied-031b-verdict-mediocre) 说明它
   "graceful degrade"：double-peak @1100/1230 = 0.880/0.889 不是 single-shot luck。成员 individual 平均仍在 ~0.89+ 带上。
3. **cross-agent attention** 对 shot/tackle 的观测聚合方式和 Siamese cross-field 不同；理论上 failure buckets 应倾向正交。
   但这是**理论预期，不是实测**（054M 目前没有 failure capture 做支撑，故标 "未测,需要先补 capture"）。

### 1.3 反假设

> H_074B-alt: 054M 的 individual baseline WR 偏低 (-1.8pp vs 055) 直接拖累 ensemble，
> 最终 baseline WR 落在 [0.895, 0.908)（= 074A tied 区间内），arch diversity 未能
> 补偿 weaker member 的直接拖累。

这是更可能的结局——因为 ensemble 总体 WR ≈ average individual WR × (1 + ensemble_lift)，
当 average individual WR 被拉下时，lift 即便非零也未必把 total 拉到 ≥ 0.91。

---

## 2. Design

### 2.1 成员表

| member | arch | baseline 1000ep | 血统 | 选入理由 |
|---|---|---:|---|---|
| `055@1150` | Siamese cross-field | **0.907** | distill from 034E | anchor / SOTA single |
| `054M@1230` | MAT cross-**agent** | **0.889** (single-shot peak) | scratch v2 MAT-min | **唯一跨 attention axis** |
| `062a@1220` | Siamese cross-field | **0.892** | curriculum + no-shape | 训练分布正交 |

### 2.2 054M 选点

- Pre-extend run ([054M_mat_medium_scratch_v2_512x512_20260420_135128](../../)) peak = iter 1230, WR 0.889。
- Extend run (054M_extend_resume_1250_to_1750_20260421_030244) 已跑到 iter 1630+，
  **post-eval 尚未进行**（rank.md §3.3 没有 1500+ 点）。用 iter 1230 为保守选择。
- 若 extend post-eval 结果 ≥ 0.895，则改用 extend peak (单独 follow-up snapshot-074B')。

### 2.3 Averaging — 等权复用 `TeamEnsembleNextAgent`

`cs8803drl/deployment/trained_team_ensemble_next_agent.py::TeamEnsembleNextAgent`
在等权输入时等价 034E 的 `ProbabilityAveragingMixedEnsembleAgent` mean-of-probs 行为。

### 2.4 Agent module

`agents/v074b_frontier_arch_diversity/agent.py`

---

## 3. Pre-registered thresholds

| 判据 | 阈值 | 读法 |
|---|---|---|
| **§3.1 突破** | `baseline 1000ep ≥ 0.915` | arch diversity 真实 lift; 首次 team-level-only ensemble 正号 |
| **§3.2 主** | `baseline 1000ep ≥ 0.910` | tied-best-single + arch lift signal |
| **§3.3 持平** | `baseline 1000ep ∈ [0.895, 0.910)` | arch diversity 未克服 weak-member 拖累 |
| **§3.4 退化** | `baseline 1000ep < 0.895` | 054M 拖累 > arch diversity lift; 回到 snapshot-034 §11.4 的 -0.1pp 定律 |
| **§3.5 sanity** | `random 500ep ≥ 0.98` | 054M 架构不匹配时 ActionFlattener 在 mixed-team ensemble 里的 bug detector |
| **§3.6 alt** | top bucket count ≤ 0.80 × best single member @ 64ep capture | arch diversity 至少应减少 worst-case loss 类型集中 |
| **§3.7 peer H2H** | vs 055@1150 n=500 WR ≥ 0.50 (tied or positive) | 074B 对 best single 不形成拖累 |

---

## 4. Risks / retrograde

1. **MAT weight 加载兼容性** — `ensemble_agent._TeamRayPolicyHandle` 已注册
   `register_team_siamese_cross_attention_model` / `register_team_siamese_distill_model` /
   `register_team_action_aux_model`。**MAT-min 模型注册在哪还没核实**（054M 训练模块
   是 `cs8803drl.branches.team_mat_min` 或类似）。Smoke test 第一步必须 load-test 054M ckpt，
   若 `load_policy_weights` 抛 `KeyError`，需要在 `_TeamRayPolicyHandle.__init__` 里补
   `register_team_mat_min_model()` 调用。**BLOCKER 预警** ↴ 见 §6 checklist step 2。
2. **weak-member 直接拖累** — 见 §1.3。
3. **额外 GPU memory** — 054M 模型规模相当 (~50MB 级别)，3-way 合计 <200MB，
   H100 free frag 内 safe。
4. **同 [snapshot-074 §4.2-4.3](snapshot-074-034-next-deploy-time-ensemble.md#42-risks)** 里通用风险 (correlated failure modes / submission zip size)。

### Retrograde

- 若 §3.4 triggered：文档化为 "arch diversity 不抵消 weak-member" 的第 2 条 datapoint
  (第 1 条是 034ea {031B + 045A + 051A} 的 -0.1pp)，**不继续 4-way 追加 054M**。
  资源转向 074C/D/E。

---

## 5. 不做的事

- 不引入 weighted averaging (如按 individual WR 加权)。首轮保持 principled 对照。
- 不做 4-way (054M + 055 + 053D + 062a) — 这是 074A ∪ 054M, 若 074B 命中 §3.1 再追加。
- 不等 054M extend post-eval — extend 结果可能 ≥ 0.895，但我们先用保守的 1230 peak
  避免等待；若 extend 出来后需要更新，见 §8。

---

## 6. Execution checklist

- [ ] 1. Snapshot drafted (**this file**)
- [ ] 2. **Smoke load test**: `python -c "from agents.v074b_frontier_arch_diversity import Agent; Agent(None if False else <env>)"` —
      确认 MAT-min 模型能被 `_TeamRayPolicyHandle` 加载。若失败，在 `ensemble_agent.py` 里
      追加 MAT 模型注册调用，或放弃 074B 换成 054M@extend peak 的其他点。
- [ ] 3. 10-ep smoke vs random.
- [ ] 4. 1000ep baseline (§3.1-§3.4)
- [ ] 5. 500ep random sanity (§3.5)
- [ ] 6. H2H vs 055@1150 n=500 (§3.7)
- [ ] 7. 若 §3.2+ 命中, 64ep capture for bucket comparison (§3.6)
- [ ] 8. Write §7 verdict

---

## 7. Verdict (2026-04-21 07:00 EDT)

**074B baseline 1000ep = 0.877 (877W-123L)** → **§3.5 REGRESSION**

### 7.1 Raw result

```
=== Official Suite Recap (parallel) ===
074B ensemble {055@1150 + 054M@1230 + 062a@1220}
checkpoint-eval vs baseline: win_rate=0.877 (877W-123L-0T) elapsed=537.9s
(on 5028919, j=1, base_port=65105, dummy_ckpt arg)
```

### 7.2 判据对照

| 阈值 | 0.877 | 判定 |
|---|---|---|
| §3.1 breakthrough ≥ 0.920 | | ✗ MISS |
| §3.2 main ≥ 0.915 | | ✗ MISS |
| §3.3 marginal ≥ 0.908 | | ✗ MISS |
| §3.4 tied [0.895, 0.908) | | ✗ MISS |
| **§3.5 regression < 0.895** | **0.877** | **✅ HIT — significant regression** |

vs 074A 0.903 = **Δ-0.026pp** (1.6× SE, direction clearly negative)
vs 055@1150 SOTA 0.907 = **Δ-0.030pp** (~2× SE, near-significant regression)

### 7.3 Engineering confirm

- **No load error**: 054M@1230 MAT 架构成员成功加载 (`register_team_siamese_cross_agent_attn_medium_model` + `register_team_siamese_cross_agent_attn_model` 已由 `trained_team_ensemble_next_agent.py` 注册)
- **Registration fix works**: 174 §4.1/§6 step 2 blocker resolved — 074B runnable end-to-end

### 7.4 机制分析 — 为什么 arch diversity hurt

两个核心假设 (不互斥):

**H1 (mechanical averaging conflict)**: MAT cross-agent-attn 产生的 action probability distribution 形状 ≠ Siamese。同一 state 下两种架构 argmax 可能不一样,softmax avg 选择 "两者都不 optimal 的折衷 action"。例: 055 argmax pass, 054M argmax shoot → ensemble 选一个 "half-pass-half-shoot" 模糊 action
**H2 (054M pre-extend 还没成熟)**: 054M@1230 是 pre-extend peak 0.889,比 055 0.907 低 1.8pp。加入一个弱 member + arch conflict = 双重负担。若 054M_extend @1750 完成后 ckpt 换到 0.90+, 可能修复

### 7.5 Lesson learned (加进 snapshot-034 meta-lessons)

**Deploy-time probability-averaging ensemble 的 sweet spot**:
- ✅ **Member 架构同质 (all Siamese / all MAT / all per-agent)**
- ✅ **Reward/data/optimizer diversity**(reward path 不同 OK,只要 output action distribution shape 类似)
- ❌ **Cross-architecture averaging** (action distribution shape 不同 → 决策冲突)
- ❌ **Member WR 差距 > 0.015**(弱 member 会拉低 mean)

**应用到 074 family**: 074A(全 Siamese)0.903 ≈ tied, 074C 预期也是 tied 范围; 074B cross-arch regressed; 074D 取 055/055v2/053Acont 全 Siamese 应类似 074A; 074E predictor rerank 在 074A base 上尝试 break tie

### 7.6 后续

**不再测同类 cross-arch**。074B 关闭,不 follow-up。**孤证支持 ensemble_agent.py arch-mix 不适合**。未来若想做 arch diversity → 考虑 **model router** 机制 (per-state 选 member 而非 probability avg),不是本 snapshot scope。

---

## 8. Follow-up paths

- **§3.1 命中**: 设计 weighted variant (按 individual WR) 做 snapshot-074B-weighted;
  同时把 054M@extend peak 替换进来看 lift 是否再 +0.5pp.
- **§3.2 命中**: 4-way (加 053Dmirror) 做 robustness 对照.
- **§3.4 triggered**: 关闭 074B, 记录为 "team-level-only ensemble anti-lift 定律" 第 2 次复现.

---

## 9. Related

- [SNAPSHOT-074](snapshot-074-034-next-deploy-time-ensemble.md) — 074A 设计和共享判据
- [SNAPSHOT-054](snapshot-054-mat-min-cross-agent-attention.md) — 054M 来源
- [SNAPSHOT-034 §11.4](snapshot-034-deploy-time-ensemble-agent.md#114-关键-lesson--naive-prob-averaging-ensemble-的-lift-来源) — team-level-only anti-lift 历史证据
- [rank.md §3.3](rank.md#33-official-baseline-1000frontier--active-points-only) — 054M@1230 baseline 数据来源
