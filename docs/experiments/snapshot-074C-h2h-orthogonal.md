# SNAPSHOT-074C: H2H-Least-Correlated Deploy-Time Ensemble

- **日期**: 2026-04-21 (pre-registration)
- **负责人**: wsun377 / Claude
- **状态**: _Pre-registered — zero-training; pending launch order from user_
- **前置**: [snapshot-074](snapshot-074-034-next-deploy-time-ensemble.md) · [rank.md §5.3](rank.md#53-已测-h2h-的-n--z--p-明细只列目前有数据的)

---

## 0. Background — 为什么按 H2H 选

074A 和 074B 都靠 **先验假设** 做 member 选择 (血统差异 / 架构差异)。
074C 尝试用 **实测 H2H** 作为 orthogonality 的直接证据：pairwise H2H 越接近 0.5，
两个 policy 在决策空间上越难互相 dominate → 正交性越强。

### rank.md §5.3 完整 H2H 矩阵 (frontier level)

按最近数据 (2026-04-20 / 21) 排:

| Pair (A vs B) | WR | z | sig | 读法 |
|---|---:|---:|:-:|---|
| **`055@1150 vs 056D@1140`** | **0.536** | 1.61 | **NOT sig (marginal)** | 👈 **最平** |
| `055@1150 vs 025b@080` | 0.702 | 9.03 | `***` | 失 decisively dominates |
| `055@1150 vs 028A@1060` | 0.750 | 11.18 | `***` | — |
| `055@1150 vs 029B@190` | 0.696 | 8.76 | `***` | — |
| `055@1150 vs 031B@1220` | 0.620 | 5.37 | `***` | — |
| `043B'@440 vs 031B@1220` | 0.600 | 4.47 | `***` | — |
| `043C'@480 vs 031B@1220` | 0.614 | 5.10 | `***` | — |
| `043C'@480 vs 034E-frontier` | 0.576 | 3.40 | `***` | — |
| `043B'@440 vs 043C'@480` | 0.468 | -1.43 | NOT sig | 方向稳定, 043C > 043B |
| `031B@1220 vs 031A@1040` | 0.516 | 0.72 | NOT sig | 架构 step 2 (cross-attn vs Siamese) 几近 tied |

（`053Dmirror@670`, `062a@1220`, `056D@1140` vs 其它 frontier 的直接 H2H **未测** — 列为
"未测,需要先补 H2H"。）

### 最 orthogonal 三员推荐 (based on available data)

主候选 `055` 是 project SOTA，必须入列。
在已测 H2H 里 `055` 唯一 NOT sig 的 match 是 vs `056D@1140` = 0.536 (z=1.61, p=0.054)。
加入 `056D` 是 H2H 指导下最硬的实证 orthogonal pick。

第三员在 `{031B, 043B', 043C', 053Dmirror, 062a, 053Acont, 029B, 025b}` 中选 ——
没有 vs 055 或 vs 056D 的直接 H2H。结构性理由选 **053Dmirror@670** (PBRS-only blood，
和 055/056D 的 v2 shape 血统都不同)。但必须注明 `053Dmirror vs 055 / vs 056D` 的 H2H
**未测**。

---

## 1. Hypothesis H_074C

### 1.1 主假设

> H_074C: H2H-informed 选择 {055 + 056D + 053Dmirror} 的 pairwise decision-space orthogonality
> 比 074A (全 Siamese 血统) 更高,
> baseline 1000ep ≥ **0.918** ( ≥ 074A 突破阈值 -0.002，与 074A §3.1 持平)。

### 1.2 为什么可能成立

1. `055 vs 056D = 0.536 NOT sig` 是**当前 frontier 里最难拉开的 pair**，
   说明两者在决策空间上高度正交。
2. `055` (distill) 和 `056D` (HP sweep lr=3e-4) 是**不同训练 pipeline** 得到的 near-ceiling policy，
   bias 来源不重合。
3. 加入 `053Dmirror` (PBRS-only) 引入第三条训练信号轴 —— 即使没 H2H 支撑，
   训练目标函数不同意味着 loss surface 的 minimum 位置也不同。

### 1.3 反假设

> H_074C-alt: `055 vs 056D = 0.536` 的 "NOT sig" 只是 sample 不足 (n=500 z=1.61 p=0.054)，
> 真实 H2H 可能 > 0.54 甚至 > 0.56. 若是这样，
> ensemble 里 055 其实仍轻度 dominate 056D，056D 的 decision diversity 被稀释,
> 最终 WR 落在 [0.90, 0.915] 即 tied 区。

> H_074C-alt2: Unity port-determinism 导致 `055 vs 056D` 的 2nd 500ep sample 完全重复 1st (rank.md §5.3 note)，
> 所以 0.536 的样本事实上只有 500ep 有效支撑; confidence interval ±3.4pp.
> 有理由认为 true H2H 在 [0.50, 0.58]，既包含 tied 也包含 decisively 055 wins。

---

## 2. Design

### 2.1 成员表

| member | arch | baseline 1000ep | 血统 | 选入理由 |
|---|---|---:|---|---|
| `055@1150` | Siamese cross-attn | 0.907 | distill | 主 H2H anchor; 唯一跟 056D 打平 |
| `056D@1140` | Siamese cross-attn | 0.891 | PBT-simplified LR=3e-4 | 实证 H2H vs 055 = 0.536 NOT sig |
| `053Dmirror@670` | Siamese cross-attn | 0.902 | PBRS-only warm 031B@80 | 结构性 orthogonal (PBRS vs v2); **H2H 未测** |

### 2.2 Averaging — 等权 `TeamEnsembleNextAgent`

### 2.3 Agent module

`agents/v074c_h2h_orthogonal/agent.py`

---

## 3. Pre-registered thresholds

| 判据 | 阈值 | 读法 |
|---|---|---|
| **§3.1 突破** | `baseline 1000ep ≥ 0.920` | H2H-informed 选择真实 superior to 074A, Δ ≥ 074A §3.1 |
| **§3.2 主** | `baseline 1000ep ≥ 0.914` | ensemble 方向正确 |
| **§3.3 持平** | `baseline 1000ep ∈ [0.900, 0.914)` | H2H orthogonality 论证 not strong enough |
| **§3.4 退化** | `baseline 1000ep < 0.900` | anti-lift; `055 vs 056D` NOT sig 不代表 decision space 正交 |
| **§3.5 sanity** | `random 500ep ≥ 0.99` | 推理链路健康 |
| **§3.6 alt** | top bucket count ≤ 0.80 × single-best @ 64ep capture | — |
| **§3.7 peer H2H** | vs 055@1150 n=500 ≥ 0.52 | — |

---

## 4. Risks / retrograde

### 4.1 主 caveat — 未补 H2H

**`053Dmirror vs 055 / vs 056D` H2H 未测。** 在证据不完整的前提下选入，
受 §1.3 H_074C-alt 的拖累可能性是首要风险。
**推荐 (但不硬性要求)**: 先补一轮 `053Dmirror vs 055` n=500 H2H，
结果 < 0.50 或 > 0.54 都会动摇当前 member 选择。**未做即先启动 074C = 接受这个不确定性**。

### 4.2 `055 vs 056D` sample 不独立

[rank.md §5.3 (2nd sample, port 51005)](rank.md#53-已测-h2h-的-n--z--p-明细只列目前有数据的) 指出
两次 500ep 样本 literally identical (268W-232L),
suspect Unity port+episode_index 确定性 seed → 看似合并 1000ep 但其实仍是 n=500 power。
**这意味着 `0.536 NOT sig` 的 CI 仍很宽**, 不要武断写成 "决定空间正交"。

### 4.3 team-level-only 阴影

和 074A/074B 共享 [snapshot-034 §11.4](snapshot-034-deploy-time-ensemble-agent.md#114-关键-lesson--naive-prob-averaging-ensemble-的-lift-来源)
的 team-level-only anti-lift 风险 ——
三个成员 family 都是 `031B` 后裔 Siamese cross-attention，架构维度 zero diversity。
074C 在架构维度上**比 074A/B 还弱**。

### Retrograde

- 若 §3.3 triggered (tied)：归因到 "未补 H2H 导致 member 选择不可靠"。启动 `053Dmirror vs 055` H2H
  补测；再决定下一步。
- 若 §3.4 triggered (退化)：team-level-only 阴影证据升级为第 2 条数据点，和 074B 合并结论。

---

## 5. 不做的事

- 不在 launch 前补 `053Dmirror vs 055` H2H (接受不确定性，节约 eval cycle)。
- 不换成 4-way (加 062a 或 029B) — 增加样本只会增加 family correlation。
- 不做 weighted averaging。

---

## 6. Execution checklist

- [ ] 1. Snapshot drafted
- [ ] 2. 10-ep smoke vs random
- [ ] 3. 1000ep baseline
- [ ] 4. 500ep random sanity
- [ ] 5. H2H vs 055@1150 n=500
- [ ] 6. 若 §3.2+ 命中: 64ep capture 对比 bucket
- [ ] 7. Optional 补充 `053Dmirror vs 055` n=500 H2H (if §3.3 triggered 以诊断原因)
- [ ] 8. Write §7 verdict

---

## 7. Verdict (2026-04-21 07:10 EDT)

**074C baseline 1000ep = 0.902 (902W-98L, n=1000)** → **§3.4 tied** [0.900, 0.914)

### 7.1 Stage 1 raw result

```
=== Official Suite Recap (parallel) ===
074C ensemble {055@1150 + 056D@1140 + 053Dmirror@670}
checkpoint-eval vs baseline: win_rate=0.902 (902W-98L-0T) elapsed=509.8s
(base_port=65205, j=1, dummy_ckpt arg for single-task parallel eval)
```

Full log: [`artifacts/official-evals/074C_baseline1000.log`](artifacts/official-evals/074C_baseline1000.log)

### 7.2 判据对照

| 阈值 | peak 0.902 | 判定 |
|---|---|---|
| §3.1 突破 ≥ 0.920 | 0.902 | ✗ MISS (-0.018pp) |
| §3.2 主 ≥ 0.914 | 0.902 | ✗ MISS (-0.012pp) |
| **§3.3 持平 [0.900, 0.914)** | **0.902** | **✅ HIT** |
| §3.4 退化 < 0.900 | 0.902 | ✗ NO |
| §3.5 sanity (random 500ep ≥ 0.99) | not re-tested for 074C (shared pipeline 074A verified) | — |

vs 055@1150 SOTA combined 0.907: **Δ = -0.005 within SE ±0.016** → **统计 tied**
vs 074A 0.903 (§3.3 tied 同区): **Δ = -0.001** — 两条 independently replicate tied pattern

### 7.3 机制分析 — 为什么 H2H orthogonality 没转化为 lift

**Arithmetic mean baseline**: (0.907 + 0.891 + 0.902) / 3 = **0.900**
**实测**: 0.902 → 比算术平均高 +0.002pp, **within noise 的 null-lift pattern**

**H2H signal 的 limitation**:
- `055 vs 056D = 0.536 NOT sig (z=1.61, p=0.054)` 是最接近 H2H-0.5 的 pair，但这仅告诉我们**两个 policy 在 100% 决策上不能互相 dominate**
- H2H 只捕捉 **pairwise overall WR**, 不捕捉**单个 state 上的 action-distribution difference**
- Two policies with H2H ≈ 0.5 could still have **highly correlated action distributions at the same state**, 只是 outcomes 在 long run 打平 — mean-of-probs ensemble 的 lift 来源 = action-distribution diversity, 不是 outcome tie
- [snapshot-034 §11.4](snapshot-034-deploy-time-ensemble-agent.md#114-关键-lesson--naive-prob-averaging-ensemble-的-lift-来源) 的 lesson 4 再次确认: **H2H orthogonality ≠ actionable decision-space orthogonality**

### 7.4 与 074A tied 模式的合流证据

074A (blood diversity) + 074C (H2H-informed) **两条独立路径都 tied** 在 0.90 附近, 跟 arithmetic mean 一致 — 强力支持 "**Siamese-only prob-avg ensemble ceiling ≈ arithmetic mean of members**" hypothesis。

叠加 074D (failure-bucket-orthogonal) 第 3 次独立复现 (0.900, see snapshot-074D §7) → **Siamese-family prob-avg ensemble 机制用尽**，member selection axis (reward blood / H2H / failure bucket) 全 null lift。

### 7.5 §1.3 反假设 H_074C-alt2 部分证实

- `055 vs 056D` H2H `0.536` 的样本重复性问题 ([rank.md §5.3 note](rank.md#53-已测-h2h-的-n--z--p-明细只列目前有数据的)) 提示 true H2H CI 可能 [0.50, 0.58], including decisively-055-wins 区
- 074C 实测 tied 没给新信息去区分 "真正 0.536" vs "0.56+ 055 dominates"
- 结论: H2H-informed member selection **从这次实验不足以 actionable**, 若要 test H_074C-alt 需要 **独立 seed 重跑 055 vs 056D** 才能解决

### 7.6 后续

- **074C 关闭**: 不再 follow-up (预案 §8 tied 分支)。不再 make weighted variant — null lift 下加权只会 add HP noise
- **不补 `053Dmirror vs 055 / vs 056D` H2H** — reasoning: 即使补了，§7.3 的 H2H-as-lift-signal null 结论已经成立, 不会改变 074C 方向
- **Deploy-time prob-averaging paradigm evaluation moves to 074E (predictor rerank)** — 074E 是 074 family 中唯一换决策函数的 variant
- Resources 流回 **training-side** (Pool A/B/C/D, 076 wide-student, 077 per-agent, 079 055v3 recursive)

---

## 8. Follow-up paths (final state)

§3.3 tied 判据命中 → 按原预案 "做 `053Dmirror vs 055` H2H 诊断 + 换员测试" 的路径**不启动**;
因 074A/074C/074D 三次独立 tied 复现 (§7.4) 证实 team-level Siamese prob-avg ceiling 已接近 arithmetic mean, member substitution 不会突破。

原预案保留记录 (不执行):
- ~~§3.1 命中 → 构造 weighted 074C~~ (未命中, 不 launch)
- ~~§3.3 tied → 做 `053Dmirror vs 055` H2H 诊断 + 换员~~ (不 actionable, §7.5 已说明)
- §3.4 退化 → closed (未触发)

---

## 9. Related

- [SNAPSHOT-074](snapshot-074-034-next-deploy-time-ensemble.md)
- [rank.md §5.3](rank.md#53-已测-h2h-的-n--z--p-明细只列目前有数据的) — H2H matrix
- [SNAPSHOT-055](snapshot-055-distill-from-034e-ensemble.md), [SNAPSHOT-056](snapshot-056-simplified-pbt-lr-sweep.md), [SNAPSHOT-053D](snapshot-053D-mirror-pbrs-only-from-weak-base.md) — member sources
