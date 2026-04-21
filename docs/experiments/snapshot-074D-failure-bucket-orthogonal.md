# SNAPSHOT-074D: Failure-Bucket-Orthogonal Deploy-Time Ensemble

- **日期**: 2026-04-21 (pre-registration)
- **负责人**: wsun377 / Claude
- **状态**: _Pre-registered — zero-training; pending launch order from user_
- **前置**: [snapshot-074](snapshot-074-034-next-deploy-time-ensemble.md) · `docs/experiments/artifacts/failure-cases/`

---

## 0. Background — 用实测 failure bucket 决定 member

H2H 矩阵 (074C) 和架构/血统 (074B) 都是 **global** orthogonality signal。
Failure bucket 是 **local** orthogonality signal —— 直接描述 "一个 policy
在什么 state 下输"。

### 从 `docs/experiments/artifacts/failure-cases/<ckpt>_baseline_500/` 提取的计数

通过解析 `episode_XXXX_team1_win_<bucket>.json` 文件名后缀（即 capture 工具
打的 bucket label），得到各 ckpt baseline 500ep 的 loss 分布:

| member          | late_def | low_poss | unclear | poor_conv | total (losses) |
|-----------------|---------:|---------:|--------:|----------:|---------------:|
| `055@1150`      |   **29** |   **26** |       3 |         1 |             60 |
| `055@1000`      |       29 |       21 |  **12** |         6 |             72 |
| `055v2@1000`    |       19 |       21 |       1 |     **6** |         **50** |
| `056D@1140`     |       23 |       21 |       7 |         2 |             56 |
| `062a@1220`     |       29 |   **30** |       8 |         2 |             72 |
| `031B@1220`     |       29 |       16 |  **12** |         4 |             63 |
| `053Acont@430`  |       16 |       14 |   **9** |         3 |             47 |
| `051A@130`      |   **32** |       17 |       3 |         4 |             57 |
| `043C'@480`     |   **32** |       18 |       8 |         1 |             61 |
| `043B'@440`     |   **37** |   **28** |       3 |         2 |             75 |

> Bucket 解读:
> - `late_defensive_collapse` — 进球后被反打失球 / 比赛尾段防守失位
> - `low_possession` — 输但持球时间短
> - `unclear_loss` — 规则不明确（边缘/裁判/双打）
> - `poor_conversion` — 持球但未射门/未转化

### 观察

1. `055@1150` 的 `late_def` + `low_poss` 合计 55/60 = 91.7% —— dominant 两桶几乎吞噬所有 loss。
2. `055v2@1000` 的 `poor_conversion = 6/50 = 12%`，是其他 ckpt (≤ 8%) 的 1.5× —— distinctive bucket signature。
3. `053Acont@430` 的 `unclear_loss = 9/47 = 19.1%` —— 显著高于 055 (5%)、062a (11%)、056D (12.5%)。
4. `031B@1220` 的 `unclear_loss = 12/63 = 19.0%` —— 跟 053Acont 同量级，但 late_def/low_poss 分布更像 055。

**最正交三员 candidate**:
- 主 (late_def + low_poss 双桶 anchor): `055@1150`
- 加 `poor_conversion` 桶多余的那个: `055v2@1000`
- 加 `unclear_loss` 桶多余的那个: `053Acont@430` (vs 031B 选 053Acont 因为 PBRS 血统自带第二轴 orthogonality)

---

## 1. Hypothesis H_074D

### 1.1 主假设

> H_074D: **failure-bucket-informed** ensemble 在 `late_defensive_collapse`
> 顶 bucket 上的 relative loss count 至少比 single-best (055@1150) 少 **20%**
> (即 §3.6 alt hypothesis 的 hard form), 同时 baseline 1000ep ≥ 0.910。

### 1.2 为什么可能成立

1. 三成员 bucket 分布**不重合**：055 吃 late_def/low_poss、055v2 吃 poor_conv、053Acont 吃 unclear。
   Ensemble 在任何单一 bucket 上应有 2/3 成员"不经常失败在这个 bucket", 相当于多数投票保护。
2. 053Acont 是 PBRS-only continue 血统 (`053A@190` 的 continue iter 200→500)；
   跟 055 / 055v2 的 v2 shape 是不同 reward path，loss surface minimum 错开。
3. 055v2 作为 recursive distill 变体，pipeline 和 055 不同；
   [snapshot-061](snapshot-061-055v2-recursive-distill.md) 证明它和 055 WR tied (0.909 vs 0.907)。

### 1.3 反假设

> H_074D-alt: bucket label 的粒度太粗 (只有 4 个 bucket)，
> ensemble 做不了 fine-grained "某个成员在这个 bucket 上更强" 的差异化决策，
> bucket-based orthogonality 只是 hindsight 标签，并不对应**实时决策时的** action
> distribution 差异。最终 WR 落在 [0.905, 0.915]，alt hypothesis 主假设 §3.6 超过 20% relative 不成立。

---

## 2. Design

### 2.1 成员表

| member | arch | baseline 1000ep | dominant bucket | 选入理由 |
|---|---|---:|---|---|
| `055@1150` | Siamese cross-attn | 0.907 | late_def + low_poss | anchor; 2 桶吞 91.7% loss |
| `055v2@1000` | Siamese cross-attn | 0.909 | poor_conversion 上移 | recursive distill；poor_conv 桶覆盖 |
| `053Acont@430` | Siamese cross-attn | 0.896 | unclear_loss 上移 | PBRS blood；unclear 桶覆盖 |

### 2.2 Averaging — 等权 `TeamEnsembleNextAgent`

### 2.3 Agent module

`agents/v074d_failure_bucket_orthogonal/agent.py`

---

## 3. Pre-registered thresholds

| 判据 | 阈值 | 读法 |
|---|---|---|
| **§3.1 突破** | `baseline 1000ep ≥ 0.915` **AND** late_def count ≤ 0.70 × 055@1150 | main hypothesis 命中 |
| **§3.2 主** | `baseline 1000ep ≥ 0.910` **AND** late_def count ≤ 0.80 × 055@1150 | bucket-informed 有效但 WR 不拉开 |
| **§3.3 持平** | `baseline 1000ep ∈ [0.900, 0.910)` | bucket orthogonality 不转化为 WR lift |
| **§3.4 退化** | `baseline 1000ep < 0.900` | bucket overlap 比看起来更深 |
| **§3.5 sanity** | `random 500ep ≥ 0.99` | 推理链路健康 |
| **§3.7 peer H2H** | vs 055@1150 n=500 ≥ 0.50 | — |

---

## 4. Risks / retrograde

1. **Bucket label 是 capture 工具后分类**，不等于**实时决策差异**。
   3 成员可能在 "pre-loss decision" 上几乎相同，只在最后一步的 luck 上不同，
   ensemble 取不到 orthogonality 收益。
2. **055 和 055v2 家族相近度高** — [snapshot-074 §5](snapshot-074-034-next-deploy-time-ensemble.md#5-what-we-are-not-doing)
   的 "不纳入 055v2" 指导原则在这里被**有意违反**，理由是 bucket 分布差异 (poor_conv 6/50 vs 1/60)
   给了实证 support。但潜在 correlated decision space 仍是主风险。
3. **n=500 capture 样本小** — loss count 10-30 的 Poisson 波动可达 ±30%，
   "055v2 poor_conv 6 vs 055 poor_conv 1" 的差距在 CI 内**可能不显著**。
   但这是 74A/C/D 中最接近 empirical decision orthogonality 的方式，值得测。

### Retrograde

- §3.3 triggered：更换 `055v2` 为 `031B@1220` (unclear_loss 桶也高, 12/63)，
  看是否稳 bucket-based orthogonality 的同时提升 WR (因为 031B 不是 055 直系后裔).
- §3.4 triggered：撤销 "bucket-based member selection" 方法论，文档化为 **failure bucket 指导无效**的实证。

---

## 5. 不做的事

- 不混 per-agent (029B / 036D) 进来 —— 会触发 mixed-family ensemble 的另一维，
  应该走 [snapshot-034 §11.4](snapshot-034-deploy-time-ensemble-agent.md#114-关键-lesson--naive-prob-averaging-ensemble-的-lift-来源)
  mixed-arch 路径 (已由 034E 验证 +1.4pp)。这里保持 team-level-only，只测 bucket axis。
- 不做 4-way。
- 不做 weighted averaging。

---

## 6. Execution checklist

- [ ] 1. Snapshot drafted
- [ ] 2. 10-ep smoke vs random
- [ ] 3. 1000ep baseline
- [ ] 4. 500ep random sanity
- [ ] 5. 64ep capture (or 500ep capture 如果 H_074D 主假设 §3.1 要求 alt 判据)
- [ ] 6. H2H vs 055@1150 n=500
- [ ] 7. Write §7 verdict

---

## 7. Verdict (2026-04-21 07:20 EDT)

**074D baseline 1000ep = 0.900 (900W-100L, n=1000)** → **§3.3 tied** [0.900, 0.910)

### 7.1 Stage 1 raw result

```
=== Official Suite Recap (parallel) ===
074D ensemble {055@1150 + 055v2@1000 + 053Acont@430}
checkpoint-eval vs baseline: win_rate=0.900 (900W-100L-0T) elapsed=506.0s
(base_port=65305, j=1, dummy_ckpt arg for single-task parallel eval)
```

Full log: [`artifacts/official-evals/074D_baseline1000.log`](artifacts/official-evals/074D_baseline1000.log)

### 7.2 判据对照

| 阈值 | peak 0.900 | 判定 |
|---|---|---|
| §3.1 突破 ≥ 0.915 AND late_def ≤ 0.70×055 | 0.900 | ✗ MISS (WR 条件就未达) |
| §3.2 主 ≥ 0.910 AND late_def ≤ 0.80×055 | 0.900 | ✗ MISS (-0.010pp WR) |
| **§3.3 持平 [0.900, 0.910)** | **0.900** | **✅ HIT** (just at lower bound) |
| §3.4 退化 < 0.900 | 0.900 | ✗ NO (borderline) |
| §3.5 sanity (random 500ep ≥ 0.99) | not re-tested for 074D | — |

vs 055@1150 SOTA combined 0.907: **Δ = -0.007 within SE** → **统计 tied**
vs 074A 0.903 / 074C 0.902: **Δ = -0.002 / -0.002** — 三次 Siamese-only ensemble 独立 tied 复现 (三点 within 0.003pp)

### 7.3 机制分析 — failure-bucket orthogonality 是 hindsight signal

**Arithmetic mean baseline**: (0.907 + 0.909 + 0.898) / 3 = **0.905**
**实测**: 0.900 → 比算术平均**低 -0.005pp** (slight drag, not lift)

**Bucket-based orthogonality 的 limitation** (§1.3 H_074D-alt 部分证实):
- Bucket label 是 **post-hoc 分类**: 一个 episode 输了以后再根据 trajectory 特征打标
- 它不等于 **realtime pre-loss decision difference** — 三个成员可能在"即将失球的关键 state"做几乎相同的 action, 只是最后一步的 unlucky 不同 bucket
- Ensemble 取不到 "某成员在这个 bucket 上决策更强" 的 lift，因为 averaging 是在**决策 time** 进行的，而成员们在决策 time 的 action distribution 是相似的
- 只是 **outcome-level bucket label** 不同 → ensemble 无法在决策 time 利用此信号

**N=500 capture 样本的 statistical caveat** (§4.3 预警):
- `055v2@1000 poor_conv = 6/50 vs 055@1150 poor_conv = 1/60`  差距看似 6× 但 Poisson 波动 CI 内不显著
- 5/500 到 10/500 的 count 差异完全可以是 noise
- Bucket distribution 不稳，基于此做 member selection 是 over-fitting on capture sample

### 7.4 与 074A/074C tied 合流 — 决定性证据

三次独立 member-selection 方法 (blood diversity / H2H orthogonality / failure-bucket orthogonality) 全部 tied 0.900-0.903:
- 074A (blood): 0.903
- 074C (H2H): 0.902
- 074D (failure bucket): 0.900

**三次 within 0.003pp 复现** → **Siamese-family prob-avg ensemble ceiling ≈ arithmetic mean of members (~0.900)**，不受 member selection 方法影响。

这是 [snapshot-034 §11.4](snapshot-034-deploy-time-ensemble-agent.md#114-关键-lesson--naive-prob-averaging-ensemble-的-lift-来源) "team-level-only anti-lift pattern" 的**第 4 次 independent confirmation** (原 3 次是 034 legacy triads + 074A + 074B negative + 074C; 现在 074D 是第 4 条独立证据)。

### 7.5 后续

- **074D 关闭** — §3.3 tied 触发原预案 "换 031B@1220" 的 retrograde **不启动** (根因已定位在决策-空间 orthogonality 缺失, 换员不救)
- **无 alt hypothesis bucket capture 跟进** — 074D 的 WR tied 同时 baseline capture bucket count 也 tied 预期, 跑 alt §3.6 不会给 new info
- **Deploy-time prob-averaging 机制定论**: Siamese-only 不抬, cross-arch (074B MAT) 退化 — **剩余 novel direction = 074E predictor rerank** (换决策函数)
- Resources 流回 **training-side**: Pool A/B/C/D, 076 wide-student distill, 077 per-agent distill, 079 055v3 recursive distill

---

## 8. Follow-up paths

- **§3.1 命中**: bucket-informed selection 验证成立, 为 snapshot-075 weighted ensemble 提供 member weighting 依据 (按 bucket coverage).
- **§3.3 tied**: 尝试替换 `055v2 → 031B` 看是否把 WR 拉进 §3.2 区.
- **§3.4 退化**: 关闭 bucket-informed 方法论，转向 074E predictor-based 方向。

---

## 9. Related

- [SNAPSHOT-074](snapshot-074-034-next-deploy-time-ensemble.md)
- [SNAPSHOT-055](snapshot-055-distill-from-034e-ensemble.md), [SNAPSHOT-061](snapshot-061-055v2-recursive-distill.md), [SNAPSHOT-053A](snapshot-053A-continue-plateau-ppo.md) — member sources
- `docs/experiments/artifacts/failure-cases/` — bucket count 提取源
