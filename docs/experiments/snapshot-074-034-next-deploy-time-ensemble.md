# SNAPSHOT-074: 034-next — Deploy-Time Ensemble on 2026-04-21 Frontier Assets

- **日期**: 2026-04-21 (pre-registration)
- **负责人**: wsun377 / Claude
- **状态**: _Pre-registered — zero-training, can launch immediately on any free inference slot_
- **前置**: [snapshot-034](snapshot-034-deploy-time-ensemble-agent.md) (034A/C/E 历史) / [rank.md §3.3](rank.md#33-official-baseline-1000frontier--active-points-only)
- **关联成员**: [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) / [snapshot-061](snapshot-061-055v2-recursive-distill.md) / [snapshot-056](snapshot-056-simplified-pbt-lr-sweep.md) / [snapshot-062](snapshot-062-curriculum-noshape-adaptive.md) / [snapshot-053D](snapshot-053D-mirror-pbrs-only-from-weak-base.md)

---

## 0. Background — 为什么现在要做 034-next

`034E-frontier = {031B + 036D + 029B}` 是 2026-04-19 的 ensemble 主候选
（baseline 500 = 0.904, combined 2000ep = 0.892，详见 [snapshot-034 §5.6](snapshot-034-deploy-time-ensemble-agent.md#56-034e-frontier-follow-upofficial-1000--peer-h2h)）。
4 天后，它作为 **教师** 被 `055` distillation 蒸馏到单一 student
（`055@1150` combined 2000ep = **0.907**，vs 034E teacher H2H = 0.590 `***`，
即 "student > teacher +9pp"，详见 [snapshot-055 §7.13-§7.14](snapshot-055-distill-from-034e-ensemble.md#713-2026-04-20-1325-edt-baseline-rerun-v2--0551150-combined-2000ep--0907-supersedes-single-shot-0911)）。

换言之：**034E 成员表已经被 055 吞掉**——`055 > 031B`、`055 > 036D`、`055 > 029B`
（H2H 全部 `***`）。继续把 034E 摆在首位是 outdated。

另一边，过去 48 小时新落地的 frontier 资产多了 4 个候选（全部 combined 2000ep 或 ≥ 1000ep single-shot, 均 ≥ 0.89）：

- `055@1150` combined 2000ep **0.907** — distill SOTA (primary anchor)
- `055v2@1000` combined 3000ep **0.909** — recursive distill，v2 verified tied 055
- `053Dmirror@670` single-shot 1000ep **0.902** — PBRS-only from weak base
- `062a@1220` combined 2000ep **0.892** — curriculum + adaptive + no-shape
- `056D@1140` single-shot 1000ep **0.891** — PBT-simplified lr sweep (lr=3e-4)

于是一个自然问题：**既然 034E 成员已过时，034-next (`074`) 应把哪 3-4 个当前 SOTA 资产组合起来?**
这是 pure deploy-time 操作——**零新训练**，任何 GPU idle 窗口都能启动 15-20 min 验证。

---

## 1. Hypothesis H_074

### 1.1 主假设

> H_074: 把 2026-04-21 4 个 frontier SOTA 资产 (`055@1150` / `053Dmirror@670` / `062a@1220` / `056D@1140`)
> 用 probability averaging 组成 deploy-time ensemble, 能在 **baseline 1000ep** 上 ≥ **0.920**,
> 即比当前 single-model SOTA (`055@1150` 0.907) 高 ≥ **+0.013** (≈ 1-2× SE 0.007)。

### 1.2 为什么有可能成立

1. **034E → 055 的差拉大证据支持"ensemble 即使不智力升级也有 stability lift"**
   - 034E ensemble 0.890 → 单 student 0.907，差 +1.7pp——但 student 的成功源于 distillation pipeline 而不是 ensemble 本身的智能。
   - 换成现在 4 个更强成员 (avg indiv ~0.90)，ensemble 至少还有 stability/calibration lift 空间。
2. **失败模式正交性候选较好**
   - `055` (distill 家族) 与 `056D` (HP sweep) failure mode 相近（都是 vanilla PPO + v2 derivative），属 **diminishing diversity**。
   - `053Dmirror` (PBRS-only from weak base) **注入 shaping-axis 正交性** — 它是唯一 PBRS-only 血统的 frontier 资产。
   - `062a` (curriculum + no-shape + adaptive gate) **注入 data-distribution 正交性** — 它从全然不同的训练课程长出来。
3. **架构 family 层面依然 mixed**
   - `055` / `053Dmirror` / `062a` 都是 team-level cross-attention（`031B` 后裔），
     `056D` 是 PBT-simplified 但同一 team-level family。**本轮缺 per-agent family**——
     这是 `034E` 包含 `029B/036D` 带来的主要 ensemble lift 来源
     ([snapshot-034 §11.4](snapshot-034-deploy-time-ensemble-agent.md#114-关键-lesson--naive-prob-averaging-ensemble-的-lift-来源))。
   - 因此 H_074 **主假设是激进的**——如果 +0.013 没达成，更可能的读法是"team-level-only
     导致 architectural diversity 不足，收益天花板约 +0.5-1pp"。

### 1.3 替代假设（若主假设 miss）

> H_074-alt: 即使 baseline WR 与 best single (055) tied, ensemble 在 top failure bucket
> (late_defensive_collapse / possession_stolen) 上的 loss 数量 **显著少于 single-best** (≥ -20% relative)。

这条是 [snapshot-034 §5.4.1](snapshot-034-deploy-time-ensemble-agent.md#5.4.1-failure_buckets_v2-对照034c-到底更像谁) 验证过的
ensemble 机制——即使总 WR 不涨，failure distribution 更 rounded 也构成可报告的 gain（特别是对 bonus 未知 agent2 的抗 exploit 价值）。

---

## 2. Design

### 2.1 成员选择（3-way primary + 4-way secondary）

**主候选 (3-way): `074A = {055@1150 + 053Dmirror@670 + 062a@1220}`**

| member | arch | baseline 1000ep | 血统 / 失败模式 | 选入理由 |
|---|---|---:|---|---|
| `055@1150` | team-level cross-attn | **0.907** | distill from 034E | current SOTA anchor; +9pp over teacher |
| `053Dmirror@670` | team-level cross-attn | **0.902** | PBRS-only from weak base (031B@80) | shaping-axis 正交；唯一 PBRS-only blood |
| `062a@1220` | team-level cross-attn | **0.892** | curriculum + adaptive + no-shape | data-dist 正交；唯一 curriculum blood |

- **不选 `056D`**：与 `055` 同 v2-shape family，failure mode 同质化（confirmed [rank.md §5.3](rank.md#53-已测-h2h-的-n--z--p-明细只列目前有数据的) 055 vs 056D H2H = 0.536 marginal），边际 diversity 低。
- **不选 `055v2@1000` 作 4-way 第 4 员**：与 `055@1150` 家族同质（recursive distill 共享 5 个 teacher 里有 4 个重合），是 055 的 "noisy clone"，参 [snapshot-034 §11.3](snapshot-034-deploy-time-ensemble-agent.md#113-假设被反驳--双重证据)
  中的 `045A vs 031A H2H = 0.492 tied` 教训：v2-桶 fingerprint 不同 ≠ 决策空间不同。

**次候选 (4-way): `074B = 074A + 056D@1140`**

纯作为 4-way robustness 测试——如果 `074A` main hypothesis 成立但边际不稳
(WR in [0.915, 0.925])，加入 `056D` 看是否把 WR 推到 ≥ 0.925 或拉回到 tied。
[snapshot-034 §8](snapshot-034-deploy-time-ensemble-agent.md#8-不做的事) 明确了"不做超过 5 个成员"，4-way 在 budget 内。

### 2.2 Voting / averaging mechanism — 沿用 034E 现有实现

现有 `ProbabilityAveragingMixedEnsembleAgent`
([cs8803drl/deployment/ensemble_agent.py §483-571](../../cs8803drl/deployment/ensemble_agent.py))
已实现：

- 每 member 输出 per-agent Discrete(27) probability distribution
- team-level member 通过 joint MultiDiscrete → per-agent marginalize
  ([ensemble_agent.py `_joint_factor_probs_to_single_agent_probs`](../../cs8803drl/deployment/ensemble_agent.py))
- 等权 mean-of-probs（非 mean-of-logits — scale-invariant，logit 尺度不同的模型不会互相压制）
- `ENSEMBLE_GREEDY=1` default greedy（与 single-policy deployment 对齐），
  与 single-ckpt eval 走 `explore=False` 口径一致

**不选 argmax-vote**：
- 离散动作 argmax vote 在 3-way 时 >=2 成员有 coin-flip tied；
- mean-of-probs 保留不确定性信息，对 PETS 风格 model averaging 更 principled（[snapshot-034 §2.1](snapshot-034-deploy-time-ensemble-agent.md#21-ensemble-类型对比) 已论证）；
- 034E 既定实现就是 mean-of-probs，复用降低工程风险。

### 2.3 Cost / latency

- **训练成本**: **零** — 所有成员已存在，无 retraining。
- **推理成本**:
  - 每 `act()` 调用: 3-4 次 GPU forward（~5-10ms each on H100）= 15-40ms/step
  - Unity step interval ≈ 100ms，推理延迟不卡 env ([snapshot-034 §2.4](snapshot-034-deploy-time-ensemble-agent.md#24-inference-成本))
  - 1000ep baseline eval 估计 15-20 min on 单 GPU
- **内存**: 3-4 × ~50MB ≈ 150-200MB, 小于任何 H100 free frag budget

### 2.4 工程实现 — 2 个新 agent 模块

```
agents/v074a_frontier_055_053D_062a/
  ├── __init__.py
  ├── agent.py     # mimic v034e_frontier_031B_3way/agent.py
  └── README.md

agents/v074b_frontier_055_053D_062a_056D/
  ├── ... (同上, 4-way)
```

每个 `agent.py` 仅 30 行 — 列出成员 checkpoint 绝对路径，调用
`ProbabilityAveragingMixedEnsembleAgent(env, members=...)`.

---

## 3. Pre-registered thresholds

| 判据 | 阈值 | 读法 |
|---|---|---|
| **§3.1 突破 (breakthrough)** | `074A baseline 1000ep ≥ 0.920` | Δ +0.013 over 055 SOTA, ≥ 2× SE → publishable ensemble lift |
| **§3.2 主 (main hypothesis)** | `074A baseline 1000ep ≥ 0.914` | Δ +0.007 over 055, ≈ 1× SE → 方向正确但 margin narrow，需 combined 2000ep 验证 |
| **§3.3 持平 (tie)** | `074A baseline 1000ep ∈ [0.900, 0.914)` | ensemble 不退化但也没实 lift; fall back 到 §3.4 alt hypothesis 看 failure bucket |
| **§3.4 退化 (regression)** | `074A baseline 1000ep < 0.900` | ensemble 伤害 best single — 回溯到 034E legacy triad 级 negative result |
| **§3.5 sanity (vs random)** | `074A random 500ep ≥ 0.99` | 不满足意味着 ensemble 推理链路有 bug (cross-arch marginalize / ActionFlattener / greedy 口径) |
| **§3.6 alt hypothesis (failure bucket)** | top bucket (late_defensive / possession_stolen) count ≤ `0.80 × single-best-in-ensemble` @ 64-ep capture | ensemble 即使 WR tied, failure dist 更 rounded 仍可报告 |
| **§3.7 peer H2H** | `074A vs 055@1150 n=500 WR ≥ 0.52` (one-tailed, marginal 正号即可) | peer-axis 不被 best single 反压；如果 < 0.50 `**` negative 则 ensemble 实为拖累 |

**`074B` (4-way) 只在 `074A` 命中 §3.2 但未达 §3.1 时启动** — 否则不跑，节约 eval cycle。

---

## 4. Simplifications / risks / retrograde

### 4.1 Simplifications

- **等权**: 不做 weighted averaging（按 individual baseline WR 加权）。[snapshot-034 §8](snapshot-034-deploy-time-ensemble-agent.md#8-不做的事) 已预言"加权引入新超参"，首轮保持 principled。
- **静态 router**: 不做 state-conditional routing（heuristic / learned gating）。
  `034F-router` 已给出 heuristic router 的 negative result
  ([snapshot-034 §8.1.6](snapshot-034-deploy-time-ensemble-agent.md#8.1.6-034f-router-已实现并完成首轮-eval-negative-result)).
- **纯 greedy eval**: 与 single-ckpt 口径对齐，`ENSEMBLE_GREEDY=1`.

### 4.2 Risks

1. **Architectural diversity 不足** — 4 个成员都是 team-level cross-attention family.
   [snapshot-034 §11.4](snapshot-034-deploy-time-ensemble-agent.md#114-关键-lesson--naive-prob-averaging-ensemble-的-lift-来源)
   已实证：`034ea = {031B + 045A + 051A}` 三个 team-level = **anti-lift -0.1pp**。
   本轮可能复现。主风险。
2. **Correlated failure modes** — `055` 学了 `034E` 的 combined 决策分布，
   再加上 `056D/062a/053Dmirror` 都在 031B 基或其变体上训练，
   deep decision correlation 可能 >> failure-bucket fingerprint 表面 diversity。
3. **Deploy cost vs grading 要求** — 作业要求 single `AgentInterface.act()` 能 run，
   4-way ensemble 满足接口但 zip 体积 4x (~200MB)，需验证 submission 流程。
4. **PBRS-only vs v2 shape disagreement** — `053Dmirror` 的 PBRS-only 在球场几何
   (position-based) 上给出的 action prior，可能和 `055/062a` 的 v2 goal-proximity prior 在关键 state 相互抵消
   (同 [snapshot-034 §8.1.2](snapshot-034-deploy-time-ensemble-agent.md#8.1.2-这意味着什么) 旧版 legacy triad 失败原因)。

### 4.3 Retrograde (回退策略)

- 若 §3.4 triggered（WR < 0.900）: 立即 **ensemble 降级为 best single (055@1150)**，
  把 074 lane 记为 negative result，文档化在 §7 verdict 里；**不尝试 HeuristicRoutingMixedEnsembleAgent
  手动 tune**（router path 已被 034F 否决）。
- 若 §3.3 triggered（tied）: 不上 submission，保持 `055@1150` 为 grading primary。
  启动 §3.6 alt hypothesis check，如 alt 命中则保留 074 为 "peer-axis / bonus 候选"。
- 若 H2H §3.7 triggered 为 negative: 同 §3.4 处理。

---

## 5. What we are NOT doing

- **不做任何 training** — 本 snapshot 是纯 deploy-time eval, 所有成员 frozen。
- **不做架构改动** — 不新实现 `EnsembleAgent`；复用 `ProbabilityAveragingMixedEnsembleAgent`.
- **不做 > 4 成员 ensemble** — 边际 diversity 收益递减, 违反 [snapshot-034 §8](snapshot-034-deploy-time-ensemble-agent.md#8-不做的事) 原则。
- **不做 weighted / learned / router 版本** — 首轮 principled 对照; 如 074A 命中再单独开 snapshot-075 做 weighted follow-up。
- **不纳入 `055v2@1000`** — 与 `055@1150` family 同质 (recursive distill 共享 4/5 teachers).
- **不纳入 `034E-frontier`** — 本身已被 `055` dominated (H2H 0.590 `***`).
- **不触碰其他 training lanes** — `074` 可在任何 GPU idle 上启动, 完全不 block `063-068` 等训练线。

---

## 6. Execution checklist

- [ ] 1. Snapshot drafted (**this file**)
- [ ] 2. Read existing `034E` wrapper code for ensemble mechanics —
      [cs8803drl/deployment/ensemble_agent.py](../../cs8803drl/deployment/ensemble_agent.py) /
      [agents/v034e_frontier_031B_3way/agent.py](../../agents/v034e_frontier_031B_3way/agent.py)
- [ ] 3. 实现 `agents/v074a_frontier_055_053D_062a/agent.py`
      - 成员: `055@1150` (team_ray) + `053Dmirror@670` (team_ray) + `062a@1220` (team_ray)
      - checkpoint 绝对路径从 rank.md §3.3 与 snapshot-055/053D/062 记录中取
- [ ] 4. 1-iter smoke test: 10 episodes vs random, 确认推理链路通
      - `scripts/eval/evaluate_official_suite.py --team0-module agents.v074a_frontier_055_053D_062a --opponents random -n 10`
- [ ] 5. baseline 1000ep eval (§3.1-§3.4 判据)
- [ ] 6. random 500ep sanity (§3.5)
- [ ] 7. H2H vs `055@1150` n=500 (§3.7)
- [ ] 8. 按 §3 判据写 §7 verdict
- [ ] 9. 如 §3.2 但未达 §3.1: 启动 `074B` (4-way 加 056D)
- [ ] 10. 如 §3.1 命中: 准备 grading submission candidate zip, 更新 rank.md §3.3

预估总 wall-clock: 15-20 min smoke + 20-30 min 1000ep baseline + 10-15 min random + 15 min H2H
= **约 1 hour on single H100** (所有步骤串行; 可跟任何 training lane 共节点 — 仅占
~200MB GPU + 1 CPU worker).

---

## 7. Verdict (2026-04-21 06:50 EDT)

**074A baseline 1000ep = 0.903 (903W-97L, n=1000)** → **§3.3 tied** [0.900-0.914]

### 7.1 Stage 1 raw result

```
=== Official Suite Recap (parallel) ===
.../055_distill_034e_ensemble.../checkpoint_001150/checkpoint-1150 vs baseline: win_rate=0.903 (903W-97L-0T)
elapsed=526.1s (on 5028919, j=1, base_port=65005)
```

注: log 只列 checkpoint-1150 是因 eval 脚本 arg 用 dummy ckpt (ensemble agent 实际从 `agents.v074a_frontier_055_053D_062a` 内部 member 表加载 3 个 ckpts;ensemble eval 与 single-ckpt eval 在日志行上不区分)

### 7.2 判据对照

| 阈值 | peak 0.903 | 判定 |
|---|---|---|
| §3.1 breakthrough ≥ 0.920 | 0.903 | ✗ MISS (-0.017pp) |
| §3.2 main ≥ 0.914 | 0.903 | ✗ MISS (-0.011pp) |
| §3.3 marginal ≥ 0.908 | 0.903 | ✗ MISS (-0.005pp) |
| **§3.4 tied [0.895, 0.908)** | **0.903** | **✅ HIT** |
| §3.5 regression < 0.890 | 0.903 | ✗ NO |

vs 055@1150 combined 2000ep 0.907: **Δ = -0.004** within SE ±0.016 → **统计 tied**, 但方向向下

### 7.3 机制分析 — 为什么 ensemble 没抬

**arithmetic mean baseline**: 3 member 等权平均 **预期 ≈ (0.907 + 0.902 + 0.892) / 3 = 0.900**
**实测**: 0.903 — 比算术平均高 +0.003pp,但 **远低于 orthogonal-failure-mode 预期的 +0.01~0.02pp 提升**

**3 个可能原因** (不互斥):
1. **Effective correlation > 0**: 055/053Dmirror/062a 虽然 reward signal 不同(v2/PBRS/no-shape),但都是 **031B Siamese cross-attn 架构 descendants + 都基于 v2-style exploitation**,失败模式 actually correlated,不 orthogonal
2. **Averaging 稀释 argmax-strong-signal**: 055 的 policy confidence 高(distill 过 3-teacher ensemble),其他 member 的 softmax 偏离时,等权 avg 把 055 拉偏
3. **No arch diversity**: snapshot-034 §11.4 已经警告过 team-level-only ensemble 历史 anti-lift pattern, 这次再验证

### 7.4 Loss bucket pending (alt hypothesis §3.6)

Alt hypothesis: 即使 baseline tied, ensemble 可能在 **failure-mode rounding** 上有价值 (例: loss distribution 更均匀,top bucket 浓度降低)。
- **待做**: Stage 2 capture 074A vs baseline n=500, 对比 loss buckets vs 055@1150 capture (see rank.md §3.3 055temp row 的 bucket 对照)
- 若 top-bucket concentration 降低 ≥ 5pp 且无新 bucket emergence → **074 有 deploy 价值作为 "rounded policy"**
- 若 bucket distribution 跟 best single 基本一致 → confirm pure anti-lift, **close 074A 方向**

### 7.5 后续: 继续探索 074B-E

用户授权测多版本。074A 只是 1/5。按新 blocker 语义,下一批立即跑:

- **074B** (arch diversity, 055+054M@1230+062a) — 已 launched 06:50 on 5028919
- **074C** (H2H-least-correlated, 055+056D+053Dmirror)
- **074D** (failure-bucket orthogonal, 055+055v2@1000+053Acont@430)
- **074E** (074A + predictor rerank)

074A 的 tied 提示: **需要真 arch/H2H 多样性**(074B/C 测),光靠 reward-path 多样性不够

### 7.6 (2026-04-21 07:30 EDT) — 074 family 全 5 variants 完成 — meta-conclusion 汇总 (append-only)

5 variants 测完后 (074A/B/C/D/E), 结论汇总到此 snapshot 主 record:

| variant | mechanism | members | baseline 1000ep | verdict |
|---|---|---|---:|---|
| **074A** | prob-avg + blood diversity | 055 + 053Dmirror + 062a | **0.903** | §3.3 tied |
| **074B** | prob-avg + arch diversity | 055 + 054M@1230 + 062a | **0.877** | §3.4 REGRESSION |
| **074C** | prob-avg + H2H-orth | 055 + 056D + 053Dmirror | **0.902** | §3.3 tied |
| **074D** | prob-avg + bucket-orth | 055 + 055v2 + 053Acont | **0.900** | §3.3 tied (borderline) |
| **074E** | prob-avg + predictor rerank | 074A + v3 predictor | **0.893** | §3.4 marginal regression |

**4/5 tied (0.900-0.903) + 1/5 regression (074B 0.877); 074E drags -0.010 below tied plateau via predictor tilt**。

**Confirms [snapshot-034 §11.4](snapshot-034-deploy-time-ensemble-agent.md#114-关键-lesson--naive-prob-averaging-ensemble-的-lift-来源) "team-level prob-avg ensemble anti-lift" pattern 4x independently** (074A/074C/074D tied + 074B regress)。Siamese family prob-avg ensemble **ceiling ≈ arithmetic mean of members ≈ 0.90**, 不受 member selection 方法影响。

**Deploy-time ensemble paradigm closed for this project** — 全 5 探测轴 (blood / arch / H2H / bucket / predictor rerank) 用尽, arch diversity 反伤, value-head rerank 反伤, member selection 方法 null lift。

**Resources flow back to training-side**:
- Pool A/B/C/D (snapshot-071/070/072/073 — 3-4 teacher distill variants, different blood/arch/reward)
- 076 wide-student distill (DIR-A, arch capacity) — RUNNING
- 077 per-agent student distill (DIR-B, cross-arch student) — RUNNING
- 079 055v3 recursive distill (BAN round 2) — RUNNING

**Grading submission primary**: 保持 **055@1150 single-model** (combined 2000ep 0.907) — 所有 ensemble/rerank 方向未能 break tied plateau。

---

## 8. Follow-up paths

根据 §7 verdict 分支:

- **如 §3.1 命中 (≥ 0.920)**: `074A` 成为 grading submission primary candidate, 项目 SOTA 从
  single-model 0.907 升级为 ensemble ≥ 0.920。写 snapshot-075 作 weighted ensemble follow-up
  (按 indiv WR 加权, 看能否再 +0.5pp); H2H 扩展到 vs `031B / 062a / 055v2 / 034E` 5-way 矩阵。
- **如 §3.2 命中但未达 §3.1 (0.914-0.919)**: 启动 `074B` (4-way 加 056D); combined 2000ep rerun
  `074A` 做 noise 证实; 对比 alt hypothesis §3.6.
- **如 §3.3 tied (0.900-0.914)**: 不升级 submission; 保留 074 作 "bonus 未知 agent2 的 rounded policy" 候选
  — 专跑 alt hypothesis §3.6 + vs unknown-agent-analog H2H (若释放); **不做其他 member substitution**
  (那是 `072` / `073` 的事)。
- **如 §3.4 regression (< 0.900)**: 文档化 negative result, 加入 [snapshot-034 §11.5](snapshot-034-deploy-time-ensemble-agent.md#115-重要-meta-lesson--ensemble--智力提升)
  meta-lesson 列表 ("team-level-only ensemble 边际效用 = 0 确认 3x"); 不继续 deploy-time 方向,
  资源转向单 model 架构 / 新训练路径 (e.g., snapshot-068/066 等 learned / progressive distill 线).

**不 follow-up 的方向** (避免 fall into 034F router 历史 sunk cost):
- heuristic router on 074 members
- learned gating (MoE-style) on frozen members
- 尝试 074C/D/E 不断替换成员 — 如果 074A/B 都没 lift, 原因是 family correlation, 换成员不救。

---

## 9. Related

- [SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md) — 034A/B/C/D/E/F 全家族历史, mean-of-probs / mixed-arch / router 所有已验证路径的完整 lesson log
- [SNAPSHOT-055](snapshot-055-distill-from-034e-ensemble.md) — `055@1150` 作为 034E teacher 的单 student, 本 snapshot 主 anchor
- [SNAPSHOT-061](snapshot-061-055v2-recursive-distill.md) — `055v2@1000` 证据; 解释为什么不纳入 074 成员表
- [SNAPSHOT-056](snapshot-056-simplified-pbt-lr-sweep.md) — `056D@1140` (4-way secondary 候选) 来源
- [SNAPSHOT-062](snapshot-062-curriculum-noshape-adaptive.md) — `062a@1220` 来源, 唯一 curriculum blood
- [SNAPSHOT-053D](snapshot-053D-mirror-pbrs-only-from-weak-base.md) — `053Dmirror@670` 来源, 唯一 PBRS-only blood
- [rank.md §3.3](rank.md#33-official-baseline-1000frontier--active-points-only) — 当前 frontier baseline 表, 本 snapshot 成员选择依据
- [作业要求 — Performance 25+25 (+5 bonus)](../references/Final%20Project%20Instructions%20Document.md#policy-performance-50-points) — grading axis, ensemble submission 合规性要求
