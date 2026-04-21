# SNAPSHOT-053D-mirror: PBRS-only from weak base — fair test of "v2 as ceiling" hypothesis

- **日期**: 2026-04-19
- **负责人**: Self
- **状态**: 预注册 / backlog (等 free 12h+ node, e.g. 056 lane 完释放)

---

## 0. 观察 + 理念 (mandatory upfront)

> 本节 captures 实验的设计动机, 必须先写清观察+理念再展开 design — 这是用户在 053Acont/051D verdict 后的明确 directive (响应过去 snapshot-051 漏写设计理念的教训).

### 0.1 触发观察 (the observation that prompted this snapshot)

2026-04-19 晚 verdict 收齐后, 出现一个 informative pattern:

| Lane | 起点 | Reward | Budget | Combined 1000ep |
|---|---|---|---|---|
| 051A | 031B@1220 (strong, 0.88) | v2 + learned (combo) | 200 iter | 0.888 |
| 051B | 031B@1220 (strong, 0.88) | learned only (no v2) | 200 iter | 0.872 (退化) |
| **051D** | **031B@80 (weak, 0.50)** | **learned only (no v2)** | **800 iter** | **0.889** |
| 053A | 031B@1220 (strong, 0.88) | v2 + outcome PBRS (combo) | 200 iter | 0.891 |
| 053Acont | 031B@1220 (strong, 0.88) | v2 + outcome PBRS (combo, continue) | 500 iter | 0.898 |

用户 [22:50 EDT] 提出 hypothesis:
> "051D 的结果也许在暗示, 我们的 053 如果拿掉 v2 shaping, 可能会带来更好的结果. 这不是 budget 的问题, 而是上限的问题."

### 0.2 理念 / 解读这个观察的两种 framing

**Framing A (saturation hypothesis, my initial framing)**:
- 051D learned-only 800 iter ≈ 051A combo 200 iter ≈ 0.889
- → "learned-only 用 4× budget 才追上 combo, v2 是 dense signal accelerator, 没了它就要更多 budget"
- → 结论: v2 是 useful but not unique; 移除 v2 不会突破 ceiling, 只会 slow

**Framing B (v2 as constraint, 用户的 hypothesis)**:
- 051D 的 weak base (031B@80) **几乎没被 v2 shape**, 而 053A/053Acont 的 base (031B@1220) **完全被 v2 shape 1220 iter**
- 053A 的 v2-baked-in priors **可能本身就是 ceiling**: PBRS 不能 push higher 因为 policy weights 早被 v2 锁定
- → "v2 is a ceiling constraint, not just an accelerator"
- → 结论: 想突破 0.898 plateau, 需要从 v2-prior-light 起点 + PBRS-only 训练

### 0.3 为什么 framing B 不是 trivially refuted

我之前用 051A vs 051B 直比反驳 framing B (combo 0.888 > learned-only 0.872 at same budget on same base). 但用户指出这不是 fair test:
- 两者都从 031B@**1220** warmstart, weights 已被 v2 完全 shape
- 051B 删 v2 = **mid-training 撤掉 dense signal** → 失稳 (跟 051A 比 -1.6pp 是 destabilization 而非 "no-v2 本质更差")
- 真正 fair test 是: **weak base** (v2-prior-light) + **learned-only or PBRS-only** + **sufficient budget** → 看能否达到甚至超过 v2-baked combo

### 0.4 053D-mirror = 上述 fair test 的 053 family 版本

镜像 051D 的设计哲学, 但换 reward 路径:
- Base = **031B@80** (weak, 同 051D)
- Reward = **outcome-PBRS only** (no v2, no learned-bucket — 只 PBRS)
- Budget = **800 iter** (同 051D)

**两件事同时测**:
1. **PBRS path vs learned-bucket path 在 fair setup 下直比** (053D-mirror vs 051D)
2. **v2 是 accelerator vs ceiling 的 framing** (053D-mirror vs 053Acont)

### 0.5 不能算 v2-free 的 caveat (诚实承认)

跟 051D 一样, 031B@80 仍含 80 iter v2 prior. 真正 v2-free 需要 BC pretrain (snapshot-051 §0.0 提到的 B-2 候选, 1-2h 工程). 053D-mirror 接受 "v2 影响显著减弱" 的近似, 不是绝对 v2-free. 若结果支持 framing B, 进一步 BC pretrain 实验可以 follow up.

---

## 1. 核心假设

### H_053D-A (PBRS path 强于 learned-bucket path)

> 在 fair setup (weak base 031B@80 + reward-only + 800 iter) 下, **PBRS-only 1000ep peak ≥ 051D combined 0.889**, 且严格 > 051D 的 late-window mean 0.888.

### H_053D-B (v2 ceiling — 用户的 hypothesis)

> 053D-mirror 1000ep peak **≥ 053Acont combined 0.898**. 如果显著超过 (>0.905), 则 confirm v2-baked-in priors 是 053Acont 的 ceiling, 移除即可 break 0.900.

### 子假设

- **H_053D-a**: 80 iter v2 prior + outcome-PBRS dense signal 足以 bootstrap 训练 (不会像纯 scratch 那样 OOD failure)
- **H_053D-b**: PBRS theory (Ng99 invariance) 保证 reward 不引入 specific behavior bias, 比 v2 hand-crafted shaping 更"中性"
- **H_053D-c**: 800 iter 给 PBRS 充分时间从 weak base 收敛 (镜像 051D 给 learned-only 充分时间的成功 pattern)

---

## 2. 设计

### 2.1 配置

| 维度 | 配置 | 说明 |
|---|---|---|
| Warmstart | **031B@80** (weak ckpt, 50 iter v2 prior) | 镜像 051D, weak v2-prior |
| Architecture | cross-attention (TEAM_SIAMESE_ENCODER + TEAM_CROSS_ATTENTION) | 跟 031B base + 053A 一致 |
| **v2 shaping** | **OFF** (`USE_REWARD_SHAPING=0`, 所有 SHAPING_*=0) | 关键 ablation 变量 |
| **Outcome PBRS** | **ON** (predictor 同 053A: A3 calibrated, λ=0.01) | 唯一 reward signal |
| Learned-bucket reward | OFF (跟 051D 不同, 053D 测 PBRS path) | |
| Budget | **800 iter** (TIMESTEPS_TOTAL=32M) | 镜像 051D 长 budget |
| LR / CLIP / etc | 同 031B base (LR=1e-4, CLIP=0.15, 4 epoch) | 控制 PPO HP 变量 |

### 2.2 工程改动

**No new code** — 完全复用 053A 的 PBRS infrastructure:
- `cs8803drl/imitation/outcome_pbrs_shaping.py` (已 done)
- `OutcomePBRSWrapper` env wrapper 已 wired in `create_rllib_env`
- 053D-mirror 只需新 launch script: `_launch_053D_mirror_pbrs_only_warm031B80.sh`
  - 复制 053A launch script
  - 改 `WARMSTART_CHECKPOINT` → 031B@80 路径
  - 改 `USE_REWARD_SHAPING=0` + 清空所有 SHAPING_*
  - 改 `MAX_ITERATIONS=800`, `TIMESTEPS_TOTAL=32000000`
  - 改 PORT_SEED 隔离

### 2.3 训练超参 (基本同 053A 但去 v2)

```bash
# Architecture (同 031B base, 跟 053A 一致)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1
TEAM_CROSS_ATTENTION_TOKENS=4
TEAM_CROSS_ATTENTION_DIM=64

# Warmstart from weak ckpt (KEY change vs 053A)
WARMSTART_CHECKPOINT=ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/.../checkpoint_000080/checkpoint-80

# PBRS reward (同 053A)
OUTCOME_PBRS_PREDICTOR_PATH=docs/experiments/artifacts/v3_dataset/direction_1b_v3/best_outcome_predictor_v3_calibrated.pt
OUTCOME_PBRS_WEIGHT=0.01
OUTCOME_PBRS_WARMUP_STEPS=10000
OUTCOME_PBRS_MAX_BUFFER_STEPS=80

# v2 shaping OFF (KEY ablation vs 053A)
USE_REWARD_SHAPING=0

# Long budget (镜像 051D)
MAX_ITERATIONS=800
TIMESTEPS_TOTAL=32000000
TIME_TOTAL_S=43200

# PPO (同 031B base)
LR=1e-4 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048
BASELINE_PROB=1.0
```

---

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: peak ≥ 0.889 | == 051D combined | PBRS path 至少不弱于 learned-bucket path |
| §3.2 主: peak ≥ 0.898 | == 053Acont (v2 combo) | **PBRS-only 在 fair setup 下达到 v2-combo 等级** → framing B 部分支持 |
| §3.3 突破: peak ≥ 0.905 | +0.7pp 越过 053Acont | **strong evidence framing B**: v2 是 ceiling, 移除 break 0.900 |
| §3.4 持平: peak ∈ [0.880, 0.889) | sub-051D | PBRS path 在 fair setup 下不优于 learned-bucket; framing A confirmed |
| §3.5 退化: peak < 0.870 | < 031B base | PBRS-only 在 weak base 上 bootstrap 失败 (H_053D-a falsified) |

---

## 4. 简化点 + 风险 + 降级 + 预案 (mandatory per established convention)

### 4.1 简化 D.A — Single seed

| 简化项 | 完整 | 当前 |
|---|---|---|
| Seed reproducibility | 3 seeds × same config, average | 1 seed |

**风险**: PPO seed sensitivity (Henderson 2018) — single 0.005-0.015 swing.
**降级预期**: peak 估值 ±1pp.
**预案**: 若 peak ∈ [0.895, 0.905] (边缘 framing B), rerun 1 seed 验证. 若 peak >> 0.905, 不需 reseed (signal 大于 noise).

### 4.2 简化 D.B — 不做 BC pretrain (B-2)

| 简化项 | 完整 | 当前 |
|---|---|---|
| v2-free starting point | BC pretrained from raw demonstrations | 031B@80 (~80 iter v2 prior) |

**风险**: 80 iter v2 prior 仍 contaminate, true v2-free test 没做 — framing B 即使 confirmed 也只是部分 evidence.
**降级预期**: peak 估值 +0.5pp lower-bound (因为 80 iter v2 prior 仍 helps bootstrap).
**预案**: 若 053D-mirror confirm framing B (peak ≥ 0.898), 启动 BC-pretrain 后续 (B-2, 工程 1-2h + train 12h).

### 4.3 简化 D.C — 单 budget point (800 iter)

**风险**: 800 iter 可能不是 PBRS-only 的最优 budget. 051D 用 800 iter saturate 在 0.889; PBRS 信号若强可能 saturate 更早 (e.g. 400 iter), 若弱可能 1000+ iter 才到 peak.
**预案**: 训练中段 (iter 300, 500, 700) check 50ep WR 趋势; 若 800 iter 仍上升明显, 加 continue lane (类似 053A → 053Acont).

### 4.4 全程降级序列

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | Default | 800 iter PBRS-only from 031B@80 | base 12h GPU |
| 1 | peak ∈ [0.895, 0.905] (边缘) | rerun seed 验证 | +12h GPU |
| 2 | step 1 confirm peak ≥ 0.898 | BC pretrain follow-up (B-2) | +14h 工程+GPU |
| 3 | peak < 0.880 | learn rate sweep ({3e-5, 1e-4, 3e-4}) | +12h × 3 |
| 4 | step 3 全失败 | declare PBRS-only path 在 weak base 下 bootstrap fail; framing A 强支持 | — |

---

## 5. 不做的事

- 不在 implementation 完成 + smoke pass 之前 launch (复用 053A infrastructure 应零 smoke)
- 不混入 v2 即使 short warm phase (会 contaminate 实验)
- 不尝试 multi-head learned reward 替代 (已知 saturate at 0.889 by 051D)
- 不与 053A / 053Acont / 051D 同时跑 (避免 PBRS predictor 加载并发) — 等当前 lanes 完
- **不 declare framing B 仅 based on single shot**: 必须 rerun 验证若 peak 在 0.895-0.910 边缘

---

## 6. 执行清单

- [ ] 1. 写 launch script `scripts/eval/_launch_053D_mirror_pbrs_only_warm031B80.sh` (~10 min, 复制 053A + 改 4 处)
- [ ] 2. Smoke check (import 不报错 + warmstart ckpt path 有效) (~5 min)
- [ ] 3. 找 free 12h+ node (等 056 ABCD 中任一完成会 free 16h)
- [ ] 4. Launch 053D-mirror, 加 .running/.done trap convention
- [ ] 5. 实时 monitor: 50ep WR 趋势 (iter 100, 300, 500, 700)
- [ ] 6. 训完 invoke `/post-train-eval` lane name `053D-mirror`
- [ ] 7. Verdict append §7
- [ ] 8. 根据 verdict 走 §4.4 降级序列

---

## 7. Verdict (待 1000ep eval 后 append)

_Pending — backlog, 等 free 12h+ node_

### 7.1 2026-04-20 17:00 EDT — Stage 1 1000ep post-eval verdict (append-only)

**Training**: 2026-04-20 10:00-16:37, **TERMINATED at iter 800** (not full 1250 — `TIME_TOTAL_S` wall hit, 450 iter unexplored). `best_reward_mean = +1.5093 @ iter 732`. Run dir: `/storage/ice1/5/1/wsun377/ray_results_scratch/053Dmirror_pbrs_only_warm031B80_20260420_094739/TeamVsBaselineShapingPPOTrainer_Soccer_8c3d4_00000_0_2026-04-20_09-48-01`.

**Stage 1 1000ep post-eval (10 ckpts, official suite parallel, total_elapsed=504.8s):**

| ckpt | 50ep inline | 1000ep verified | W-L |
|---:|---:|---:|---:|
| 100 | 0.92 | 0.794 | 794-206 |
| 130 | 0.92 | 0.827 | 827-172-1T |
| 530 | — | 0.868 | 868-132 |
| 540 | 0.94 | 0.871 | 871-129 |
| 550 | — | 0.877 | 877-123 |
| **670** | — | **0.902** | **902-98** 🔝 |
| 680 | **0.96** (inline peak) | 0.897 | 897-103 |
| 690 | — | 0.888 | 888-112 |
| 780 | 0.92 | 0.880 | 880-120 |
| 790 | — | 0.868 | 868-132 |

**Verdict**:
- **Peak single-shot 1000ep = 0.902 @ iter 670** (902W-98L-0T, SE ±0.016).
- vs **031B combined 0.880**: Δ = **+0.022pp**, z ≈ 0.98 → **NOT yet statistically significant** (needs combined 2000ep to tighten SE).
- vs **031B-noshape 0.875**: Δ = **+0.027pp**, z ≈ 1.2 → still not sig.
- vs **055@1150 combined 0.907 (current SOTA)**: Δ = **-0.005pp** → **essentially tied with SOTA** (within 0.3σ).
- **Inline 50ep vs 1000ep gap**: inline 50ep peak 0.96 @ 680 vs 1000ep 0.897 → **+0.063pp optimistic** (consistent with 200ep noise memo).
- **Training truncated at iter 800 (wall hit)** — 450 iter unexplored, may have climbed higher if budget allowed.
- **Plateau iter 530-690 holds mean ~0.88**, with 670 peaking 0.902 — not single-point luck.

**Preliminary SOTA status**: **NOT confirmed yet** (single-shot 1000ep only, need combined 2000ep to validate). Currently ranks **approx 3rd**, behind 055 (0.907) and 056D / 056extend (0.89) tier. **Outcome reading**: falls between framing §4.4 Outcome A (≥0.905) and Outcome B (≈0.898) — lean Outcome A if combined verifies, else Outcome B.

**Follow-up recommendations** (to feed §8 decision tree):
1. **Rerun ckpts 500-1000ep** to tighten SE → Stage 2 rerun for combined ≥ 1500 ep on peak ckpts (530 / 670 / 680 / 690).
2. If combined > 0.90 confirmed, run **H2H vs 055@1150** to test whether self-PBRS (no teacher) matches distill ceiling.
3. **053Dmirror-v2**: resume 053Dmirror @ ckpt 730-790 with longer `TIME_TOTAL_S` to explore iter 800-1250 window that wall-terminated.

---

## 8. 后续路径 (基于 verdict 的决策树)

### Outcome A (peak ≥ 0.905, framing B 强支持)
- 立即启动 BC-pretrain (B-2) 进一步去 v2 prior, 测真 v2-free PBRS 上限
- 把 053Acont base ckpt 训练 retro 改 PBRS-only continue (从 031B@80 PBRS-only 训出来的中段 ckpt 接续)

### Outcome B (peak ≈ 0.898, framing B 部分支持)
- v2 不是 hard ceiling 但也不 unique 必要
- 投资 distillation (snapshot-055) 把 053D-mirror knowledge 压缩进 single network
- 不必再投资 v2 ablation (信号已饱和)

### Outcome C (peak < 0.889, framing A confirmed)
- v2 是 dense signal accelerator, 不是 ceiling
- 053D-mirror lane 关闭
- 053Acont 0.898 是当前 PBRS path 实际 peak, 转 distill / curriculum / RND

---

## 9. 相关

- [SNAPSHOT-053](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md) — 053A/Acont 来源, PBRS infrastructure
- [SNAPSHOT-051 §0.0](snapshot-051-learned-reward-from-strong-vs-strong-failures.md#00-051c051d-设计理念回填-2026-04-19) — 051D 设计理念 (本 snapshot 镜像的 learned-bucket 版本)
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B base 来源
- [outcome_pbrs_shaping.py](../../cs8803drl/imitation/outcome_pbrs_shaping.py) — PBRS wrapper 复用

### 理论支撑

- **Ng et al. 1999** "Policy invariance under reward transformations" — PBRS 理论 backbone, 保证 reward shaping 不改变 optimal policy. 本 snapshot 利用此性质论证: PBRS 可以提供 dense signal 而不引入 specific behavioral bias (vs v2 hand-crafted shaping 的 specific priors)
- **Hinton et al. 2015** distillation — 若 framing B confirmed, 053D-mirror 可作为 distillation 055 的 teacher 之一 (multi-source teachers 体现 reward path orthogonality)
