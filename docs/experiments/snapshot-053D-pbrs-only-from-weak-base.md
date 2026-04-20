# SNAPSHOT-053D: PBRS-only from weak base — fair test of "is v2 a ceiling on PBRS?"

- **日期**: 2026-04-19
- **负责人**: Self
- **状态**: 预注册 (snapshot-only); pending free 16h+ node 后 launch
- **预计 budget**: 800 iter ≈ 12h GPU

## 0. 观察 + 理念 (mandatory upfront, **per user 直接要求**)

### 0.1 触发观察 (2026-04-19 23:30 EDT 用户提出)

053Acont 完成后 053 family 整体看起来是 **plateau saturate ~0.898**。同时 051D verdict 出来:
- 051D = 031B@**80** (weak base, ~50% WR) + learned-only (no v2) + **800 iter** → combined 0.889
- 051A = 031B@**1220** (strong base) + v2+learned (combo) + **200 iter** → combined ~0.888 (single-shot)

我**第一反应** wrongly framed 这个对比为 "051A vs 051B (相同 base, 200 iter, ±v2)" 来反驳 v2 是 ceiling. 用户 critique 我:

1. **051A vs 051B 不是 fair "v2 vs no-v2" test**: 两者 warmstart 都从 031B@1220 (1220 iter v2 训练 baked-in). 删 v2 让 051B 退化只是因为 **mid-training 撤掉 dense signal 让 policy 失稳**, 不是 "no-v2 本质更差". 类比: 一直吃糖的人突然戒糖会颤抖, 但不代表糖是必需品.
2. **051D 的真正设计意图** 才是 fair test: weak base (v2 prior 极弱) + learned-only + 充分 budget. 答案是: learned-only 在 fair setup 下能匹配 combo, 不需要 v2.
3. **应用到 053**: 053A/053Acont 都从 031B@1220 (full v2 baked) warmstart + 加 PBRS combo. v2 的"贡献"主要是 baked-in 的 prior + 200-500 iter 内 inertia. 真正的 PBRS-only fair test 应该 mirror 051D: **PBRS-only from weak base (031B@80) + 800 iter**.

### 0.2 核心 hypothesis

> **v2 hand-crafted shaping 在 053 combo 中的角色不是 accelerator, 而是 ceiling constraint**. 因为 v2 的具体 priors (turtle penalty, possession bonus, deep_zone penalty) 把 policy 推向 **specific behavioral mode**, 限制 explore 空间. PBRS-only (Ng99 policy invariance) 不带这种 prior, 给充足 budget 应能 explore 更高 ceiling.

**预期 (preregistered)**:
- 053D peak combined ≥ **0.900**: confirm v2 是 ceiling, PBRS-only superior
- 053D peak combined ≈ 0.890 (跟 051D 同档): PBRS path 跟 learned-bucket path 在 fair setup 下相当, **v2 也不是 ceiling 但 PBRS 不比 learned-bucket 强多少**
- 053D peak combined < 0.880: PBRS-only 在弱 base 上不够 carry 训练, v2 是 functional bridge

### 0.3 跟 053A/053Acont 的对比意义

| Lane | warmstart | warmstart v2-prior | Reward (训练时) | iter | 测的问题 |
|---|---|---|---|---|---|
| 053A | 031B@1220 | 大量 (1220 iter) | v2 + outcome-PBRS | 200 | combo 在 saturated 起点的边际值 |
| 053Acont | 031B@1220 (via 053A@200) | 大量 + 200 iter combo | v2 + PBRS continue | 300 (200→500) | combo 长跑能否 break 053A 0.891 |
| **053D** (this) | **031B@80 (weak)** | **少量 (80 iter)** | **outcome-PBRS only (no v2)** | **800** | **PBRS 单独能否在 fair setup 下 break combo ceiling** |
| (mirror) **051D** | 031B@80 | 少量 (80 iter) | learned-only (no v2) | 800 | learned-bucket 同问题 |

053D **直接镜像 051D 设计** — 唯一变量是 reward 类型 (PBRS vs learned-bucket). 如果 053D > 051D 0.889 → PBRS 在 fair setup 也强 (与 051D 同档证明 reward type 比 v2 重要).

### 0.4 Caveat (mandatory honest disclosure)

跟 051D 一样的 caveat: 031B@80 **仍含 80 iter v2 prior**, 不是 v2-free 的纯净测试. 真正 v2-free 测试需要:
- BC pretrain (从未做过, 工程 1-2h)
- 或 scratch (前 300-500 iter OOD, learned reward 给不出梯度, 历史从未跑通)

053D 是"v2 影响显著减弱"的近似测试, 不是 absolute v2-free. 如果 053D ≥ 0.900, **decisive evidence v2 是 ceiling**; 如果 053D < 0.900, 还可以追加 BC-pretrain pure-v2-free version (053E backlog).

## 1. 核心假设

### H_053D-main

> 053D (031B@80 + outcome-PBRS only + 800 iter) **1000ep peak combined ≥ 0.900**, decisively > 053Acont 0.898, confirming v2 priors limit ceiling.

### 子假设

- **H_053D-a**: PBRS calibrated V(s) 在 weak base 上能提供 enough density 让 PPO 正常学 (not OOD-doomed)
- **H_053D-b**: 800 iter 足够让 PBRS 单独 carry 训练到 saturate (vs 051D 800 iter learned-only 也成功)
- **H_053D-c**: 没有 v2 inertia, policy 自由 explore non-v2-favored behavior modes (e.g., 不被 turtle penalty 强制 forward)

## 2. 设计

### 2.1 配置 (mirror 051D + swap reward)

```bash
# Warmstart: weak 031B base (mirror 051D)
WARMSTART_CHECKPOINT=/home/hice1/wsun377/Desktop/cs8803drl/ray_results/031B_team_cross_attention_scratch_v2_512x512_20260418_233553/TeamVsBaselineShapingPPOTrainer_Soccer_ea2de_00000_0_2026-04-18_23-36-13/checkpoint_000080/checkpoint-80

# Architecture: same cross-attn (mirror 051D + 053A)
TEAM_SIAMESE_ENCODER=1 TEAM_CROSS_ATTENTION=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION_TOKENS=4 TEAM_CROSS_ATTENTION_DIM=64

# Reward: PBRS only (NO v2)
USE_REWARD_SHAPING=0
# All SHAPING_* unset
OUTCOME_PBRS_PREDICTOR_PATH=/home/hice1/wsun377/Desktop/cs8803drl/docs/experiments/artifacts/v3_dataset/direction_1b_v3/best_outcome_predictor_v3_calibrated.pt
OUTCOME_PBRS_WEIGHT=0.01     # 同 053A
OUTCOME_PBRS_WARMUP_STEPS=10000
OUTCOME_PBRS_MAX_BUFFER_STEPS=80

# Budget: 800 iter (mirror 051D)
TIMESTEPS_TOTAL=32000000 MAX_ITERATIONS=800 TIME_TOTAL_S=43200 CHECKPOINT_FREQ=10

# PPO (同 053A / 051D)
LR=0.0001 CLIP_PARAM=0.15 NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000 SGD_MINIBATCH_SIZE=2048
BASELINE_PROB=1.0
```

### 2.2 工程实现

**0 工程** — 全部 env vars 已支持:
- WARMSTART_CHECKPOINT 已支持
- USE_REWARD_SHAPING=0 已支持 (snapshot-051 §0.2)
- OUTCOME_PBRS_PREDICTOR_PATH 已支持 (snapshot-053 §3.2)
- 直接复用 `cs8803drl/training/train_ray_team_vs_baseline_shaping.py`

仅需写 launch script (5 min).

### 2.3 PORT_SEED

避开 active lanes:
- 054=31, 053Acont=25, 055=37, 056ABCD=41-44, 057=35, 058=39
- **053D 用 PORT_SEED=27**

## 3. 预注册判据

| 判据 | 阈值 | verdict | 行动 |
|---|---|---|---|
| §3.1 突破 | combined ≥ 0.900 | **v2 是 ceiling, PBRS-only superior** | 写 053-paradigm verdict, distillation/ensemble 把 053D 加入 |
| §3.2 持平 (PBRS ≈ learned-bucket fair) | combined ∈ [0.885, 0.900) | PBRS 不比 learned-bucket 强多少, v2 是 weak constraint | reward family 整体 saturate, 转 paradigm shift (distill/curriculum/RND) |
| §3.3 持平 v2-baked combo | combined ∈ [0.880, 0.885) | learned-only 慢匹配, v2 是 accelerator 不是 ceiling | 053A 范式确认 |
| §3.4 退化 | combined < 0.880 | PBRS-only 在 weak base 上 carry 不动 | v2 是 functional bridge, hypothesis refuted |

## 4. 简化 + 风险 + 降级 (per user mandatory)

### 4.1 简化 — 只测 1 个 PBRS λ value

`OUTCOME_PBRS_WEIGHT=0.01` 沿用 053A 配置, 不 sweep λ.

**风险**: 在 weak base 上, λ=0.01 可能不是 optimal (053A 的 λ 是在 strong base 上 tuned 的). 如 PBRS signal 太小淹没在 sparse goal reward 里, 训练 stall.

**降级 plan**:
- L1: 如 1000ep peak < 0.880, sweep λ ∈ {0.005, 0.01, 0.02, 0.05} 用 4 个 free node (后续)
- L2: 把 OUTCOME_PBRS_WARMUP_STEPS 减到 0 (让 weak base 早期就用 PBRS)

### 4.2 简化 — 不重训 predictor 适配 weak base

A3 calibrated predictor 是在 frontier-vs-frontier H2H 上训练的, vs **weak vs baseline state distribution** 可能 misaligned (predictor OOD).

**风险**: predictor 输出在 weak state 上 noise 大, ΔV 信号被 noise 主导. PBRS theory 保证 reward hacking 不改 optimal policy 但可能减慢收敛.

**降级 plan**:
- L1: 接受 noisy predictor, 让 800 iter 长 budget 平均掉 noise
- L2: 在 weak vs baseline trajectory 上 fine-tune predictor (1-2h 工程)

### 4.3 简化 — 不 mid-training swap reward

不引入 v2-decay schedule (避免 051B 那种 mid-training 失稳). PBRS-only 全程一致.

### 4.4 降级序列

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | Default | 800 iter, λ=0.01, A3 predictor | base 12h |
| 1 | peak < 0.880 | λ sweep 4 lane | +12h × 4 |
| 2 | step 1 全失败 | predictor fine-tune on weak data | +2h 工程 + 12h |
| 3 | step 2 失败 | declare v2 functional bridge, 053D 路径关闭 | — |

## 5. 不做的事

- 不做 BC-pretrain warmstart (那是 053E if 053D 不结论)
- 不混入 v2 mid-training (避免 051B 失稳 trap)
- 不 swap calibrated predictor (用 A3 same as 053A)
- 不 launch 直到 free 16h+ node available (056ABCD 完会释放)

## 6. 执行清单

- [ ] 1. 写 launch script (~5 min): `scripts/eval/_launch_053D_pbrs_only_from_weak_base.sh`
- [ ] 2. 等 056 lane 任意完成 → free 一个 16h fresh node
- [ ] 3. Health check + launch (PORT_SEED=27, BASE 56955)
- [ ] 4. Monitor: 早期 (50ep WR @ iter 50) 应能脱离 0.50 base; 否则 PBRS 信号在弱 base 失效, kill 转 §4 step 1
- [ ] 5. 训完 invoke `/post-train-eval` lane name `053D`
- [ ] 6. Stage 2 capture peak ckpt
- [ ] 7. Stage 3 H2H: vs 051D@740 (direct PBRS vs learned-bucket fair test) + vs 053Acont@430 (v2-combo vs no-v2)
- [ ] 8. Verdict append §7

## 7. Verdict

_Pending_

## 8. 后续路径

### Outcome A — 突破 (combined ≥ 0.900)
- v2 是 ceiling 实证
- 直接挑 distillation 055 / curriculum 058 中的 v2-free variants 同样优化
- 034f next-gen ensemble 把 053D + 053Acont + 043A' 作为 mode-orthogonal candidates

### Outcome B — 同 051D 档 (≈ 0.889)
- PBRS path 跟 learned-bucket path 在 fair setup 下没差 — reward family 整体 saturate
- 转 paradigm shift (distill/curriculum/RND/PBT) 是真正出路

### Outcome C — 退化 (< 0.880)
- v2 是 functional bridge for early bootstrap
- 053 paradigm verified 但 v2 是 indispensable
- 053D backlog 关闭, 不再投资 PBRS-only

## 9. 相关

- [snapshot-051 §0.0](snapshot-051-learned-reward-from-strong-vs-strong-failures.md#00-设计理念--design-rationale-051c051d-回填--2026-04-19) — 051D 设计理念 (053D 镜像它)
- [snapshot-053](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md) — outcome-PBRS 范式 + A3 calibrated predictor
- [snapshot-053 §11](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md#11-053a-continue-iter-200500--plateau-style-stability--verdict-2026-04-19) — 053Acont verdict (053D 比较对象)
- [outcome_pbrs_shaping.py](../../cs8803drl/imitation/outcome_pbrs_shaping.py) — PBRS wrapper code (复用)

### 理论支撑

- **Ng et al. 1999** "Policy invariance under reward transformations" (PBRS theorem) — 保证不管 PBRS V(s) 选什么, optimal policy 不变, 只可能改 convergence speed
- **051D verdict** (snapshot-051 §9) — empirical evidence learned-only path 在 weak base + 长 budget 下能 carry 训练
- **Ng & Russell 2000** "Algorithms for Inverse Reinforcement Learning" — 启发 PBRS V(s) 来自 outcome predictor 的合理性
