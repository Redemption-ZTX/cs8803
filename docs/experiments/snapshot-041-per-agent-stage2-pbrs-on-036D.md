# SNAPSHOT-041: Per-Agent Stage 2 PBRS Handoff on 036D Base

- **日期**: 2026-04-18
- **负责人**:
- **状态**: 预注册 / batch 待写
- **依赖**: [SNAPSHOT-036D](snapshot-036d-learned-reward-stability-fix.md) 的 `036D@150` 已坐实 1000ep mean **0.843** / max **0.860**（首次 learned-reward fine-tune 在 1000ep max 上超 029B@190 warmstart）

## 0. Known gap 诊断（动机来源）

回顾 [SNAPSHOT-040 §0](snapshot-040-team-level-stage2-on-031A.md#0-known-gap-诊断)：per-agent SOTA `029B@190 (0.868)` 的形成路径是**两阶段**：

```
Stage 1: 017 BC@2100 (0.842) → 026B-warm PBRS (peak 0.864 / 主候选 @170 0.842)
Stage 2: 026B@170 → 029B v2-handoff (0.846 → 0.868, +2.6pp)
```

040 把 Stage 2 镜像跑到 team-level 高 base 上 (031A@1040 = 0.860)。041 是它的 **per-agent 镜像**：
- 040: `031A@1040 (0.860, Siamese architecture)` + advanced shaping
- **041 (本): `036D@150 (0.860, learned reward + 029B-warmstart)` + advanced shaping**

`036D@150` 是项目第一个 per-agent 的 learned-reward base 突破 029B 平台。这是 per-agent lane 上**第一次有比 029B@190 还更强的 base 可以接 Stage 2**。

## 1. 核心假设

> **036D@150 的 0.860 来自 029B-warmstart + light learned reward (λ=0.003) + warmup10。把 026/038 体系内已知的 advanced shaping (PBRS / event / depenalized v2 / entropy-only) 作为 Stage 2 叠加在 036D 这个 per-agent 高 base 上，能否复现 per-agent `026 → 029` 的 +2.6pp 增益，把 1000ep 推到 0.88+？**

子假设：

- **H1 (PBRS handoff)**: 036D + PBRS goal-prox handoff（029B 同款配方）→ +1.5~+2.5pp
- **H2 (event)**: 036D + event-based shaping → +0~+1.5pp
- **H3 (entropy)**: 036D + entropy=0.01 fine-tune → +0~+1.5pp（单独考验 entropy 对 learned-reward 的稳定化）

如果 H1 出明显正号 (+2pp 以上)，**036D + PBRS 直接成为新的 SOTA 候选 0.88+**，把项目推到距 9/10 阈值仅 -2pp，且**可与 040B (team-level) 互为对照**回答："Stage 2 PBRS 在哪个底座架构上更有效"。

## 2. 与 SNAPSHOT-040 的关系（最关键对照）

| 维度 | SNAPSHOT-040 (team-level) | **SNAPSHOT-041 (per-agent, 本)** |
|---|---|---|
| Warmstart base | `031A@1040` (Siamese architecture, scratch trained) | `036D@150` (flat MLP, learned reward + 029B-warmstart) |
| 架构 | Siamese encoder `[256,256]×2 share + merge [256,128]` | flat MLP `[512,512]` (与 029B 同款) |
| Base 1000ep WR | **0.860** | **0.860** (并列！) |
| Base 来源 | architecture innovation | learned-reward + warmstart 链 |
| Stage 1 base reward | v2 only | v2 + light learned-reward (λ=0.003) |
| Stage 2 (本 snapshot) | + advanced shaping | + advanced shaping (镜像设计) |
| 镜像设计 | 040A/B/C/D = 026A/B/C/D in team-level | 041A/B/C/D = 026A/B/C/D in 036D-lineage per-agent |

**关键对照**：040 和 041 的 base WR 完全相等 (0.860)，但底座来源完全不同（架构 vs reward learning）。Stage 2 同 shaping 在两个 base 上的增益对比可以直接回答：
- 如果 040B > 041B：team-level 架构对 PBRS 更敏感
- 如果 041B > 040B：per-agent learned-reward 路径对 PBRS 更友好
- 如果两者接近：PBRS 增益与底座来源解耦

## 3. 候选 lane 矩阵

四条 Stage 2 lane，镜像 026/038/040 的 A/B/C/D：

| lane | 起点 | shaping 设计 | shaping env vars | 镜像 |
|---|---|---|---|---|
| **041A** | 036D@150 | depenalized v2 (去除 `time_penalty`/`opp_progress_penalty`/`deep_zone`) | `SHAPING_TIME_PENALTY=0` `SHAPING_OPP_PROGRESS_PENALTY=0` `SHAPING_DEEP_ZONE_*=0` | 026A / 038A / 040A |
| **041B** | 036D@150 | + goal-prox PBRS（029B handoff 同款） | `SHAPING_GOAL_PROXIMITY_SCALE=0.005` `SHAPING_GOAL_PROXIMITY_GAMMA=0.99` `SHAPING_GOAL_CENTER_X=15.0` `SHAPING_GOAL_CENTER_Y=0.0` | 026B / 029B / 038B / **040B** |
| **041C** | 036D@150 | + event-based (shot/tackle/clearance) | `SHAPING_EVENT_SHOT_REWARD=0.05` 等 | 026C / 038C / 040C |
| **041D** | 036D@150 | v2 + entropy=0.01 fine-tune | `ENTROPY_COEFF=0.01` shaping 不变 | 026D / 038D / 040D |

四条 lane 都用 **完全相同的 flat MLP `[512,512]` + 029B-warmstart 配方**（`FCNET_HIDDENS=512,512`），区别只在 reward 端。

**关键决策**：是否保留 036D 的 learned reward (`LEARNED_REWARD_SHAPING_WEIGHT=0.003 + warmup10`)？
- **保留**（推荐）：041 = 036D continuation + 加 PBRS。是"双层 shaping 是否叠加"的实验
- **撤掉**：041 = 029B + PBRS（与 029B 自己的 PBRS handoff 重复，无新意）

**首轮选保留 036D learned reward**，让 041 真正测试 "learned reward + analytical shaping 是否可叠加"。这也意味着 041 会**继承 036D 的 inf 问题**（31.7%）——见 §6 R3。

## 4. 训练超参

```bash
# warmstart from 036D@150 (per-agent base)
WARMSTART_CHECKPOINT=ray_results/PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107/MAPPOVsBaselineTrainer_Soccer_72c1c_00000_0_2026-04-18_12-41-29/checkpoint_000150/checkpoint-150

# learned reward 保留 (036D 配方)
LEARNED_REWARD_MODEL_PATH=ray_results/reward_models/036_stage2/reward_model.pt
LEARNED_REWARD_SHAPING_WEIGHT=0.003
LEARNED_REWARD_WARMUP_STEPS=10000

# v2 shaping (29B 同款)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001
SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01
SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# 041B-specific: 加 PBRS
SHAPING_GOAL_PROXIMITY_SCALE=0.005
SHAPING_GOAL_PROXIMITY_GAMMA=0.99
SHAPING_GOAL_CENTER_X=15.0
SHAPING_GOAL_CENTER_Y=0.0

# PPO (与 036D 同)
LR=1e-4
CLIP_PARAM=0.15
NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000
SGD_MINIBATCH_SIZE=2048
FCNET_HIDDENS=512,512

# Stage 2 budget
MAX_ITERATIONS=200             # 与 040 同款
EVAL_INTERVAL=10
EVAL_EPISODES=50
CHECKPOINT_FREQ=10
```

每条 lane budget: `~3-4h` (200 iter × ~1min/iter on H100)。

## 5. 预注册判据

### 5.1 主判据（成立门槛）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| official 1000ep | ≥ 0.86 | 至少不退化于 036D@150 base |
| **official 1000ep peak** | **≥ 0.875** | +1.5pp 增益（036D SE ≈ 0.011，意味着 +1.5pp 接近 1.4σ 边缘） |
| H2H vs 036D@150 | ≥ 0.52 | 直接证明 Stage 2 改进，不只是 baseline-only artifact |

### 5.2 突破判据（强成立）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| official 1000ep peak | ≥ 0.88 | **+2.0pp 突破**，对应 029B 路径上 026B@170 → 029B@190 那一跳 |
| H2H vs 029B@190 | ≥ 0.55 | 比 036D 自己的 H2H (vs 029B = 0.507 平局) 真正拉开 |
| failure capture: 任一 v2 bucket 比 036D 改善 ≥ 5pp | — | 机制层面证据 |

### 5.3 失败判据

| 条件 | 解读 |
|---|---|
| 1000ep < 0.85 | Stage 2 PBRS 把 036D 已学到的 wasted_possession 改善 (-9.2pp) 冲掉 |
| 1000ep ≥ 0.86 但 H2H vs 036D < 0.50 | PBRS 只在 baseline 上学了等价策略，无真实技能提升 |
| inf 率 ≥ 50% | 双 shaping 把 036D 已经 31.7% 的 inf 推得更糟 |

### 5.4 Lane 优先级

按预期 ROI：

1. **041B (PBRS handoff)** — 最高优先级，直接对应 per-agent 唯一突破路径
2. **041D (entropy)** — 次高，验证 entropy 对 learned-reward inf 是否有抑制作用
3. **041C (event)** — 中等
4. **041A (depenalized v2)** — 最低，sanity 对照

**首轮先跑 041B 单条**：如果 041B 出明显正号，041D 接续；其余视情况。

## 6. 风险

### R1 — Reward shock + 双层 shaping 叠加

**机制担心**：036D@150 已经被 learned reward 调过 150 iter。再加 PBRS 让 reward space 突变两次（learned + analytical）。这可能：
- 让 policy 完全失去 036D 学到的 wasted_possession 改善
- 触发更多 inf（PBRS 引入更大的 advantage 方差 → 更易溢出）

**缓解**：
- 监控前 20 iter 的 `wasted_possession` 趋势（capture 0.10/0.20pp 周期）
- 如果 `unclear_loss` 飙升 ≥ 18%，提前 abort

### R2 — Saturation（"两次 +2pp"难复现）

**029B (per-agent SOTA) 已经是 v2-only 的接近天花板**。036D 用 learned reward 又挤出 +1.4pp 到 0.860。再加 PBRS 想再挤 +1.5-2pp，要求 per-agent v2 architecture 的真实容量在 ≥ 0.88——这个**没有先验证据**。

**判据**：如果 4 条 041 lane 平均 ≤ +0.5pp 且方差小，写明 "per-agent + learned reward + PBRS 叠加 = saturation, lane 终结"。

### R3 — Inf 继承

036D 本身有 31.7% inf 率（[snapshot-036D §10.5](snapshot-036d-learned-reward-stability-fix.md)）。041 继承 learned reward 配置，会带着这个 inf 率。

**两个可能**：
- **乐观**：PBRS 加入让 advantage 信号更密集 → policy 更新更平滑 → inf 率反而下降
- **悲观**：PBRS 引入新的 reward outlier → inf 率上升到 50%+

**缓解**：训练完成后第一件事看 inf 率。如果 ≥ 50%，先回 [snapshot-036E](snapshot-036e-logit-clamp-fix.md) 那条线（如果有）做 logit clamp，再考虑 041 rerun。

### R4 — H2H vs 036D 的尴尬位置

如果 041B baseline ≥ 0.87 但 H2H vs 036D < 0.52（甚至 < 0.50），说明 PBRS 让 policy 学到了 baseline-specific exploits，**真实技能没提升**。这正是 [SNAPSHOT-029 §H1](snapshot-029-post-025b-sota-extension.md) 否决 029A 的判据。041B 必须满足 H2H 才算"真"成立。

## 7. 不做的事

- 不做 self-play / opponent pool 变动（保留 reward 端是唯一变量，与 040 对齐）
- 不调 LR / clip / sgd_iter（036D 同款 = 唯一公平 ablation）
- 不一次起 4 条 lane（先 041B verify，决定后续）
- 不撤掉 036D 的 learned reward（首轮要测 "double shaping" 假设）

## 8. 执行清单

1. **首轮先跑 041B**（PBRS handoff on 036D base）：
   - 写 batch `soccerstwos_h100_cpu32_mappo_041B_pbrs_on_036D150_512x512.batch`
   - 验证 `WARMSTART_CHECKPOINT` 加载 036D weights 正常 (smoke 1 iter 看 reward 合理)
   - 200 iter，CHECKPOINT_FREQ=10
2. 训练完成后用 [post-training pipeline](../architecture/engineering-standards.md#post-training-eval-pipeline-orchestrator)：
   ```bash
   bash scripts/eval/run_post_training_pipeline.sh \
     --run-dir ray_results/041B_mappo_pbrs_on_036D150_<ts> \
     --lane-type per_agent \
     --opponents "036D@150,029B@190,025b@080,031A@1040" \
     --jobid <SLURM_JOB>
   ```
3. 按 §5.1 / §5.2 判据决定是否启动 041A/C/D
4. failure capture v2 bucket 比对 036D / 029B
5. verdict 入库 §9 首轮结果

## 11. 首轮结果（2026-04-19，041B 完成）

### 11.1 训练 + warmstart

- Trial: `041B_mappo_pbrs_on_036D150_512x512_20260419_004530/MAPPOVsBaselineTrainer_Soccer_a4a2c_..._00-45-51`
- 200 iter / 8M steps 完成，20 ckpts (每 10 iter)
- warmstart 正常加载 036D@150 (`status: warmstart_applied`)
- merged_training_summary 关键指标：
  - `best_reward_mean: -0.3826 @ iter 43`
  - `final_reward_mean: -0.4119`
  - **reward 全程负值**（PBRS 加上去引入负 advantage 主导，但 win 率仍高 — reward 与 outcome 解耦）

数据完整性 caveat：
- internal eval 在 iter 启动时正好撞上 fieldnames bug（trainer 进程比 fieldnames edit 早启动），导致 50 ep 内评 CSV `dict contains fields not in fieldnames` failed retry
- user 已 manual backfill 完整 50ep 数据
- 训练 ckpts 完好，1000ep 后续测正常

### 11.2 1000ep 官方 baseline eval

按 `top 5% + ties + ±1` doctrine 选 ckpts → 3 ckpts (40, 50, 60)；fallback 规则（<5 ckpts → top 10%）→ 6 ckpts。

| iter | 50ep WR (backfilled) | **1000ep WR** | Δ (50→1000) |
|---:|---:|---:|---:|
| 10 | 0.84 | 0.829 | -0.011 |
| 20 | 0.90 | 0.835 | -0.065 |
| 30 | 0.90 | 0.823 | -0.077 |
| 40 | 0.82 | 0.822 | +0.002 |
| 50 | **0.94** | 0.844 | **-0.096** ← 最大缩水 |
| **60** | 0.88 | **0.852** ← peak | -0.028 |

**Peak 1000ep = 0.852 @ ckpt 60**

**Raw recap (for verification)**:

```
=== 041B_baseline1000.log ===
.../041B_.../checkpoint_000010/checkpoint-10 vs baseline: win_rate=0.829 (829W-171L-0T)
.../041B_.../checkpoint_000020/checkpoint-20 vs baseline: win_rate=0.835 (835W-165L-0T)
.../041B_.../checkpoint_000030/checkpoint-30 vs baseline: win_rate=0.823 (823W-177L-0T)
.../041B_.../checkpoint_000040/checkpoint-40 vs baseline: win_rate=0.822 (822W-178L-0T)
.../041B_.../checkpoint_000050/checkpoint-50 vs baseline: win_rate=0.844 (844W-156L-0T)
.../041B_.../checkpoint_000060/checkpoint-60 vs baseline: win_rate=0.852 (852W-148L-0T)
```

### 11.3 Verdict — 路径否决，**轻微退化**

按 §5 预注册：

| 判据 | 阈值 | 实测 | 结果 |
|---|---|---|:---:|
| §5.1 official 1000ep | ≥ 0.86 (不退化于 base) | peak 0.852, mean 0.834 | ❌ **退化** |
| §5.1 official 1000ep peak | ≥ 0.875 (+1.5pp) | 0.852 | ❌ |
| §5.2 1000ep peak | ≥ 0.88 (突破) | 0.852 | ❌ |

**041B peak 0.852 < 036D@150 base 0.860** —— 不仅没有 026B→029B 的 +2.6pp 增益，**反而轻微退化 -0.008pp**。所有 6 个 ckpts (peak 0.852, mean 0.834) 都低于 036D base。

H1 (PBRS handoff +1.5~+2.5pp) **明确否决**。041B 路径终结。

### 11.4 与 040 的对照（v2 ceiling 假设）

040 全 4 lane (031A v2 + extras) 在 [0.863, 0.865] 区间，几乎打平 031A base 0.860 (+0.003)。041B (036D + PBRS) **唯一退化**到 0.852 (-0.008)。

| Lane | base | 修改 | 1000ep peak | Δ vs base |
|---|---|---|---:|---:|
| 031A scratch | — | Siamese + v2 | 0.860 | (base) |
| 036D@150 | 029B@190 + v2 | + learned reward (λ=0.003 + warmup) | 0.860 | (base) |
| 040A | 031A@1040 + v2 | + depenalized v2 | 0.863 | +0.003 |
| 040B | 031A@1040 + v2 | + PBRS | 0.863 | +0.003 |
| 040C | 031A@1040 + v2 | + event | 0.865 | +0.005 |
| 040D | 031A@1040 + v2 | + entropy=0.01 | 0.863 | +0.003 |
| **041B** | **036D@150 + v2 + learned** | **+ PBRS** | **0.852** | **-0.008** ← 唯一退化 |

所有 lane 都在 [0.852, 0.865] 区间。**v2 ceiling 假设证据进一步加强**：跨架构、跨 reward 路径，所有 v2-derivative lane 都打不破 ~0.86。

041B 是唯一退化的，可能因为它叠加了**双 shaping**（learned reward + PBRS），reward shock 把 036D 已学到的优势推走。

### 11.5 不做 capture / H2H

- **跳过 capture**: 041B peak 0.852 < base 0.860，policy 比 036D 更弱不更强，failure 结构没新机制
- **跳过 H2H vs 036D / 029B**: 已知 baseline 轴退化，H2H 大概率也是平局或略输 base，无新信息

### 11.6 项目级影响 + 下一步

041B 路径**终结**——和 040 系列一起证明：**Stage 2 shaping handoff 在 v2-架构 high base 上无 ROI**。

下一步可能方向：
- **MaxEnt 路径** (新提案)：完全移除 v2，用 pure sparse + 高 entropy 或 learned reward only。**测试"v2 是 ceiling"假设**，可能突破 0.86
- **架构突破**: [042 cross-arch transfer](snapshot-042-cross-architecture-knowledge-transfer.md) (029B → Siamese)，绕开 v2 但不动 v2
- **Self-play diversity**: [043 frontier-only league](snapshot-043-frontier-selfplay-pool.md)，不动 reward 但加对手多样性
- 044 specialist 已 verdict (sparring 设计失败，044C 合并到 043)

### 11.7 数据 logs 路径

- run_dir: `ray_results/041B_mappo_pbrs_on_036D150_512x512_20260419_004530`
- 1000ep eval log: `docs/experiments/artifacts/official-evals/041B_baseline1000.log`
- per-ckpt logs: `docs/experiments/artifacts/official-evals/parallel-logs/041B_baseline1000/`
- merged training summary: `ray_results/041B_.../merged_training_summary.txt`

## 9. 相关

- [SNAPSHOT-036D](snapshot-036d-learned-reward-stability-fix.md) — base lane (036D 是 Stage 0 / 0)
- [SNAPSHOT-029](snapshot-029-post-025b-sota-extension.md) — 029B handoff 是直接对照路径
- [SNAPSHOT-026](snapshot-026-reward-liberation-ablation.md) — Stage 2 shaping 来源 (per-agent 镜像)
- [SNAPSHOT-040](snapshot-040-team-level-stage2-on-031A.md) — 直接对照 (team-level + Stage 2，全 saturation)
- [rank.md](rank.md) — 数值依据
