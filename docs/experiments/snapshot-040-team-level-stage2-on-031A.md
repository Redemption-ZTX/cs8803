# SNAPSHOT-040: Team-Level Stage 2 Handoff on 031A SOTA Base

- **日期**: 2026-04-18
- **负责人**:
- **状态**: `040A/B/C/D` batch 已就绪 / 待启动
- **依赖**: [SNAPSHOT-031 §11](snapshot-031-team-level-native-dual-encoder-attention.md#11-首轮结果031a2026-04-18) 的 `031A@1040` 已坐实为项目 SOTA（baseline 1000ep avg `0.860` + H2H 双胜 `029B@190` / `025b@080`）

## 0. Known gap 诊断（动机来源）

回顾 [SNAPSHOT-038 §0](snapshot-038-team-level-liberation-handoff-placeholder.md#0-known-gap-诊断)：per-agent SOTA `029B@190 (0.868)` 的形成路径是**两阶段**：

```
Stage 1: 017 BC@2100 (0.842) → 026B-warm PBRS (peak 0.864 / 主候选 @170 0.842)
Stage 2: 026B@170 → 029B v2-handoff (0.846 → 0.868, +2.6pp)
```

[SNAPSHOT-038 §10.11](snapshot-038-team-level-liberation-handoff-placeholder.md#1011-errata-2026-04-18-同-shaping-跨架构对比-038-team-level-vs-026-per-agent) 已记录：38 lane 在 `028A@1060 (0.783)` 这个**低底座**上跑 026 镜像 shaping 的 Stage 1，per-base 增量（+1.3 ~ +2.3pp）和 026 同字母 lane 的 +0.0~+2.2pp 相当——**team-level shaping 本身并不弱于 per-agent**，但绝对 WR 落后是因为底座差 5.9pp 起跑。

`031A@1040 = 0.860 (1000ep avg)` 第一次让 team-level 拿到了一个**和 per-agent 平台对齐的 base**（031A 1000ep avg 0.860 vs `017 BC@2100` 1000ep 0.804）。这是项目第一次具备做 **team-level Stage 2 handoff** 的硬件条件——之前所有 team-level base 都太低（028A 0.783, 030A 0.809, 032nextC 0.822），从这些点叠加 shaping 即使 +2pp 也只到 0.84，碰不到 029B 平台。

## 1. 核心假设

> **当前 031A@1040 的 0.860 来自架构改造（Siamese dual encoder）+ v2 baseline shaping，没有用任何 advanced shaping。把 026/038 体系内已知的 advanced shaping (PBRS / event / depenalized v2 / entropy-only) 作为 Stage 2 叠加在 031A 这个高 base 上，能否复现 per-agent `026 → 029` 的 +2.6pp 增益，把 1000ep 推到 0.88+？**

子假设：

- **H1（PBRS handoff）**: 031A + PBRS handoff（如 029B 那样从 v2 切到 v2+PBRS）→ +1.5~+2.5pp
- **H2（event handoff）**: 031A + event-based shaping → +0~+1.5pp（C-warm 在 per-agent 上是早期高点 + 较大波动，team-level 上可能复现）
- **H3（depenalized v2）**: 031A + depenalized v2 → +0~+1pp（A-warm 在 per-agent 上是失败 lane，预期低；作 sanity 对照）
- **H4（entropy-only）**: 031A + entropy=0.01 fine-tune → +0~+1.5pp（038D 在 028A 上 +2.3pp 是 stage 1 最强，但其 unclear_loss 16.3% 警告 turtle 风险）

如果 H1 出明显正号（+2pp 以上），031A + PBRS handoff 直接成为新的 SOTA 候选，1000ep 0.88 把项目推到距 9/10 阈值仅 -2pp。

## 2. 与 SNAPSHOT-038 的关系

| 维度 | SNAPSHOT-038 | **SNAPSHOT-040 (本)** |
|---|---|---|
| Warmstart base | 028A@1060 (0.783, flat MLP) | **031A@1040 (0.860, Siamese dual encoder)** |
| 架构 | flat MLP `[512,512]` | **Siamese encoder `[256,256]×2 share + merge [256,128]`** |
| 已知 base 1000ep WR | 0.783 | **0.860** (+7.7pp 起跑) |
| Stage 1 | 4 条 lane (A/B/C/D) 已完成，max +2.3pp | （不重做 Stage 1，直接做 Stage 2） |
| Stage 2 启动条件 | 未达预注册阈值，**未启动** | **已具备前置条件**（031A 对齐 per-agent 平台） |

snapshot-038 的失败教训：底座 0.783 离 029B 平台 0.846 有 6.3pp gap，shaping 单 stage 只 +2.3pp 不够补。snapshot-040 把底座换成 0.860，**反过来变成"shaping 能不能再多挤 +2pp"**——这个问题 026→029 路径上是 yes（+2.6pp），所以是合理预期。

## 3. 候选 lane 矩阵

四条 Stage 2 lane，各自镜像 026/038 的 A/B/C/D：

| lane | 起点 | shaping 设计 | shaping env vars | 镜像 |
|---|---|---|---|---|
| **040A** | 031A@1040 | depenalized v2 (去除 `time_penalty`/`opp_progress_penalty`/`deep_zone`，只保留 `ball_progress`+`possession`) | `SHAPING_TIME_PENALTY=0` `SHAPING_OPP_PROGRESS_PENALTY=0` `SHAPING_DEEP_ZONE_*=0` | 026A / 038A |
| **040B** | 031A@1040 | + goal-prox PBRS（在 v2 之上加 PBRS，如 029B handoff 设计）| `SHAPING_GOAL_PROXIMITY_SCALE=0.005` `SHAPING_GOAL_PROXIMITY_GAMMA=0.99` `SHAPING_GOAL_CENTER_X=15.0` `SHAPING_GOAL_CENTER_Y=0.0` | 026B / 029B / 038B |
| **040C** | 031A@1040 | + event-based (shot/tackle/clearance) | `SHAPING_EVENT_SHOT_REWARD=0.05` `SHAPING_EVENT_TACKLE_REWARD=0.03` `SHAPING_EVENT_CLEARANCE_REWARD=0.02` | 026C / 038C |
| **040D** | 031A@1040 | v2 + entropy=0.01 fine-tune | `SHAPING_*` 不变, `ENTROPY_COEFF=0.01` | 026D / 038D |

四条 lane 都使用 **完全相同的 SiameseTeamModel 架构**（`TEAM_SIAMESE_ENCODER=1`），区别只在 reward 端，符合 ablation cleanness。

## 4. 训练超参

```bash
WARMSTART_CHECKPOINT=ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/.../checkpoint_001040/checkpoint-1040
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
LR=1e-4
CLIP_PARAM=0.15
NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000
SGD_MINIBATCH_SIZE=2048
MAX_ITERATIONS=200             # Stage 2 通常 200 iter 够
EVAL_INTERVAL=10
EVAL_EPISODES=50
CHECKPOINT_FREQ=10
```

**关键**：用 `WARMSTART_CHECKPOINT` 路径（不是 `RESTORE_CHECKPOINT`），让 trainer 重新初始化 optimizer state，只继承 policy weights——这是 029B handoff 同款做法。

每条 lane budget: `~3-4h`（200 iter × ~1min/iter on H100）。

## 5. 预注册判据

### 5.1 主判据（成立门槛）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| official 1000ep | ≥ 0.86 | 至少不退化于 031A@1040 base |
| **official 1000ep peak** | **≥ 0.875** | 显式给出 +1.5pp 增益（031A SE ≈ 0.011，意味着 +1.5pp 接近 1.4σ 边缘） |
| H2H vs 031A@1040 | ≥ 0.52 | 直接证明 Stage 2 改进有真实优势，不是 baseline-only artifact |

### 5.2 突破判据（强成立）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| official 1000ep peak | ≥ 0.88 | **+2.0pp 突破**，对应 029B 路径上 026B@170 → 029B@190 那一跳 |
| H2H vs 029B@190 | ≥ 0.56 | 比 031A 单线的 0.552 还要再加一点 |
| failure capture: 任一 v2 bucket 比 031A 改善 ≥ 5pp | — | 机制层面证据 |

### 5.3 失败判据

| 条件 | 解读 |
|---|---|
| 1000ep < 0.85 | Stage 2 shaping 把 031A 已学到的 policy 推坏；reward shock 主导 |
| 1000ep ≥ 0.86 但 H2H vs 031A < 0.50 | shaping 只在 baseline 上学了等价策略，无真实技能提升 |
| `unclear_loss` ≥ 18% 或 L mean ≥ 50 步 | turtle pattern（038D 警告） |

### 5.4 Lane 优先级

按预期 ROI：

1. **040B (PBRS handoff)** — 最高优先级，直接对应 per-agent 唯一突破路径
2. **040D (entropy)** — 次高，038D 在低底座上 +2.3pp 是当时 Stage 1 最强；但要监控 turtle
3. **040C (event)** — 中等，per-agent 上是 noisy 高点
4. **040A (depenalized v2)** — 最低，per-agent 上 lane 是失败案例 (026A)，但作 sanity 对照保留

**保守执行口径仍以 040B 为首优先**；但四条 `040A/B/C/D` 现在都已具备 matched runnable batch，如果当前阶段更重视并行推进，也可以四条一起启动。

## 6. 风险

### R1 — Reward shock（031A 已学到 v2-only 的 policy，加 PBRS/event 时 reward 分布突变）

**缓解**：
- LR 不变（1e-4），让网络慢慢吸收新 reward
- 监控前 20 iter 的 `policy_loss` 和 `value_loss`：如果 value_loss 飙升 >5×，证明 shock 太大，需要加 reward warmup（类似 036D 的 `warmup_steps`）

### R2 — Saturation（031A 已经 0.860，离 baseline 上限 ~0.90 只 4pp 空间，shaping 难再涨）

**这是 026 vs 038 对比里 user 提示的 base 饱和度问题在 040 上的体现**。如果 040 全部 lane 都只 +0.5pp 以内，证据指向 "shaping 在 high base 上边际效用收敛"，而不是 shaping 本身失效。

**判据**：如果 4 条 lane 平均 ≤ +0.5pp 且方差小，写明"高 base saturation 假设成立，031 shaping handoff 路径终结"。

### R3 — Architecture × shaping interaction（Siamese encoder 可能对 shaping 信号的响应方式和 flat MLP 不同）

**机制担心**：dual encoder 的 feat0/feat1 是同一个函数处理两个 obs slice，shaping 信号通过 reward 反向传播会同时调整两个 branch。这可能让 shaping 学得更"对称"或"更慢"——未知。

**缓解**：监控 encoder cos_sim 在 Stage 2 训练中是否漂移（031A 训练时维持在 [0.3, 0.9]）。如果 cos_sim 飙到 >0.95，证明 siamese collapse，需要回到 flat MLP 上做 Stage 2（绕过 031 架构的限制）。

## 7. 不做的事

- 不在本 snapshot 内做 self-play / opponent pool 变动（保留 reward 端是唯一变量）
- 不调 LR / clip / sgd_iter（031A 同款 = 唯一公平 ablation）
- 不做 dual handoff (Stage 1 + Stage 2 一起换)——分阶段才能定位增益来源

## 8. 执行清单

1. **首轮 lane 已全部 runnable**：
   - [040A depenalized-v2](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_040A_depenalized_v2_on_031A1040_512x512.batch)
   - [040B PBRS handoff](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_040B_pbrs_on_031A1040_512x512.batch)
   - [040C event lane](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_040C_event_lane_on_031A1040_512x512.batch)
   - [040D v2 + entropy](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_040D_v2_entropy_on_031A1040_512x512.batch)
   - 四条都使用 `WARMSTART_CHECKPOINT =` [031A@1040](../../ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/checkpoint_001040/checkpoint-1040)，保持 SiameseTeamModel 架构固定
   - 四条都使用 matched `200 iter / checkpoint_freq=10 / lr=1e-4 / clip=0.15 / num_sgd_iter=4`
2. 训练完成后，对 top 5 ckpt 跑 `official baseline 1000` (parallel=5)
3. 按 §5.1 / §5.2 判据决定是否启动 040A/C/D
4. 040B 主候选 ckpt 做 H2H vs 031A@1040 + 029B@190 (各 n=1000)
5. failure capture v2 bucket
6. verdict 入库 `§9 首轮结果`

## 11. 首轮结果（2026-04-19，4 lane 全部完成）

### 11.1 训练 reward 异常停滞

**4 条 lane 训练 200 iter，reward 几乎不变**：

| Lane | iter1 | iter50 | iter100 | iter200 | Δ200iter |
|---|---:|---:|---:|---:|---:|
| 040A depenalized v2 | +2.23 | +2.23 | +2.24 | +2.24 | **+0.01** |
| 040B PBRS handoff | +2.37 | +2.37 | +2.35 | +2.35 | **-0.02** |
| 040C event lane | +2.06 | +2.07 | +2.07 | +2.07 | **+0.01** |
| 040D v2 entropy | +2.21 | +2.22 | +2.21 | +2.20 | **-0.01** |

iter1 reward 已经在 +2.0~+2.4（warmstart 加载 031A@1040 后即时 episode reward 水平），整 200 iter 都在 ±0.03 噪声内。policy 实质上**没产生方向性更新**——跟 [§6.2 R2 saturation 风险](#r2--saturation031a-已经-0860离-baseline-上限-090-只-4pp-空间shaping-难再涨)预言一致。

数据完整性确认：
- 4 lane 全部完成 200 iter（040A `formal_rerun2/740bc`, 040B `formal/a25f1`, 040C `formal_rerun/ed5e2`, 040D `formal/18a36`），各 20 ckpts
- 040B/D 各有第二个 trial (`ad`, `ee`)——前者 port collision 失败，后者污染于 fieldnames bug，**全部忽略**
- inf_kl 全部 0%（team-level shaping trainer 没有 learned-reward 的数值不稳问题）
- warmstart 全部正确加载 `031A@1040` weights

### 11.2 1000ep 官方 baseline eval（saturation 实证）

按 top 5%+ties+±1 doctrine 选 ckpt，4 lane 各 3-6 个 ckpt：

| Lane | ckpts | 1000ep range | peak | mean | Δ vs 031A@1040 (0.860) |
|---|---|---|---:|---:|---:|
| **040A** depenalized v2 | 40, 50, 60 | 0.852-0.863 | **0.863** | 0.856 | **+0.003** |
| **040B** PBRS handoff | 130, 140, 150, 170, 180, 190 | 0.837-0.863 | **0.863** | 0.853 | **+0.003** |
| **040C** event lane | 40, 50, 60 | 0.836-0.865 | **0.865** | 0.851 | **+0.005** |
| **040D** v2 entropy | 130, 140, 150 | 0.841-0.863 | **0.863** | 0.852 | **+0.003** |

**4 条 lane peak 全部在 [0.863, 0.865] 区间，与 031A@1040 base (2000-game avg 0.860) 在 1000ep SE ±0.016 内**。

**Raw recap (for verification)**:

```
=== 040A_baseline1000.log ===
.../040A_.../checkpoint_000040/checkpoint-40 vs baseline: win_rate=0.852 (852W-148L-0T)
.../040A_.../checkpoint_000050/checkpoint-50 vs baseline: win_rate=0.853 (853W-147L-0T)
.../040A_.../checkpoint_000060/checkpoint-60 vs baseline: win_rate=0.863 (863W-137L-0T)

=== 040B_baseline1000.log ===
.../040B_.../checkpoint_000130/checkpoint-130 vs baseline: win_rate=0.855 (855W-145L-0T)
.../040B_.../checkpoint_000140/checkpoint-140 vs baseline: win_rate=0.855 (855W-145L-0T)
.../040B_.../checkpoint_000150/checkpoint-150 vs baseline: win_rate=0.837 (837W-163L-0T)
.../040B_.../checkpoint_000170/checkpoint-170 vs baseline: win_rate=0.862 (862W-138L-0T)
.../040B_.../checkpoint_000180/checkpoint-180 vs baseline: win_rate=0.845 (845W-155L-0T)
.../040B_.../checkpoint_000190/checkpoint-190 vs baseline: win_rate=0.863 (863W-137L-0T)

=== 040C_baseline1000.log ===
.../040C_.../checkpoint_000040/checkpoint-40 vs baseline: win_rate=0.852 (852W-148L-0T)
.../040C_.../checkpoint_000050/checkpoint-50 vs baseline: win_rate=0.865 (865W-135L-0T)
.../040C_.../checkpoint_000060/checkpoint-60 vs baseline: win_rate=0.836 (836W-164L-0T)

=== 040D_baseline1000.log ===
.../040D_.../checkpoint_000130/checkpoint-130 vs baseline: win_rate=0.841 (841W-159L-0T)
.../040D_.../checkpoint_000140/checkpoint-140 vs baseline: win_rate=0.863 (863W-137L-0T)
.../040D_.../checkpoint_000150/checkpoint-150 vs baseline: win_rate=0.852 (852W-148L-0T)
```

### 11.3 Verdict — Saturation 假设确认

按 §5 预注册：

| 判据 | 阈值 | 实测 | 结果 |
|---|---|---|:---:|
| §5.1 official 1000ep peak | ≥ 0.875 (+1.5pp) | 4 lane peak 0.863-0.865 (+0.3~+0.5pp) | ❌ **均未达成** |
| §5.2 1000ep peak | ≥ 0.88 (突破) | 同上 | ❌ |
| §5.3 失败判据 (1000ep < 0.85) | — | 部分 ckpt < 0.85 但 peak 都 ≥ 0.85 | 边界（不算"推坏"，但也无增益） |

**§6.2 R2 saturation 风险预言成真**: 031A@1040 的 0.860 是 v2-architecture + Siamese 在 baseline 轴的 ceiling，shaping 改动（PBRS / event / depenalized / entropy）**边际效用收敛到 0**。4 条 lane 平均 Δ +0.003pp，标准差 0.001pp——**没有任何 shaping 在 baseline 轴上有实质增益**。

### 11.4 不做 Stage 2 / Stage 3

按 §8 执行清单原计划要做 failure capture + H2H，但鉴于 §11.3 的 saturation 结果：
- **跳过 failure capture**: 4 lane policy 跟 031A@1040 base 几乎相同，failure 结构也会几乎一样，无新机制信息
- **跳过 H2H**: 与 base / frontier 都会接近平局（差 < SE），无新排名信息

GPU 节省给后续真正有差异的 lane（041 / 044C / 042）。

### 11.5 机制解读（为什么 saturation）

reward 几乎不变 + warmstart 工作 = **policy 在初始 reward landscape 局部最优附近震荡**。三个可能解释：

1. **031A 已经在 v2-架构 ceiling**: 0.860 是这个 architecture × reward 组合能学到的最强；新 shaping 把 advantage signal 略微改变方向，但 policy 在这个 reward landscape 上没有更高 reward 的可达点
2. **Shaping 信号相对 dense reward 太弱**: PBRS scale=0.005、event scale=0.05 等，加到原 +2.3 dense reward 上是 2-5% 量级，被噪声淹没
3. **PPO LR=1e-4 + clip=0.15 对 fine-tune 已饱和 policy 太保守**: scratch 友好配置反而冻结了 fine-tune

三个不互斥；但本 snapshot 不深究，因为 verdict 是"路径终结"。

### 11.6 项目级影响 + 下一步

**040 路径暂时终结**——shaping handoff 在 031A 这个 high base 上不能再挤增益。snapshot-040 §1 核心假设 H1/H2/H3/H4 全部否决。

下一步建议（项目主线，已有 snapshot 跟进）:
- **[041](snapshot-041-per-agent-stage2-pbrs-on-036D.md)**: per-agent 镜像。**036D@150** 是 0.860 但来源不同（learned reward），可能没饱和。轮询 1000ep 出来再判
- **[042](snapshot-042-cross-architecture-knowledge-transfer.md)**: 029B encoder transfer 到 Siamese 架构——绕开 v2-shaping ceiling
- **[043](snapshot-043-frontier-selfplay-pool.md)** / **[044](snapshot-044-adversarial-specialists-league.md)**: self-play 路径，绕开 baseline-axis saturation 用 peer diversity 推 main agent

### 11.7 数据 logs 路径

| Lane | 1000ep eval log |
|---|---|
| 040A | `docs/experiments/artifacts/official-evals/040A_baseline1000.log` + `parallel-logs/040A_baseline1000/` |
| 040B | `docs/experiments/artifacts/official-evals/040B_baseline1000.log` + `parallel-logs/040B_baseline1000/` |
| 040C | `docs/experiments/artifacts/official-evals/040C_baseline1000.log` + `parallel-logs/040C_baseline1000/` |
| 040D | `docs/experiments/artifacts/official-evals/040D_baseline1000.log` + `parallel-logs/040D_baseline1000/` |

## 9. 相关

- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — base lane (031A 是 Stage 0)
- [SNAPSHOT-026](snapshot-026-reward-liberation-ablation.md) — Stage 2 shaping 来源 (per-agent 镜像)
- [SNAPSHOT-029](snapshot-029-post-025b-sota-extension.md) — 029B handoff 是直接对照路径
- [SNAPSHOT-038](snapshot-038-team-level-liberation-handoff-placeholder.md) — 同 shaping 在低底座上的失败对比
- [SNAPSHOT-041](snapshot-041-per-agent-stage2-pbrs-on-036D.md) — per-agent 镜像，可能没饱和
- [rank.md](rank.md) — 数值依据
