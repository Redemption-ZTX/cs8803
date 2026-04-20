# SNAPSHOT-038: Team-Level Liberation + Handoff — Known Gap Activation

- **日期**: 2026-04-18
- **负责人**:
- **状态**: **active** — 首轮执行完整 `Path 2`（4-way Stage 1 liberation ablation）
- **用途**: 将 team-level 的“Stage 1 liberation + Stage 2 handoff”已知程序缺口，从占位议题升级为主动主线

## 0. Known gap 诊断

per-agent SOTA 029B@190 (0.868) 的形成路径**不是一次性** 的单一 shaping 实验，而是两个阶段：

| 阶段 | 作用 | 涉及 snapshot |
|---|---|---|
| **Stage 1 — Liberation ablation** (4 变体同时测) | 在 017+v2 底座上探索不同 reward/event shaping | [SNAPSHOT-026 A/B/C/D](snapshot-026-reward-liberation-ablation.md) |
| **Stage 2 — B-warm handoff** (选中期 checkpoint → 切回 clean shaping) | 保留探索期的有益行为，用 clean shaping 抛掉 shaping-specific exploit | [SNAPSHOT-029 §B](snapshot-029-post-025b-sota-extension.md) |

**team-level 侧目前只走了 Stage 1 的一部分**，且是**串行、非同步**测的，**完全没走 Stage 2 (handoff)**：

| team-level lane | Stage 1 类比 | Stage 2 类比 |
|---|---|---|
| [030A](snapshot-030-team-level-advanced-shaping-chain.md) (field-role) | 类 025b 单步增量 | — |
| [030D](snapshot-030-team-level-advanced-shaping-chain.md) (goal-prox PBRS) | 类 026B 一种变体 | **缺（预注册的 030-C 被搁置）** |
| [033A](snapshot-033-team-level-native-coordination-reward.md) (spacing+coverage PBRS) | 另一种变体 | — |
| [032A](snapshot-032-team-level-native-coordination-aux-loss.md) (aux loss) | 侧线，不属 shaping 轴 | — |

**差距**：team-level 用"一次性 shaping"尝试取代了 per-agent 走的"探索 + 切换"两阶段过程。如果 per-agent 的 +2.6pp 增益（0.842 → 0.868）**至少部分来自 handoff 本身**而非最终 shaping 的威力，team-level 复现不了这个增益的原因就是**程序缺口**，不是架构问题。

## 1. 核心假设

`H_038`：**team-level 架构也能从"Stage 1 liberation 探索 → Stage 2 handoff 到 clean shaping"两阶段过程中拿到 +2-3pp 的 baseline WR 提升**，就像 per-agent 029B 从 026B@170 handoff 到 v2 拿到了 +2.6pp。

证伪条件:
- 任何形式的 team-level handoff 都拿不到超过 "原 Stage 1 自身最优 checkpoint +0.01" 的提升
- 这说明 team-level 的 reward landscape 不受益于 handoff，per-agent 的 +2.6pp 里 handoff 成分小于我们以为

## 2. Per-agent 参照链详细

按实测数：

```
017@2100  (0.842 official 500)
  │ + field-role binding
  └─ 025b@080  (0.842 official, +0.00)   ← 单步增量几乎无增益
  │
  │ + liberation ablation (026 A/B/C/D)
  ├─ 026A-warm                           ← 一种 liberation 变体
  ├─ 026B-warm (PBRS, 峰值 @250 = 0.864)   ← 最强 Stage 1 产物
  ├─ 026C-warm (event-based)
  └─ 026D-warm
        │
        │ Stage 2: 取 026B 中期 @170 (不是 @250)，切回 v2 shaping
        └─ 029B@190  (0.868 official 500 / 0.846 1000ep)
                                           ← **+2.6pp 来自 handoff 过程**
```

关键 detail — handoff 用的是 **026B@170**，**不是** 026B 自己的 peak @250。因为:
- @250 已经"锁进" PBRS exploit
- @170 保留了 PBRS 的 exploration 益处但未 overfit 到 PBRS reward
- 切到 v2 后 @170 版本继续进步，@250 版本停滞

这是 handoff 的**机制假设**：中期有益探索 + 后期 clean polish。

## 3. Team-level 现状盘点

team-level 目前最好的 Stage 1 产物（均 028A@1060 warmstart）:

| Model | Stage 1 shaping | 1000ep baseline WR | 备注 |
|---|---|---:|---|
| 030D@320 | goal-prox PBRS | 0.816 | 最接近 026B 的直接类比 |
| 032nextC@130 | v2 only, symm-aux head off | 0.822 | 意外的高点；严格讲不是 shaping lane |
| 030A@360 | field-role | 0.809 | — |
| 030A@300 | field-role | — | 可能更早的 mid-training 点，未测 1000ep |

**对应 Stage 2 handoff 的缺失实验**：从上述某一点（尤其 030D 中期或 032nextC 中期）→ **warmstart with v2 only, continue 200 iter** 看是否能从 0.82 → 0.84+。

## 4. 路径 1 — 狭义 handoff（030-C 激活）

**目标**：最小 incremental 实验，直接复刻 per-agent 029B 的 handoff 动作。

| 配置 | 值 |
|---|---|
| warmstart checkpoint | 030D@150-200 之间某点（需先做 mid 1000ep 锁定"中期强但未锁死"窗口）|
| shaping | v2 only (关掉 goal-prox PBRS) |
| 训练预算 | 200 iter, ~6h GPU |
| 对照 | `030-C-control` = 同 warmstart + 保留 PBRS（= 030D 继续训）|

**预注册判据**:
- `030-C vs 028A@1060` H2H (500ep) ≥ 0.55
- `030-C` official 500 ≥ 0.85（明显超出 030D 峰值 0.862 不强求，但 ≥ 025b/030A 0.84 线必要）
- `030-C` 1000ep ≥ 0.83

失败判据（任一触发则此路径不再沿伸）:
- official 500 < 0.82
- 1000ep 比 030D 起点更低

**预计信息密度**: 中-高。回答"team-level 是否受益于 handoff 本身"。

## 5. 路径 2 — 完整 4-way liberation + handoff（本轮主计划）

**目标**：真正复刻 per-agent 026+029 全路径，做 team-level 原生的 liberation ablation。

4 条同 base、同 budget 的 Stage 1 lane，每条不同 shaping，**并按 `026 A/B/C/D` 做干净镜像**：

| Lane | shaping | 对标 per-agent |
|---|---|---|
| `038-A` | 去惩罚 v2：保留 `ball_progress + possession`，移除 penalty-style shaping | `026A` |
| `038-B` | goal-proximity PBRS + possession | `026B` |
| `038-C` | event-based：shot / tackle / clearance | `026C` |
| `038-D` | v2 shaping 不变 + `entropy_coeff=0.01` | `026D` |

Stage 1 budget: 4 × 200 iter ≈ 24h GPU
Stage 2 (handoff): 选 Stage 1 里 mid-training 最强的 1-2 个 checkpoint → v2 handoff, 200 iter ≈ 6-12h GPU

实现口径说明：
- 这里不再使用旧占位版里的 `pass-event` / `spacing+coverage` / `opp-progress amplified` 混合定义。
- 本 snapshot 的价值就是和 [SNAPSHOT-026](snapshot-026-reward-liberation-ablation.md) 做**干净对照**，所以首轮 Stage 1 必须按 `026 A/B/C/D` 的信号设计逐条镜像。
- 这也意味着：原本在 team-level 已经单独测过的 [030D](snapshot-030-team-level-advanced-shaping-chain.md) 与 [033A](snapshot-033-team-level-native-coordination-reward.md)，在 `038` 里不再直接复用为 lane 定义；它们现在更多是先验参考，而不是 `038` 的实验定义。

**预计信息密度**: 很高，但成本高。只在路径 1 也无法推进时才值得启动。

## 6. 激活更新

`038` 原始预注册把 `036C` 当作 gating 条件；但到 2026-04-18 为止，项目状态已经发生了两个变化：

1. `036C` 首轮 verdict 已经给出：它在 baseline 轴上是正信号，但还不足以封闭 team-level 的已知程序缺口。
2. `036` 的数值稳定性修复已由其他人接手，因此 `038` 不再需要继续等待 `036` 才能推进。

因此，当前口径更新为：

- **`038` 立即激活**
- **首轮直接走完整 `Path 2`**
- `Path 1` 不再作为主计划，只保留为后续 Stage 2 handoff 的缩小版 fallback

## 7. 不做的事

- **不做**继续等待 `036C` 完整收尾再行动；`036` 的剩余工作已与 `038` 解耦
- **不做**混用 team-only lane 定义来“近似 liberation”；首轮必须保持 `026 A/B/C/D` 对照干净
- **不做**在未先锁定 Stage 1 mid-training 强点时就盲选 handoff 起点。handoff 起点质量决定 Stage 2 成败
- **不做**改变 warmstart 至 `029B@190` 或更强 per-agent 点。本 snapshot 专测 "team-level 架构能否 handoff"，用 per-agent warmstart 会污染变量

## 8. 首轮执行口径（已激活）

### 8.1 Stage 1 统一 skeleton

四条 lane 统一使用：

- warm-start: `028A@1060`
- trainer: team-level PPO shaping trainer
- network: `512x512`
- PPO params: `lr=1e-4`, `clip=0.15`, `num_sgd_iter=4`, `sgd_minibatch=2048`
- budget: `200 iter`
- 评测: `baseline/random @ 50ep` 内评；frontier checkpoint 再做 `official 500/1000`

### 8.2 本轮四条 lane 的精确定义

| Lane | 运行名模式 | shaping 定义 |
|---|---|---|
| `038-A` | `038A_team_depenalized_v2_stage1_on_028A1060_*` | `026A` 镜像：`ball_progress + possession`，无 penalty-style shaping |
| `038-B` | `038B_team_goal_prox_stage1_on_028A1060_*` | `026B` 镜像：goal-proximity PBRS + possession |
| `038-C` | `038C_team_event_lane_stage1_on_028A1060_*` | `026C` 镜像：shot / tackle / clearance sparse events |
| `038-D` | `038D_team_v2_entropy_stage1_on_028A1060_*` | `026D` 镜像：v2 shaping 不变 + `entropy_coeff=0.01` |

### 8.3 Stage 2 条件

首轮不预先假定谁进入 handoff。Stage 2 只在以下条件满足时激活：

- 至少一条 Stage 1 lane 在 matched budget 下明显高于 `028A@1060`
- 并且该 lane 存在“中期强点”而不是单一 late spike
- 再从该 lane 的中期 checkpoint 切回 clean v2，执行 handoff continuation

## 10. [2026-04-18 ~15:50 EDT] Stage 1 四条 lane 首轮结果

4 条 lane 均训到 MAX_ITERATIONS=200 正常完成（`Done training`），无 inf 或 crash。每条选 top 5 checkpoint 做 `official 1000ep` (parallel=5)。

### 10.1 `official baseline 1000` 结果

| Lane | shaping 设计 | 1000ep range | 1000ep max (ckpt) | Δ vs `028A@1060` (0.783) |
|---|---|---|---:|---:|
| `038A` | 去惩罚 v2（ball_progress + possession） | 0.783–0.796 | **0.796 @ ckpt 160** | +1.3 pp |
| `038B` | goal-prox PBRS + possession | 0.783–0.797 | **0.797 @ ckpt 110** | +1.4 pp |
| `038C` | event-based (shot/tackle/clearance) | 0.765–0.800 | **0.800 @ ckpt 50** | +1.7 pp |
| `038D` | v2 + `entropy_coeff=0.01` | 0.793–**0.806** | **0.806 @ ckpt 40 ≈ ckpt 60** | **+2.3 pp** |

### 10.2 50ep → 1000ep 回归

每 lane 50ep max ~0.88–0.92，到 1000ep 下降 **≈ 8–11 pp**（和 [030/032/033/036C 的系统性 pattern 一致](snapshot-030-team-level-advanced-shaping-chain.md)）。50ep peak 在 team-level lane 上仍然不是可靠指标。

### 10.3 读法（克制）

- 4 条 lane 的 1000ep max 彼此差距 **≤ 1.0 pp**，都落在 028A warmstart 的 ±2.4 pp 的 95% CI（SE 0.012）之内
- 没有任何一条单独"明显高于 028A@1060"——§8.3 `Stage 2 激活条件`的**第一条**处在 **边界上**（+2.3pp 接近 CI 上界但未清晰超出）
- `038D` 相对其他 3 条有 +0.6 到 +1.0 pp 方向性优势，但这个差异**也**在单次 1000ep SE 之内
- 相对已有 team-level 数据点（`030A` 0.809 / `030D` 0.816 / `032nextC` 0.822），4 条 038 lane **均未超过**
- 相对 per-agent baseline-axis 头名 `029B@190` 0.846，4 条 038 lane **均明显低** 4–6 pp

### 10.4 Stage 2 handoff 决策（保留预注册，非武断）

按 §8.3 预注册判据:
- 条件 1（matched budget 下明显高于 028A@1060）: **边界 / 部分满足**（038D +2.3pp 方向性正，但未过 CI 上界）
- 条件 2（中期强点而不是单一 late spike）: 从 50ep 曲线看 **038D ckpt 40–100 区间都有 0.82+ 局**，不是单 late spike

所以**手动判定**：
- **不**对 038A/B/C 启动 handoff（增益更小，Stage 2 信息密度会更低）
- **暂缓**对 038D 启动 handoff，**先做 failure capture** 看 ckpt 40 的失败结构是否已接近 029B 水平。如果 `wasted_possession` 和 `low_possession` 显著低于 028A 结构，再决定启动 Stage 2

### 10.5 Failure capture (in progress)

正在对 [`038D ckpt 40`](../../ray_results/038D_team_v2_entropy_stage1_on_028A1060_512x512_20260418_120734/TeamVsBaselineShapingPPOTrainer_Soccer_c2fab_00000_0_2026-04-18_12-07-56/checkpoint_000040/checkpoint-40) 做 500ep failure capture（[rank.md §0.2](rank.md#02-正确读-h2h-的办法) 规范）。结果出后会追加到 §10.6。

### 10.6 v2 bucket 对照（038D@40 已完成）

| Bucket | 038D@40 (L=98) | 030D@320 ref (L=90) | 030A@360 ref (L=79) |
|---|---:|---:|---:|
| defensive_pin | 38.8% | 46.7% | 43.0% |
| territorial_dominance | 39.8% | 47.8% | 45.6% |
| wasted_possession | **42.9%** | 37.8% | 36.7% |
| possession_stolen | 28.6% | 36.7% | 38.0% |
| progress_deficit | 19.4% | 27.8% | 21.5% |
| **unclear_loss** | **16.3%** | 11.1% | 12.7% |

L episode steps:

| | mean | median | max |
|---|---:|---:|---:|
| 038D@40 | **43.2** | 30 | **256** |
| 030D@320 ref | 33.6 | 23 | 164 |
| 030A@360 ref | 37.0 | 29 | 133 |

### 10.7 038D 行为读法

- `unclear_loss` 16.3% **高于** 030D/030A 参考（11.1% / 12.7%），接近 [036C@270 reward gaming 信号](snapshot-036-learned-reward-shaping-from-demonstrations.md) (16.9%)
- L mean steps 43.2，**最高** L = 256，turtle 特征明显
- `defensive_pin / territorial_dominance` 反而**更低** —— policy 不在自己半场被压，更多在中场打转
- `wasted_possession` 42.9% **更高** —— 能控球但不进球，典型转化失败
- `possession_stolen` 28.6% 最低 —— 球权确实更稳

合起来：**038D entropy=0.01 让 policy 探索了更多"中场拖时间但不进球"的状态**。这不是 reward model gaming（038D 没用 learned reward），而是 entropy 增加自然带来的 exploration → "拉锯战" 失败模式。

### 10.8 verdict (Stage 1 收口)

按预注册 §8.3 Stage 2 激活条件:

| 条件 | 038D 状态 |
|---|---|
| 1. matched budget 下明显高于 028A@1060 | **未明显高**（+2.3pp on max in 1000ep SE 之内） |
| 2. 中期强点而不是单一 late spike | **满足**（@40, @60, @20 在 50ep / 1000ep 都是高分） |
| 3. failure capture 显示结构改善 | **未改善**（unclear_loss/wasted_poss 上升） |

按预注册：**Stage 2 handoff 不启动**。038 Stage 1 的 4 条 lane 都未给出"明显高于 warmstart 的 Stage 1 mid-training point"。

不武断说"team-level liberation 整线无效"——只能说"现有 4 个具体 shaping (depenalized v2 / goal-prox PBRS / event lane / entropy=0.01) 在 028A@1060 底座上没复现 per-agent 026→029 那样的 +2.6pp 跨阶段增益"。

### 10.9 4 条 lane 的原始 Training Summary（一手存档）

按 user request 把每条 lane 的 print summary 直接归档:

**038A (depenalized v2)**:
```
Training Summary
  stop_reason: Trial status: TERMINATED
  run_dir:     ray_results/038A_team_depenalized_v2_stage1_on_028A1060_512x512_20260418_121346
  best_reward_mean: +2.2373 @ iteration 87
  best_checkpoint:  .../checkpoint_000120/checkpoint-120
  final_checkpoint: .../checkpoint_000200/checkpoint-200
  best_eval_checkpoint: .../checkpoint_000180/checkpoint-180
  best_eval_baseline:  0.900 (45W-5L-0T) @ iteration 180
  best_eval_random:    1.000 (50W-0L-0T)
Done training
```

**038B (goal-prox PBRS)**:
```
Training Summary
  stop_reason: Trial status: TERMINATED
  run_dir:     ray_results/038B_team_goal_prox_stage1_on_028A1060_512x512_20260418_120728
  best_reward_mean: +2.2395 @ iteration 169
  best_checkpoint:  .../checkpoint_000200/checkpoint-200
  final_checkpoint: .../checkpoint_000200/checkpoint-200
  best_eval_checkpoint: .../checkpoint_000090/checkpoint-90
  best_eval_baseline:  0.880 (44W-6L-0T) @ iteration 90
  best_eval_random:    0.980 (49W-1L-0T)
Done training
```

**038C (event lane)**:
```
Training Summary
  stop_reason: Trial status: TERMINATED
  run_dir:     ray_results/038C_team_event_lane_stage1_on_028A1060_512x512_20260418_120730
  best_reward_mean: +2.0798 @ iteration 171
  best_checkpoint:  .../checkpoint_000190/checkpoint-190
  final_checkpoint: .../checkpoint_000200/checkpoint-200
  best_eval_checkpoint: .../checkpoint_000080/checkpoint-80
  best_eval_baseline:  0.900 (45W-5L-0T) @ iteration 80
  best_eval_random:    1.000 (50W-0L-0T)
Done training
```

**038D (v2 + entropy=0.01)**:
```
Training Summary
  stop_reason: Trial status: TERMINATED
  run_dir:     ray_results/038D_team_v2_entropy_stage1_on_028A1060_512x512_20260418_120734
  best_reward_mean: +2.2009 @ iteration 117
  best_checkpoint:  .../checkpoint_000200/checkpoint-200
  final_checkpoint: .../checkpoint_000200/checkpoint-200
  best_eval_checkpoint: .../checkpoint_000100/checkpoint-100
  best_eval_baseline:  0.920 (46W-4L-0T) @ iteration 100
  best_eval_random:    0.980 (49W-1L-0T)
Done training
```

四条 lane 的 `best_eval_baseline` 是 **50ep training-internal**，有 8-12pp 抽样上限：实际 **1000ep** 见 §10.1 表 + [rank.md §3.3](rank.md#33-official-baseline-1000frontier--active-points-only)。

### 10.10 同期对比 (036D learned reward)

[snapshot-036D §10.3](snapshot-036d-learned-reward-stability-fix.md#103-official-1000ep关键结果) 同期出 1000ep max **0.860 (ckpt 150)**，**首次** 1000ep max 超 029B@190 warmstart (0.846)。

这暗示：**team-level shaping 路径目前没产生新的突破，而 per-agent learned reward + 稳定化 fix 才是当前最有希望的方向**。snapshot-038 的下一步可考虑：
- 暂缓 Stage 2 handoff
- 等 036D@150 的 H2H + capture 确认是否真超 029B
- 如确认，team-level liberation 优先级降低；项目主线收敛回 per-agent learned-reward 路径

### 10.11 [errata 2026-04-18] 同 shaping 跨架构对比 (038 team-level vs 026 per-agent)

**之前 §10.1 / §10.4 把 `038A/B/C/D` 都和 `028A@1060` warmstart 比较 ——这是同 base 的 "matched-budget continuation" 比较，但**不是**同 shaping 的"跨架构"比较。** user 提示：038 的设计本来就是 026 在 team-level 上的镜像（A/B/C/D shaping 一一对应），所以正确的"shaping 跨架构"对比是 **038 vs 026 同字母**。

**但**——比较的关键是要**控制起点**。026 和 038 不是从同一个 baseline 出发的：

| Lane | Warmstart base | Base WR (official 500) | Base WR (1000ep) |
|---|---|---:|---:|
| **026 (per-agent)** | `BC@2100` (017 main run) | 0.842 | 0.804 |
| **038 (team-level)** | `028A@1060` (team-level) | 0.783 | 0.783 |
| **Δ base (per-agent − team-level)** |  | **+5.9pp** | **+2.1pp** |

**`5.9pp` 是 per-agent vs team-level 在 warmstart 阶段就已经存在的架构差距**——和 shaping 设计无关。任何"绝对 WR 跨架构对比"都会**自动继承**这个 5.9pp 差距 。所以"038 比 026 落后 1-7pp" 里**大部分是 base 差距继承**，**不是** shaping 在 team-level 上失效。

#### 10.11.1 绝对 WR 表（仅供参考，实际意义有限）

**方法论 caveat**：026 lane 当时只跑了 [`official baseline 500-window`](snapshot-026-reward-liberation-ablation.md#155-汇总表)（更早的 eval 协议），038 lane 跑的是 `official baseline 1000` (n=1000)。**1000ep 比 500ep 系统性低 2-4pp**，所以下表里 026 数 都是"500ep 上界 (samples-inflated)"，038 是 1000ep 真实点：

| Shaping | 026 (per-agent) | 038 (team-level) | 绝对 Δ |
|---|---:|---:|---:|
| A (depenalized v2) | 0.810 (500ep peak @280) | 0.796 (1000ep peak @160) | -0.014 |
| B (PBRS goal-prox) | 0.842 (500ep @170) / 0.864 (peak @250) | 0.797 (1000ep peak @110) | -0.045 / -0.067 |
| C (event lane) | 0.846 (500ep peak @30) | 0.800 (1000ep peak @50) | -0.046 |
| D (entropy=0.01) | 0.824 (500ep peak @140) | 0.806 (1000ep peak @40) | -0.018 |

#### 10.11.2 **真正对等的对比：per-base 增量 (Δ over own warmstart base)**

把 base 差距扣掉之后，看每条 shaping 在自己架构上**贡献了多少**：

| Shaping | 026 gain over BC@2100 (0.842) | 038 gain over 028A@1060 (0.783) | Δ (team − per-agent) |
|---|---:|---:|---:|
| A (depenalized v2) | -0.032 (peak @280) | +0.013 (peak @160) | **+0.045** ↑ team |
| B (PBRS goal-prox) | +0.000 (@170) / +0.022 (peak @250) | +0.014 (peak @110) | **+0.014 / -0.008** |
| C (event lane) | +0.004 (peak @30) | +0.017 (peak @50) | **+0.013** ↑ team |
| D (entropy=0.01) | -0.018 (peak @140) | +0.023 (peak @40) | **+0.041** ↑ team |

**翻转的读法**：扣掉 base 差距后，**4 条 shaping 在 team-level 上的 Stage 1 增量都 ≥ per-agent 的同字母 counterpart**（B 在 peak 上微弱落后 -0.008，但在 @170 主候选位置上 team-level +1.4pp 优于 per-agent 0.0pp）。

#### 10.11.3 这个翻转的两种解释

1. **乐观读法**："team-level 架构对 shaping 的反应**至少不差**于 per-agent，可能更好" —— 但要证明这个，需要把 team-level base 推到 per-agent 同等高度（~0.84）再加 shaping
2. **保守读法**："026 起点已经接近 per-agent 平台天花板 (~0.842)，shaping 难再涨；038 起点远未饱和 (0.783) 还有 headroom，加什么都涨一点" —— 这是 **base 饱和度** artifact，不是 shaping 真实优势

无论哪种解释，**之前 §10.11 草稿的 "team-level shaping 整体劣于 per-agent" 结论站不住脚** —— 那个比较把 base 差距和 shaping 差距混淆了。

#### 10.11.4 修正后的 verdict

- **同 base continuation (vs 028A@1060)**: 4 条 038 lane +1.3~+2.3pp，**没明显跳出 028A 窄带**——这个判定不变
- **同 shaping 跨架构 (vs 026 同字母)**:
  - **绝对 WR**: 038 落后 1-7pp，但这 1-7pp 大部分是 base 差距继承 (5.9pp)
  - **per-base Δ**: 038 反而 ≥ 026，shaping 在 team-level 上的边际效用**至少不差**
- **Stage 2 handoff 决策不变**: 即使 shaping 在 team-level 上效用相当，从 028A@1060 (0.783) 出发的 Stage 2 终点也很难超过 029B@190 (0.846)；要追上需要 base 本身先到 0.84+
- **真正的"team-level shaping 是否有效"问题，必须用一个 0.84+ 的 team-level base 作 warmstart 才能干净回答** —— 当前 031A@1040 (0.860 1000ep) 是这样的候选，但 031A 用的是 dual-encoder + 自带 v2 shaping，已经包含了 38A 的设计；要做严格对比应该 031A@1040 + (038B/C/D 同字母 shaping) 各跑一条
- 优先级 unchanged: 031A 已是当前 SOTA，**先把 031A 做透（H2H matrix + capture + ckpt 1170）再决定是否 launch "031A + shaping" Stage 3**

## 9. 相关

### 10.7 结论（首轮收口，不预判未来方向）

- 4 条 Stage 1 lane 都**成立**（训练正常完成、无 inf、1000ep ≥ 028A warmstart）
- **没有任何一条跳出"028A + continuation"的窄带**，在 1000ep 尺度上**与 025b/017/029B 仍有 4–6 pp 差距**
- 是否值得继续 Stage 2 handoff，先看 `038D@40` 的 failure capture 结构；不急于下"team-level liberation 无法打穿"的断言

## 9. 相关

- [SNAPSHOT-026](snapshot-026-reward-liberation-ablation.md) — per-agent 的 Stage 1 参照
- [SNAPSHOT-029](snapshot-029-post-025b-sota-extension.md) — per-agent 的 Stage 2 参照，尤其 B-warm handoff 细节
- [SNAPSHOT-030](snapshot-030-team-level-advanced-shaping-chain.md) — team-level shaping chain，030-C 的原位占位
- [SNAPSHOT-033](snapshot-033-team-level-native-coordination-reward.md) — team-PBRS 单 lane 结果
- [SNAPSHOT-036](snapshot-036-learned-reward-shaping-from-demonstrations.md) — 当前主线，激活条件依赖其 verdict
- [rank.md](rank.md) — 所有数值依据
