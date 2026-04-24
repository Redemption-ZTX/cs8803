## SNAPSHOT-082: Hierarchical Two-Stream Siamese Encoder — arch revision to break the 0.91 plateau (DIR ①)

- **日期**: 2026-04-21
- **负责人**: Self
- **状态**: 预注册（pre-registered, not launched）— implementation + launcher + smoke test ready.
- **前置**: [snapshot-031](snapshot-031-team-level-native-dual-encoder-attention.md) (031A/B foundation) / [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) (current SOTA recipe 0.907) / [snapshot-075 §5.1](snapshot-075-strategic-synthesis-toward-0.93.md) (DIR-A wide-student 076 wide test)
- **同期对照**: [snapshot-076](snapshot-076-wide-student-distill.md) (wide student, same 031B-monolithic encoder) / [snapshot-079](snapshot-079-055v3-recursive-distill.md) (recursive distill)

---

## 0. 背景

### 0.1 0.91 tie-saturation evidence

截至 2026-04-21 23:30 EDT（见 [rank.md](rank.md)），frontier combined 2000ep/3000ep：

| Lane | Composition | Combined WR | Note |
|---|---|---:|---|
| 055@1150 | distill 034E → 031B student | **0.907** | prior single-model SOTA |
| 055v2_extend@1750 | recursive 5-teacher distill | **0.9163** | current SOTA, +0.009 over 055 |
| 056D, 058, 062a etc. | curriculum / LR sweep / PBRS / T-sweep | 0.892–0.916 | all within ±1σ of plateau |
| 074F | weighted 1750-centric ensemble | best member + **0.007** only | multi-model squeeze ≈ exhausted |

所有 distill / recursive / curriculum / PBRS / T-sweep / ensemble 变体都在 [0.89, 0.92) 区间 saturate。**`student arch` 是唯一没被系统 ablate 的 first-class axis**（076 扩 width 但保持 monolithic MLP；所有 distill 的 student 都是 031B-arch 的单块 [336→256→256→cross-attn] encoder）。

### 0.2 Hypothesized bottleneck: encoder representation quality

- 031B 的 encoder 把 336-dim per-agent obs 当单一 flat vector 喂 MLP；
- soccer_twos 的 336-dim 是 **ray-cast perception** (site-packages/soccer_twos/wrappers.py:172 hardcodes `shape=(336,)`；`_obs[:336]` 才被交给 policy，position/velocity/ball 来自 info dict)；
- ML-Agents RayPerceptionSensor 默认 `tags=7, distance=1 → 8 features per ray`，3 stacked frames；`336 = 3 × 112`，即三帧堆叠的射线观测；
- 单块 MLP 必须同时学 **"最近帧 vs 历史帧"**（temporal 语义）+ **"前向 sensor vs 后向 sensor"**（spatial 语义），没有任何 inductive bias 指引它先做 temporal summary 再做 spatial fusion。
- 076 试图通过 **扩 width (0.46M → 1.0M)** 回收 encoder capacity，但若 bottleneck 是 **结构性 inductive bias 而不是 capacity**，wide student 会 tie 而非 break。本 lane 的 H_082 正是切开这两个假设。

### 0.3 Related literature

- Two-stream CNN (Simonyan & Zisserman 2014, NeurIPS)：action recognition 把 spatial 和 temporal 流分开，后期 fuse，显著超过 monolithic。
- Atari DQN frame-stacking vs recurrent encoders (Hausknecht & Stone 2015)：stacked frames 用 separate temporal modeling 优于 flat concat。
- MAT (Wen 2022) 虽然 cross-agent attention 优势来自 agent-level 结构，但 encoder 内部仍 monolithic — 留有 hierarchical 改造空间。

---

## 1. Hypothesis H_082

### 1.1 主假设

> **H_082 (主)**: 把 per-agent 336-dim obs 拆成 `env_slice (224 = 两个历史帧)` + `self_slice (112 = 最近帧)`，分别经 `env_encoder MLP(192,192)` 和 `self_encoder MLP(64,64)`，concat 后 `agent_feat (256-dim) → 031B 4×64 token 的 within-agent cross-attention → merge MLP(256,128) → 标准 policy/value heads`，其余 (v2 shaping + LR=1e-4 + 1250 iter + baseline opponent) 与 031B / 055 scratch 一致。**Combined 2000ep peak ≥ 0.920**（+0.013 over 055 raw scratch 0.907，+0.004 over 055v2_extend@1750 的 0.9163；如果 HIT 则同时验证 "encoder inductive bias 是 0.91 plateau 的 load-bearing cause" 这一结构性假设）。

### 1.2 Stretch

> **H_082-stretch**: combined 2000ep peak ≥ 0.925 — hierarchical encoder 比 monolithic 在同 param count 下提供更强表征，scratch arch revision 能单独突破 0.92。

### 1.3 子假设

- **H_082-a**: student param count 从 031B 的 0.46M 降到约 0.40M（见 §2.3），但 *inductive bias 更强* → 即使 capacity 小 5–10%，peak WR ≥ 031B 的 0.88。验证 076 的 wide-student 假设是否必要。
- **H_082-b**: self/env 两条流 loss-metrics（`self_stream_norm_mean`、`env_stream_norm_mean`、`encoder_cos_sim_mean`）能清晰看到 **self 流收敛快、env 流收敛慢**（符合"近场信号简单、历史感知复杂"的 expected structure）。如果两条流 activation norm 趋同或完全 degenerate 到一条，说明 split 无效。
- **H_082-c**: PPO 在 0.4M 两流 encoder 上仍稳定；不会因分流 引入 gradient skew 或 entropy collapse（监控 `attention_entropy_mean` 与 policy entropy）。

### 1.4 Anti-hypothesis

- **H_082-null**: 336-dim obs 内部 **没有** clean temporal/spatial split —— "前 224 维 vs 后 112 维" 的 heuristic 划分可能和 Unity 二进制真实语义错配（我们无法拆 Unity binary）。此时两条流编码相当于 randomly 划分 feature 再分别 MLP，不提供有效 bias，combined 2000ep peak ∈ [0.895, 0.915) tied 031B/055。
- **H_082-bad**: hierarchical encoder 比 monolithic **差** —— 在 small PPO model 上 concat-then-merge 浪费 capacity（Two-stream CNN 原始论文也报告过部分 early-fusion 优于 late-fusion 的 setting）。如果 combined 2000ep < 0.890，需 abandon 并 flag 为 regression。

---

## 2. Design

### 2.1 Observation split rationale

**关键探索发现**（见 [/cs8803drl/branches/team_siamese_two_stream.py](../../cs8803drl/branches/team_siamese_two_stream.py) docstring）:

- soccer_twos 的 `observation_space.shape == (336,)` 是编译环境硬编码（[wrappers.py:172](#)）；
- `_obs[:336]` 是 policy 看到的全部，position/rotation/velocity/ball 只在 **info dict** 里，不参与 obs；
- 即：336 dims 全是 **ray-perception** 数据。snapshot-021 §4.1 "own velocity + ball relative + goals" 的描述源自早期误读（那些字段其实从 info 导出再 append）。

因此 "self_dim(小) vs env_dim(大)" 的 naive 划分无法做到 `self = own-state / env = rays`，因为没有 own-state 在 obs 里。最清晰的可辩护 split 是 **temporal hierarchy**：

- `self_slice` = 尾部 112 dims ≈ 最近一帧射线 （"agent 当前看到什么"）
- `env_slice` = 前 224 dims ≈ 两个历史帧 （"过去两帧，供推断运动 / 持续性"）

这对应 `336 = 3 × 112` 的 stacked-frame 结构（7 tag one-hot + 1 distance = 8 features/ray；11 forward rays + 3 back rays = 112 features/frame；3 frames stacked）。

注：split 无法从 Unity binary 反推真实 feature ordering（需要 de-assemble），但 `_encode_agent` 的切分方向（`env_slice = obs[:env_dim]`, `self_slice = obs[env_dim:]`）是可通过 `TEAM_SIAMESE_SELF_SLICE_DIM` env var 动态调，后续 snapshot 可以 ablate 不同 split 而无需改代码。

### 2.2 Architecture

```
Per agent (336-dim obs):
  env_slice (obs[:224])  → Linear(224→192) → ReLU → Linear(192→192) → ReLU → env_feat (192)
  self_slice (obs[224:]) → Linear(112→64)  → ReLU → Linear(64→64)   → ReLU → self_feat (64)
  agent_feat = cat(self_feat, env_feat)  → 256-dim

Two agents (from 672-dim team obs):
  agent_feat_0, agent_feat_1  (each 256-d)

031B within-agent cross-attention (unchanged):
  tokens_i = agent_feat_i.view(B, 4, 64)   # 4 tokens × 64 head_dim
  attn_0 = softmax(Q@Kᵀ/√64) · V  with Q from tokens_0, K,V from tokens_1
  attn_1 = ... (symmetric, shared Q/K/V projections)
  attn_i_flat = attn_i.reshape(B, 256)

Merge:
  merged_input = cat(agent_feat_0, attn_0_flat, agent_feat_1, attn_1_flat)  → 1024-d
  Linear(1024 → 256) → ReLU → Linear(256 → 128) → ReLU → policy_logits / value
```

### 2.3 Param count budget

| Component | Params |
|---|---:|
| self_encoder (112→64→64) | ~11.4K |
| env_encoder (224→192→192) | ~80.3K |
| q/k/v_proj (64×64 each, no bias) | 12.3K |
| merge_mlp (1024→256→128) | ~295K |
| logits_layer (128→18) | 2.3K |
| value_layer (128→1) | 129 |
| **Total** | **~0.40M** |

Reference: 031B ~0.46M, 076 wide ~1.0M, 076-wider target ~1.0M. 082 **比 031B 略小** — 刻意保持 param parity 近侧，避免 076 wide 的 capacity 混淆变量。如果 082 ≥ 031B WR 而 params 更少，证据会更强。

### 2.4 Env vars (new flags)

- `TEAM_SIAMESE_TWO_STREAM=1` — enable flag（required: `TEAM_SIAMESE_ENCODER=1` + `TEAM_CROSS_ATTENTION=1`）
- `TEAM_SIAMESE_SELF_HIDDENS=64,64` — self-stream encoder hidden sizes
- `TEAM_SIAMESE_ENV_HIDDENS=192,192` — env-stream encoder hidden sizes
- `TEAM_SIAMESE_SELF_SLICE_DIM=0` — 0 means default `half_obs_dim // 3 = 112`; set to other int to override
- 仍复用 `TEAM_SIAMESE_MERGE_HIDDENS`、`TEAM_CROSS_ATTENTION_TOKENS`、`TEAM_CROSS_ATTENTION_DIM`

**Constraint**: `self_hiddens[-1] + env_hiddens[-1] == TEAM_CROSS_ATTENTION_TOKENS * TEAM_CROSS_ATTENTION_DIM`，否则构造器会抛 `ValueError`（smoke-tested）。默认值 `64 + 192 = 256 = 4 × 64` 满足。

### 2.5 Training recipe (copy from 031B / 055 scratch)

| Item | Value | Source |
|---|---|---|
| Framework | Ray RLlib 1.4 PyTorch | 项目约束 |
| Opponent | baseline (BASELINE_PROB=1.0) | 031B 对齐 |
| LR / clip | 1e-4 / 0.15 | 031B 对齐 |
| Batch | rollout_fragment=1000, train_batch=40k, SGD_minibatch=2048, num_sgd_iter=4 | 031B 对齐 |
| Reward | v2 shaping (snapshot-030) | 031B / 055 对齐 |
| Budget | 1250 iter, 50M steps, 12h wall | 031B / 055 对齐 |
| Eval | baseline + random @ 200ep every 10 iter | 常规 |

### 2.6 Why NOT 3-stream / learned split

- 3-stream（self / ray / ball）需要 own-state 在 obs 中 — 暂无（官方 evaluator 约束 AgentInterface 只看 raw obs）；未来 snapshot 可先 append own-state 再做 3-stream。
- Learned split（mask / gating）会引入额外可学参数覆盖 heuristic split 的可能性错配，但也会增加训练稳定性风险；**保守选择先打固定 heuristic，若 HIT 再迭代**。

---

## 3. 预注册 Thresholds

*Combined 2000ep = 1000 blue + 1000 orange；所有判决以 combined 为准。*

### 3.1 §3.1 Breakthrough (stretch-win)

- combined 2000ep peak ≥ **0.925**
- 同时 vs 055v2_extend@1750 combined 2000ep ≥ 0.921 (+1pp over current SOTA)
- 结论：hierarchical encoder inductive bias 是 0.91 plateau 的 load-bearing cause；后续 DIR ① iteration（3-stream / learned split / larger self_stream）值得重仓。

### 3.2 §3.2 Main HIT

- combined 2000ep peak ∈ [0.918, 0.925)
- Δ vs 055 scratch 0.907 ≥ +0.011pp (>2×SE, approaching 2σ)
- 结论：encoder arch 贡献 ≥ +0.011pp，与 076 wide 结果对比（76 若 tie）证明 "inductive bias 优于 pure capacity"。

### 3.3 §3.3 Marginal / tied

- combined 2000ep peak ∈ [0.900, 0.918)
- Δ vs 055 scratch ∈ [-0.007, +0.011pp]
- 结论：hierarchical encoder 不差于 031B 但也没 break plateau；可能的 follow-up：
  - 改 split heuristic（`SELF_SLICE_DIM=88` 对齐 forward-sensor 长度；或做逐 ray-block 的 3-stream）
  - 套 distill（082-arch student + 034E ensemble teacher = 082D）

### 3.4 §3.4 Regression (退市)

- combined 2000ep peak < 0.890
- 或 single-shot 1000ep < 0.840 @ 任一 checkpoint（严重 scratch 训练不稳）
- 结论：heuristic split 反而破坏 representation；关掉该 lane，不发起 082-iter。

---

## 4. Risks & Retrograde

### 4.1 Known risks

- **R1 split heuristic 错配**: 336 dims 在 Unity binary 内部真实 ordering 可能不是 `stride=112 per frame`。缓解：`TEAM_SIAMESE_SELF_SLICE_DIM` env var 允许 0-成本 ablate（后续 snapshot 跑 `SELF_SLICE_DIM=88 / 168 / 224` 对比）。
- **R2 concat-then-merge capacity 浪费**: Two-stream 论文里 early-fusion 偶尔优于 late-fusion。缓解：merge_mlp 保持和 031B 一致的 1024→256→128 topology，不引入 082 特有的 fusion 层；两条流 late fuse 在 cross-attention 之前、而非之后，保守。
- **R3 Scratch 不稳**: PPO on new arch scratch 偶尔 entropy collapse（见 052 transformer regression）。缓解：默认 encoder 初始化用 PyTorch 标准 Linear init；不做 ZeroInit（054M 曾用 ZeroInit 让新模块 start from 031B 等价，但这里两条流和 031B 非 gradual 兼容）。监控 `attention_entropy_mean`、`self_stream_norm_mean`、`env_stream_norm_mean` 三个内部 metric 作为前兆。
- **R4 与 distill 不兼容**: 目前 082 不实现 distill 变体（因为 teacher checkpoint 的 encoder topology 不同，`extract_torch_weights` 会 unexpected-key）。如果 HIT，后续 snapshot 才做 082D ensemble-distill 版本。

### 4.2 Retrograde 方案

如 §3.4 regression: `unset TEAM_SIAMESE_TWO_STREAM`，改跑 031B-equivalent baseline（同 launcher 删掉 082 flags 即可）。不需要 roll back code — 082 model 只在 flag 打开时才走该 branch。

---

## 5. Not doing (有意识排除)

- **LSTM / recurrent encoder** — 已有 3 帧堆叠作为 temporal proxy；LSTM 需要改 rollout shape，排除在本 snapshot 外。
- **Ray-block attention (per-ray token-level attention)** — 需要明确 ray ordering，336/8=42 rays 但含 back sensor 参数不等数，不满足"N × K × D"整齐 token 化；排除。
- **Teacher distillation** — 本 snapshot **纯 arch test**（DIR ① 的 clean experiment）。distill 让 teacher 梯度盖过新 arch 的真实贡献，无法 isolate 变量。082 HIT 后才做 082D。
- **3-stream split (self / ray / ball)** — 需要 obs 中含 own state，当前 336-dim 没有。
- **Learned gating / attention-based split** — 引入额外可学参数，违反 "first isolate heuristic split effect" 原则。

---

## 6. Execution checklist

- [x] 实现 `SiameseTwoStreamTeamTorchModel` → [/cs8803drl/branches/team_siamese_two_stream.py](../../cs8803drl/branches/team_siamese_two_stream.py)
- [x] 注册 `team_siamese_two_stream_model` 到 ModelCatalog
- [x] 训练脚本 env-var 解析 + 验证 + 构造分支 → `cs8803drl/training/train_ray_team_vs_baseline_shaping.py`（minimal additions, gated 在 `TEAM_SIAMESE_TWO_STREAM=1`）
- [x] 部署模块注册 → `cs8803drl/deployment/trained_team_ray_agent.py`、`cs8803drl/deployment/trained_team_ensemble_next_agent.py`
- [x] Launcher → [/scripts/eval/_launch_082_two_stream_siamese_scratch.sh](../../scripts/eval/_launch_082_two_stream_siamese_scratch.sh)
- [x] Smoke test（构造 + forward + backprop + param count + shape-mismatch error path）**PASS**（0.402M params）
- [ ] 发射（pending user confirmation）
- [ ] Combined 2000ep eval at peak checkpoint
- [ ] §3 verdict

---

## 7. Verdict — sub-SOTA, lane closed (append-only, 2026-04-22)

### 7.1 训练完成

- run_dir: `/storage/ice1/5/1/wsun377/ray_results_scratch/082_two_stream_siamese_scratch_20260421_202731/TeamVsBaselineShapingPPOTrainer_Soccer_19fb1_00000_0_2026-04-21_20-27-53/`
- 1250 iter scratch，PORT_SEED=82，约 ~11h wallclock
- v2 shaping / LR=1e-4 / baseline opponent / 50M steps 与 031B/055/083 完全同 budget

### 7.2 Stage 1 baseline 1000ep eval (top 5% + ties + ±1, 24 ckpts, parallel-7, 994s)

| Top ckpt | 1000ep WR | NW-ML |
|---:|---:|:---:|
| **🏆 1090** | **0.885** | 885-115 |
| 1240 | 0.885 | 885-115 |
| 1210 | 0.883 | 883-117 |
| 1230 | 0.883 | 883-117 |
| 1100 | 0.882 | 882-118 |
| 1080 | 0.876 | 876-124 |
| 1200 | 0.873 | 873-127 |
| 880 | 0.873 | 873-127 |
| 1220 | 0.875 | 875-125 |
| 900 | 0.875 | 875-125 |
| 1040 | 0.875 | 875-125 |

**peak = 0.885 @ ckpt-1090/1240 (tied), mean ≈ 0.868, range [0.840, 0.885]**

### 7.3 §3 判据

| 阈值 | 082 实测 | verdict |
|---|---|---|
| §3.1 主 ≥ 0.920 | ❌ 0.885 (-0.035) | 未达 |
| §3.2 stretch ≥ 0.925 | ❌ | 未达 |
| §3.3 tied [0.895, 0.915) | ❌ 0.885 | 未达 |
| **§3.4 退化 < 0.890** | **✅ 0.885** | **sub-plateau** |

**verdict on baseline axis**: 082 single-shot peak 0.885 **低于 031B 0.882 plateau**（实际统计上相当），且低于 055/055v2/083 所有 post-distill frontier 约 −3pp。

### 7.4 Combined 2000ep / H2H 不做

- 082 single-shot peak 0.885 已明显落在 plateau 之下，combined 2000ep 只会再往均值 0.87 收敛，不会翻案
- 直接 H2H vs 055v2@1750 (0.9163) 必然 LOSE `***`，信息含量极低
- 082 vs 083 （双轴 arch alternative 对比）：082 temporal split 不如 083 spatial split，`0.885` vs `0.909` combined ≈ −2.4pp

### 7.5 机制读解

- **"temporal split 224/112" 假设证伪**: 把 ray-perception 按"前 224 = 2 历史帧 / 后 112 = 最近帧"拆开，两条 MLP 再 concat，没有为 encoder 提供有用 bias
- 可能原因：Unity binary 的 336-dim memory layout 不一定是 `[frame0, frame1, frame2]` 这种 C-contiguous 顺序（`soccer_twos` 未文档化内部顺序），heuristic split 和真实 temporal semantic 错配
- 对比：083 spatial split `(3,14,8)` reshape mean-over-frames 得 `(14, 24)` ray tokens —— 这个假设也是 heuristic，但 reshape 后的 14 个 ray tokens 自学 spatial attention 得到 +2.4pp，说明 spatial layout 可能比 temporal layout 更接近真实 binary 顺序
- **统一教训**：**arch-level inductive bias 在 scratch 上最多到 plateau**，想突破 SOTA 要靠 policy 层 distill/ensemble

### 7.6 决策

1. **082 lane 关闭** — sub-SOTA, 不做 Stage 2/3
2. **不 package 为 agent** — per user directive
3. **不再探索 temporal-axis encoder arch** — 082 结果压死这条
4. 与 083 共同结论：encoder arch 不是 plateau bottleneck，pivot 到 Stone Layered L2 / DIR-H 融合方向
5. Follow-up §8 的 082-iter-A (split sweep) 和 082D (arch + distill) **不再启动** —— 即便 split sweep 小幅回升也不会过 0.90 plateau

### 7.7 Raw Stage 1 recap

```
=== Official Suite Recap (parallel) ===
checkpoint-810  vs baseline: win_rate=0.866 (866W-134L-0T)
checkpoint-820  vs baseline: win_rate=0.848 (848W-152L-0T)
checkpoint-830  vs baseline: win_rate=0.840 (840W-160L-0T)
checkpoint-880  vs baseline: win_rate=0.873 (873W-127L-0T)
checkpoint-890  vs baseline: win_rate=0.872 (872W-128L-0T)
checkpoint-900  vs baseline: win_rate=0.875 (875W-125L-0T)
checkpoint-910  vs baseline: win_rate=0.842 (842W-158L-0T)
checkpoint-920  vs baseline: win_rate=0.856 (856W-144L-0T)
checkpoint-1040 vs baseline: win_rate=0.875 (875W-125L-0T)
checkpoint-1050 vs baseline: win_rate=0.859 (859W-141L-0T)
checkpoint-1060 vs baseline: win_rate=0.868 (868W-132L-0T)
checkpoint-1080 vs baseline: win_rate=0.876 (876W-124L-0T)
checkpoint-1090 vs baseline: win_rate=0.885 (885W-115L-0T)
checkpoint-1100 vs baseline: win_rate=0.882 (882W-118L-0T)
checkpoint-1120 vs baseline: win_rate=0.871 (871W-129L-0T)
checkpoint-1130 vs baseline: win_rate=0.860 (860W-140L-0T)
checkpoint-1140 vs baseline: win_rate=0.872 (872W-128L-0T)
checkpoint-1150 vs baseline: win_rate=0.871 (871W-129L-0T)
checkpoint-1160 vs baseline: win_rate=0.861 (861W-139L-0T)
checkpoint-1200 vs baseline: win_rate=0.873 (873W-127L-0T)
checkpoint-1210 vs baseline: win_rate=0.883 (883W-117L-0T)
checkpoint-1220 vs baseline: win_rate=0.875 (875W-125L-0T)
checkpoint-1230 vs baseline: win_rate=0.883 (883W-117L-0T)
checkpoint-1240 vs baseline: win_rate=0.885 (885W-115L-0T)
[suite-parallel] total_elapsed=994.2s tasks=24 parallel=7
```

Log: [082_baseline1000.log](../../docs/experiments/artifacts/official-evals/082_baseline1000.log)

---

## 8. Follow-up

- **If HIT (§3.1 or §3.2)**: 启动 082-iter-A =「split sweep」（SELF_SLICE_DIM ∈ {88, 112, 168, 224}）+ 082D = 082-arch + 034E ensemble distill（把 082 的 encoder inductive bias 和 distill 叠加）。
- **If marginal (§3.3)**: 暂停 DIR ①，转 DIR ②（snapshot-076 wide student / snapshot-077 per-agent student / snapshot-078 DAGGER 中未做完的）。
- **If regression (§3.4)**: 关闭 DIR ①。证据表明 monolithic MLP encoder 已饱和 — plateau 来自 reward 信号 / 环境 bandwidth 而非 encoder structure，后续应 invest DIR ③ adaptive reward (snapshot-081)。

---

## 9. Related snapshots

- [snapshot-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031B arch foundation (shared Siamese encoder + within-agent cross-attention)
- [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) — 0.907 distill SOTA recipe; 同 student arch
- [snapshot-075](snapshot-075-strategic-synthesis-toward-0.93.md) — 0.91 plateau 的整体性分析 + 五条 frontier direction
- [snapshot-076](snapshot-076-wide-student-distill.md) — 同期 DIR-A (wide monolithic encoder) 对照；082 是同 axis 的 structural alternative
- [snapshot-052](snapshot-052-full-transformer.md) — 反例：更大 refinement block 在 small PPO 上 regress，教训是"prefer minimal structural change"（082 严格遵守：只改 encoder，保留 attention + merge 不变）
