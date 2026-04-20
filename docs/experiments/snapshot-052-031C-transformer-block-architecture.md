# SNAPSHOT-052: 031C Transformer Block — Architecture Step 3 (handoff doc)

- **日期**: 2026-04-19
- **负责人**: TBD (handoff to teammate)
- **状态**: lane 关闭 (2026-04-19) — 052A REGRESSION -8.2pp / 052 REGRESSION -10.8pp vs 031B 0.882, 架构 step 3 决定性失败 (见 §7)

## 0A. 2026-04-19 实施收缩说明

在真正对照当前 `031B` 代码后，先落地的是 **`031C-min`**，而不是本页原始设想里的“完整 transformer + 4-head MHA”版本。

当前代码库里的 `031B` 实现是：
- **手写 single-head token attention**
- merge 输入保持 `[feat0, attn0, feat1, attn1]` 共 `1024` dim

所以首轮最小可解释实现收成：
- **保留 `031B` attention 机制不变**
- **保留 `031B` merge 拓扑不变**
- **只增加 shared FFN + residual + LayerNorm refinement**

也就是说，当前已实现的 `031C-min` 测的是：

> `031B + transformer-style FFN/residual/norm`

而**不混入**：
- 真 `nn.MultiheadAttention` 重构
- `4 heads` 改动
- merge 输入从 `1024 → 512` 的拓扑变化

这样首轮结果最容易解释；若 `031C-min` 为正，再继续做完整 MHA refactor。

## 0B. 2026-04-19 实施拆分现状

在 `031C-min` 的基础上，现已额外落实 **`031C-mha`**，并继续补齐了 **full `031C`**：
- 仍保留 `031B` 的 `1024`-dim merge 拓扑
- 仍保留 shared FFN/residual/norm refinement
- **只把手写 single-head token attention 换成真 `nn.MultiheadAttention`**

与此同时，**full `031C`** 已作为本 snapshot 的主线版本落地：
- `true MHA`
- `FFN + residual + LayerNorm`
- merge 从 `[feat0, z0, feat1, z1]` 收成 `concat(z0, z1)` 的 `512`-dim transformer-style merge

因此当前 `052` 已经被拆成三条可比较的子线：
- **`052A / 031C-min`**: 测 FFN/residual/norm 本身的增益
- **`052B / 031C-mha`**: 在 `031C-min` 上再测 true MHA 的增益
- **`052 / full 031C`**: 原始 snapshot 主线，测 true MHA + refinement + transformer merge 的合成效果

这比原始“一步到位 full transformer”更容易解释：
- `031B -> 031C-min` = refinement block 的净效应
- `031C-min -> 031C-mha` = true MHA 的净效应
- `031C-mha -> full 031C` = merge 拓扑收缩的净效应

## 0. 背景与定位

[SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) 系列架构升级路径：

| step | 架构 | 1000ep peak | Δ vs prev step |
|---|---|---|---|
| 0 | 028A flat MLP team-level | 0.783 | base |
| 1 (Siamese) | 031A: Siamese encoder + concat merge | 0.860 | **+7.7pp** |
| 2 (Cross-attention) | 031B: + 1-head cross-attention (dim=64, tokens=4) | **0.882** | +2.2pp |
| **3 (Transformer-min)** | **031C-min: + FFN + LayerNorm + residual** | **TBD** | **预期 +0.5-2pp** |

**diminishing returns 严重**: step 1 +7.7pp, step 2 +2.2pp, step 3 expected +0.5-2pp。但即使 step 3 拿到 +1pp，031B 0.882 → 031C 0.892 = **直接达到 0.90 grading 门槛**。

这是**给 user "更高分数 + 稳定性 + 多条路径" 目标**的一条独立路径，不依赖 ensemble (snapshot-034) / DAGGER (snapshot-051) / cross-train (snapshot-046) 等任何 follow-up 路径。

## 1. 核心假设

### H_031C

> 把 cross-attention 升级成完整 transformer block (attention + FFN + LayerNorm + residual)，**1000ep peak ≥ 0.890**，比 031B (0.882) 提升 ≥ +0.8pp。如果 ≥ 0.90，**直接打到 grading 门槛**。

### 子假设

- **H_031C-a**: FFN (feedforward) 给 attention 后的 features 提供更复杂变换，capacity 提升
- **H_031C-b**: LayerNorm 让训练更稳定，避免 attention 数值漂移（031B max\|kl\|=10.71 已偏高）
- **H_031C-c**: residual 让深层 attention block 的 gradient 顺畅传播

## 2. 设计

### 2.1 架构 spec（当前已落实的是 `031C-min`）

```
Input: 672-dim joint obs (= concat(agent_0_obs_336, agent_1_obs_336))

Step 1 — Siamese encoder (031B 同款):
    obs_336 → Linear(336→256) → ReLU → Linear(256→256) → ReLU → feat_256 (per agent)

Step 2 — Cross-attention (031B 同款):
    feat_256_per_agent → Tokenize (4 tokens × 64 dim each)
    Existing 031B token attention (single-head, hand-written)
    → attended_feat_256 per agent

Step 3 NEW — Transformer FFN + Norm + Residual:
    attended_feat = LayerNorm(attended_feat + cross_attn_output)
    ffn_out = Linear(256→512) → GELU → Linear(512→256)
    feat_final = LayerNorm(attended_feat + ffn_out)

Step 4 — Merge to joint:
    concat(feat_agent_0, feat_final_agent_0, feat_agent_1, feat_final_agent_1) → 1024 dim
    Linear(1024→256) → ReLU → Linear(256→128) → ReLU → policy/value heads
```

**当前 `031C-min` 关键设计决策**:
- **不改 attention 头机制** — 先只测 FFN/residual/norm 的净增益
- **不改 merge 拓扑** — 维持 `031B` 的 `1024` dim merge 输入，避免把“表示 refinement”与“merge 重构”混在一起
- **FFN expansion ratio 2x** (256→512→256) — 标准 transformer FFN 是 4x，这里 2x 节省参数（已经是小模型）
- **GELU activation** in FFN — 比 ReLU 略好的现代选择，跟 transformer 标准一致
- **Pre-norm vs post-norm**: 当前实现默认 **post-norm**，但保留 env var 方便切到 prenorm

### 2.2 训练超参（继承 031B）

```bash
# Architecture (031C 新增)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128
TEAM_CROSS_ATTENTION=1
TEAM_CROSS_ATTENTION_TOKENS=4
TEAM_CROSS_ATTENTION_DIM=64

# 031C-min NEW additions:
TEAM_TRANSFORMER_MIN=1
TEAM_TRANSFORMER_FFN_HIDDEN=512           # FFN 中间维
TEAM_TRANSFORMER_FFN_ACTIVATION=gelu      # GELU
TEAM_TRANSFORMER_NORM=postnorm            # post-norm 标准

# PPO 同 031B
LR=1e-4
CLIP_PARAM=0.15
NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000
SGD_MINIBATCH_SIZE=2048
GAMMA=0.99
GAE_LAMBDA=0.95
ENTROPY_COEFF=0.0

# v2 shaping 同 031B
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001
SHAPING_BALL_PROGRESS=0.01
SHAPING_OPP_PROGRESS_PENALTY=0.01
SHAPING_POSSESSION_BONUS=0.002
SHAPING_DEEP_ZONE_OUTER_THRESHOLD=-8
SHAPING_DEEP_ZONE_OUTER_PENALTY=0.003
SHAPING_DEEP_ZONE_INNER_THRESHOLD=-12
SHAPING_DEEP_ZONE_INNER_PENALTY=0.003

# Training budget — 同 031B (1250 iter scratch)
MAX_ITERATIONS=1250
TIMESTEPS_TOTAL=50000000
EVAL_INTERVAL=10
EVAL_EPISODES=50
CHECKPOINT_FREQ=10
```

### 2.3 工程实现

#### 2.3.1 修改 `cs8803drl/branches/team_siamese.py`

当前已实现的是 `SiameseTransformerMinTeamTorchModel`，继承自 `SiameseCrossAttentionTeamTorchModel`：

```python
class SiameseTransformerTeamTorchModel(SiameseCrossAttentionTeamTorchModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        # FFN block (per-agent)
        custom_config = model_config.get("custom_model_config", {})
        ffn_hidden = int(custom_config.get("transformer_ffn_hidden", 512))
        ffn_activation = custom_config.get("transformer_ffn_activation", "gelu")
        norm_pos = custom_config.get("transformer_norm", "postnorm")
        feat_dim = self.cross_attn_output_dim  # 256 from cross-attn
        self._ffn = nn.Sequential(
            nn.Linear(feat_dim, ffn_hidden),
            nn.GELU() if ffn_activation == "gelu" else nn.ReLU(),
            nn.Linear(ffn_hidden, feat_dim),
        )
        self._ln_attn = nn.LayerNorm(feat_dim)
        self._ln_ffn = nn.LayerNorm(feat_dim)
        self._norm_pos = norm_pos
    def _refine_attended_feature(self, feat, attn_output):

        # Post-norm transformer block
        if self._norm_pos == "postnorm":
            x = self._ln_attn(feat + attn_output)
            ffn_out = self._ffn(x)
            return self._ln_ffn(x + ffn_out)
        else:  # prenorm
            x = feat + attn_output
            ffn_in = self._ln_attn(x)
            ffn_out = self._ffn(ffn_in)
            return x + self._ln_ffn(ffn_out)
```

注：完整版 `031C` 的 `true MHA` refactor 仍保留在本页原始设计里，但**当前已落地的不是那一步**。

#### 2.3.2 注册新 model

在 `team_siamese.py` 末尾加：

```python
def register_team_siamese_transformer_model():
    ModelCatalog.register_custom_model(
        "team_siamese_transformer_model",
        SiameseTransformerTeamTorchModel,
    )
```

#### 2.3.3 修改 train script

`cs8803drl/training/train_ray_team_vs_baseline_shaping.py` 已扩展支持新的 `TEAM_TRANSFORMER_MIN` env var：

```python
# In train script's model_config setup
if os.environ.get("TEAM_TRANSFORMER_MIN", "0") == "1":
    custom_model_name = TEAM_SIAMESE_TRANSFORMER_MIN_MODEL_NAME
    register_team_siamese_transformer_min_model()
    model_config["custom_model_config"].update({
        "transformer_ffn_hidden": int(os.environ.get("TEAM_TRANSFORMER_FFN_HIDDEN", "512")),
        "transformer_ffn_activation": os.environ.get("TEAM_TRANSFORMER_FFN_ACTIVATION", "gelu"),
        "transformer_norm": os.environ.get("TEAM_TRANSFORMER_NORM", "postnorm"),
    })
elif os.environ.get("TEAM_CROSS_ATTENTION", "0") == "1":
    # ... existing cross-attention path ...
```

#### 2.3.4 真 MHA 是后续二阶段，不是 `031C-min`

当前 `031C-min` 不动这一层。若 `031C-min` 首轮结果为正，再新开一步做：
- 把手写 single-head token attention 改成真 `nn.MultiheadAttention`
- 再开放 `TEAM_CROSS_ATTENTION_HEADS`

### 2.4 Smoke test 计划 (在 launch 训练前)

类似 snapshot-031 §12.8 实现进度的 smoke pattern:

```python
# scripts/smoke/smoke_031C_transformer_model.py
import torch
from cs8803drl.branches.team_siamese import (
    SiameseTransformerTeamTorchModel,
    register_team_siamese_transformer_model,
)
register_team_siamese_transformer_model()
model = SiameseTransformerTeamTorchModel(
    obs_space=Box(low=-inf, high=inf, shape=(672,)),
    action_space=MultiDiscrete([3,3,3,3,3,3]),
    num_outputs=18,  # 6 dims × 3 vals
    model_config={"custom_model_config": {
        "siamese_encoder_hiddens": "256,256",
        "siamese_merge_hiddens": "256,128",
        "cross_attention_tokens": 4,
        "cross_attention_dim": 64,
        "cross_attention_heads": 4,
        "transformer_ffn_hidden": 512,
        "transformer_ffn_activation": "gelu",
        "transformer_norm": "postnorm",
    }},
    name="test",
)
# Verify forward
batch = torch.randn(8, 672)
logits, state = model({"obs": batch})
assert logits.shape == (8, 18)
value = model.value_function()
assert value.shape == (8,)
# Verify backward
loss = (logits.sum() + value.sum())
loss.backward()
print("smoke PASS: forward + backward + shape match")
# Param count
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,} (vs 031B ~= ?)")
```

## 3. 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 主: 1000ep peak ≥ 0.890 | +0.8pp vs 031B | **架构 step 3 成立** |
| §3.2 突破: 1000ep peak ≥ 0.900 | +1.8pp vs 031B | **直接达到 grading 门槛**, 项目 declare success |
| §3.3 持平: 1000ep peak ∈ [0.875, 0.890) | -0.7pp~+0.8pp | step 3 **没用**, diminishing returns 确认 |
| §3.4 退化: 1000ep peak < 0.870 | < 031A base | step 3 **伤害**, transformer 在小模型上 over-parameterized |

## 4. 风险

### R1 — 参数膨胀 → over-fit / 训练慢

FFN 加 2x dim = 256 + (256×512 + 512×256) = ~262K extra params per agent (× 2 agents)。031B 总参数 ~1M, 加 ~500K 是 50% 增加。**缓解**: 同 budget 训练 (50M steps), 看是否 saturate; 如果 peak 在 800 iter 前 plateau, 减小 FFN hidden 到 256。

### R2 — LayerNorm 在 PPO 下不稳定

`LayerNorm` 在 supervised learning 是 well-understood, 在 PPO 上可能引入 gradient scale issue。**缓解**: 用 `pre-norm` (norm before attention/ffn) 而不是 post-norm 是更稳定的 modern 选择 (e.g., GPT, BERT 现代版本都 prenorm)。snapshot-052 默认 postnorm 但保留 prenorm 作为 fallback if 不稳。

### R3 — Multi-head 在 dim=64 太小

4 heads × 16 dim/head = 256, head dim 16 偏小。可能不如 single head 好。**缓解**: 把 `TEAM_CROSS_ATTENTION_DIM` 改成 128 (2 heads × 64) 或 256 (4 heads × 64), 但参数翻倍。或保留 `TEAM_CROSS_ATTENTION_HEADS=1` 单独测 transformer FFN 影响。

### R4 — 训练时长不够

031B 训了 1250 iter 才稳定 0.882。031C 多了 FFN, 可能需要 1500-2000 iter 才 converge。**缓解**: 第一轮 1250 iter 够了, 看 trajectory 是否还在上升，再决定续跑。

## 5. 不做的事

- **不在 031C 启动前测试 transformer FFN 的单独 ablation** (FFN-only without LayerNorm/residual) — 那是 single-feature ablation, 不在本 snapshot scope
- **不改 reward / shaping** (复用 031B 同 v2)
- **不改 base model size** (encoder hidden 不变)
- **不在没 smoke pass 之前 launch 14h 训练**
- **不跟 031A/031B 同时跑同节点** (避免 GPU OOM, transformer 加显存)

## 6. 执行清单 (handoff 给同学)

1. **阅读** [team_siamese.py](../../cs8803drl/branches/team_siamese.py) 现有 `SiameseCrossAttentionTeamTorchModel` 实现 (~30 min)
2. **实现** `SiameseTransformerTeamTorchModel` class (~3-4h, 见 §2.3.1)
3. **注册** model + 修改 train script 添加 `TEAM_TRANSFORMER` env var 路径 (~1h, 见 §2.3.2/2.3.3)
4. **写 smoke test** `scripts/smoke/smoke_031C_transformer_model.py` (~1h, 见 §2.4)
5. **跑 smoke**，确认 forward/backward + shape (~10 min)
6. **写 batch** `scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_031C_transformer_scratch_v2_512x512.batch` (~30 min, 复制 031B batch 改 model env vars)
7. **跑 1-iter training smoke** 确认 Ray + checkpoint 工作 (~10 min)
8. **Launch 1250 iter scratch** on free node (PORT_SEED uniqueness)
9. **训完 invoke** [`/post-train-eval`](../../.claude/skills/post-train-eval/SKILL.md) lane name `031C`
10. **Verdict** append 到本 snapshot §7

总工程: **5-7 小时 + 14h GPU 训练 + 30 min eval**.

## 7. Verdict — 架构 step 3 决定性 REGRESSION (2026-04-19, append-only)

### 7.1 训练完成

- **052A** (031C-min, refinement only): 1250 iter scratch on atl1-1-03-011-18-0, run dir `ray_results/052A_team_transformer_min_scratch_v2_512x512_20260419_102129/`
- **052** (full 031C, MHA + refinement + merge): 1250 iter scratch on atl1-1-03-012-13-0, run dir `ray_results/052_team_transformer_full_scratch_v2_512x512_20260419_103340/`
- 两条都 best_eval_random ≥ 0.96 (Random 判据全 robust pass)
- 两条 50ep 内部 peak 都 ≥ 0.88 (052A) / 0.90 (052), 但 1000ep 实测远低 — confirms 50ep noise ±0.07 is huge for 1250-iter runs

### 7.2 Stage 1 1000ep eval (top 5%+ties+±1, parallel-7)

**052A** (24 ckpts, 010-20-0 port 60005, 1192s):

| Top ckpt | 1000ep WR | NW-ML |
|---:|---:|:---:|
| **🏆 1080** | **0.800** | 800-200 |
| 1140 | 0.799 | 799-201 |
| 1150 | 0.793 | 793-207 |
| 1050 | 0.792 | 792-208 |
| 1090 | 0.789 | 789-211 |
| 1060 | 0.789 | 789-211 |

**peak = 0.800 @ ckpt 1080, mean ~0.78, range [0.767, 0.800]**

**052** (20 ckpts, 011-13-0 port 61205, 1191s):

| Top ckpt | 1000ep WR | NW-ML |
|---:|---:|:---:|
| **🏆 870** | **0.774** | 774-226 |
| 1150 | 0.773 | 773-227 |
| 1140 | 0.762 | 762-238 |
| 830 | 0.757 | 757-243 |
| 1160 | 0.756 | 756-244 |
| 520 | 0.754 | 754-246 |

**peak = 0.774 @ ckpt 870, mean ~0.75, range [0.720, 0.774]**

### 7.3 Decomposition (architecture step 3 拆解)

| Step | 1000ep peak | Δ vs prev | per-component net |
|---|---|---|---|
| 0: 028A flat MLP | 0.783 | base | — |
| 1: 031A Siamese | 0.860 | **+7.7pp** ★ | Siamese encoder 净效应 |
| 2: 031B + cross-attention | 0.882 | **+2.2pp** ★ | single-head token attention 净效应 |
| **3a: 052A + FFN/Norm/residual (031C-min)** | **0.800** | **-8.2pp** ⚠️ | refinement block 净效应 (REGRESSION) |
| **3b: 052 + true MHA + merge 1024→512 (full 031C)** | **0.774** | -2.6pp from 052A | MHA + merge 收缩 净效应 (further REGRESSION) |

**架构 step 3 累计 -10.8pp vs 031B**, 远低于 §3.4 退化阈值 (<0.870)。

### 7.4 严格按 [§3 判据](#3-预注册判据)

| 阈值 | 052A | 052 | verdict |
|---|---|---|---|
| §3.1 主 ≥ 0.890 (step 3 成立) | ❌ 0.800 (-9pp) | ❌ 0.774 (-12pp) | 都未达 |
| §3.2 突破 ≥ 0.900 (grading 门槛) | ❌ | ❌ | |
| §3.3 持平 [0.875, 0.890) | ❌ | ❌ | |
| **§3.4 退化 < 0.870** | **✅ 0.800** | **✅ 0.774** | **两个都 REGRESSION** |

### 7.5 Stage 2 failure capture — 失败模式分析

500ep capture 在两 peak ckpt 上跑 (port 62205/62405, ~15 min)。v2 桶 vs 031B@1220 (0.876):

| Bucket | 031B@1220 | 052A@1080 | 052@870 | Δ (052A-031B) | Δ (052-031B) |
|---|---|---|---|---|---|
| defensive_pin | 43.5% | 41.7% | 43.2% | -1.8 | -0.3 |
| territorial_dominance | 43.5% | 41.7% | 47.2% | -1.8 | +3.7 |
| **wasted_possession** | 37.1% | 37.0% | **49.6%** | 0 | **+12.5** ⚡ |
| **possession_stolen** | 27.4% | **37.0%** | 20.8% | **+9.6** ⚡ | -6.6 |
| progress_deficit | 25.8% | 25.9% | 24.0% | +0.1 | -1.8 |
| unclear_loss | 19.4% | 13.9% | 14.4% | -5.5 | -5.0 |

**Mean metrics on losses**:

| metric | 031B | **052A** | **052** |
|---|---|---|---|
| mean_ball_x | 0.38 | **0.75** | 0.41 |
| team0_possession_ratio | 0.49 | 0.47 | **0.56** |
| team0_progress_toward_goal | 3.63 | 4.36 | **4.81** |

**机制读解**:
- **052A (refinement only)**: FFN + LayerNorm + residual 让 policy 更 aggressive (mean_ball_x +0.37), 但 ball control 反而 fragile (**possession_stolen +9.6pp**). 学到的是「更冲的 + 更易丢球」组合。R2 风险 (LayerNorm vs PPO 不兼容) 实证。
- **052 (full 031C)**: 进一步加 MHA + 1024→512 merge 收缩 → **control 率最高 0.56 但 wasted_possession +12.5pp**. 学到的是「持球时间更长 + conversion 不行」组合。MHA + merge 收缩没补偿 refinement 的损失, 反而方向相反。

**两个失败模式都不是「学不会」, 而是「学到不同的弱 policy」**:
- 052A: aggressive-but-fragile
- 052: possession-heavy-but-wasteful

### 7.6 关键 lesson

1. **架构 diminishing returns 翻转**: step 1 (+7.7) → step 2 (+2.2) → step 3 (-8 ~ -11)。Diminishing returns 不只是 "增益变小", 还可能 "变负"
2. **小模型 ≠ scaled-down 大模型**: transformer block (FFN + LayerNorm + residual) 在 ~1.5M params PPO 上**不能直接套用**。LayerNorm 改 gradient flow → PPO 的 advantage normalization + grad clip 假设不再成立
3. **"标准 transformer" 在 RL 上不是 free lunch**: 需要 RL-specific 调整 (e.g., ReZero, residual scaling, proper init)
4. **MHA 在小 cross-attn dim 上无效**: 4 heads × 64 dim = head_dim 16 太小, capacity 反而被打散

### 7.7 决策 — 架构 step 3 lane 关闭

- **052A / 052 / 052B(planned MHA-only) 都不再训** — 决定性 REGRESSION
- **架构 axis 暂时无新 step**: 031B 0.882 是当前架构 ceiling, 想突破需要**完全不同的 architecture family** (e.g., LSTM / state-space model / agent-attention 而非 token-attention)
- **Stage 3 H2H 不做** (regression 决定性, ROI 极低; 用户已确认)
- 后续突破 0.90 路径转交其他 lane: 053A (PBRS) / 046E ablation pending / self-play league backlog

### 7.8 Raw recap

```
=== 052A baseline 1000ep top peaks (24 ckpts, parallel-7, 1192s) ===
ckpt-1080 vs baseline: win_rate=0.800 (800W-200L-0T)
ckpt-1140 vs baseline: win_rate=0.799 (799W-201L-0T)
ckpt-1150 vs baseline: win_rate=0.793 (793W-207L-0T)
ckpt-1050 vs baseline: win_rate=0.792 (792W-208L-0T)
ckpt-1090 vs baseline: win_rate=0.789 (789W-211L-0T)
ckpt-1060 vs baseline: win_rate=0.789 (789W-211L-0T)
ckpt-1160 vs baseline: win_rate=0.782 (782W-218L-0T)
ckpt-1100 vs baseline: win_rate=0.776 (776W-224L-0T)
ckpt-890  vs baseline: win_rate=0.771 (771W-229L-0T)
ckpt-970  vs baseline: win_rate=0.770 (770W-230L-0T)

=== 052 baseline 1000ep top peaks (20 ckpts, parallel-7, 1191s) ===
ckpt-870  vs baseline: win_rate=0.774 (774W-226L-0T)
ckpt-1150 vs baseline: win_rate=0.773 (773W-227L-0T)
ckpt-1140 vs baseline: win_rate=0.762 (762W-238L-0T)
ckpt-830  vs baseline: win_rate=0.757 (757W-243L-0T)
ckpt-1160 vs baseline: win_rate=0.756 (756W-244L-0T)
ckpt-520  vs baseline: win_rate=0.754 (754W-246L-0T)
ckpt-490  vs baseline: win_rate=0.753 (753W-247L-0T)
ckpt-800  vs baseline: win_rate=0.748 (748W-252L-0T)
ckpt-810  vs baseline: win_rate=0.747 (747W-253L-0T)
ckpt-530  vs baseline: win_rate=0.746 (746W-254L-0T)

=== 052A capture (500ep, peak ckpt 1080) ===
WR=0.784 (392-108-0)  L step mean=33.1 median=24.0

=== 052 capture (500ep, peak ckpt 870) ===
WR=0.750 (375-125-0)  L step mean=34.7 median=25.0
```

完整 logs: [052A_baseline1000](../../docs/experiments/artifacts/official-evals/052A_baseline1000.log) / [052_baseline1000](../../docs/experiments/artifacts/official-evals/052_baseline1000.log) / [052A_checkpoint1080 capture](../../docs/experiments/artifacts/official-evals/failure-capture-logs/052A_checkpoint1080.log) / [052_checkpoint870 capture](../../docs/experiments/artifacts/official-evals/failure-capture-logs/052_checkpoint870.log)

## 8. 相关

- [SNAPSHOT-031 §12](snapshot-031-team-level-native-dual-encoder-attention.md#12-031-b-激活cross-attention2026-04-18-预注册) — 031B 设计 (cross-attention)
- [SNAPSHOT-031 §13.10](snapshot-031-team-level-native-dual-encoder-attention.md#1310-031b-后续发展路径地图) — TIER 2 多头/transformer/LSTM backlog 来源
- [team_siamese.py](../../cs8803drl/branches/team_siamese.py) — 待修改的 model 文件
- [train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py) — 待修改的训练脚本
- [SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md) / [SNAPSHOT-050](snapshot-050-cross-student-dagger-probe.md) / [SNAPSHOT-051](snapshot-051-learned-reward-from-strong-vs-strong-failures.md) — 并行突破路径，031C 是其中之一
