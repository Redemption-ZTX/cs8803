# SNAPSHOT-043: Frontier Self-Play Pool Fine-Tune

- **日期**: 2026-04-19
- **负责人**:
- **状态**: `043A' / 043B' / 043C'` formal 已全部完成；当前更稳的口径是：三条都已进入 **baseline 1000 的高平台带**（`043B'@440 = 0.904`，`043A'@80 = 0.901`，`043C'@480 = 0.895`，`034E = 0.890`）。其中 `043B'` 保留当前最高的 baseline 尖峰，而 `043C'` 已补完 `failure capture + peer H2H + direct H2H`，现有证据开始**轻微偏向 `043C'` 是更 rounded 的 overall variant**；但 `043B vs 043C = 0.468 / 0.532` 的 direct H2H 仍不够大，不能写成硬替代
- **依赖**: `031B@1220` warm-start；`043A'@80` resume；当前 harder frontier pool 使用 `031B@1220 / 029B@190 / 036D@150`

## 0.1 2026-04-19 实装版重解释：`043A'`

最初的 `043` prereg 是一个更大的想法包：

- trainable = `031A`
- frontier pool = `029B / 025b / 028A / 036D`
- 还默认需要一个全新的 `FrozenTeamCheckpointPolicy` 路径

但当前真正落地时，我们把它收成了更小、也更可实施的版本：

> **`043A' = 031B@1220 warm-start + baseline/diversity curriculum`**

具体定义现在是：

- **trainable**: `031B@1220`（而不是旧 prereg 的 `031A@1040`）
- **opponent pool**:
  - baseline `50%`
  - `031A@1040` `20%`
  - `029B@190` `15%`
  - `036D@150` `15%`
- **sampling**: episode-level fixed opponent（每局只 sample 一次，不做 mid-episode switching）
- **infra**:
  - 复用已存在的 [FrozenTeamPolicy](../../cs8803drl/core/frozen_team_policy.py)
  - 新增 [FrozenSharedCCPolicy](../../cs8803drl/core/frozen_shared_cc_policy.py)
  - 在 [create_rllib_env](../../cs8803drl/core/utils.py) 中显式安装 `baseline + frozen frontier pool`
  - trainer 侧通过 `OPPONENT_POOL_*` env vars 接线

这样做的目的是把它和 [SNAPSHOT-046](snapshot-046-cross-train-pair.md) 分开：

- `046E` 是 **intensity**：单一强 frozen 对手
- `043A'` 是 **diversity**：更宽的 opponent 分布

当前这个 snapshot 之后如果继续前进，应该一律以 `043A'` 这个实现版为准，而不是再回到旧版 `031A + 4-opponent` 全量 prereg。

## 0. Known gap 诊断（动机来源）

**当前所有 lane 训练 opponent_mix = `baseline 100%`** (`BASELINE_PROB=1.0`)。这有两个隐性副作用：

1. **Policy 学到的可能是 baseline-specific exploit**——win 0.86 vs baseline 不等于 win 0.86 vs 任何对手
2. **H2H 数据已证 frontier 之间差距远小于 vs baseline**: `029B@190 vs baseline = 0.868` 但 `029B vs 025b = 0.508` (持平)

这暗示 0.86 baseline WR 里**有相当一部分是 "baseline 弱点利用"，不是 general strong play**。

> **如果让 policy 同时面对 baseline + frontier 对手，强迫学 general strategy，假设：vs baseline WR 反而上涨 (副作用)，因为 policy 真正变强。**

类似自我对弈 / multi-task RL 已知规律：训练分布越广，单 task 表现往往更稳。

## 1. 核心假设

> **`031A@1040` (1000ep 0.860) 当前的 baseline 优势里有 ≥0.05 是 "baseline-specific exploit"。把训练 opponent 改成 `baseline 40% + frontier 60% (mix 029B/025b/028A/036D)`，强迫 policy 学 general strategy，预期：**
>
> - **H1 (主)**: 1000ep vs baseline ≥ 0.87 (+1pp)，因为 policy 真变强
> - **H2 (强成立)**: H2H vs 任一 frontier 持续提升，超过自己原始 H2H
> - **H3 (失败 - mode collapse)**: vs baseline ≤ 0.80，被 frontier 搅乱了 baseline 训练

如果 H1 + H2 同时成立，得到一个**真正的 generally-strong policy**，是项目第一个跨 baseline/peer 双轴都强的 lane。乐观下可能直接突破 0.90 (9/10)。

## 2. 三个候选 trainable 起点

| 起点 | 当前 baseline 1000ep | H2H vs frontier | 备注 |
|---|---:|---|---|
| **031A@1040** | 0.860 | vs 029B 0.552, vs 025b 0.532, vs 028A 0.568 | **当前 SOTA**, 最强的 general-fit 起点 |
| 029B@190 | 0.846 (warmstart 配方) | vs 025b 0.508, vs 028A 0.538 | per-agent SOTA, frontier H2H 偏弱 |
| 036D@150 | 0.860 | vs 029B 0.507 | learned-reward base, frontier H2H 平 |

**首选 031A@1040 作为 trainable** — 它的 H2H 已经是项目最强，self-play 加持有最好的 starting point。

## 3. Opponent pool 设计

### 3.1 Pool 组成

| Opponent | sample 概率 | 角色 |
|---|---:|---|
| baseline | 40% | 保留作业判据轴；不能掉太多 baseline WR |
| 029B@190 | 15% | per-agent SOTA |
| 025b@080 | 15% | per-agent peer-axis 隐性头名 |
| 028A@1060 | 15% | team-level base (架构同 trainable) |
| 036D@150 | 15% | per-agent learned-reward base |

**为什么 baseline 40% 而不是 60%**:
- 60% 太接近现有训练，可能没足够 frontier exposure
- 40% 留一半给 frontier，强迫学 general
- 30% 太少，可能让 baseline WR 退化太多

**为什么不包含 031A 自己**: trainable 是 031A@1040，加自己就是 cyclic self-play (旧 snapshot-043 默认想法)，不利于学新策略

### 3.2 Sampling 策略

每 episode 开始时 sample 一个 opponent，整个 episode 用同一个 (避免 policy switching mid-game 的混乱)。

```python
def policy_mapping_fn(agent_id, *args, **kwargs):
    if int(agent_id) in TEAM0_AGENT_IDS:
        return "default"  # trainable 031A

    # team1 episode-level opponent sampling
    if not hasattr(episode, "_opp"):
        rand = np.random.random()
        if rand < 0.40:
            episode._opp = "baseline"
        elif rand < 0.55:
            episode._opp = "frozen_029B"
        elif rand < 0.70:
            episode._opp = "frozen_025b"
        elif rand < 0.85:
            episode._opp = "frozen_028A"
        else:
            episode._opp = "frozen_036D"
    return episode._opp
```

### 3.3 FrozenTeamCheckpointPolicy adapter

[SNAPSHOT-023](snapshot-023-frozenteamcheckpointpolicy-opponent-adapter.md) 已经设计但**没实施**。这个 snapshot 需要先把 adapter 落地。

接口：
```python
class FrozenTeamCheckpointPolicy(Policy):
    """Wraps an existing trainer checkpoint as a frozen opponent in a
    multiagent_player env. Adapts team-level joint output to per-agent action."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.checkpoint_path = config["checkpoint_path"]
        self.lane_type = config["lane_type"]  # "team_level" or "per_agent"
        self._policy = None  # lazy load

    def _ensure_loaded(self):
        if self._policy is None:
            # Load via deployment module pattern (similar to trained_team_ray_agent)
            ...

    def compute_actions(self, obs_batch, ...):
        self._ensure_loaded()
        # For team_level opponent: feed joint obs (need teammate's), output joint action, take own slice
        # For per_agent opponent: feed own 336-dim obs, output 9-dim action
        ...

    def learn_on_batch(self, samples):
        return {}  # no-op
```

实现在 `cs8803drl/branches/frozen_team_checkpoint.py`。

## 4. 训练超参

```bash
# warmstart from 031A@1040
WARMSTART_CHECKPOINT=ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/.../checkpoint_001040/checkpoint-1040

# Architecture (same as 031A)
TEAM_SIAMESE_ENCODER=1
TEAM_SIAMESE_ENCODER_HIDDENS=256,256
TEAM_SIAMESE_MERGE_HIDDENS=256,128

# Opponent pool (NEW)
OPPONENT_POOL_BASELINE_PROB=0.40
OPPONENT_POOL_FRONTIER=029B,025b,028A,036D    # 各 15%
OPPONENT_FROZEN_CKPT_029B=ray_results/PPO_mappo_029B_.../checkpoint-190
OPPONENT_FROZEN_CKPT_025b=ray_results/PPO_mappo_025b_.../checkpoint-80
OPPONENT_FROZEN_CKPT_028A=ray_results/PPO_team_level_bc_bootstrap_028A_.../checkpoint-1060
OPPONENT_FROZEN_CKPT_036D=ray_results/PPO_mappo_036D_.../checkpoint-150
OPPONENT_FROZEN_LANE_029B=per_agent
OPPONENT_FROZEN_LANE_025b=per_agent
OPPONENT_FROZEN_LANE_028A=team_level
OPPONENT_FROZEN_LANE_036D=per_agent

# v2 shaping (同 031A)
USE_REWARD_SHAPING=1
SHAPING_TIME_PENALTY=0.001
... (same v2 config)

# PPO
LR=1e-4
CLIP_PARAM=0.15
NUM_SGD_ITER=4
TRAIN_BATCH_SIZE=40000
SGD_MINIBATCH_SIZE=2048

# Budget
MAX_ITERATIONS=500            # frontier 难度比 baseline 高，给更多 iter
EVAL_INTERVAL=10
EVAL_EPISODES=50              # 50ep vs baseline only (主作业判据)
CHECKPOINT_FREQ=10
```

总训练时间: 500 iter × ~1.2 min/iter (FrozenPolicy inference 加 ~20% overhead) = ~10h。

## 5. 预注册判据

### 5.1 主判据（成立门槛）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| **vs baseline 1000ep peak** | **≥ 0.86** | **不能退化于 031A 起点**（self-play 不能毁了 baseline 优势） |
| H2H vs 031A@1040 (orig) | ≥ 0.51 | 训练后的 self 比 frozen-self 略强 |
| H2H vs frontier mix (avg) | ≥ 原始 031A 的对应 H2H | self-play 至少不让 frontier H2H 退化 |

### 5.2 突破判据（强成立）

| 项 | 阈值 | 逻辑 |
|---|---|---|
| **vs baseline 1000ep peak** | **≥ 0.88** | self-play 真的让 baseline WR 涨 +2pp |
| H2H vs 029B@190 | ≥ 0.58 | 比 031A 自己的 0.552 再强 +3pp |
| H2H vs 025b@080 | ≥ 0.55 | 比 031A 自己的 0.532 再强 +2pp |
| H2H vs 028A@1060 | ≥ 0.60 | 比 031A 自己的 0.568 再强 +3pp |
| failure capture: baseline 失败结构改善 | wasted_possession ≤ 35% | "更稳的 general policy" 应该不犯同样错 |

### 5.3 失败判据

| 条件 | 解读 |
|---|---|
| vs baseline 1000ep < 0.80 | self-play mode collapse, 把 baseline 优势吃掉了 |
| H2H vs frontier 退化 | over-specialized to specific opponent in pool, 失去 general |
| 训练 reward 不收敛 / 持续震荡 | opponent pool 太难, 起点 031A 不够强 |

### 5.4 Lane 优先级

按预期 ROI：

1. **043A (031A @ baseline40+frontier60)** — 主线，最有希望
2. **043B (029B @ same pool)** — 备选 trainable，看 per-agent 起点是否更适合 self-play
3. **043C (031A @ baseline 60% + frontier 40%)** — 更保守的 mix，看 baseline-保留度

**首轮先跑 043A**：如果 ≥ 0.86 (没退化)，再考虑 043B/C。

## 6. 风险

### R1 — Mode collapse to specific opponent

self-play pool 里有 4 个 frozen opponents，policy 可能 over-fit 到打败"最容易 exploit 的那一个"，对其他 opponents 反而退化。

**缓解**：
- pool 概率均匀分布 (各 15%)
- 监控每 50 iter 的 per-opponent reward 趋势：如果某个 opponent 的 reward 开始飙升（exploit 了），降低它的 sample 概率
- 训练完成后做 H2H **逐个 opponent 对比**

### R2 — Baseline WR 退化

baseline 40% 训练量不一定够维持 0.86 baseline WR。如果 frontier 60% 把 policy 拉去对付强对手，baseline-specific 的 exploit pattern 可能被遗忘。

**缓解**：
- 50ep 内评 keep `EVAL_OPPONENTS=baseline,random` 监控 baseline WR
- 如果 baseline WR 在 iter 100 时 ≤ 0.80，提前 abort 并改 baseline 60%

### R3 — Adapter 实现复杂度

`FrozenTeamCheckpointPolicy` 需要：
- 处理 team-level 和 per-agent opponent 的不同 action 空间
- per-agent opponent 给 team1 的两个 agent 各跑一次 inference
- 适配 RLlib 的 `Policy` 接口

这个 ~200 行新代码，且需要在 multi-worker 环境下正确序列化。

**缓解**：
- 先实现 per-agent only adapter (team_level 暂不支持)
- per-agent only 已经覆盖 029B / 025b / 036D 三个 opponent (75% of frontier exposure)
- team_level (028A) 优先级低，可后续补

### R4 — GPU memory & throughput

FrozenPolicy inference 在 sample worker 上跑，每 step 4 个 opponent inference (双 agent × 2 sides)，可能让 worker throughput 下降 30-50%。

**缓解**：
- `NUM_ENVS_PER_WORKER` 从 5 降到 3，减少单 worker memory 压力
- 接受训练时间从 10h → 12-14h

### R5 — 作业判据偏离

9/10 判据是 **vs baseline + vs random**。self-play 是间接路径——"更强的 policy" 不必然 vs baseline 更强（甚至可能更弱，见 R2）。

**这不是"做错"，是"高方差选择"**。snapshot 的 §5.3 已预先规定 baseline 退化是失败判据。

## 7. 不做的事

- **不用动态 pool**：opponent ckpts 固定，不在训练中加新 ckpt（避免 cyclic self-play 不稳定）
- **不混 reward shaping handoff**：保留 v2 不变（self-play 是唯一变量）
- **不考虑双向 self-play (trainable on both sides)**：复杂度爆炸
- **不在 043 内迭代 mix ratio**：先跑 43A 单一 mix 出结果，再决定是否做 043B/C

## 8. 执行清单

### 8.1 2026-04-19 已落实的 `043A'` scaffolding

当前仓库里已经具备一版可直接启动的 `043A'` 最小实现：

- [FrozenSharedCCPolicy](../../cs8803drl/core/frozen_shared_cc_policy.py)
  - 让 frozen per-agent `shared_cc` / MAPPO checkpoint 能作为 `team_vs_policy` 的 opponent callable
  - 通过两次顺序调用的 teammate-obs adapter，服务 `029B / 036D` 这类 per-agent frontier opponent
- [FrozenTeamPolicy](../../cs8803drl/core/frozen_team_policy.py)
  - reset hook 已泛化为任何 `reset_episode()` 对象，可直接承接 opponent pool
- [create_rllib_env](../../cs8803drl/core/utils.py)
  - 现在支持 `opponent_mix = {baseline_prob, frozen_opponents}`
  - opponent sampling 收成 episode-level fixed pool，而不是旧的逐步随机逻辑
- [train_ray_team_vs_baseline_shaping.py](../../cs8803drl/training/train_ray_team_vs_baseline_shaping.py)
  - 新增 `OPPONENT_POOL_BASELINE_PROB`
  - 新增 `OPPONENT_POOL_FRONTIER_SPECS`
  - 与 `TEAM_OPPONENT_CHECKPOINT` 显式互斥
- [043A' batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_043Aprime_warm_vs_frontier_pool_031B_cross_attention_512x512.batch)
  - warm-start = `031B@1220`
  - pool = `baseline 50% + 031A 20% + 029B 15% + 036D 15%`
  - `NUM_ENVS_PER_WORKER=3`
  - 首轮 verdict budget = `500 iter / 12M steps`

因此当前 `043` 的工程状态已经不再是“adapter 待实施”，而是：

> **`043A'` runnable 且 formal 已完成；以下保留为工程落地记录。**

### 8.2 2026-04-19 正式训练与 follow-up 结果：`043A'` 成为 baseline-oriented frontier 候选

formal run：

- [043Aprime_warm_vs_frontier_pool_031B_formal](../../ray_results/043Aprime_warm_vs_frontier_pool_031B_formal)
- trainable = `031B@1220` warm-start
- budget 实跑完成：`500 iter / 12,000,000 timesteps`

训练内 `checkpoint_eval.csv` 的 `baseline 50ep` 峰值出现在 `checkpoint-90 = 0.960`，因此按 `top 5% + ties + 前后1点` 规则，补做了 15 个点的 `official baseline 1000` 并行复核。关键结果如下：

| checkpoint | official baseline 1000 |
|---|---:|
| **80** | **0.901** |
| 440 | 0.895 |
| 280 | 0.894 |
| 190 | 0.891 |
| 430 | 0.889 |
| 90 / 100 / 110 / 130 | 0.886~0.888 |

最重要的结论不是单点 `0.901` 本身，而是：

- `80 / 190 / 280 / 430 / 440` 都落在 `0.889~0.901`
- 说明 `043A'` 不是偶然尖峰，而是真正把 **training-distribution / diversity curriculum** 推成了一条 baseline 强正号路线

`checkpoint-80` 的 follow-up 也已完成：

- **failure capture 500 vs baseline**:
  - `461W-39L = 0.922`
  - `fast_win_rate = 0.896`
  - `saved_episode_count = 39`
  - `mean episode steps = 42.6`
  - `mean losing steps = 39.5`
- **H2H vs `031B@1220`**:
  - `262W-238L = 0.524`
  - 方向上是正号，但幅度很小，仍更像“窄优势”而不是 peer 轴硬突破
- **H2H vs `034E-frontier`**:
  - `221W-279L = 0.442`
  - 明确负号，说明它还不是整体上压过当前 ensemble 主线的 policy

因此当前最稳的 verdict 是：

> **`043A'` 强力验证了“训练分布升级”这条路线在 baseline 轴上的价值，并给出了 `0.901` 的正式候选点；但 peer 轴证据仍弱于 `034E-frontier`，所以更适合被写成 `baseline-oriented frontier candidate`，而不是新的 overall champion。**

### 8.3 `failure_buckets_v2` 视角：`043A'` 为什么 baseline 很强，但 peer 轴没有一起翻过去

为了避免被 v1 的直觉标签带偏，我们继续用 [failure_buckets_v2.py](../../cs8803drl/imitation/failure_buckets_v2.py) 把 `043A'@80` 与几条核心参照线放到同一口径下对比：

| model | n | defensive_pin | territorial_dominance | wasted_possession | possession_stolen | progress_deficit | unclear_loss | tail_mean_ball_x med | poss med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `043A'@80` | 39 | 35.9% | 38.5% | 43.6% | 35.9% | 15.4% | 7.7% | `+4.115` | `0.473` |
| `031B@1220` | 62 | 43.5% | 43.5% | 37.1% | 27.4% | 25.8% | 19.4% | `+2.401` | `0.452` |
| `031A@1040` | 85 | 47.1% | 50.6% | 42.4% | 32.9% | 15.3% | 12.9% | `+1.372` | `0.500` |
| `034C-frontier` | 64 | 45.3% | 42.2% | 31.2% | 35.9% | 25.0% | 20.3% | `-0.707` | `0.433` |
| `036D@150` | 88 | 54.5% | 55.7% | 38.6% | 38.6% | 21.6% | 10.2% | `-4.895` | `0.448` |
| `029B@190` | 80 | 51.2% | 47.5% | 48.8% | 32.5% | 23.8% | 8.8% | `-3.629` | `0.543` |

这个对照很能解释 `043A'` 这次“看起来有点出奇”的结果：

1. **它确实学到了极强的前场压制与 baseline exploit。**
   - `tail_mean_ball_x median = +4.115`
   - 明显高于 `031B (+2.401)` 和 `031A (+1.372)`
   - 这和它 `official 1000 = 0.901`、`capture 500 = 0.922`、`fast_win_rate = 0.896` 是一致的

2. **但它并没有把 `031A/031B` 系的 conversion 问题彻底解决。**
   - `wasted_possession = 43.6%`
   - 与 `031A (42.4%)` 接近，甚至高于 `031B (37.1%)`
   - 所以它不是“更完整的 031B”，而更像“更前压、更会打 baseline 的 031 系变体”

3. **它的失败模式是双峰混合，而不是单一病灶。**
   - 既有 `wasted_possession` (`43.6%`)
   - 也有 `possession_stolen` (`35.9%`)
   - 说明 diversity curriculum 没把行为收成单一 specialist，而是同时保留了“压住但没收掉”和“被反抢后翻盘”两类败局

4. **它输得反而更“干净”，不是乱输。**
   - `progress_deficit = 15.4%`
   - `unclear_loss = 7.7%`
   - 这两个都低于 `031B / 034C`
   - 因此 `043A'` 不是那种杂乱、不可解释的 policy；它是有明确强项，也有明确短板

所以现在更准确的画像是：

> **`043A'` 不是简单的“更 aggressive 的 031B”，而是一条前场压制极强、baseline exploit 很深、但仍保留 `031A/031B` 系 conversion 漏洞的 curriculum line。**

这也自然解释了 follow-up：

- 对 baseline 很强：`0.901 / 0.922`
- 对 `031B` 只有窄优势：`0.524`
- 对 `034E-frontier` 仍然不够：`0.442`

换句话说，`043A'` 当前更像：

- **baseline 轴上的新强候选**
- 而不是 **peer + baseline 双轴都已坐实的新主线**

### 8.4 下一步优化：从 `043A'` 的 baseline specialist，推向更硬的 peer curriculum

基于上面的 `v2` 画像，`043A'` 的下一步不应该是“原配方再跑一遍”，而应该是**直接修改训练分布**：

- **把 `031A` 从 frozen pool 中拿掉**
  - `031A` 与 `031B` 家族过近，当前看更像是在强化同一类 `wasted_possession` 风格
- **把 frozen `031B@1220` 放进 pool**
  - `043A'` 对 `031B` 只有 `0.524`，因此新 curriculum 应直接把 `031B` 作为更强的同家族 target
- **下调 baseline 比例**
  - `043A'` 的 baseline 很强，但对 `034E` 明显落后，说明 `baseline 50%` 仍然过重

因此当前推荐的两个 follow-up lane 是：

| lane | resume 起点 | baseline | frozen pool |
|---|---|---:|---|
| `043B'` | `043A'@80` | `40%` | `031B 20% + 029B 20% + 036D 20%` |
| `043C'` | `043A'@80` | `35%` | `031B 25% + 029B 20% + 036D 20%` |

两条都已经落实成 batch：

- [043B' batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_043Bprime_resume080_vs_harder_frontier_pool_031B_cross_attention_512x512.batch)
- [043C' batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_043Cprime_resume080_vs_heavier_frontier_pool_031B_cross_attention_512x512.batch)

当前更稳的优化目标也应该改成：

- baseline `1000ep >= 0.89`
- `H2H vs 031B >= 0.55`
- 至少不再被 `034E` 明确压过

### 8.5 2026-04-19 follow-up：`043B'` 把 `043` 从 baseline specialist 推成了 overall frontier 候选

`043B'` 使用的是上面定义的更硬 curriculum：

- resume = `043A'@80`
- baseline `40%`
- frozen pool = `031B@1220 20% + 029B@190 20% + 036D@150 20%`

formal run：

- [043Bprime_resume080_vs_harder_frontier_pool_031B_cross_attention_rerun1](../../ray_results/043Bprime_resume080_vs_harder_frontier_pool_031B_cross_attention_rerun1)

训练内 `checkpoint_eval.csv` 的 `baseline 50ep` 会把 `140 / 260 / 270 / 440` 都抬到 `0.96`，但 `official baseline 1000` 证明这条线**不能直接信单个 online peak**。按 `top 5% + ties + 前后1点` 做 parallel `official 1000` 后，关键结果如下：

| checkpoint | official baseline 1000 |
|---|---:|
| **440** | **0.904** |
| 130 | 0.894 |
| 430 | 0.893 |
| 270 | 0.886 |
| 250 | 0.885 |
| 450 | 0.883 |
| 140 / 280 | 0.874 |
| 150 | 0.877 |
| 260 | 0.868 |

这个 official readout 比 `043A'` 更有信息量：

- `043A'` 的 strongest official 点在早期 `80`
- `043B'` 的 strongest official 点出现在后段 `440`
- 而且 `130 / 250 / 270 / 430 / 440` 形成了一段真实高平台

所以更稳的解读是：

> **把 `031A` 从 pool 拿掉、换进 frozen `031B`、并把 baseline 比例从 `50%` 降到 `40%`，确实把 `043` 从 baseline-only specialist 往更完整的 frontier policy 推了一步。**

`checkpoint-440` 的 follow-up 也已经完成：

- **failure capture 500 vs baseline**:
  - `426W-74L = 0.852`
  - `fast_win_rate = 0.812`
  - `saved_episode_count = 74`
  - `mean episode steps = 41.0`
  - `mean losing steps = 30.9`
- 主要败局标签：
  - `late_defensive_collapse = 37`
  - `low_possession = 28`
  - `opponent_forward_progress = 4`
  - `unclear_loss = 3`
  - `poor_conversion = 2`
- **H2H vs `031B@1220`**:
  - `300W-200L = 0.600`
  - 明确正号，且两侧都 > `0.58`
- **H2H vs `034E-frontier`**:
  - `277W-223L = 0.554`
  - 对当前 ensemble 主线给出真实正优势

这组结果和 `043A'` 的差异很关键：

| lane | official baseline 1000 | H2H vs 031B | H2H vs 034E |
|---|---:|---:|---:|
| `043A'@80` | `0.901` | `0.524` | `0.442` |
| **`043B'@440`** | **`0.904`** | **`0.600`** | **`0.554`** |

也就是说，`043B'` 不只是把 baseline 顶住了，它还把 peer 轴也一起拉起来了。

但 `failure capture 500 = 0.852` 也提醒我们：

- `043B'@440` 仍然是一条 **高上限但不够圆的 policy**
- 它比 `043A'` 更强，但输给 baseline 时更容易集中在
  - `late_defensive_collapse`
  - `low_possession`

因此当前最稳的 verdict 应该写成：

> **`043B'@440` 已经把 `043` 从“baseline-oriented frontier candidate”推进成了“baseline + peer 都真正有竞争力的 overall frontier candidate”。但这里更稳的读法不是“它靠 baseline 1000 的 +0.9pp 已经明确压过 043A'/043C'”，而是“它在同一条 baseline 高平台里，同时额外拿到了正向 peer H2H 证据”。**

### 8.6 2026-04-19 follow-up：`043C'` 不是“更差”，而是更 rounded 的重-frontier 版本

`043C'` 使用的是更激进的 mix：

- resume = `043A'@80`
- baseline `35%`
- frozen pool = `031B@1220 25% + 029B@190 20% + 036D@150 20%`

formal run：

- [043Cprime_resume080_vs_heavier_frontier_pool_031B_cross_attention_rerun1](../../ray_results/043Cprime_resume080_vs_heavier_frontier_pool_031B_cross_attention_rerun1)

训练内 `checkpoint_eval.csv` 的 `baseline 50ep` 最亮点出现在 `checkpoint-490 = 0.980`，并且 `250 / 330 / 370 = 0.960`，看起来像一条**更偏后程成形**的 lane。
但按 `top 5% + ties + 前后1点` 再补 `official baseline 1000` 后，结论明显收紧：

| checkpoint | official baseline 1000 |
|---|---:|
| **480** | **0.895** |
| 380 | 0.893 |
| 260 | 0.890 |
| 240 / 330 | 0.889 |
| 360 | 0.886 |
| 250 | 0.885 |
| 320 | 0.884 |
| 370 / 500 | 0.877 |
| 340 | 0.870 |
| 490 | 0.864 |

如果只看 baseline 轴，这批结果有两个特别关键的信号：

1. **`043C'` 确实不是坏线。**
   - 它仍然能打到 `0.895`
   - 所以更重的 frontier curriculum 本身没有把 lane 直接毁掉

2. **它没有在 baseline 轴上把 `043B'` 明显拉开。**
   - `043B'@440 = 0.904`
   - `043C'@480 = 0.895`
   - 这 `0.9pp` 本身在当前量级上并不构成硬差距
   - 同时 `490` 这个最亮的 online 点在 `official 1000` 上直接掉到 `0.864`

如果只到这里，更稳的解释仍然会是：

> **把 baseline 比例从 `40%` 再压到 `35%`，没有带来额外的 baseline 轴收益；但这并不自动等于 `043C'` 更差。**

不过现在 `043C'@480` 的 peer follow-up 已经补完：

- **failure capture 500 vs baseline**
  - `440W-60L = 0.880`
  - `fast_win_rate = 0.844`
  - `saved_episode_count = 60`
  - 主要败局标签：
    - `late_defensive_collapse = 32`
    - `low_possession = 18`
    - `unclear_loss = 8`
    - `territory_loss = 1`
    - `poor_conversion = 1`
- **H2H vs `031B@1220`**
  - `307W-193L = 0.614`
- **H2H vs `034E-frontier`**
  - `288W-212L = 0.576`

把这些和 `043B'@440` 并排看，会得到一个更有信息量的对照：

| lane | official baseline 1000 | capture 500 | H2H vs 031B | H2H vs 034E |
|---|---:|---:|---:|---:|
| `043B'@440` | `0.904` | `0.852` | `0.600` | `0.554` |
| `043C'@480` | `0.895` | `0.880` | `0.614` | `0.576` |

这意味着：

- `043B'` 保留了 **更高的 baseline 尖峰**
- `043C'` 则在
  - `failure capture`
  - `vs 031B`
  - `vs 034E`

  这三条上都更强

所以现在更稳的解释已经不再是“`043C'` 没补完证据”，而是：

> **`043B'` 更像 baseline 轴尖峰更高的版本，`043C'` 更像更 rounded 的 overall 版本。**

### 8.7 2026-04-19 direct compare：`043B'@440` vs `043C'@480`

为了避免只靠“分别打别人”的间接读法，我们又补了 direct H2H：

- **`043B'@440 vs 043C'@480`**
  - `234W-266L = 0.468`
  - 等价地说：**`043C'@480` 对 `043B'@440` 是 `0.532`**
  - 方向上支持 `043C > 043B`
  - 但幅度只有 `+3.2pp`，在 `500ep` 下仍不到足以写成硬替代的程度

因此当前 `043` 家族最稳的阶段性收口应该写成：

1. **`043B'@440` 与 `043C'@480` 组成当前 `043` 的双主候选**
2. **`043B'`** — baseline 轴尖峰更高
3. **`043C'`** — failure / peer 轴更均衡
4. **`043A'@80`** — 第一代 baseline-oriented frontier candidate

换句话说，`043C'` 的主要价值不是“新冠军点”，而是把 `043` 的 mix-ratio sweet spot 收得更清楚了：

- `baseline 50%`：baseline 很亮，但 peer 轴不足
- `baseline 40%`：更像 baseline 尖峰
- **`baseline 35%`：更像更 rounded 的 peer-aware 平衡点**

> **下面的 Phase 1~4 是最初 prereg 阶段保留下来的历史工程计划。`043A'/043B'/043C'` 的实际实现与结论以上面的 8.1~8.5 为准。**

### Phase 1: Adapter 实施
1. 写 `cs8803drl/branches/frozen_team_checkpoint.py`，per-agent only 优先
2. smoke test: load 029B@190 as frozen, run 1 episode in env
3. unit test: opponent module wraps both per-agent and team-level

### Phase 2: Training script 改写
4. 写 `cs8803drl/training/train_ray_frontier_selfplay.py` (基于 `train_ray_selfplay.py`)
5. wire opponent pool sampling via env vars (`OPPONENT_POOL_*`)
6. 写 batch `soccerstwos_h100_cpu32_team_level_043A_frontier_selfplay_on_031A1040_512x512.batch`

### Phase 3: 训练
7. 先 1-iter smoke 在 GPU 节点
8. 200 iter mini-run 监控 reward 收敛 + per-opponent 分布
9. 如果 200 iter 不崩，500 iter 完整 run

### Phase 4: 评估
10. eval pipeline (post-training pipeline 加 frontier 对手)
11. failure capture 比对 baseline-only 训练的 031A
12. 写 §9 verdict

## 9. 相关

- [SNAPSHOT-023](snapshot-023-frozenteamcheckpointpolicy-opponent-adapter.md) — adapter 原始预注册 (本 snapshot 激活它)
- [SNAPSHOT-018/019](snapshot-018-mappo-v2-opponent-pool-finetune.md) — 旧的 baseline-only opponent pool 实验 (不同 lane 但同精神)
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) — 031A 来源 (主 trainable 起点)
- [rank.md](rank.md) — frontier ckpts 的当前 H2H 数据 (本 snapshot §5.2 阈值依据)
