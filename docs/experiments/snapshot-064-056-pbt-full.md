## SNAPSHOT-064: 056-PBT-full — True Population-Based Training (Tier A3-PBT-full)

- **Date**: 2026-04-20
- **Status**: 预注册 / design review pending (user will discuss with peers before committing to code)

### §0 背景

- 056 series (simplified PBT LR sweep) 已完成: peak 056D `lr=3e-4` combined 2000ep 0.891，marginal tied 031B (+0.009pp，sub-noise)。Peak LR 为 3e-4 (3× baseline)。056A `lr=3e-5` (0.78) / 056B `lr=7e-5` (0.86) 证明小 LR 降 peak。
- **SIMPLIFIED vs FULL 区别**: 056A/B/C/D = 4 条**独立** lane, **NO population exchange, NO HP mutation, NO evolution**。这是 HP sweep, 不是 PBT。
- 真 PBT (Jaderberg 2017 "Population Based Training of Neural Networks"): population of N workers, periodic synchronous exchange (bottom-K copies top-K's weights + mutated HPs), continuous training。在 DeepMind StarCraft / DMLab / Atari 上都 validated 显著增益 (+5-20%)。
- 本 project 尚未真跑过 PBT, 只有 HP sweep。User 确认值得投入 ~1-2 天工程验证。

### §1 假设

**H_064**: 真 PBT (**8-member population on 2 GPUs** + delayed weight exchange + HP mutation) vs 056D `lr=3e-4` independent run, **combined 2000ep peak ≥ 0.905** (+0.014pp vs 056D 0.891, crossing 2σ SE threshold)。

子假设:
- **H_064-a**: weight exchange 让 worst worker 节省重新学 basic control 的 iter 预算, 专注优化 HP。
- **H_064-b**: HP mutation (LR ×0.8 / ×1.25, clip ±0.05) 动态 tune 最优, 比固定单点 HP 好。
- **H_064-c**: 8-member population 才开始足够像真实 PBT，而不是 "最差 1 条抄最好 1 条" 的 noisy adaptive sweep。
- **H_064-d**: 056D 的分化发生较晚（peak 在 1110-1140，056extend 也提示 1140 非饱和），所以 exploit 应该晚启动，而不是前 200-300 iter 就频繁触发。

### §2 设计

#### 2.1 Infrastructure: Ray `PopulationBasedTraining` (方案 A, 2-GPU version)

- 用 Ray 1.4 内置 `ray.tune.schedulers.PopulationBasedTraining`。
- **2 GPU × 8 trials 并发**, fractional GPU: 每 trial `0.25 GPU`。
- 优先级是 **card count > card type**。若能申到 `2×H200` 可直接用，但对本 workload 的关键收益主要来自 **population=8**，不是单卡从 H100 升到 H200。
- Ray trainer 内部处理 weight exchange + HP mutation。

#### 2.2 Population & schedule

- **Population size**: 8
- **Burn-in**: `iter < 500` 禁止 exploit / copy
- **Exchange frequency**: every `250 iter` after burn-in (target windows: `500 / 750 / 1000`)
- **Selection**: truncation selection — bottom 25% (worst 2 of 8) copy top 25% (best 2 of 8)'s weights
- **Exploit gate**: donor 和 target 的 ranking score 差距必须 `>= 0.04` 才允许 copy，避免 1pp 级噪声触发错误 exploit
- **Why delayed exploit**: 056D 的 best 正式点在 `1110-1140`，而且 056extend 继续训练后仍出现更高 inline 点；这说明前期 raw reward 分化太早利用会放大噪声。

#### 2.3 Ranking signal (replaces raw `episode_reward_mean`)

- **No-shape 仍然保留**: `USE_REWARD_SHAPING=0` 是本 snapshot 的前提，因为 `031B-noshape` 与 PBRS 系列都提示 v2 shaping 更像限制项，而不是最优解。
- 但 **ranking 不再直接用 raw `episode_reward_mean`**。
- 每 `100 iter` 跑一次 **baseline-only mini-eval, 100 episodes**。
- 定义:
  - `eval_wr_t =` 第 `t` 次 mini-eval 的 win rate
  - `ranking_score_t = mean(eval_wr_t, eval_wr_{t-1})`
- 若 trial 还只有 1 次 mini-eval，则用当前 `eval_wr_t`。
- tie-break 才看 smoothed `episode_reward_mean`（最近若干 iter 的移动平均），避免 sparse/no-shape 下单次 reward 尖峰主导复制决策。

#### 2.4 HP search + mutation space

| HP | Initial range | Mutation | Justification |
|---|---|---|---|
| `LR` | log-uniform [1e-4, 5e-4] | ×0.8 or ×1.25 | 056 现有证据指向高 LR 区间有利；不再把 population 浪费在 `3e-5 / 7e-5` 这类已知偏低区 |
| `CLIP_PARAM` | uniform [0.10, 0.18] | ±0.02 | 031B uses 0.15；先做窄范围 trust-region search |

Initial population: 8 trials with random samples from these ranges (Ray `tune.loguniform`, `tune.uniform`)。

**Fixed for first full pass**:
- `NUM_SGD_ITER = 4`
- `ENTROPY_COEF = 0.0`

Rationale: 056 真正给出方向证据的是 **LR**，不是一整包 PPO HP；first full pass 先把 mutation 收窄到 `LR + clip`，减少 search noise 与工程风险。

#### 2.5 Architecture + shaping (fixed across population)

- **Arch**: 031B team-level (Siamese + cross-attn, 256/256/256/128 dims)
- **Shaping**: **NO v2** (sparse outcome only) — reward = 2×(WR-0.5)
- **Teacher distill**: NONE (PBT 独立于 distill)
- **Opponent**: `BASELINE_PROB=1.0` (no curriculum)

#### 2.6 训练超参 (per trial)

```bash
# 固定:
TEAM_SIAMESE_ENCODER=1 TEAM_CROSS_ATTENTION=1
USE_REWARD_SHAPING=0
BASELINE_PROB=1.0
TIMESTEPS_TOTAL=50000000 MAX_ITERATIONS=1250
NUM_WORKERS=2 NUM_ENVS_PER_WORKER=2
TRAIN_BATCH_SIZE=20000 SGD_MINIBATCH_SIZE=1024
# Per-trial HP (PBT 管理):
LR, CLIP_PARAM (from search space, mutated every 250 iter after burn-in)
```

Why shrink per-trial footprint:
- 当前 trainer 默认是 `8 workers × 5 envs`, 这是单 lane 正常训练配置，不适合直接复制到 `8-trial PBT`。
- 本 snapshot 的核心是 **population dynamics**, 不是每个 member 都保持 full single-run footprint。

#### 2.7 GPU / node budget

- **2 GPU × 8 trials** (fractional `0.25 GPU` each)
- 预算更像 `17-20h`，而不是原版 `15-17h`
- 单节点若有 `2×H100` 足够；若能拿到 `2×H200` 也可直接上，但不是必要条件

### §3 预注册判据

| 判据 | 阈值 | verdict |
|---|---|---|
| §3.1 marginal: peak ≥ 0.895 | +0.004 vs 056D | PBT has measurable effect |
| §3.2 main: peak ≥ 0.905 | +0.014 | PBT detectable at 2σ SE; **main hypothesis met** |
| §3.3 breakthrough: peak ≥ 0.915 | approach 055 distill | PBT is a path to SOTA |
| §3.4 持平: peak ∈ [0.882, 0.895) | sub-marginal | PBT 不改 peak; HP sweep is optimal already |
| §3.5 regression: peak < 0.870 | PBT 反伤 | exchange disrupts learning |

Secondary metrics (for post-hoc analysis):
- HP trajectory per trial (which HP settings converge to)
- Exchange rate (how often bottom actually copies top)
- Population diversity (std across 8 workers at each iter)
- Ranking stability (mini-eval score gap at each exploit window)

### §4 简化 + 风险 + 降级

#### 4.1 简化 A — Ray built-in vs custom orchestration

- 用 Ray built-in — simpler but fractional GPU 性能仍可能只有 ~70%。
- **风险**: Ray 1.4 PBT 可能有 bug (版本老); trainer 内部 checkpoint transfer 可能有开销。
- **降级**: 若 Ray PBT 崩, 落回 custom orchestration (方案 C, 写手动 peer comparison + ckpt swap 脚本)。

#### 4.2 简化 B — 8-member population (vs Jaderberg's 40+)

- 8 仍远小于经典 PBT 的 40+；它只是从 "最差 1 条抄最好 1 条" 提升到勉强有 population 结构。
- **降级**: 若 peak < 0.895 且 ranking/stability 看起来合理, 扩到 12-16 members（需要更多 GPU 或进一步压 trial footprint）。

#### 4.3 简化 C — HP mutation 只 2 个参数 (LR/clip)

- 其他 HP (`num_sgd_iter`, `entropy`, batch, gamma, lambda) 先不 search。
- **降级**: 若 peak 平, 下一轮 sweep 其他 HP。

#### 4.4 简化 D — NO shaping

- PBT 在 sparse reward 下 ranking 可能 noise 大，尤其前期。
- **缓解**: no-shape 只用于训练目标；selection ranking 改走 periodic mini-eval，而不是 raw reward。
- **降级**: 若 mini-eval ranking 仍几乎随机, 才考虑加 shaping 作为 ranking proxy；这一步不是默认项。

#### 4.5 全程降级序列

| Step | 触发 | 动作 | 增量 |
|---|---|---|---|
| 0 | Default | Ray PBT, 8-member / 2-GPU / no-shape + eval-ranking | base 17-20h |
| 1 | exploit windows 看起来全是噪声 | exploit 再后移 (`750/1000`) or 提高 gate margin | +0.5 day dev |
| 2 | Ray PBT 崩 | 落回 custom orchestration | +1 day dev |
| 3 | step 1/2 都失败 | 再考虑 shaping-based ranking proxy | +17h |
| 4 | step 3 失败 | PBT 路径关闭; HP sweep is optimal |

### §5 不做的事

- 不跟 distill (055v2) 同时跑 (避免 HP landscape 交叉影响)。
- 不 search batch_size / gamma / lambda (先只聚焦 PBT 本身是否有价值)。
- 不在 code 完成 + smoke test 前 launch。

### §6 执行清单

- [ ] 1. Snapshot 预注册 (this file) + **user design review**
- [ ] 2. Phase 1: 写 PBT trainer entry point (~4h)
  - Copy `train_ray_team_vs_baseline_shaping.py` → `train_ray_team_pbt.py`
  - Replace `tune.run(...)` with `tune.run(..., scheduler=PopulationBasedTraining(...))`
  - 配置 HP search space (`param_space={...}`)
- [ ] 3. Phase 2: 2-GPU fractional allocation + checkpoint config (~2h)
  - `resources_per_trial={"gpu": 0.25, "cpu": 2}`
  - Ensure ckpt freq / `keep_checkpoints_num` compatible with PBT
  - Per-trial footprint downsize (`NUM_WORKERS=2`, `NUM_ENVS_PER_WORKER=2`, smaller batch)
- [ ] 4. Phase 3: mini-eval ranking hook (~2h)
  - every `100 iter` run `100ep` baseline eval
  - compute rolling `ranking_score`
- [ ] 5. Phase 4: Smoke test (~2h)
  - 8 trials × 80 iter, verify mini-eval logging works
  - force one exploit window in a shortened test
- [ ] 6. Launch on 1 node / 2 GPU (17-20h)
- [ ] 7. Verdict per §3

### §7 Verdict

_Pending design review + code completion_

### §8 后续路径

- **Outcome A (breakthrough ≥ 0.915)**: PBT is a SOTA path. Scale up: 12-16 members, longer training, combine with distill (055 PBT-distill combo)。
- **Outcome B (persistent ≥ 0.895 < 0.915)**: PBT better than HP sweep by margin. Try 2.5 (bigger population) for diminishing returns check。
- **Outcome C (< 0.895)**: PBT 不值 +engineering cost; confirm HP sweep path is optimal for this setup; close 056-PBT-full line。
- **Secondary**: compare HP trajectory to 056D's fixed `3e-4` — does PBT converge to similar LR, or discover different regions?

### §9 相关

- [snapshot-056](snapshot-056-simplified-pbt-lr-sweep.md) — 056 simplified predecessor
- [snapshot-055](snapshot-055-distill-from-034e-ensemble.md) — 055 distill current SOTA
- [BACKLOG.md](BACKLOG.md) — 056-PBT-full listed under 056 series
- **Theoretical**: Jaderberg et al. 2017 "Population Based Training of Neural Networks" (DeepMind, first PBT paper); Vinyals et al. 2019 "Grandmaster level StarCraft II" (uses PBT at scale); Ray Tune `PopulationBasedTraining` docs at docs.ray.io (release 1.4)
- **Code targets (未创建)**:
  - `cs8803drl/training/train_ray_team_pbt.py`
  - `scripts/eval/_launch_056pbt_full_8member_2gpu.sh`
