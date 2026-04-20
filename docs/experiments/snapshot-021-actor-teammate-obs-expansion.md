# SNAPSHOT-021: Actor Teammate-Obs Expansion

- **日期**: 2026-04-15
- **负责人**:
- **状态**: 原始 `021b` 本地保真诊断为负结果，但归一化 `021b-norm` 在 local true-info 下已恢复为稳定正结果（`checkpoint-450 = 0.780 @ 500ep`）；`021c-B` 首轮 official-aligned 运行已完成并给出中强正结果（最佳 official `baseline 500 = 0.794 @ checkpoint-480`）

## 0. 2026-04-16 更正：当前应拆成 021b / 021c 两步

本 snapshot 原本是一个单一实验：

- 训练时给 actor 增加 `teammate pos+vel + normalized time`
- 再按常规 internal eval / official-style deployment 做选模

但首轮 run 的结果极不正常：

- run: [PPO_mappo_teammate_obs_scratch_512x512_20260416_002424](../../ray_results/PPO_mappo_teammate_obs_scratch_512x512_20260416_002424)
- best internal `baseline 50`: `0.180 @ checkpoint-390`
- best internal `random 50`: `0.300`
- `vs random` 出现大量 tie，表现更像功能性异常，而不是普通意义上的“效果差”

2026-04-16 的诊断结果表明：

- 训练 env 返回的 `info` 结构是**正确的 Ray 多智能体格式**
  - top-level keys = `[0, 1, 2, 3]`
  - `info[0].keys() = ['player_info', 'ball_info']`
- 因此 [extract_player_state_by_agent()](../../cs8803drl/core/obs_teammate.py) **并没有因为顶层 key 结构错误而恒返回空 dict**
- 训练期 entropy 虽然下降，但**没有塌到接近 0**
  - `iter 1: 3.29`
  - `iter 100: 1.99`
  - `iter 390: 1.67`
  - `iter 500: 1.74`
- internal eval / deployment 使用的 teammate-state decoder 误差非常大：
  - `mean_abs_err_per_dim = [15.85, 9.36, 20.18, 15.10]`
  - `mean_l2_err = 35.71`
  - `median_l2_err = 32.30`
  - `p90_l2_err = 61.73`

因此当前最合理的拆分是：

- **021b**：保留“真实 teammate state + normalized time”这套原始设计，但评测不再走 decoder；改为 **local diagnostic eval**，直接从 env `info` 读取真实 teammate state，回答“这个想法本身有没有学到”
- **021c**：重新设计成 **official-aligned** 的可重跑版本，要求训练和部署都只使用 official 推理时真的拿得到的信息；当前主推方案是 **auxiliary teammate prediction head**，再正式检验 A 假设

在此基础上，`021b` 已完成一次本地保真诊断：

- checkpoint: [checkpoint-390](../../ray_results/PPO_mappo_teammate_obs_scratch_512x512_20260416_002424/MAPPOVsBaselineTrainer_Soccer_33d13_00000_0_2026-04-16_00-24-47/checkpoint_000390/checkpoint-390)
- eval mode: `local_true_teammate_info`
- opponent: `ceia_baseline_agent`
- episodes: `200`
- result: `0.135` (`27W-173L-0T`)

这条结果很关键，因为它说明：

- 首轮异常 **不只是** “official-style deployment 用 decoder 补 teammate state” 导致的 eval 偏差
- 即使在“训练语义 = 评测语义”的本地保真条件下，这个 checkpoint 仍然几乎打不赢 baseline

一句话结论先收成：

**`021a` 的异常 run 不能单独否决 A 假设，而原始未归一化 `021b` 表明“直接把真实 teammate+time 拼进 actor obs” 这条实现如果不做量纲对齐，确实学不成。**

随后，围绕 [code-audit-001 §2.6.5](../architecture/code-audit-001.md#265--obs-扩展字段未归一化first-layer-activation-被高量级维度主导) 提出的“tail 量纲失配”主因假设，我们又做了归一化复跑：

- run: [PPO_mappo_teammate_obs_norm_scratch_512x512_20260416_172518](../../ray_results/PPO_mappo_teammate_obs_norm_scratch_512x512_20260416_172518)
- `TEAMMATE_STATE_SCALE = 15,7,8,8`
- best internal `baseline 50`:
  - [checkpoint-450](../../ray_results/PPO_mappo_teammate_obs_norm_scratch_512x512_20260416_172518/MAPPOVsBaselineTrainer_Soccer_d2904_00000_0_2026-04-16_17-25-42/checkpoint_000450/checkpoint-450) = `0.640`

更关键的是，对同一个 [checkpoint-450](../../ray_results/PPO_mappo_teammate_obs_norm_scratch_512x512_20260416_172518/MAPPOVsBaselineTrainer_Soccer_d2904_00000_0_2026-04-16_17-25-42/checkpoint_000450/checkpoint-450) 做 **local true-info** 诊断后，先得到一个 `200` 局的小样本强信号：

- episodes: `200`
- `team0_win_rate = 0.860` (`172W-28L-0T`)

因此，`021` 到目前为止更准确的拆解应是：

- **未归一化 direct-concat**：失败
- **归一化 direct-concat（local true-info）**：成功
- **official-aligned auxiliary route (`021c-B`)**：成功，但当前强度仍低于主冠军线

随后 `021c-B` 的 first-round official-aligned 运行已经完成：

- run: [PPO_mappo_aux_teammate_scratch_512x512_20260416_065850](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850)
- internal `baseline 50` 最好窗口：
  - [checkpoint-450](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000450/checkpoint-450) = `0.860`
  - [checkpoint-490](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000490/checkpoint-490) = `0.860`
- 按 `top 5% + ties + 前后 2 点` 做 official `baseline 500` 复核后，最佳点为：
  - [checkpoint-480](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000480/checkpoint-480) = **`0.794`**

这说明：

- `021c-B` 明确优于 `021b` 的 direct-concat 方案
- 但当前仍未超过 [SNAPSHOT-018](snapshot-018-mappo-v2-opponent-pool-finetune.md) 的 `0.812` 或 [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md) 的 `0.842`
- 因此当前最合理的读法是：
  - **teammate 感知作为 auxiliary 表示学习是有效的**
  - 但它目前还是一条“中强正结果线”，不是新的冠军线

## 1. 背景

[SNAPSHOT-013 §12](snapshot-013-baseline-weakness-analysis.md#12-bc%E2%86%92mappo-数据对-11-的进一步强化2026-04-15-后补) 与 [SNAPSHOT-017 §11](snapshot-017-bc-to-mappo-bootstrap.md#11-failure-bucket-深度分析与判据-verdict) 把 9 条 lane 的 `low_possession` 占失败比锁定在 **22.6% - 28.1%** 的极窄带，形成**跨 intervention 不变量**。根因假设候选有两个：

| 候选 | 机制 |
|---|---|
| **A (obs)** | actor 只看 own_obs（无论 single_player / multiagent_player 模式，actor input 都不含队友 state） → 无法推断该不该补位 |
| **B (reward)** | shaping 三项核心 (`ball_progress / possession_bonus / time_penalty`) 对两个 team0 agent **完全对称** → 两 agent 学同构行为 → positional collapse |

本 snapshot 定义 **A 路线的实验**。B 路线的实验见 [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md)，两条线并行设计、独立验证。

## 2. 关键事实：之前没点明的约束

截至本轮之前，项目所有分析默认假设 actor 能访问队友 state，但实际：

- 所有 lane（v1/v2/v4 PPO 的 single_player + shared policy；MAPPO/Pool/BC 的 multiagent_player + shared actor）的 **actor 输入都只有 own_obs**
- 队友位置/速度**只在 centralized critic 里出现**（MAPPO lane），且仅在训练时使用，inference 时无用
- 队友信号对 actor 的可获得性限于“own rays 打到队友身上”这种间接探测

**这个隐蔽约束是 `low_possession` 22-28% 跨 lane 不变量最可能的根因 A。**

## 3. 假设

### 主假设

如果把队友的 state 字段（位置、速度，可选最近 action）拼进 actor 的 obs，actor 将获得足够信号来做补位判断，`low_possession` 占失败比会从 22-28% 降至 ≤ 15%。

### 备择假设

如果加 teammate obs 后 `low_possession` 仍然 ≥ 20%，说明 **B (reward 对称)** 才是主因，或两个因素都必须动。

## 4. 实验设计

### 4.1 021b：保留原始 teammate+time 设计，但改成本地保真诊断

在 env wrapper 层扩展 per-agent observation：

| 字段 | 维度 | 来源 |
|---|---|---|
| 现有 own rays + own velocity + ball relative + goals | 336 | env 原生 |
| **teammate position** (x, y) | +2 | env info dict |
| **teammate velocity** (vx, vy) | +2 | env info dict |
| **normalized episode time** (`episode_step / max_steps`) | +1 | wrapper 自维护 |
| **teammate last action** (one-hot 27) | +27 (可选) | policy 自维护 |

最小版：加 4 维 teammate pos + vel，再加 1 维 normalized time → actor obs 从 336 → **341**
完整版：再加 27 维 last-action embedding → **368**

`021b` 保留这套原始语义不变：

- 训练仍然看 **真实 teammate state + normalized time**
- 评测不再用 [trained_shared_cc_teammate_obs_agent.py](../../cs8803drl/deployment/trained_shared_cc_teammate_obs_agent.py) 里的 decoder
- 改为 **local diagnostic eval harness**，在本地对局环境里也直接从 env `info` 读取真实 teammate state

这条 lane 的目标不是 official 排名，而是诊断：

- 如果在“训练语义 = 本地评测语义”的条件下，策略恢复正常行为
- 那当前异常就应归因于 deployment/eval 语义错位，而不是 obs 假设本身失效

当前实际结果已经给出：

- [checkpoint-390](../../ray_results/PPO_mappo_teammate_obs_scratch_512x512_20260416_002424/MAPPOVsBaselineTrainer_Soccer_33d13_00000_0_2026-04-16_00-24-47/checkpoint_000390/checkpoint-390) 在 `021b` 本地保真诊断下对 baseline 仅有 `0.135 @ 200 ep`
- 因此 `021b` 已经从“待判断的诊断 lane”变成了**负结果**

这意味着：

- decoder 误差确实是 `021a` 的一个问题
- 但它**不是唯一问题**
- 更大的问题是：**最朴素的 true-info teammate+time obs 直拼方案，本身就没有学出有效策略**

### 4.2 021c：official-aligned 重跑版本

`021c` 的目标是让：

- 训练时 actor 用到的 teammate/time 信号
- 与 official evaluator 的 `act({0: obs0, 1: obs1})` 语义完全对齐

因此 `021c` 的硬约束必须写死成：

- **训练时任何新增信号，都必须能在 official `AgentInterface.act(obs)` 调用里用同样方法得到**
- 也就是说，`021c` **不能**再依赖 env `info` 真值
- 也不能再依赖“训练看真值、部署看重建值”的跨语义桥接

按这个约束，`021c` 的合法实现只有三类：

| 方案 | 训练期输入/监督 | 推理期可用信号 | 对 A 假设的检验强度 |
|---|---|---|---|
| `021c-A` 更好的 decoder | `decoder(raw obs)`；训练和推理都用同一 decoder | `raw obs -> decoder(raw obs)` | 取决于 decoder 精度 |
| `021c-B` auxiliary teammate prediction head | actor 仍只吃 `raw obs`；另加辅助头，用 `info` 里的 teammate state 当监督标签 | `raw obs`；推理时只用 actor 主干内部表示 | **强，且最符合 official 约束** |
| `021c-C` 不扩展 obs，只改模型结构 | `raw obs` | `raw obs` | 弱，不真正直接测 A |

当前推荐明确收敛到 **`021c-B`**：

- 训练时不把真实 teammate state 直接塞进 actor 输入
- 而是在共享表示上加一个 auxiliary prediction head，监督“从我的 `raw obs` 预测 teammate state”
- 监督标签可以来自训练 env 的 `info`
- 推理时 auxiliary head 可以完全不用，submission 只依赖 `act(obs)` 内部可得信号

这条路线的好处是：

- 完全满足 official API 约束
- 不再赌当前误差很大的 decoder
- 仍然是在测“teammate 感知能否被内化到 policy 表示中”

这条 lane 才承担正式 official `baseline 500` 复核的职责。

### 4.3 021b / 021c Warm-start：都无法使用

无论是 `021b` 的真实 teammate+time obs，还是 `021c` 的 official-aligned teammate signal，只要 actor 输入维度改变，现有 checkpoint 的 policy network 第一层权重都不兼容。必须 **from scratch 重训**。

这是本 lane 的最大成本：~8h GPU + 可能多 seed。

### 4.4 021c 训练配置

最接近 [SNAPSHOT-014 MAPPO+shaping-v2](snapshot-014-mappo-fair-ablation.md) 的 scratch 设置，只改 obs：

- 训练脚本：基于 [train_ray_mappo_vs_baseline.py](../../cs8803drl/training/train_ray_mappo_vs_baseline.py) 改造
- actor 主干输入保持 official 可得的 `raw obs`（336 维）
- 新增 auxiliary prediction head：从 actor 共享表示预测 teammate state
- 训练时用 env `info` 里的真实 teammate state 作辅助监督标签
- 推理时不依赖 `info`，也不要求 auxiliary head 输出
- `variation = multiagent_player`, `multiagent = True`
- `FCNET_HIDDENS = 512,512`
- `custom_model = shared_cc_model`（CC input 不变；actor 主干 obs 语义保持 official 对齐）
- `gamma = 0.99, lambda = 0.95`
- shaping：**v2 不变**（保持和 MAPPO+v2 lane 唯一差别是 obs）
- 500 iter, ~20M steps, ~8h H100

#### 4.4.1 Auxiliary teammate prediction head 具体规格

为避免歧义，预注册下列设计参数：

**架构**：
- 在 actor 共享 trunk 的最后一层 hidden state（512 维）后挂一个 **2 层 MLP** 辅助头
  - hidden_size = 128
  - 输出维度 = 4（teammate `x, y, vx, vy`）
- 推理时**完全不用** auxiliary head，submission wrapper 只前向 actor 主干

**Label 归一化**（明确设计选择）：
- teammate state 的 4 维量纲不同：`(x, y)` 在 field 半径 ~15 / 7 单位范围，`(vx, vy)` 在 ~5 单位范围
- 直接对原始单位做 MSE 会让 position 维度的误差主导梯度，velocity 学不动
- **本设计默认对 label 做 per-dim 归一化**：用固定尺度 `scale = [15.0, 7.0, 5.0, 5.0]` 除一遍
  - 训练时：`label_norm = true_state / scale`，aux head 输出也直接对齐归一化空间
  - 监控时：`mae_per_dim` **以归一化后单位记录**，并同时输出**反归一化后的 field-单位 mae**（×scale 即可），方便人读
- **如果**未来发现归一化常数不合理（例如 vx 实际跑到 ±8），按一次离线统计重新校准 scale；不在训练 loop 里在线估计

**损失**：
- `aux_loss = MSE(pred_teammate_state_norm, true_teammate_state_norm)`
- 真值标签从 `extract_player_state_by_agent(info)` 抽取（仅训练期）
- 缺标签时（reset / info 缺失）该 sample 的 aux_loss 跳过，PPO loss 不变

**联合训练**：
- 总 loss = `ppo_loss + aux_weight * aux_loss`
- **aux_weight 默认 = 0.1**（让 aux 贡献和 PPO 主梯度的量级大致 balanced）
- 如果训练前 100 iter PPO entropy 异常下降 ≥ 50%，调低 aux_weight 到 0.05 重跑

**Warm-up**：
- 不做 separate aux pretraining；从 iter 0 起 joint training
- 这避免引入"aux head 训练阶段不学 PPO"的额外变量

**监控指标 + 软判据**（写入训练 log；**不作为硬 gate**）：
- `aux_train_mse_norm` 每 iter 平均（归一化空间）
- `aux_train_mae_per_dim_field`（4 维分别，反归一化回 field 单位）

期望区间（**diagnostic target，非必通门**）：

| 时间点 | 期望 mae_per_dim 量级（field 单位） |
|---|---|
| iter 100 | 进入 ≤ 5 量级（field 半径的 1/3） |
| iter 300 | 进入 ≤ 2 量级（field 半径的 1/7） |

**判读规则**：
- 落入期望区间 → aux head 正常 internalize 了 teammate 感知，配合主 PPO 一起看
- 高于期望但 ≤ 10 → aux head 学得偏弱，但**不直接判失败**——主判据（500-ep WR ≥ 0.81 + low_poss ≤ 15%）仍然是最终裁决
- **若持续 ≥ 10**（例如 iter 200 还在 12+）→ aux 信号几乎没在表示里 internalize，**判为高风险信号**——可调 aux_weight 或检查 label 流；但**不强制中断**当前 run

这样 aux 监控承担**早期告警**而不是 must-pass gate——避免在阈值校准不充分的情况下提前 kill 一个其实有效的 run。

**Smoke 验证**：
- 1-iter smoke 必须打印：
  - actor obs shape = 336（确认未扩展）
  - aux_head output shape = 4
  - 一个 batch 的 `aux_mae_per_dim` 数值（粗略对齐到 < field radius 量级即可）

#### 4.4.2 工程改动清单

- 新建 [cs8803drl/branches/teammate_aux_head.py](../../cs8803drl/branches/teammate_aux_head.py)：定义 `SharedCCWithTeammateAuxModel`，继承 `SharedCentralCriticTorchModel` + 加 aux head + 加 aux loss callback
- 修改 [train_ray_mappo_vs_baseline.py](../../cs8803drl/training/train_ray_mappo_vs_baseline.py)：加环境变量 `AUX_TEAMMATE_HEAD=1` + `AUX_TEAMMATE_WEIGHT=0.1`，触发使用新 model
- **不改** `obs_teammate.py` 或 `create_rllib_env`——021c 不动 obs space
- 新 batch 脚本 `scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_aux_teammate_scratch_512x512.batch`

### 4.5 当前工程含义：旧实现对应的只是首轮异常原型

- 已实现的部分：
  - [cs8803drl/core/utils.py](../../cs8803drl/core/utils.py) 的 `create_rllib_env` 已加 obs wrapper 分支（`OBS_INCLUDE_TEAMMATE` + `OBS_INCLUDE_TIME`）
  - [cs8803drl/core/obs_teammate.py](../../cs8803drl/core/obs_teammate.py) 已从 env info 提取 teammate state，并追加 normalized episode time
  - [cs8803drl/deployment/trained_shared_cc_teammate_obs_agent.py](../../cs8803drl/deployment/trained_shared_cc_teammate_obs_agent.py) 已实现 official-style deployment wrapper
- 首轮原型已暴露的问题：
  - 训练时 actor 看到的是真实 teammate state
  - eval/deployment 时 actor 看到的是 decoder 重建的 teammate state
  - 当前 decoder 误差过大，导致 internal eval 不可信
- `021b` 要解决的核心：
  - **在不改原始 obs 设计的前提下，先做保真诊断**
  - 不再允许“训练看真值、评测看重建值”
- `021c` 要解决的核心：
  - **训练语义 = official 推理语义**
  - 任何 submission 时用到的信号都必须来自 `act(obs)` 当下可得信息
  - 若需要用 `info`，它只能作为训练期 auxiliary supervision label，不能直接进入推理输入

### 4.6 021b / 021c 评估协议

- `021b`：
  - 不走 official evaluator
  - 使用 **local diagnostic eval harness**，在与训练一致的真实 teammate/time 语义下评测
  - 只用于判断“想法本身有没有学到”，不参与主排名
- `021c`：
  - 训练内 `baseline 50` 筛选
  - `top 5% + ties → baseline 500` 官方复核
  - 最终 1-2 shortlist 补 `random 500`
  - 对 500-ep best ckpt 做 save-all failure capture，重点看 `low_possession` 占比

## 5. 预声明判据

### 5.1 021c 主判据

**500-ep 官方 WR vs baseline ≥ 0.81**

这个阈值和 [SNAPSHOT-018 §5.1](snapshot-018-mappo-v2-opponent-pool-finetune.md#51-主判据胜出必过) 一致，至少追平 opponent-pool lane 的 `0.812`。不要求直接超过 BC SOTA `0.842`（BC 已包含 warm-start 优势）。

### 5.2 021c 机制判据（本 lane 核心）

**`low_possession` 占失败比 ≤ 15%**（从 22-28% 带打到 baseline BvB 的 0% 附近）。

这是根因 A 假设的直接检验：

- 若 `low_poss` 降到 ≤ 15% → A 假设成立，后续 obs / auxiliary 表示路线值得继续推进（如更强的 teammate prediction target、历史窗口、可学习压缩等）
- 若 `low_poss` 还在 20-28% → A 假设被否决，obs 不是主因，问题更可能在 reward 对称或别的维度

### 5.3 021b 诊断判据

`021b` 不承担 official 排名，因此不设项目级 SOTA 阈值。它只回答一个问题：

- 如果把评测语义也改回“真实 teammate state + time”
- 当前 lane 是否还会表现为明显的功能性异常

最关键的观察量是：

- `vs random` 的 tie 是否仍然异常偏高
- `vs baseline/random` 的 local eval 是否显著高于当前 decoder 版内部结果

如果 `021b` 的本地保真评测恢复出**非退化行为**，就足以说明当前首轮 run 的主要问题来自 deployment/eval 失配。

### 5.4 021c 失败情形

| 触发条件 | 解读 |
|---|---|
| 500-ep WR < 0.75 | official-aligned teammate 表征不够表达队友信息，或 auxiliary loss 路线没有成功把 teammate 感知内化到表示 |
| `low_poss` ≥ 20% 且 WR < 0.80 | A 假设被否决，不值得继续 obs 扩展；转而优先 B 假设 |
| 训练前 100 iter 出现 NaN 或 entropy 崩 | obs wrapper 或 eval 对齐实现有 bug，先修工程再重跑 |

### 5.5 当前首轮异常 run verdict

- 当前 run 的异常结果**不足以否决 A 假设**
- 当前 run 更像是在暴露一个**训练/部署语义错位**问题
- 因此当前正式 verdict 是：
  - **首轮异常已诊断，但实验结论无效**
  - 需要进入 `021b`（保真诊断）与 `021c`（official-aligned 重跑）

## 6. 风险

### R1 — obs 信号是否足够表达“队友正在干什么”

只加 teammate pos + vel + normalized time 能让 actor **感知**队友，并区分比赛前后段，但可能仍不足以让 actor **预测**队友即将做什么。如果完整版（+27 维 last-action）才管用，首轮最小版会失败，给出错误的“A 假设否决”信号。

缓解：首轮失败后，开第二轮加 last-action embedding 再跑一次。只有两轮都失败才能断定 A 假设错。

### R2 — teammate/time 信号的语义一致性

需要验证 soccer_twos env 的 info dict 在每一步都能稳定提供 teammate pos/vel，同时必须确认训练路径与评测路径使用的是同一种 teammate/time 语义。

2026-04-16 的补充结论：

- 训练 env 的 `info` 顶层结构已确认**不是这里的 bug**
- 真正的风险点是：**训练用真实 teammate state，而部署只能重建 teammate state**
- 因此本风险项已经从“info 可用性”升级为“**训练/部署语义一致性**”

缓解：

- 1-iter smoke 阶段打印 info 结构，确认 top-level keys 与 `player_info` payload 正常
- `021b`：本地评测时直接打印 actor obs 最后 5 维，确认训练路径与本地诊断路径使用的是**同一种真实 teammate/time 信号**
- `021c`：将“official-available”定义严格限制为 `act(obs)` 当下可得信号；若使用 `info`，只能作为 auxiliary label，而不能直接进入推理输入

### R2.1 — official API 下无 `info` 访问

official evaluator 的约束比“尽量对齐”更硬：

- `AgentInterface.act()` 只接收 `observation`
- evaluator 主循环只调用 `act({0: obs0, 1: obs1})`
- `load_agent()` 在构造 agent 后会立刻 `env.close()`

所以：

- `021b` 允许训练和本地诊断都直接使用真实 `info`
- `021c` 则必须训练时也只使用推理时可得的信号

如果最终出现：

- `021b` 恢复非退化行为
- 但 `021c` 没有恢复

那么结论不是“A 假设错了”，而是：

- **teammate 感知本身有学习价值**
- 但它**不能通过当前 observation 提取路径被可靠利用**
- 下一步应转向更强的 model structure 或 auxiliary-loss 路线，而不是简单 obs wrapper

### R3 — from-scratch 训练的噪声

单 seed 可能给出 `0.75~0.82` 的任意点估计。如果实测落到 `0.80`，不知道是“obs 有用但小增量”还是“seed luck”。

缓解：至少跑 **2 seed**，对比 MAPPO+v2 scratch (`0.786`) 的 baseline 做 paired test。

### R4 — 训练成本高

`8~12h GPU × 2 seed = 16~24h GPU`。对比 [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md) 的 warm-start fine-tune（~5h）显著更贵。

**推荐执行顺序**：先跑 [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md)（便宜），若 `022` 成功（`low_poss` 降到 ≤ 15%）→ A 假设可能不必测；若 `022` 失败 → `021c` 值得开。`021b` 只在需要分辨“训练没学到”与“评测没对齐”时启动。

## 7. 不做的事（明确边界）

- 不改 reward shaping（v2 冻结）——保持和 [SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md) scratch 基线唯一差别是 obs
- 不改模型主干结构（512×512 + CC 不变）
- 不改 `gamma / lambda / batch / 训练量`
- 不把 `021b` 的 local 诊断结果直接当成 official 排名
- 不继续沿用当前 decoder 版 internal eval 作为正式结论依据
- 不把 `021b` 的真实-info 本地结果包装成 official 可提交结果

## 8. 与 [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md) 的关系

两个 snapshot 独立验证根因 A / B，决策矩阵如下：

| 021c 结果 | 022 结果 | 诊断 | 下一步 |
|---|---|---|---|
| `low_poss ≤ 15%` | `low_poss ≤ 15%` | A 和 B 都是真因，可独立缓解 | 选便宜的（022）作主力 |
| `low_poss ≤ 15%` | `low_poss` 未降 | **A 是主因** | 继续 obs 方向，放弃 reward 对称修复 |
| `low_poss` 未降 | `low_poss ≤ 15%` | **B 是主因** | 021 的 obs 路径放弃 |
| 都未降 | | 两假设都不单独成立 | 合并（021 obs + 022 reward）同时做 |

`021b` 不参与这个矩阵；它只是帮助我们判断当前首轮异常到底是“训练没学到”，还是“评测没对齐”。

## 9. 相关

- [SNAPSHOT-013: baseline weakness analysis](snapshot-013-baseline-weakness-analysis.md) §11 / §12
- [SNAPSHOT-014: MAPPO 公平对照](snapshot-014-mappo-fair-ablation.md)
- [SNAPSHOT-017: BC→MAPPO bootstrap](snapshot-017-bc-to-mappo-bootstrap.md) §11
- [SNAPSHOT-022: agent-id asymmetric shaping](snapshot-022-role-differentiated-shaping.md)
- [code-audit-001 §2.6.3](../architecture/code-audit-001.md#263--official-api-推理时无-info-访问) — official API 无 `info` 访问
- [code-audit-001 §2.6.4](../architecture/code-audit-001.md#264--train-用-env-info-真值eval-用-lossy-decoder真实风险但不是当前首轮失败的唯一主因) — decoder 分裂（真实风险，但现已降级为次因）
- [code-audit-001 §2.6.5](../architecture/code-audit-001.md#265--obs-扩展字段未归一化first-layer-activation-被高量级维度主导) — 当前最强主因假设：unnormalized teammate tail dominance

## 10. 首轮异常原型已完成内容

1. 扩展 [cs8803drl/core/obs_teammate.py](../../cs8803drl/core/obs_teammate.py) 的 obs wrapper：在 teammate 4 维之外再追加 1 维 normalized episode time
2. 修改 [cs8803drl/core/utils.py](../../cs8803drl/core/utils.py) 的 `create_rllib_env`，按 `OBS_INCLUDE_TEAMMATE` 与 `OBS_INCLUDE_TIME` 环境变量开启
3. 完成 smoke：obs 维度从 `336 -> 341`
4. 新 batch 脚本 [soccerstwos_h100_cpu32_mappo_teammate_obs_scratch_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_teammate_obs_scratch_512x512.batch)
5. 完成首轮训练 run：[PPO_mappo_teammate_obs_scratch_512x512_20260416_002424](../../ray_results/PPO_mappo_teammate_obs_scratch_512x512_20260416_002424)
6. 完成首轮异常诊断：
   - info 顶层结构正常
   - entropy 无“塌到 0”的证据
   - deployment decoder 误差极大，当前 internal eval 不可信

## 11. 021b / 021c 下一步执行清单

1. 实现 `021b` 的 local diagnostic eval harness：
   - deployment wrapper 不再用 decoder
   - 本地对局环境里直接从 env `info` 读取真实 teammate state
2. 用 `021b` 对当前 lane 做保真诊断，判断“真实 teammate/time 语义下是否恢复非退化行为”
3. 设计并实现 `021c-B`：
   - actor 主干只吃 official `act(obs)` 可得的 `raw obs`
   - 新增 auxiliary teammate prediction head
   - 训练时用 `info` 里的真实 teammate state 作 supervision label，但不把它直接塞进推理输入
4. 为 `021c` 新建 smoke：
   - 打印 auxiliary label 与 prediction 的数值范围
   - 验证训练路径与 official-style deployment 路径的一致性
5. 重跑 `021c` 500 iter × 2 seed 训练
6. `top 5% + ties → baseline 500` 选模

## 12. 首轮异常 run 完整 verdict（append-only, 2026-04-16）

本节锁定首轮 `PPO_mappo_teammate_obs_scratch_512x512_20260416_002424` 的负结果，作为后续 021b/021c 设计依据。**不修改 §1-§9 的预注册内容。**

### 12.1 训练数据 dump

| 指标 | 值 |
|---|---|
| 训练总 iter | 500 |
| 训练 timesteps | ~20M |
| best_internal `baseline 50` | **0.180** @ ckpt 390 |
| best_internal `random 50` | **0.300** (15W-11L-**24T**) @ best ckpt |
| iter≥100 `baseline 50` mean | ≈ 0.07 |
| iter≥100 `baseline 50` max | 0.18 |
| best_reward_mean | -1.7068 |
| entropy 起始 / 末段 | 3.29 → 1.74（无 collapse）|
| iter 452/456 出现 `kl=inf, total_loss=inf` | 是（瞬时数值抖动，非全程崩坏）|

### 12.2 四个候选诊断的检验结果

| 候选 | 验证结果 | 是否成立 |
|---|---|---|
| (a) info 顶层 keys 错位 → teammate state 恒零 | 实测 `info top_level_keys=[0,1,2,3]`，`info[0].keys()=['player_info','ball_info']` | ❌ 否决 |
| (b) Entropy collapse → policy 退化为单 action | iter 1→500: 3.29→1.74，下降但远高于 0 | ❌ 否决 |
| (c) **Train-eval 表征分裂**：训练读 info 真值，eval 走 decoder 拟合 | decoder offline 测量：mean_l2_err = **35.71**, mean_abs_err = `[15.85, 9.36, 20.18, 15.10]`，相对 field radius ~15 等于 **predict ≈ random** | ✅ 成立，但不是唯一主因 |
| (d) **Obs tail 未归一化**：341 维 tail 量级远大于原始 `own_obs` | smoke 显示原始 `336` 维 own_obs 落在 `[0,1]`；而 teammate tail 直接喂 field 单位，`x≈±15, y≈±7, v≈±8`，训练期 first-layer 存在明显 activation dominance 风险 | ✅ **当前最强主因假设** |

### 12.3 验证总结

首轮异常的解释需要从“单一主因”收紧成“**主因优先级排序**”：

1. **当前最强主因假设是 obs tail 未归一化**
   smoke 已确认训练期 actor 确实吃到真实 teammate state + time，但这些新增 tail 的量级是：
   - 原始 `own_obs`：抽样观测全部落在 `[0,1]`
   - teammate `x/y/vx/vy`：直接使用 field 单位，约 `±15 / ±7 / ±8 / ±8`
   - time：step 1 时仅 `1/1500 ≈ 0.00067`

   这意味着新增的 5 维 tail 与原始 336 维特征之间存在明显的动态范围错配。对于 scratch PPO 来说，这会让第一层更容易被高量级的 teammate 位置/速度维度主导，进而压制原始 rays / ball / opponent 相关信号。

2. **Train-eval 表征分裂是真问题，但现在更像次因**
   decoder 误差仍然非常大（`mean_l2_err = 35.71`），因此 official-style deployment 路径当然不可信；但 `021b` 的 local true-info 诊断只有 `0.135`，说明即使把评测改回真实 teammate state，策略也没有恢复。这把 decoder 分裂从“唯一主因”降成了“存在但非主因”的位置。

3. **obs 假设没有被整体否决，但“直接拼 raw teammate+time”这条具体实现路线失败**
   当前数据支持的最稳结论不是“teammate 感知完全没用”，而是：**最朴素的 direct-concat 版本在未归一化条件下没有学成。**

### 12.4 verdict 落定

- **首轮 run 的 `0.18` baseline WR 不能用来否决根因 A 假设**
- 但原因已不再表述为“只是 train-eval decoder bug”，而应改成：
  - 训练 wrapper 读 `info` 正常
  - decoder 分裂存在，但不是唯一主因
  - **当前最强主因假设是未归一化 teammate tail 导致 first-layer activation dominance**
- 因此 `021b` 的下一步不是继续复读原始配置，而是：
  - **优先做一个归一化版重跑**（初始建议 `scale=[15,7,8,8]` 的保守 field-scale）
  - 再用本地 true-info diagnostic 检验 direct-concat 路线是否恢复
- `021c` 的意义也随之更清楚：
  - 它不再是“修 decoder”
  - 而是**换一条更合理的 teammate 感知学习路径（auxiliary head）**，继续检验 A 假设

### 12.5 与 [code-audit-001 §2.6.4](../architecture/code-audit-001.md#264--train-用-env-info-真值eval-用-lossy-decoder表征分裂破坏训练) / [§2.6.5](../architecture/code-audit-001.md#265--obs-扩展字段未归一化first-layer-activation-被高量级维度主导) 的双向引用

本 snapshot 的首轮异常现在拆成两层：

- audit §2.6.4 记录 **decoder 分裂** 这个真实存在但已降级的 train-eval 风险
- audit §2.6.5 记录 **obs tail 未归一化** 这个当前更强的训练期主因假设

两者都来自本 snapshot 的具体数据，互相 cross-link 用于防止未来 lane 再次把“代理信号保真度”和“输入量纲匹配”混在一起。

### 12.6 021b 本地保真诊断结果（2026-04-16 后补）

在 [evaluate_teammate_obs_local.py](../../scripts/eval/evaluate_teammate_obs_local.py) 下，对 [checkpoint-390](../../ray_results/PPO_mappo_teammate_obs_scratch_512x512_20260416_002424/MAPPOVsBaselineTrainer_Soccer_33d13_00000_0_2026-04-16_00-24-47/checkpoint_000390/checkpoint-390) 做 `local_true_teammate_info` 诊断：

| 指标 | 值 |
|---|---|
| opponent | `ceia_baseline_agent` |
| episodes | `200` |
| `team0_wins` | `27` |
| `team1_wins` | `173` |
| ties | `0` |
| `team0_win_rate` | **0.135** |
| `episode_steps_all` | mean `100.1`, median `63.0`, p75 `121.0`, min `8`, max `792` |
| `episode_steps_team0_win` | mean `106.5`, median `68.0`, p75 `125.0`, min `8`, max `325` |
| `episode_steps_team1_win` | mean `99.1`, median `63.0`, p75 `117.0`, min `10`, max `792` |

这条结果把 `021` 的解释再收紧了一步：

- `021a` 的差结果不能仅归因于 decoder-based deployment/eval
- `021b` 已经说明：即使评测也直接使用真实 teammate state，当前这条训练出来的策略仍然非常弱

因此 `021` 当前更准确的判断是：

- **训练-评测语义分裂是真问题**
- 但**训练方案本身也没有学起来**
- 所以 `021c` 的意义不再是“修正 021a 的 eval 偏差”，而是**换一条更合理的 teammate 感知学习路径（auxiliary head）重新检验 A 假设**

### 12.7 021c-B 首轮 official `baseline 500` 结果（2026-04-16 后补）

`021c-B` 的首轮 official-aligned run 为：

- run: [PPO_mappo_aux_teammate_scratch_512x512_20260416_065850](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850)

训练内 `baseline 50` 曾在后段给出较高窗口：

- [checkpoint-450](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000450/checkpoint-450) = `0.860`
- [checkpoint-490](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000490/checkpoint-490) = `0.860`

按 `top 5% + ties + 前后 2 点` 规则，对 `400-500` 的 `11` 个 checkpoint 做 official `baseline 500` 复核，结果如下：

| checkpoint | official `baseline 500` |
|---|---:|
| [checkpoint-400](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000400/checkpoint-400) | `0.760` |
| [checkpoint-410](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000410/checkpoint-410) | `0.744` |
| [checkpoint-420](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000420/checkpoint-420) | `0.792` |
| [checkpoint-430](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000430/checkpoint-430) | `0.776` |
| [checkpoint-440](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000440/checkpoint-440) | `0.764` |
| [checkpoint-450](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000450/checkpoint-450) | `0.792` |
| [checkpoint-460](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000460/checkpoint-460) | `0.772` |
| [checkpoint-470](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000470/checkpoint-470) | `0.764` |
| [checkpoint-480](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000480/checkpoint-480) | **`0.794`** |
| [checkpoint-490](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000490/checkpoint-490) | `0.762` |
| [checkpoint-500](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000500/checkpoint-500) | `0.778` |

这批 official `500` 的读法和 training-internal `50` 不一样：

- internal `50` 把后段窗口高估到了 `0.84~0.86`
- official `500` 把整条线收回到更可信的 `0.76~0.79` 区间
- 所以 `021c-B` 当前更像一条**中高位平台线**，而不是冠军级突破线

当前最准确的结论是：

- `021c-B` 明确优于 `021b` 的 direct-concat 方案
- 但它仍未超过 [SNAPSHOT-018](snapshot-018-mappo-v2-opponent-pool-finetune.md) 的 `0.812` 或 [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md) 的 `0.842`
- 因此它当前应被视为：**A 假设的正结果证据，但不是新的主冠军线**

### 12.8 021c-B failure capture：`420 / 450 / 480`

对 official 最强的三个点做 `baseline 500` failure capture：

| checkpoint | official `baseline 500` | failure capture |
|---|---:|---:|
| [checkpoint-420](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000420/checkpoint-420) | `0.792` | `0.756` |
| [checkpoint-450](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000450/checkpoint-450) | `0.792` | `0.786` |
| [checkpoint-480](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000480/checkpoint-480) | **`0.794`** | **`0.804`** |

按本次保存 episode 数重新对齐 primary label 后，失败结构如下：

| checkpoint | `late_defensive_collapse` | `low_possession` | `poor_conversion` | `unclear_loss` | verdict |
|---|---:|---:|---:|---:|---|
| `420` | `54/122` | `35/122` | `8/122` | `16/122` | 可以放掉 |
| `450` | `52/107` | `30/107` | `15/107` | `9/107` | 强备选 |
| `480` | `42/98` | `36/98` | `6/98` | `12/98` | **当前主候选** |

从 failure structure 看：

- `480` 最大优势是 **`poor_conversion` 压得最低**
- `450` 虽然整体也强，但在终结质量上明显更脆
- `420` 的 official 分数虽然接近，但 capture 掉得最明显，稳定性不够

因此 `021c-B` 目前的最合理收束是：

- [checkpoint-480](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000480/checkpoint-480) = 当前主候选
- [checkpoint-450](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000450/checkpoint-450) = 备选
- [checkpoint-420](../../ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065850/MAPPOVsBaselineTrainer_Soccer_4e0cb_00000_0_2026-04-16_06-59-14/checkpoint_000420/checkpoint-420) = 可排除

一句话 verdict：

**`021c-B` 已经证明”teammate 感知作为 auxiliary 表示学习”是有效路线；但当前 best point 仍只到 `0.794`，所以它是值得继续推进的中强正结果，而不是新的冠军模型。**

### 12.9 `021b-norm`：归一化 direct-concat 的 local true-info 复跑（2026-04-17 后补）

围绕 audit §2.6.5 的“tail 量纲失配”假设，新增归一化版 direct-concat 复跑：

- run:
  - [PPO_mappo_teammate_obs_norm_scratch_512x512_20260416_172518](../../ray_results/PPO_mappo_teammate_obs_norm_scratch_512x512_20260416_172518)
- scale:
  - `TEAMMATE_STATE_SCALE = 15,7,8,8`

训练内 decoder-style eval 仍然不稳，但已经明显好于未归一化版本：

| checkpoint | internal `baseline 50` | internal `random 50` |
|---|---:|---:|
| [checkpoint-440](../../ray_results/PPO_mappo_teammate_obs_norm_scratch_512x512_20260416_172518/MAPPOVsBaselineTrainer_Soccer_d2904_00000_0_2026-04-16_17-25-42/checkpoint_000440/checkpoint-440) | `0.600` | `0.960` |
| [checkpoint-450](../../ray_results/PPO_mappo_teammate_obs_norm_scratch_512x512_20260416_172518/MAPPOVsBaselineTrainer_Soccer_d2904_00000_0_2026-04-16_17-25-42/checkpoint_000450/checkpoint-450) | **`0.640`** | `0.520` |
| [checkpoint-460](../../ray_results/PPO_mappo_teammate_obs_norm_scratch_512x512_20260416_172518/MAPPOVsBaselineTrainer_Soccer_d2904_00000_0_2026-04-16_17-25-42/checkpoint_000460/checkpoint-460) | `0.420` | `0.860` |

最关键的是，本次对 [checkpoint-450](../../ray_results/PPO_mappo_teammate_obs_norm_scratch_512x512_20260416_172518/MAPPOVsBaselineTrainer_Soccer_d2904_00000_0_2026-04-16_17-25-42/checkpoint_000450/checkpoint-450) 做了归一化后的 **local true-info** 诊断：

- log:
  - [021b_norm_checkpoint450_vs_baseline_200.log](artifacts/local-evals/021b_norm_checkpoint450_vs_baseline_200.log)
- opponent:
  - `ceia_baseline_agent`
- episodes:
  - `200`
- `team0_wins`:
  - `172`
- `team1_wins`:
  - `28`
- `team0_win_rate`:
  - **`0.860`**
- `episode_steps_all`:
  - mean `56.3`, median `40.0`, p75 `70.0`, min `7`, max `255`
- `teammate_state_scale`:
  - `15,7,8,8`

随后按同一语义继续扩到 `500` 局，并保存 failure cases：

- log:
  - [021b_norm_checkpoint450_local_trueinfo_baseline_500_v2.log](artifacts/local-evals/021b_norm_checkpoint450_local_trueinfo_baseline_500_v2.log)
- summary:
  - [summary.json](artifacts/failure-cases/021b_norm_checkpoint450_local_trueinfo_baseline_500_v2/summary.json)
- opponent:
  - `ceia_baseline_agent`
- episodes:
  - `500`
- `team0_wins`:
  - `390`
- `team1_wins`:
  - `110`
- `team0_win_rate`:
  - **`0.780`**

这批 `500` 局结果更适合作为 `021b-norm` 的正式诊断口径。它说明归一化后的 direct-concat 不是偶然恢复，而是真正能稳定学成，但强度也没有 `200` 局小样本看上去那么夸张。

在修复本地 recorder 对 possession/progress debug 的可见性后，`021b-norm @450` 的新版 failure capture 主桶为：

| bucket | count |
|---|---:|
| `late_defensive_collapse` | `43/110` |
| `low_possession` | `38/110` |
| `poor_conversion` | `13/110` |
| `unclear_loss` | `11/110` |
| `territory_loss` | `4/110` |

其中：

- `low_possession` 的保存 loss 为 `38`
- `poor_conversion` 的保存 loss 为 `13`
- `late_defensive_collapse` 仍然是最大单桶

这条结果非常关键，因为它把 `021b` 的 verdict 从“direct-concat 路线失败”改写成了更精确的版本：

- **未归一化 direct-concat**：失败
- **归一化 direct-concat + local true-info**：稳定正结果（`0.780 @ 500ep`）

因此，当前更准确的机制结论是：

- A 假设并没有被 `021b` 否掉
- 相反，`021b-norm @450 = 0.780 @ 500ep` 说明：**如果 actor 能稳定看到经过量纲对齐的 teammate signal，这条路本身是有效的**
- `021b` 当前仍不能成为 official lane，只是因为它依赖本地 true-info 诊断语义；deployment / official API 下依然拿不到这份 signal

一句话收束：

**`021b-norm` 证明“归一化后的 direct-concat”在匹配语义下是稳定正结果；`021c-B` 证明“在 official 约束下”也能通过 auxiliary route 部分兑现这件事。两者合在一起，A 假设现在已经是有力正证据，而不是待定猜想。**

## 13. 021c-B §5 预声明判据 explicit verdict + A 假设诊断

### 13.1 §5.1 主判据 verdict

**500-ep 官方 WR vs baseline ≥ 0.81**

- 实测 11-ckpt max = **0.794** @ ckpt 480
- ❌ **FAIL**（差 0.016）
- 11-ckpt mean = 0.773（差 0.037）

### 13.2 §5.2 机制判据 verdict——`low_possession` 占失败比

预声明：**`low_possession` ≤ 15%**（从 22-28% 降到 baseline-BvB 的 0% 附近）

实测（从 §12.8 failure capture 计算 low_poss 占比）：

| ckpt | total losses | low_poss count | **low_poss %** | 对照 MAPPO+v2 @470 |
|---|---|---|---|---|
| 420 | 122 | 35 | **28.7%** | 24.0% |
| 450 | 107 | 30 | **28.0%** | 24.0% |
| **480** | 98 | 36 | **36.7%** | 24.0% |

❌ **FAIL**——不是”没降”，是**反向恶化**。ckpt 480 的 36.7% 是**项目所有 lane 里最高**（此前跨 lane 不变量上界是 v4 PPO 的 28.1%）。

### 13.3 机制解释：teammate 感知在对称 reward 下**增强同构行为**

绝对计数对比（021c-B @480 vs MAPPO+v2 @470, 500-ep failure capture）：

| 失败桶 | MAPPO+v2 @470 | 021c-B @480 | Δ | 含义 |
|---|---|---|---|---|
| late_defensive_collapse | 62 | 42 | **−20** (−32%) | aux head **帮防守**（两 agent 知道彼此位置，协同回防更有效） |
| **low_possession** | 30 | 36 | **+6** (+20%) | aux head **害进攻**（两 agent 知道彼此位置，在对称 reward 下更强烈地同步行动 → positional collapse 恶化） |
| poor_conversion | 11 | 6 | −5 (−45%) | 射门改善 |
| unclear_loss | 16 | 12 | −4 | 灰色区改善 |
| **总失败** | 125 | 98 | −27 | net WR 微升 (+0.008) |

**核心矛盾**：late_collapse 降 32%（aux head 帮防守），但 low_poss 升 20%（aux head 害进攻）。两个效应几乎抵消，net WR +0.008 微乎其微。

**根因链**：
1. Aux head 训练成功——actor 内部表示确实 internalize 了 teammate 位置信息（late_collapse 降是证据）
2. 但 reward 对称 → 两个 agent 拿到**一模一样的 gradient signal**
3. 当两个 agent 都知道对方位置且 reward 信号相同时，optimal strategy under symmetric reward = “做和队友一样的事”
4. 结果：**teammate 感知 + 对称 reward = 更强的行为相关性**——追球更同步、放弃更同步
5. 防守时”一起回防”是正确的 → late_collapse 降
6. 进攻时”一起追球”是错误的（两人挤在一起 → 空位没人接应）→ low_poss 升

**一句话**：**A 假设给 actor 的 teammate 信息不是”如何分工”的信号，是”如何更好地同步”的信号。在对称 reward 下，同步是合理的最优策略——恰好和我们想要的分工相反。**

### 13.4 §8 决策矩阵定位

snapshot-021 §8 的四格矩阵：

| 021c 结果 | 022 结果 | 诊断 | 下一步 |
|---|---|---|---|
| `low_poss ≤ 15%` | `low_poss ≤ 15%` | A 和 B 都是真因 | 选便宜的作主力 |
| `low_poss ≤ 15%` | 未降 | A 是主因 | 继续 obs |
| 未降 | `low_poss ≤ 15%` | B 是主因 | 放弃 obs |
| 都未降 | | 两假设都不成立 | 合并 A+B |

**021c-B 着陆位置**：`low_poss` 未降（反升 36.7%）→ **第三行或第四行**，取决于 022 结果。

但 021c-B 的数据还给了一个**不在原矩阵里的情报**：”A 单独不但没修 low_poss，还把它搞坏了”。这比”没降”严重——它说明：

- **如果先做 A 再做 B**（先 aux head 再加 role-diff shaping），B 需要同时抵消 A 的负效应
- **如果直接做 B 不做 A**，可能效果更干净

所以决策矩阵的第三行（”B 是主因 → 放弃 obs”）现在有**额外支撑**——不只是”A 没用”，是”A 可能反向干扰 B”。

### 13.5 对后续路线的即时影响

**1. 021c-B 不作为 submission agent，不继续延长训练**
- WR 0.794 不超 MAPPO+v2 0.786 的 margin- low_poss 恶化 → 结构比基线更差
- 延长到 iter 1000 可能推 WR 几个百分点，但 low_poss 问题不会消失

**2. 022（reward asymmetry）变成唯一有解力的单因子路线**
- 如果 022 降 low_poss → B 是主因 → 用 reward asymmetry 就够
- 如果 022 也不降 → A+B 合并是最后手段（但 A 单独有负效应，需要 A+B 相互制衡才行）

**3. 021c-B 作为 report 素材有高价值**
- “aux head 降 late_collapse 但升 low_poss”是一个**有解释力的非显然结果**
- 比简单的”A 假设 FAIL”更值 15 分的 technical reasoning
- report 叙事：”teammate 感知在对称 reward 下增强了行为相关性，和分工目标相反”——这是一个**可以写进论文级别**的 insight

**4. 如果 022 也 FAIL → 两者合并（A + B）值得尝试**
- 设计：aux head（给 teammate 信息） + role-diff shaping（给非对称 gradient）
- aux head 的同步效应 + role-diff 的差异化信号可能**正好互补**
- 但这是更复杂的组合实验，时间和 GPU 都要考虑

## 14. 021b-norm failure capture v2 修正 + A 假设终局 verdict（2026-04-17 append-only）

### 14.1 v1 分类脚本 bug 与 v2 修正

§12.9 的 failure capture 数据存在**两个版本**：

| 版本 | 脚本状态 | low_poss | poor_conv | unclear | 来源 |
|---|---|---|---|---|---|
| v1 | **分类 bug**（low_poss/poor_conv 误归 unclear）| 0 (0.0%) | 0 (0.0%) | 53 (50.5%) | [021b_norm_..._500](artifacts/failure-cases/021b_norm_checkpoint450_local_trueinfo_baseline_500) |
| **v2** | **修复后** | **38 (34.5%)** | **13 (11.8%)** | 11 (10.0%) | [021b_norm_..._500_v2](artifacts/failure-cases/021b_norm_checkpoint450_local_trueinfo_baseline_500_v2) |

v1 的 `low_poss=0, poor_conv=0` 是脚本把这两个桶的 episode 错误归入 `unclear_loss` 的 artifact。**以 v2 为准。**

### 14.2 021b-norm 的 low_poss：跨 lane 对照

| Lane | eval 模式 | WR | losses | **low_poss %** |
|---|---|---|---|---|
| MAPPO+v2 @470（对照）| official | 0.750 | 125 | **24.0%** |
| BC @2100 | official | 0.828 | 86 | 26.7% |
| 022 role-diff @270 | official | 0.840 | 80 | 21.2% |
| 024 field-role @270 | official | 0.830 | 85 | 32.9% |
| 021c-B aux @480 | official | 0.780 | 110 | **40.0%** |
| **021b-norm @450** | **local true-info** | **0.780** | 110 | **34.5%** |

**021b-norm 的 low_poss = 34.5%**——高于跨 lane 22-28% 基线带，和 021c-B 的 40.0% **方向一致**。

### 14.3 统一机制结论

无论通过哪条路径给 actor teammate 信息，low_poss 占比**都恶化**：

| 路径 | teammate 信号 | low_poss | 对比 baseline 24% |
|---|---|---|---|
| 021c-B（aux head, official-aligned）| rays 间接推断 | **40.0%** | +16 pp |
| **021b-norm（direct concat, local true-info）** | **真实 info + 归一化** | **34.5%** | **+10.5 pp** |

**两条路径都恶化。** 直接给真实 teammate state（021b-norm）比间接推断（021c-B）程度轻一些（34.5% vs 40.0%），但方向完全一致：**teammate 感知 + 对称 reward = 更强的行为同步 → low_poss 恶化**。

这和 §13.3 的机制解释一致：
1. Actor 知道队友在哪 → 两 agent 做更强的同步决策
2. 对称 reward 下同步 = 都追球或都不追 → 占球率极化
3. 防守时"一起回防"是正确的 → late_collapse 降
4. 进攻时"一起追球"是错误的 → low_poss 升

**021b-norm 的数据**确认这不是 021c-B aux head 精度不够的问题——**即使给 actor 精确真值，机制仍然相同**。

### 14.4 A 假设最终 verdict（结合 021b-norm + 021c-B）

| 维度 | 判断 | 证据 |
|---|---|---|
| **WR 提升** | ✅ 有效 | 021b-norm 0.780 > scratch baseline ~0.70；021c-B 0.794 > MAPPO+v2 0.786 |
| **late_collapse 改善** | ✅ 有效 | 021c-B: 62→42 (−32%)；021b-norm: 43/110 (39%) |
| **low_poss 改善** | ❌ **反向恶化** | 021c-B: 24%→40%；021b-norm: 24%→34.5% |
| **poor_conv 改善** | ⚠️ 中性 | 021c-B: 6/98；021b-norm: 13/110（数量变化不一致）|

**A 假设的准确 verdict**：

> **teammate 感知对 WR 和 late_collapse 有正效应，但在对称 reward 下对 low_possession 有负效应。A 假设不是"对还是错"的二值判断——它修了一个桶（防守协调），搞坏了另一个桶（占球分工），net 效应在 WR 上微正。**

### 14.5 对 §12.9 结论的修正

§12.9 末尾写道：

> "A 假设现在已经是有力正证据，而不是待定猜想。"

现在需要**收紧**为：

> "A 假设对 WR 有正效应（0.780 local true-info），但**不修 low_possession**（反升至 34.5%）。它是**有条件的正证据**：teammate 感知帮防守、害进攻，net 效应取决于失败桶的组合。这和 021c-B 的结论一致——只是证据更强，因为 021b-norm 用的是精确真值而非间接推断。"

### 14.6 对项目级 A/B 诊断的最终影响

结合 §13（021c-B）和 §14（021b-norm）的全量数据：

| 干预 | low_poss 占比 | WR | 诊断 |
|---|---|---|---|
| 12 条 RL-only lane | 22-28% | 0.65-0.84 | 不变量 baseline |
| A — aux head (021c-B) | **40%** | 0.794 | A 恶化 low_poss |
| A — direct true-info (021b-norm) | **34.5%** | 0.780 | A 恶化 low_poss（真值也一样）|
| B — agent-id (022) | 21-26% | 0.818 | B 中性 |
| B — spawn-depth (024) | 26-33% | 0.842 | B 中性偏高 |

**`low_possession ≈ 25% of failures` 在所有 14 条 lane（含 A / B / A+B 变体）下从未降低。唯一的偏离方向是 A 路径的恶化（34-40%）。** 这已不再是"待验证的推论"，而是**实证终局**。
