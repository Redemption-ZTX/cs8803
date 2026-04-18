# SNAPSHOT-006: Fixed-Teammate 与 Dual-Expert 主线重想

- **日期**: 2026-04-11
- **负责人**:
- **目标**:
  - 停止继续堆 PPO 结构变体，回到更贴近任务定义的 single-player rethink
  - 验证“部署期协调器”“场景化专家训练”“训练时固定队友”三条替代思路
  - 找到一个比现有 `~0.79-0.80` 平台更值得继续投入的主线
- **配置**:
  - 部署协调器: [trained_dual_expert_agent.py](../../cs8803drl/deployment/trained_dual_expert_agent.py)
  - 专家协调与场景: [expert_coordination.py](../../cs8803drl/branches/expert_coordination.py)
  - 主训练入口: [train_ray_team_vs_random_shaping.py](../../cs8803drl/training/train_ray_team_vs_random_shaping.py)
  - fixed-teammate 部署封装: [trained_fixed_teammate_agent.py](../../cs8803drl/deployment/trained_fixed_teammate_agent.py)
  - fixed-teammate batch: [soccerstwos_h100_cpu32_singleplayer_fixed_teammate.batch](../../scripts/batch/adaptation/soccerstwos_h100_cpu32_singleplayer_fixed_teammate.batch)
  - fixed-teammate scratch 对照 batch: [soccerstwos_h100_cpu32_singleplayer_fixed_teammate_scratch.batch](../../scripts/batch/adaptation/soccerstwos_h100_cpu32_singleplayer_fixed_teammate_scratch.batch)
  - wide actor 对照 batch: [soccerstwos_h100_cpu32_singleplayer_fixed_teammate_wide512x256.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_singleplayer_fixed_teammate_wide512x256.batch)
  - 官方评估: [scripts/eval/evaluate_official_suite.py](../../scripts/eval/evaluate_official_suite.py)
  - 训练内筛选口径: `50` 局
  - 最终确认口径: 官方 `200` 局

## 1. 纯协调器验证

使用 [checkpoint-225](../../ray_results/PPO_continue_ckpt160_cpu32_20260408_183648/PPO_Soccer_79ad0_00000_0_2026-04-08_18-37-08/checkpoint_000225/checkpoint-225) 同时扮演“进攻专家”和“防守专家”，只在部署期由 [trained_dual_expert_agent.py](../../cs8803drl/deployment/trained_dual_expert_agent.py) 做角色切换。

- 官方 `20` 局曾短暂出现 `17/20 = 0.85`
- 官方 `200` 局最终确认:
  - `vs baseline`: `156/200 = 0.780`
  - `vs random`: `198/200 = 0.990`

**结论**:

- 推理期协调本身不够，之前的小样本亮点来自采样波动
- 这条线没有突破 single-player 既有平台

## 2. Attack Expert 训练验证

在修复“训练队友分布与评估队友分布不一致”之后，重跑进攻专家：

- 运行目录: [PPO_attack_expert_warm225_20260409_141445](../../ray_results/PPO_attack_expert_warm225_20260409_141445)
- 最佳训练内筛选:
  - [checkpoint-60](../../ray_results/PPO_attack_expert_warm225_20260409_141445/PPO_Soccer_0876f_00000_0_2026-04-09_14-15-05/checkpoint_000060/checkpoint-60)
  - `12/50 = 0.24`

训练信号表现：

- `episode_reward_mean` 只从大约 `-1.55` 改善到 `-1.16`
- loss / vf_loss / kl 数值稳定，但胜率始终很低

**结论**:

- 这不是优化器发散，而是“attack expert 任务定义错了”
- 当前场景化专家训练没有对齐真实比赛目标，应停止当作主线

## 3. Fixed-Teammate 主线重定义

为了让训练分布更接近部署分布，新主线改为：

- 训练时队友固定为 [checkpoint-225](../../ray_results/PPO_continue_ckpt160_cpu32_20260408_183648/PPO_Soccer_79ad0_00000_0_2026-04-08_18-37-08/checkpoint_000225/checkpoint-225)
- 评估时沿用同样的固定队友部署方式
- reward shaping 关闭，尽量贴近真实比赛 reward

### 3.1 无效 run：restore 语义错误

- 运行目录: [PPO_singleplayer_fixed_teammate_20260411_145957](../../ray_results/PPO_singleplayer_fixed_teammate_20260411_145957)
- `progress.csv` 只有 1 条数据行
- 首尾都是 `training_iteration = 226 / timesteps_total = 4144000`

**结论**:

- 这不是有效训练，只是恢复旧 checkpoint 后立刻停掉
- 因此该 run 的 `33/50 = 0.66` 不可用于实验结论

### 3.2 语义修复

已将主训练入口补成真正的 `fresh run + warm-start`：

- [train_ray_team_vs_random_shaping.py](../../cs8803drl/training/train_ray_team_vs_random_shaping.py) 新增 `WARMSTART_CHECKPOINT`
- [soccerstwos_h100_cpu32_singleplayer_fixed_teammate.batch](../../scripts/batch/adaptation/soccerstwos_h100_cpu32_singleplayer_fixed_teammate.batch) 改为默认使用 warm-start，而非 restore

## 4. 当前进行中的两条 fixed-teammate run

以下状态记录于 2026-04-11 文档编写时刻，属于**中间快照**，不作为最终结论：

### 4.1 标准宽度 `256,128`

- 运行目录: [PPO_singleplayer_fixed_teammate_20260411_151253](../../ray_results/PPO_singleplayer_fixed_teammate_20260411_151253)
- 当前最新进度（写文档时）:
  - `training_iteration = 113`
  - `timesteps_total = 2,712,000`
  - `episode_reward_mean = 1.9221`

### 4.2 更宽 actor `512,256`

- 运行目录: [PPO_singleplayer_fixed_teammate_wide512x256_20260411_151654](../../ray_results/PPO_singleplayer_fixed_teammate_wide512x256_20260411_151654)
- 当前最新进度（写文档时）:
  - `training_iteration = 81`
  - `timesteps_total = 1,944,000`
  - `episode_reward_mean = -0.2309`

**注意**:

- 这条 `512,256` run 不是干净的模型宽度对照
- 它同时改变了初始化方式（从零开始）和模型宽度
- 因此它不能回答“更宽模型是否优于 `256,128` warm-start”
- 后续宽度判断应基于同初始化、同固定队友设置的对照矩阵

### 4.3 下一步：干净初始化对照

为了回答当前最关键的问题，下一步主对照应当是：

- 同一 fixed-teammate 训练设置
- 同一 `256,128` actor
- 同一固定队友 [checkpoint-225](../../ray_results/PPO_continue_ckpt160_cpu32_20260408_183648/PPO_Soccer_79ad0_00000_0_2026-04-08_18-37-08/checkpoint_000225/checkpoint-225)
- 唯一差别：`warm-start` vs `scratch`

对应脚本：

- [soccerstwos_h100_cpu32_singleplayer_fixed_teammate.batch](../../scripts/batch/adaptation/soccerstwos_h100_cpu32_singleplayer_fixed_teammate.batch)
- [soccerstwos_h100_cpu32_singleplayer_fixed_teammate_scratch.batch](../../scripts/batch/adaptation/soccerstwos_h100_cpu32_singleplayer_fixed_teammate_scratch.batch)

## 当前判断

- 纯协调器没有真实提升
- 场景化专家训练偏离目标过远
- “训练时固定队友 + 关闭 shaping” 是当前最值得继续的 single-player rethink
- `512,256` from-scratch run 只能作为探索记录，不能当作干净宽度结论
- 真正该优先跑的是 `256,128 warm-start` vs `256,128 scratch` 的初始化对照

## 下一步

1. 先完成 `256,128 warm-start` 与 `256,128 scratch` 的 fixed-teammate 初始化对照
2. 再决定是否值得做同初始化下的宽度对照
3. 对有效 run 的 best checkpoint 做官方 `200` 局确认
4. 训练结束后执行第二阶段物理目录整理

## 相关

- [snapshot-003-official-evaluator-realignment.md](snapshot-003-official-evaluator-realignment.md)
- [snapshot-004-role-ppo-and-shared-policy-ablation.md](snapshot-004-role-ppo-and-shared-policy-ablation.md)
- [snapshot-005-observation-memory-and-centralized-critic-ablation.md](snapshot-005-observation-memory-and-centralized-critic-ablation.md)
- [directory-governance.md](../management/directory-governance.md)
