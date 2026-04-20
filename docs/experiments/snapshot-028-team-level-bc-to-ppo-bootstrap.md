# SNAPSHOT-028: Team-Level IL→BC→PPO Bootstrap

- **日期**: 2026-04-17
- **负责人**:
- **状态**: 已完成首轮 verdict

## 0. 续跑说明

`028-A` 首轮正式 RL run 在 home quota 打满后中断，原始 run root 为：

- [PPO_team_level_bc_bootstrap_028A_512x512_formal](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal)

断点与恢复信息：

- 续跑点： [checkpoint-460](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_10dc1_00000_0_2026-04-17_11-21-11/checkpoint_000460/checkpoint-460)
- 中断原因：home quota exhausted
- 成功续跑 trial： [TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54)

当前应以 run root 下的 **merged summary** 作为正式总摘要。这里的 canonical 输出包含 **文本摘要 + 合并曲线图** 两部分：

- [merged_training_summary.txt](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/merged_training_summary.txt)
- [training_loss_curve_merged.png](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/training_loss_curve_merged.png)

该 merged summary 已确认能拼出：

- 原始 trial：`it 1 -> 462`
- 续跑 trial：`it 461 -> 1250`

因此，`028-A` 的最终 best/final 结论应以 merged summary 和根目录 [checkpoint_eval.csv](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/checkpoint_eval.csv) 为准，而不是只看某个 trial 的终端输出；图形上也应优先查看 `training_loss_curve_merged.png`。

## 1. 为什么需要 028

[SNAPSHOT-027](snapshot-027-team-level-ppo-coordination.md) 做的是 team-level scratch，对标的是 MAPPO+v2 scratch（见 027 §2b）。即使 027 出正结果，它回答的仍然只是"team-level 架构本身的价值"，无法直接挑战当前 SOTA。

当前 SOTA 路径：

```
baseline teacher 轨迹（015）
  → player-level BC（015/017）
  → BC→MAPPO warm-start（017）→ BC@2100 = 0.842
  → field-role binding stable（025b@80）= 0.842 + H2H 压过 BC
```

**整条 SOTA 都依赖 BC bootstrap**。在 per-agent 架构下，BC bootstrap 对最终 WR 贡献 +0.16 到 +0.19（见 017 对比）。如果 team-level 架构确实有协调优势，那么**叠加 BC bootstrap** 才是公平挑战 SOTA 的版本——同 bootstrap 策略、同 shaping、不同架构。

本 snapshot 的职责是：**把 017 的 IL→BC→RL 管线在 team level 重建一遍，回答"team-level 架构 + BC bootstrap 能否真正突破 0.842 天花板"**。

这里必须强调：`028` 的对标不是“复用 017 的 player-level BC 再换一个 PPO 头”，而是要把 **teacher → BC → PPO** 三个阶段都切换到 **team policy** 语义：

- `017`：`multiagent / per-agent BC -> per-agent MAPPO`
- `028`：`team-level BC -> team-level PPO`

因此，**player-level BC 资产不能直接复用为 028 的 Stage 2**。可复用的是：

- team-level teacher 采集脚本
- team-level BC trainer / deployment wrapper
- 以及后续要新增的 `team-level BC -> team-level PPO` warm-start 接线

旧的 team-level dataset / BC checkpoint 可以作为 smoke、shape 对齐、warm-start 映射开发的脚手架，但 `028` 的正式结论应建立在本 snapshot 自己重新产出的 Stage 1/2 结果之上。

## 2. 与既有 lane 的关系

| snapshot | 架构 | bootstrap | 对标关系 |
|---|---|---|---|
| [008 Base-C](snapshot-008-starter-aligned-base-model-lane.md) | team-level | scratch, 无 shaping | 从未跑过 |
| [009 Base-D](snapshot-009-base-team-vs-baseline-lane.md) | team-level | scratch, 无 shaping | 0.82, 参考下限 |
| [014 MAPPO+v2 scratch](snapshot-014-mappo-fair-ablation.md) | per-agent | scratch | 027 的对标 |
| [017 BC→MAPPO](snapshot-017-bc-to-mappo-bootstrap.md) | per-agent | BC bootstrap | **本 snapshot 的对标** |
| [027 team-level scratch](snapshot-027-team-level-ppo-coordination.md) | team-level | scratch | 028 的无 bootstrap 对照 |
| **028（本 snapshot）** | **team-level** | **BC bootstrap** | **挑战 SOTA 的正式版本** |

核心对照矩阵：

| | scratch | BC bootstrap |
|---|---|---|
| per-agent | [014 MAPPO+v2](snapshot-014-mappo-fair-ablation.md) | [017 BC@2100](snapshot-017-bc-to-mappo-bootstrap.md) = 0.842 (SOTA base) |
| team-level | [027](snapshot-027-team-level-ppo-coordination.md) | **028（本 snapshot）** |

这是一个干净的 2×2——能同时拆解"架构"和"bootstrap"两个因素各自的贡献。

## 3. 核心假设

### 主假设

如果 team-level 架构有实际协调价值，且 BC bootstrap 的收益在架构转换后能保留，那么：

- 028 的 official `baseline 500` 应当 **> 017 的 0.842**
- 且失败桶中 `low_possession` 应低于 017 的 35-39%（代表协调确实改善了持球）
- H2H vs 025b@80 应 ≥ 0.500

### 备择假设 A

如果 028 ≈ 017 (0.82-0.84)：

- BC bootstrap 的收益在两种架构下都能榨取
- 但 team-level 架构本身在 baseline 对手下没有额外优势
- 架构转换是"换了种方式拿到同样的上限"

### 备择假设 B

如果 028 < 017：

- team-level 架构对 BC bootstrap 的迁移有损耗（例如联合动作的 BC 学习质量差于 per-agent BC）
- 或协调能力在 baseline 对手下本来就不是瓶颈（baseline 本身不利用协调弱点）

## 4. 工程管线

本 snapshot 的工程量比 027 大得多。必须分阶段落地：

### 4.1 Stage 1 — team-level teacher 轨迹采集

baseline 本身是 team agent（一个 agent 控制两人）。直接记录：

- `obs_team = np.concatenate((obs[0], obs[1]))` — 672 维
- `action_team = np.concatenate((action[0], action[1]))` — 6 维 MultiDiscrete

**采集策略**：
- baseline vs random 对局：覆盖 baseline 在优势局面下的打法
- baseline vs baseline 对局：覆盖 baseline 在对等局面下的打法
- 目标 rollout 数：约 100k-200k 时间步（和 015 player-level BC 的量级对齐）

**复用口径**：
- 采集脚本本身已存在 team-level 版本，可直接复用
- 但 `028` 的正式 Stage 1 数据应在本 snapshot 名下重新采集
- 015 留下来的 team dataset 只作为 smoke / 对齐资产，不直接充当 028 的正式 teacher data

### 4.2 Stage 2 — team-level BC trainer

- 输入：`(obs_team, action_team)` 对
- 模型：`Box(672) → MLP → MultiDiscrete([3,3,3,3,3,3])` 分类头
- loss：6 个独立的 cross-entropy，相加
- 训练量：5k-10k steps（参考 015 的量级）

**复用口径**：
- 现有 team-level BC 模型 / trainer / deployment wrapper 已存在，可直接复用
- 但 `028` 的正式 Stage 2 仍应基于 `028` 自己的 Stage 1 数据重新训练一版 team-level BC
- 旧的 team-level BC checkpoint 只用于 smoke、warm-start 映射开发与格式验证

### 4.3 Stage 3 — BC → team-level PPO warmstart

- 加载 team-level BC 的 weights
- 直接作为 team-level PPO 的 policy 初始化（同维度）
- 比 017 简单：不需要 per-agent→shared 的 key 映射，因为两边都是 team-level

**新增工程**：
- `warmstart_team_level_policy_from_bc()` 函数
- 在 027 的 team-level PPO 训练入口里加 `BC_WARMSTART_CHECKPOINT` 支持

这是本 snapshot 当前唯一真正缺失、且不能靠旧 run 结果跳过的工程环节。

### 4.4 Stage 4 — team-level PPO 正式训练

- warmstart from team-level BC
- v2 shaping 不变
- 预算同 027（50M steps / 16h）
- 如果 BC bootstrap 有效，预期 500 iter 内就能达到或超过 017 的水平

### 4.5 当前已落地的 batch

`028` 首轮主线已经具备从 teacher 到 RL 的完整 batch 骨架：

- Stage 1 self-play teacher 采集：
  - [soccerstwos_h100_cpu32_collect_team_teacher_selfplay_028A.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_collect_team_teacher_selfplay_028A.batch)
- Stage 1 baseline-vs-random teacher 采集：
  - [soccerstwos_h100_cpu32_collect_team_teacher_vs_random_028A.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_collect_team_teacher_vs_random_028A.batch)
- Stage 2 team-level BC：
  - [soccerstwos_h100_cpu32_bc_team_bootstrap_028A_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_bc_team_bootstrap_028A_512x512.batch)
- Stage 4 team-level PPO warm-start：
  - [soccerstwos_h100_cpu32_team_level_bc_bootstrap_028A_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_bc_bootstrap_028A_512x512.batch)

当前默认的 `BC_WARMSTART_CHECKPOINT` 仍指向旧的 team-level BC checkpoint，仅用于 smoke、映射开发与链路验证。`028` 的正式结果仍应以本 snapshot 自己重新采集 teacher data、重新训练 BC 后的 checkpoint 为准。

## 5. 执行矩阵

| lane | bootstrap | shaping | network | iter | GPU 预估 |
|---|---|---|---|---|---|
| **Stage 1** | — | — | — | 1 次采集 | ~2h CPU |
| **Stage 2** | — | — | [512, 512] | BC 5-10k steps | ~30min GPU |
| **028-A** | team-level BC | v2 shaping | [512, 512] | 50M steps RL | ~16h GPU |
| **028-B**（可选）| team-level BC | v2 shaping | [768, 512] | 50M steps RL | ~16h GPU |

首轮先以 **028-A = 512x512** 为正式主线。原因不是“512 一定最好”，而是：

- 当前已有 team-level BC 资产与 `512x512` 同宽，最适合先把 `team BC -> team PPO warm-start` 路径做干净
- 对 `028` 这种“完整对标 SOTA”的实验，先把 **teacher / BC / warm-start / RL** 的公平闭环建立起来，比一开始就追更宽网络更重要
- `768x512` 应视作后续容量扩展，而不是 028 首轮是否成立的前提

## 6. 预声明判据

### 6.1 主判据

| 阈值 | 逻辑 |
|---|---|
| **official 500 ≥ 0.842** | 追平 017 BC@2100 = team-level BC bootstrap 不劣于 per-agent BC bootstrap |
| **official 500 ≥ 0.85** | 真正突破 0.842 天花板（team-level 架构有净增益）|
| **H2H vs 025b@80 ≥ 0.500** | 在当前冠军位前面至少打成五五开 |

### 6.2 机制判据

- `low_possession` **严格低于** 017 @2100 的对应占比（如果 team-level 协调有效，这个桶应该先下降）
- `late_defensive_collapse` 不恶化

### 6.3 失败判据

| 条件 | 解读 |
|---|---|
| official 500 < 0.80 | BC→team-level PPO 的迁移损失大，架构转换破坏 BC 学到的能力 |
| official 500 在 0.80-0.84 且 H2H < 0.450 | team-level BC bootstrap 的上限和 per-agent BC bootstrap 相当，没有架构优势 |
| Stage 2 BC eval WR < 0.50 | team-level BC 本身没学好 baseline 策略 → BC 管线需要修 |

## 7. 风险

### R1 — team-level BC 的样本效率

player-level BC 的输入是 336 维、输出 Discrete(27)。team-level BC 的输入 672 维、输出 6 个独立 Discrete(3)。
参数量翻倍，但 teacher 样本数不变。可能需要采更多 teacher 轨迹。

缓解：如果 BC eval WR 低于预期，把 teacher rollout 数从 100k 扩到 200k-400k。

### R2 — baseline 的 team-level action 分布可能过于单一

baseline 是确定性策略（或近确定性）。在 team-level 下，两人的 joint action 分布可能极度尖锐——BC 学到的是"1-of-N 确定性映射"而不是"行为分布"。这让 PPO warmstart 后的 entropy 极低，早期探索能力弱。

缓解：
- BC 时加 label smoothing 或 entropy regularization
- PPO warmstart 后前 50 iter 用更高的 entropy_coeff（如 0.01）逼出探索

### R3 — 027 基础设施依赖

028 需要 027 已经落地的 team-level 训练入口、部署 wrapper、env 配置。如果 027 的某个环节有 bug（例如 MultiDiscrete 没保住被 flatten 了），028 会同样踩坑。

缓解：**028 不能独立开工，必须在 027 的 smoke test 通过后再启动**。

### R4 — 架构转换对 BC 学到的协调能力的"破坏"

BC 在 team-level 学到的是 baseline 的联合动作分布。但 baseline 本身是 per-agent 思维的（它的两个 agent 独立计算动作）。所以 team-level BC 学到的"协调"可能只是 baseline 两个 per-agent 策略的统计叠加，不是真正的协调模式。

这种情况下，BC bootstrap 给 team-level PPO 的初始化和 per-agent BC bootstrap 没有本质区别——只是换了种表示。

缓解：这是**假设本身的风险**，不能通过工程缓解；只能由 028 的结果验证或证伪。如果 028 ≈ 017，说明这就是实际情况。

## 8. 不做的事

- **不做 self-play**（先对齐 baseline 目标）
- **不做 role-diff shaping**（不叠 022/024/025 的 role binding）
- **不做 centralized critic**（team-level actor 已看全局）
- **不做 opponent pool**（先测最干净的 vs baseline 信号）
- **不并行跑 028-B**（先看 028-A，再决定宽度消融）

## 9. 与 SOTA 的直接映射

| stage | per-agent SOTA | **team-level 本 snapshot** |
|---|---|---|
| teacher | baseline（015）| baseline（team-level 重新采集）|
| BC | player-level BC（015/017）| **team-level BC（Stage 1+2，重新训练）** |
| RL warmstart | BC→MAPPO（017）→ BC@2100 | **BC→team-level PPO（Stage 3+4）** |
| fine-tune | 025b field-role binding → 0.842 + H2H | （未来可做，不在本 snapshot 内）|

本 snapshot 做前三个 stage。如果 028-A 出正结果，后续可以考虑在 028 基础上加 team-level 的"field-role binding 等价物"（如 team-level reward 里给某个固定 slot 更偏进攻/防守的系数），但那是 028 之后的事。

## 10. 执行清单

0. **前置**：等 [SNAPSHOT-027](snapshot-027-team-level-ppo-coordination.md) smoke test 通过（确认 team-level 基础设施 work）
1. Stage 1 — teacher 轨迹采集：
   - 复用现有 team-level trajectory collector
   - 在 028 名下重新跑 baseline vs random + baseline vs baseline，采 100k-200k steps
   - 验证 obs_team shape = 672, action_team shape = (6,)
2. Stage 2 — team-level BC trainer：
   - 复用现有 team-level BC trainer / deployment wrapper
   - 基于 028 Stage 1 数据重新训练一版 BC（首轮 512x512）
   - 对 BC 模型做 official `baseline 50` eval（sanity check）
3. Stage 3 — warmstart 函数：
   - 实现 `warmstart_team_level_policy_from_bc()`
   - 在 027 训练入口加 `BC_WARMSTART_CHECKPOINT` 分支
4. Stage 4 — 1-iter smoke：
   - 验证 warmstart 后 iter-1 的 `baseline 50` ≥ BC eval WR（不破坏 BC 能力）
   - 当前工程 smoke 已确认：
     - 旧 team-level BC checkpoint 可成功接入 team-level PPO
     - warm-start 日志打印 `copied=6, adapted=0, skipped=0`
     - `checkpoint-1` 能正常写出 PPO checkpoint；首轮 smoke 中未等待 `checkpoint_eval.csv` 完整落盘，不视为 warm-start 失败
5. Stage 4 — 长训（50M steps / 16h）
6. 按 §6 做 official `baseline 500` 选模 + failure capture
7. H2H vs 025b@80 + H2H vs 017@2100
8. verdict 落本文件 §11+（append-only）

## 11. 首轮正式结果（official + failure capture 已完成）

`028-A` 最终完整跑到 `50M steps / 1250 iter`，merged summary 显示训练本体是健康的：

- merged best reward：`+2.1943 @ iter 959`
- final reward：`+2.1668 @ iter 1250`
- canonical summary： [merged_training_summary.txt](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/merged_training_summary.txt)

但和 [SNAPSHOT-027](snapshot-027-team-level-ppo-coordination.md) 一样，这里的 internal eval 不能直接当正式结论。我们按 **top 5% + ties + `±2 checkpoint window`** 的协议对高窗做了 official `baseline 500` 复核，结果如下：

- official 峰值： [checkpoint-1220](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001220/checkpoint-1220) `= 0.844`
- 次强点： [checkpoint-1110](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001110/checkpoint-1110) `= 0.818`
- 更稳的中段点： [checkpoint-1060](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060) `= 0.810`

随后对 `1220 / 1110 / 1060` 做 `500ep` failure capture，结果是：

- [checkpoint-1220](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001220/checkpoint-1220): `0.844 official -> 0.796 capture`
- [checkpoint-1110](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001110/checkpoint-1110): `0.818 official -> 0.762 capture`
- [checkpoint-1060](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060): `0.810 official -> 0.806 capture`

对应的失败桶结构也把“峰值”和“可信主候选”分开了：

- `1060`
  - `late_defensive_collapse = 48/97`
  - `low_possession = 30/97`
  - `poor_conversion = 6/97`
  - `unclear_loss = 12/97`
- `1110`
  - `late_defensive_collapse = 51/119`
  - `low_possession = 39/119`
  - `poor_conversion = 11/119`
  - `territory_loss = 6/119`
  - `unclear_loss = 11/119`
- `1220`
  - `late_defensive_collapse = 49/104`
  - `low_possession = 31/104`
  - `poor_conversion = 15/104`
  - `unclear_loss = 7/104`

这组结果支持一个更稳的 readout：

1. `028A` 的确比 `027A` 更强。
   `027A` 的 best official 只有 `0.804`，而 `028A` 至少在 late window 里打出了真实的 `0.81~0.84` 区间。

2. internal eval 明显高估了 `028A` 的 late spike。
   典型例子是 `1220: 0.94 internal -> 0.844 official -> 0.796 capture`，说明这不是一个可以直接当主候选的“稳定冠军点”。

3. `028A` 更可信的主候选应收口到 **[checkpoint-1060](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060)**，而不是 official 峰值 `1220`。
   `1060` 的 official/capture gap 最小，且 `poor_conversion` 明显更低，是更平衡的 team-level BC warm-start 候选点。

因此，`028A` 当前最稳的首轮结论是：

- **team-level BC warm-start 明确有效**
- **它真实地优于 team-level scratch（027）**
- **official 峰值是 `1220 = 0.844`，但更可信的主候选是 `1060 = 0.810 official / 0.806 capture`**
- **是否能进一步挑战 [SNAPSHOT-025b](snapshot-025b-bc-champion-field-role-binding-stability-tune.md) 或 [SNAPSHOT-029](snapshot-029-post-025b-sota-extension.md) 的主线，需要由后续 H2H 决定**

这些 H2H 现在也已经补完：

- [checkpoint-1060](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060) vs [027A@650](../../ray_results/PPO_team_level_v2_scratch_768x512_20260417_095059/TeamVsBaselineShapingPPOTrainer_Soccer_8384d_00000_0_2026-04-17_09-51-20/checkpoint_000650/checkpoint-650) = `0.592`
- [checkpoint-1060](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060) vs [017@2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) = `0.432`
- [checkpoint-1060](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060) vs [025b@80](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80) = `0.432`
- [checkpoint-1060](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001060/checkpoint-1060) vs [029B@190](../../ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190) = `0.462`
- [checkpoint-1220](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001220/checkpoint-1220) vs [017@2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) = `0.466`
- [checkpoint-1220](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001220/checkpoint-1220) vs [025b@80](../../ray_results/PPO_mappo_field_role_binding_bc2100_stable_512x512_20260417_060418/MAPPOVsBaselineTrainer_Soccer_da95d_00000_0_2026-04-17_06-04-42/checkpoint_000080/checkpoint-80) = `0.428`
- [checkpoint-1220](../../ray_results/PPO_team_level_bc_bootstrap_028A_512x512_formal/TeamVsBaselineShapingPPOTrainer_Soccer_85a0f_00000_0_2026-04-17_19-16-54/checkpoint_001220/checkpoint-1220) vs [029B@190](../../ray_results/PPO_mappo_029B_bwarm170_to_v2_512x512_formal/MAPPOVsBaselineTrainer_Soccer_457ea_00000_0_2026-04-17_12-12-46/checkpoint_000190/checkpoint-190) = `0.428`

这些对打把 `028A` 的最终位置钉得很清楚：

1. `028A` 的确明显优于 `027A`。
   `1060 vs 027A@650 = 0.592`，说明 BC warm-start 在 team-level PPO 上带来了真实且可重复的增益，而不是只把 official baseline 分数轻微抬高。

2. `028A` 还没有进入冠军位讨论。
   不管取更稳的 `1060` 还是 official 峰值 `1220`，对 [017](snapshot-017-bc-to-mappo-bootstrap.md)、[025b](snapshot-025b-bc-champion-field-role-binding-stability-tune.md)、[029B](snapshot-029-post-025b-sota-extension.md) 的 H2H 都是明显负值。新增的 `1060 vs 029B@190 = 0.462` 也说明 `029B` 的优势不能简单归因于“team-level base 本身较弱”。

3. `1060` 作为主候选比 `1220` 更合理。
   `1220` 虽然给出 `0.844` 的 official 峰值，但 failure capture 和 H2H 都不支持把它当成稳定主点；`1060` 的 official/capture 对齐最好，而且在 H2H 中也比 `1220` 更干净。

因此，`028A` 的首轮正式 verdict 现在可以收成：

- **team-level BC warm-start 明确成立**
- **它真实地优于 team-level scratch（027）**
- **它当前仍低于 `017 / 025b / 029B` 这些 per-agent 强线**
- **其中 `1060 vs 029B@190 = 0.462` 进一步说明：当前 team-level BC base 还没有接近 `029B` 这条 per-agent 强挑战者**
- **`028A` 的最佳身份是“team-level 主线已验证可行，但首轮还不是冠军挑战成功”**

## 12. 相关

- [SNAPSHOT-015: player-level BC teacher trajectory](snapshot-015-behavior-cloning-team-bootstrap.md)
- [SNAPSHOT-017: BC→MAPPO bootstrap](snapshot-017-bc-to-mappo-bootstrap.md)
- [SNAPSHOT-025b: current champion](snapshot-025b-bc-champion-field-role-binding-stability-tune.md)
- [SNAPSHOT-027: team-level scratch](snapshot-027-team-level-ppo-coordination.md)
