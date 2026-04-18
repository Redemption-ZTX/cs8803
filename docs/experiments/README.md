# 实验记录

> 训练脚本分析与超参详情见 [代码审计](../architecture/code-audit.md)（最新: [code-audit-000](../architecture/code-audit-000.md)）。
> 工程规范中的实验流程见 [engineering-standards.md](../architecture/engineering-standards.md#实验迭代)。
> 评分标准见 [作业要求 — Experimental Results](../references/Final%20Project%20Instructions%20Document.md#experimental-results-25-points)。

## 实验索引

| 编号 | 标题 | 日期 | 状态 |
|------|------|------|------|
| [SNAPSHOT-000](snapshot-000-prior-team.md) | 前任团队实验结果分析 | 2026-04-07 | 已完成 |
| [SNAPSHOT-001](snapshot-001-reward-shaping-runtime-validation.md) | Reward shaping 真实环境运行验证 | 2026-04-08 | 已完成 |
| [SNAPSHOT-002](snapshot-002-h100-overlay-env-validation.md) | H100 overlay 环境与 GPU smoke test | 2026-04-08 | 已完成 |
| [SNAPSHOT-003](snapshot-003-official-evaluator-realignment.md) | 官方 evaluator 对齐、本地评估修正、single-player warm-start 重选 | 2026-04-09 | 已完成 |
| [SNAPSHOT-004](snapshot-004-role-ppo-and-shared-policy-ablation.md) | role PPO / warm-start / shared-policy 方向消融 | 2026-04-09 | 进行中 |
| [SNAPSHOT-005](snapshot-005-observation-memory-and-centralized-critic-ablation.md) | summary-observation / LSTM / centralized-critic 方向复盘 | 2026-04-09 | 已完成 |
| [SNAPSHOT-006](snapshot-006-fixed-teammate-and-dual-expert-rethink.md) | dual-expert 协调失败、attack expert 失配、fixed-teammate 主线重想 | 2026-04-11 | 进行中 |
| [SNAPSHOT-007](snapshot-007-base-lane-reset-and-directory-reorg.md) | 回到 SSOT/starter 重新定义 base lane，并完成结构化目录重构 | 2026-04-11 | 已完成 |
| [SNAPSHOT-008](snapshot-008-starter-aligned-base-model-lane.md) | starter 对齐的 scratch base model lane：Base-A / Base-B / Base-C | 2026-04-11 | 进行中 |
| [SNAPSHOT-009](snapshot-009-base-team-vs-baseline-lane.md) | 直接对准 baseline 的 scratch base lane：Base-D / Base-E | 2026-04-11 | 已完成 |
| [SNAPSHOT-010](snapshot-010-shaping-v2-deep-zone-ablation.md) | shaping-v1 失败样本复盘、v2 deep-zone / negative-C A/B 与 500 局失败分析 | 2026-04-12 | 已完成 |
| [SNAPSHOT-011](snapshot-011-shaping-v3-progress-gated-ablation.md) | v3 progress-gated shaping：在 v2 基础上只修正正向 progress 给分口径；首轮训练结果不支持继续沿此方向迭代 | 2026-04-13 | 已完成 |
| [SNAPSHOT-012](snapshot-012-imitation-learning-bc-bootstrap.md) | imitation learning / behavior cloning bootstrap：baseline teacher 轨迹采集与 BC 起点设计 | 2026-04-13 | 第一阶段已完成（转入 `015`） |
| [SNAPSHOT-013](snapshot-013-baseline-weakness-analysis.md) | baseline weakness analysis：从失败样本分析转向对手可利用模式分析；`baseline vs baseline / random` 首轮分析已完成 | 2026-04-13 | 已完成（首轮分析） |
| [SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md) | MAPPO / centralized-critic 公平对照：首轮官方 500 结果表明 `MAPPO + shaping-v2` 追平当前项目最好成绩 | 2026-04-13 | 已完成（首轮对照） |
| [SNAPSHOT-015](snapshot-015-behavior-cloning-team-bootstrap.md) | team-level behavior cloning：最小 BC trainer、checkpoint 格式与 deployment wrapper 落地；正式 team-BC run 已完成并产出 `checkpoint_000030` | 2026-04-13 | 已完成（首轮 BC train） |
| [SNAPSHOT-016](snapshot-016-shaping-v4-survival-anti-rush-ablation.md) | shaping-v4 survival / anti-rush：首轮官方 `baseline 500` 形成稳定 `0.768` 平台，成为当前最好的 PPO shaping 版本 | 2026-04-13 | 已完成（首轮对照） |
| [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md) | `BC -> MAPPO` bootstrap：player-level BC bridge、正式 warm-start 训练、官方 `baseline 500` 复核与 4 点 failure capture 已完成；当前最好点为 `checkpoint-2100 = 0.842` | 2026-04-14 | 已完成首轮结果 |
| [SNAPSHOT-018](snapshot-018-mappo-v2-opponent-pool-finetune.md) | MAPPO+v2 warm-start opponent-pool fine-tune：static frozen-pool 首轮完成，`ckpt290` 以 `0.812` 成为当前最强 baseline 主线，并补完对旧 MAPPO 强线的 head-to-head / failure capture | 2026-04-14 | 已完成（首轮对照） |
| [SNAPSHOT-019](snapshot-019-mappo-v2-opponent-pool-anchor-heavy-rebalance.md) | opponent-pool anchor-heavy rebalance：保持 `v2` warm-start 与 `v2 shaping` 不动，只把 pool 改为 `baseline 30 / anchor 30 / v1 20 / bs0 20`；首轮官方 `baseline 500` 显示早段 `checkpoint-40/140 = 0.788`，failure capture 表明这是典型的 early-window lane | 2026-04-15 | 已完成首轮结果 |
| [SNAPSHOT-020](snapshot-020-mappo-v4-fair-ablation.md) | `MAPPO + v4 shaping` 公平对照：首轮官方 `baseline 500` 最好点为 `checkpoint-490 = 0.764`，failure capture 显示 `late_defensive_collapse + low_possession` 仍是主问题；当前证据不支持将 `v4 shaping` 优先并入 MAPPO 主干 | 2026-04-15 | 已完成首轮结果 |
| [SNAPSHOT-021](snapshot-021-actor-teammate-obs-expansion.md) | actor teammate-obs expansion：原始未归一化 `021b` 为负结果，但归一化 `021b-norm` 在 local true-info 下已恢复为稳定正结果（`checkpoint-450 = 0.780 @ 500ep`，failure capture 已补）；`021c-B` 首轮 official-aligned run 已完成，最佳 official `baseline 500` 为 `checkpoint-480 = 0.794` | 2026-04-15 | `021b-norm` local 正结果 + `021c-B` 首轮正结果已完成 |
| [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md) | agent-id asymmetric shaping：修复后首轮 rerun 已完成，official `baseline 500` 最好点为 `checkpoint-280 = 0.818`；failure capture 说明该线显著压低 `low_possession`，但 head-to-head 仍低于 `017` 与 `024` | 2026-04-15 | 已完成修复后首轮结果 |
| [SNAPSHOT-023](snapshot-023-frozenteamcheckpointpolicy-opponent-adapter.md) | FrozenTeamCheckpointPolicy opponent adapter：team-level PPO (如 v4 PPO) 以 frozen opponent 身份进入 multiagent_player pool 的工程预案；conditional on 020 正向翻转 | 2026-04-15 | 预注册 / on-hold |
| [SNAPSHOT-024](snapshot-024-striker-defender-role-binding.md) | striker/defender role binding：修复后首轮 rerun 已完成，official `baseline 500` 最好点为 `checkpoint-270 = 0.842`；failure capture 将 `270` 确认为平衡主候选，head-to-head 显示其已小优 `018` 但仍低于 `017` | 2026-04-15 | 已完成修复后首轮结果 |
| [SNAPSHOT-025](snapshot-025-bc-champion-field-role-binding.md) | BC-champion warm-start + field-role binding：首轮 official `baseline 500` 已完成，official 最高点为 `checkpoint-140 = 0.822`，但结合 failure capture 后更可信的主候选是 `checkpoint-20 = 0.818`；结论是“冠军底座上的短程有益 fine-tune”，而非新冠军线 | 2026-04-17 | 已完成首轮结果 |
| [SNAPSHOT-025b](snapshot-025b-bc-champion-field-role-binding-stability-tune.md) | `025` 的稳定性修复并行臂：在把 `bad iters` 从 `84/200` 压到 `5/200` 之后，首轮 official `baseline 500` 峰值达到 `checkpoint-80 = 0.842`；failure capture 仍保留 `checkpoint-70 = 0.836 official / 0.840 capture` 作为结构更干净的强备选，而 head-to-head 已显示 `025b@70/80` 均压过 `017`，其中 `025b@70` 也压过 `024`，使 `025b` 进入当前冠军位 | 2026-04-17 | 已完成首轮结果 |
| [SNAPSHOT-026](snapshot-026-reward-liberation-ablation.md) | reward liberation ablation：首轮 `A/B/C/D-warm` 的 official / failure capture / 关键 H2H 已完成；`B-warm@250` 给出 official 峰值 `0.864`，但更可信的主候选已收紧到 `B-warm@170 = 0.842 official / 0.822 capture`。结论是“PBRS 是最强 liberation 候选，并在失败结构上有真实改善，但 `026` 还不是总冠军线” | 2026-04-17 | 已完成首轮结果 |
| [SNAPSHOT-027](snapshot-027-team-level-ppo-coordination.md) | team-level PPO scratch：回到上游 `team_vs_policy` 架构，一个 policy 同时看到两人 obs (672 维) 并输出联合动作 (MultiDiscrete)，从架构层面解决 per-agent 无法协调的根本限制；**对标 MAPPO+v2 scratch 而非 SOTA**。`027-A` 虽因 quota 中断且遇到 `checkpoint-700` 损坏，但已从 `checkpoint-690` 续跑并完成 `710+` eval 回填；按 `top 5% + ties + window` 的 official `baseline 500` 复核后，best official 收口为 `checkpoint-650/1230 = 0.804`，因此这条线被正式定性为“成立的 team-level scratch 正结果，但 ceiling 约 0.80，不是冠军线” | 2026-04-17 | 已完成首轮结果 |
| [SNAPSHOT-028](snapshot-028-team-level-bc-to-ppo-bootstrap.md) | team-level IL→BC→PPO bootstrap：以 **team-policy teacher → team-policy BC → team-policy PPO** 完整对标当前 SOTA；`028-A` 已完成 official / failure capture / 关键 H2H：official 峰值为 `checkpoint-1220 = 0.844`，但更可信的主候选收口为 `checkpoint-1060 = 0.810 official / 0.806 capture`。H2H 显示 `1060` 已明显压过 `027A@650`（`0.592`），但仍明显低于 `017 / 025b / 029B`，因此正式结论是“team-level BC warm-start 明确成立，但首轮还不是冠军挑战成功” | 2026-04-17 | 已完成首轮 verdict |
| [SNAPSHOT-029](snapshot-029-post-025b-sota-extension.md) | post-025b SOTA extension（per-agent 三路并行）：H1（`029-A=025b+PBRS`）否决（official 0.842 + H2H 0.458 输 025b）；H2（`029-B=B-warm@170→v2 handoff`）部分成立——**`@190 official 500=0.868` 为项目首次稳定突破，H2H 平 025b、压过 017/026B，成为新 SOTA**；H3（`029-C=opp pool`）弱正（`@270 official 0.848`）。`029B@190` failure capture 显示 `low_poss=26.2%`（PBRS 学到的 20.2% 被 v2 接手后冲掉，机制层面 H2 边界失败）| 2026-04-17 | 已完成首轮 verdict |
| [SNAPSHOT-030](snapshot-030-team-level-advanced-shaping-chain.md) | team-level advanced shaping chain（port 实验）：把 per-agent 的 advanced shaping trick（field-role binding / opp pool / PBRS）port 到 team-level base (028A@1220) 上。**前置方法论说明**：当前所有 trick 都是为 per-agent 设计，port 失败只能证明"per-agent 方法栈不适配 team-level 架构"，不能否决 team-level 架构本身的潜力——team-level-native 方法另起 snapshot | 2026-04-18 | 预注册 / 待实现 |

## 产物归档

- 官方 evaluator 原始日志：[`artifacts/official-evals/`](artifacts/official-evals/)
- checkpoint 扫描 CSV：[`artifacts/official-scans/`](artifacts/official-scans/)
- SLURM / batch 输出：[`artifacts/slurm-logs/`](artifacts/slurm-logs/)

## 续跑摘要口径

对于因 quota 中断后续跑的实验，canonical summary 不再是单个 trial 终端末尾打印的 `Training Summary`，而是 run root 下由 [print_merged_training_summary.py](../../scripts/tools/print_merged_training_summary.py) 生成的两份产物：

- `merged_training_summary.txt`
- `training_loss_curve_merged.png`

其中前者负责汇总断点前后各段的 best/final/segment 信息，后者负责提供跨 trial 合并后的训练曲线。

## 命名规范

- `snapshot-NNN-简短标题.md` — 每次实验或分析一个独立文件
- 编号递增，NNN 从 000 开始
- README.md 只做索引，不放具体实验内容

<!-- 模板：新建 snapshot-NNN-xxx.md 时复制以下结构

# SNAPSHOT-NNN: 实验标题

- **日期**: YYYY-MM-DD
- **负责人**:
- **目标**:
- **配置**:
  - 脚本:
  - 超参:
  - reward:
- **结果**:
  - 训练步数:
  - 最终胜率:
  - 曲线:
- **结论**:
- **相关**: （链接到相关 ADR / code-audit 章节）

-->
