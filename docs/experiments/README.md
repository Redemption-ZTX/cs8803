# 实验记录

> **📊 所有 policy 的 baseline WR / random WR / peer H2H 汇总见 [rank.md](rank.md)**（项目单一真相源；每次做完 eval 必须同步更新；现已严格区分 `official 500 / official 1000 / capture 500 / combined 1000 / H2H`，并为已有 H2H 补齐 `n / z / p`；含 H2H log 正确读法 §0）
>
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
| [SNAPSHOT-028](snapshot-028-team-level-bc-to-ppo-bootstrap.md) | team-level IL→BC→PPO bootstrap：以 **team-policy teacher → team-policy BC → team-policy PPO** 完整对标当前 SOTA；`028-A` 已完成 official / failure capture / 关键 H2H：official 峰值为 `checkpoint-1220 = 0.844`，但更可信的主候选收口为 `checkpoint-1060 = 0.810 official / 0.806 capture`。H2H 显示 `1060` 已明显压过 `027A@650`（`0.592`），但对 `029B@190` 仍只有 `0.462`，因此正式结论仍是“team-level BC warm-start 明确成立，但首轮还不是冠军挑战成功” | 2026-04-17 | 已完成首轮 verdict |
| [SNAPSHOT-029](snapshot-029-post-025b-sota-extension.md) | post-025b SOTA extension（per-agent 三路并行）：H1（`029-A=025b+PBRS`）否决（official 0.842 + H2H 0.458 输 025b）；H2（`029-B=B-warm@170→v2 handoff`）部分成立——`@190 official 500=0.868`，H2H 平 025b、压过 017/026B，形成**强挑战者 / 亚军位**；H3（`029-C=opp pool`）弱正（`@270 official 0.848`）。`029B@190` failure capture 显示 `low_poss=26.2%`（PBRS 学到的 20.2% 被 v2 接手后冲掉，机制层面 H2 边界失败）| 2026-04-17 | 已完成首轮 verdict |
| [SNAPSHOT-030](snapshot-030-team-level-advanced-shaping-chain.md) | team-level advanced shaping chain（port 实验）：`030-A` 与 `030-D` 首轮已完成。`030A@360` 以 `0.832 official / 0.842 capture` 成为更稳的 field-role 候选；`030D@320` 则打出更高 ceiling（`0.862 official / 0.820 capture`），并在 H2H 中对 `028A@1060` 给出 `0.536` 的方向性优势，但对 `025b@80` 的 rerun 已收紧到 `0.450`，说明其仍真实低于顶线。追加 H2H 进一步显示 `030D@320` 对 `030A@360` 只有 `0.504`，因此当前更稳的口径是“`030D` ceiling 更高、`030A` 更稳”，并已补上 `030D-control` 用于拆分 PBRS 与 continuation 效应 | 2026-04-18 | 已完成首轮结果 + follow-up 对照已就绪 |
| [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md) | team-level native 系列 #1（**网络架构**）：Siamese dual encoder（两人 obs 共享权重编码后融合）+ 可选 cross-agent attention。`031-A = dual encoder scratch + v2 shaping` 首轮已坐实**项目新 SOTA**（1000ep avg `0.860` + H2H 双胜 `029B@190` `0.552 ***` / `025b@080` `0.532 *`）；按预注册 §5 触发 `031-B` 激活（cross-attention 设计/超参/判据见 §12） | 2026-04-18 | `031-A` 完成首轮 / `031-B` 预注册已激活 |
| [SNAPSHOT-032](snapshot-032-team-level-native-coordination-aux-loss.md) | team-level native 系列 #2（**训练目标**）：`032-A` 与 `032-A-control` 首轮已完成。official `baseline 500` 上 `control@200 = 0.836` 略高于 `032-A@170 = 0.826`；H2H 中 `032-A@170` 对 `control@200 = 0.528`、对 `028A@1060 = 0.536` 都给出方向性正号，但当前样本下仍偏边缘，因此更稳的结论是“aux 没把 baseline ceiling 明显抬高，但在 direct H2H 上给出了值得继续追的方向性信号” | 2026-04-18 | 已完成首轮 verdict |
| [SNAPSHOT-032-next](snapshot-032-next-symmetric-action-aux.md) | `032` 的最小修正版：通过 symmetric bidirectional aux + 稳定 `aux_*` logging，基本排除了“plumbing 没接通”的解释。首轮 `official baseline 1000` 显示 `032-next-control@130 = 0.822` 明显高于 `032-next-A@110 = 0.793`；而 `progress.csv` 中 `aux_action_loss/acc` 又确实表明 aux 在学习，因此当前更稳的结论是“问题更偏 aux 目标本身，而不是实现接线” | 2026-04-18 | 已完成首轮结果 |
| [SNAPSHOT-033](snapshot-033-team-level-native-coordination-reward.md) | team-level native 系列 #3（**环境/reward**）：`033-A = 028A@1060 + spacing/coverage PBRS` 首轮 official / failure capture / base-H2H 已完成。official 最高点并列为 `checkpoint-80/130 = 0.826`；其中 `80` 的 capture 更稳（`0.786`），`130` 则在 H2H 中以 `0.518` 小胜 `028A@1060`。当前更稳的口径是“存在窄窗正信号，但尚不足以写成稳定升级” | 2026-04-18 | 已完成首轮结果 |
| [SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md) | deploy-time ensemble agent（PETS 启发）：旧版 per-agent triad `029B+025b+017` 已被首轮结果否决（`034-A = 0.806 / 0.984`，`3×029B` sanity `= 0.850 / 0.992`）；frontier mixed 则经历了两代主线：第一代 `031A+036D(+029B)` 已翻正，其中 `034C-frontier = 0.890 / 0.996`，但 `baseline 1000 = 0.843`、更像 peer-play rounded policy；第二代改用 `031B` 作为新 anchor 后，`2×031B = 0.914 / 0.996`、`031B+036D+029B = 0.904 / 0.998`（official `500`），并进一步给出 `baseline 1000 = 0.890`、`H2H vs 031B = 0.596`、`vs 029B = 0.590`、`vs 034C = 0.544`。当前 `034E-frontier` 已是 `034` 与 ensemble 主候选；后续 `034F-router`（heuristic conditional routing）首轮 `official 1000 = 0.862`，已被收为 negative result。 | 2026-04-19 | legacy triad 已否 / `034E` 主候选 / `034F` 否决 |
| [SNAPSHOT-035](snapshot-035-ppo-stability-backlog.md) | PPO stability backlog（pre-registration only）：从 HW1/HW2/HW3 学到的三个候选——smaller entropy_coeff / advantage normalization / twin value heads。本 snapshot 仅登记候选，不写完整设计；任一启动时另开 `035a/b/c` 子 snapshot | 2026-04-18 | backlog / 详细设计待启动时补 |
| [SNAPSHOT-036](snapshot-036-learned-reward-shaping-from-demonstrations.md) | learned reward shaping from demonstrations（HW4 MaxEnt IRL 启发）：路径 C 首轮 `036C-warm` 已完成，`official baseline 1000` 最好点为 `checkpoint-270 = 0.833`，而结合 `failure capture 500` 后更稳的 baseline-oriented 主候选收口为 `checkpoint-150 = 0.832 official / 0.844 capture`。首轮 peer H2H `036C@270 vs 029B@190 = 0.422` 说明它仍低于当前顶线；同时训练期出现多次 `kl/total_loss=inf`，因此当前更合理的 follow-up 是先修数值稳定性，再决定是否 rerun | 2026-04-18 | 已完成首轮结果（数值异常待修） |
| [SNAPSHOT-037 RETRACTED](snapshot-037-architecture-dominance-peer-play.md) | **已撤回**：2026-04-18 期间因读 H2H log 方向假设错误，曾构造了"team-level 架构击败 per-agent SOTA"的叙述；核实后全部反转。保留文件为错误反面教材 + log 格式说明 | 2026-04-18 | **RETRACTED** |
| [SNAPSHOT-038](snapshot-038-team-level-liberation-handoff-placeholder.md) | team-level liberation + handoff known-gap 已激活：首轮不再走旧占位版的混合 lane，而是按 `026 A/B/C/D` 做干净的 team-level 镜像对照。当前 Stage 1 统一以 `028A@1060` 为 warm-start，直接执行 `038-A/B/C/D` 四条 matched-budget liberation ablation，再从最强 mid-training 点进入 Stage 2 handoff | 2026-04-18 | active / Path 2 Stage 1 batch 已就绪 |
| [SNAPSHOT-036D](snapshot-036d-learned-reward-stability-fix.md) | 036C 的稳定性修复版（parallel 于 025→025b 的关系）；**与 [039](snapshot-039-airl-adaptive-reward-learning.md) 并行执行**。PPO 超参已在 025b 稳定化 baseline，所以 036D 的 fix 转向 reward-side：`λ: 0.01→0.003` + `finite-check + hard clip` + `warmup_steps=10000`（≈10 iter）。代码已实现：`learned_reward_shaping.py` 加 `warmup_steps` 参数 + finite-check；`create_rllib_env` + `train_ray_mappo_vs_baseline` 读 `LEARNED_REWARD_WARMUP_STEPS` 环境变量；smoke 通过（warmup 精确 skip N 步，NaN→0，Inf→±λ clip 全部工作）。Batch 就绪 | 2026-04-18 | 实现完成，待提交训练 |
| [SNAPSHOT-039](snapshot-039-airl-adaptive-reward-learning.md) | AIRL-inspired adaptive reward learning（Line 2 重工程 MaxEnt-flavored）；**与 [036D](snapshot-036d-learned-reward-stability-fix.md) 并行执行**。`§10 fix` 现已实现：adaptive callback 不再依赖失效的 wrapper-discovery 路径，`3 iter` smoke 已确认 `3/3` refresh 成功且 `broadcasted_to=8/9_workers`；当前 formal rerun 已启动。仍保留一个小 caveat：日志尚未区分 local worker 与 remote rollout workers，因此 `8/9` 目前按“8 个真正 rollout workers 已更新，local worker 可能无 wrapper”解读 | 2026-04-19 | 首轮 broken verdict 已出 / **fix 已实现并进入 formal rerun** |
| [SNAPSHOT-040](snapshot-040-team-level-stage2-on-031A.md) | team-level Stage 2 handoff on 031A SOTA base：把 026/038 体系内已知的 advanced shaping (PBRS / event / depenalized v2 / entropy-only) 作为 Stage 2 叠加在 `031A@1040 (1000ep 0.860)` 这个**项目首个对齐 per-agent 平台的 team-level base** 上，复现 per-agent `026 → 029` 路径上的 +2.6pp 增益。四条 lane (`040A/B/C/D`) 镜像 026/038；`040B` 仍是最高优先级，但四条现均已 runnable | 2026-04-18 | `040A/B/C/D` batch 已就绪 / 待启动 |
| [SNAPSHOT-041](snapshot-041-per-agent-stage2-pbrs-on-036D.md) | **方向 1 (per-agent Stage 2 镜像)**：把 026/038/040 的 advanced shaping 叠加在 per-agent learned-reward base `036D@150 (1000ep 0.860)` 上，与 [040](snapshot-040-team-level-stage2-on-031A.md) 形成"同 0.860 base / 不同来源" (架构 vs reward) 的直接对照。**041B 已完成**: 1000ep peak `0.852 @ ckpt 60` < 036D base 0.860 → **唯一退化 lane (-0.008pp)**，PBRS + learned reward 双 shaping 叠加冲掉 036D 优势。**041 路径终结**。详见 [§11](snapshot-041-per-agent-stage2-pbrs-on-036D.md#11-首轮结果2026-04-19041b-完成) | 2026-04-19 | 已 verdict / 路径否决 |
| [SNAPSHOT-042](snapshot-042-cross-architecture-knowledge-transfer.md) | **方向 2 (跨架构知识迁移)**：把 per-agent SOTA `029B@190` 学到的行为蒸馏进 team-level Siamese 架构 (`031A` 同款)。当前主线已从最初的 A1 weight surgery 转到 **`042A3 = factor-wise KL co-train`**：teacher=`029B@190`，student=`031A`-style Siamese，distill loss 在 team-level PPO 上与主损失并行训练。A3 现已实现并正常训练；早期 checkpoint-eval loader bug 已修，后续只需按同 root 继续收结果 | 2026-04-19 | **A3 已实现并在跑** |
| [SNAPSHOT-043](snapshot-043-frontier-selfplay-pool.md) | **方向 3 (frontier self-play / diversity curriculum)**：旧 prereg 的 `031A + baseline40/frontier60` 全量想法已收成更可实施的 `043A' / 043B' / 043C'` 族。`043A'`（`031B@1220` warm-start + `baseline 50% + 031A 20% + 029B 15% + 036D 15%`）formal 已完成并给出 `official baseline 1000 = 0.901 @ checkpoint-80`，但 peer 轴仍输 `034E`。随后 `043B'` 从 `043A'@80` 续跑，拿掉 `031A`、换进 frozen `031B@1220 + 029B + 036D`，并把 baseline 比例降到 `40%`；当前 best official 点已达到 **`checkpoint-440 = 0.904`**，follow-up 给出 **`H2H vs 031B = 0.600`**、**`vs 034E = 0.554`**。`043C'` 则进一步把 baseline 比例压到 `35%`，best `official baseline 1000 = 0.895 @ checkpoint-480`，但 follow-up 同时给出 **`capture 500 = 0.880`**、**`H2H vs 031B = 0.614`**、**`vs 034E = 0.576`**，并在 direct H2H 中对 `043B'` 拿到 `0.532` 的窄优势。当前更稳的口径是：**`043B'` 与 `043C'` 组成 `043` 的双主候选；前者 baseline 尖峰更高，后者更 rounded，但 direct H2H 幅度还不够写成硬替代。** | 2026-04-19 | `043A'/043B'/043C'` formal 已完成；`043B/043C` 现为双主候选 |
| [SNAPSHOT-044](snapshot-044-adversarial-specialists-league.md) | **AlphaStar-style league self-play**：先训两个 sparring specialist——**矛 (044A)** 用 `fast_win_rate` 作 train reward + eval metric (只奖励速胜)，**盾 (044B)** 用 `non_loss_rate` (ties=wins, 不输就行)。**关键简化**：specialist 目标通过 train reward + eval metric 直接对齐，不需要复杂 reward shaping。Phase 3 (`044C`) main agent (`031A@1040` 起点) 训 league pool = `baseline 30 + 矛 20 + 盾 20 + frontier 30`，比 [043 frontier-only](snapshot-043-frontier-selfplay-pool.md) 多了 specialist diversity。**044A/B 已完成**: spear 1000ep peak win_rate 0.766，shield 0.790，**双 specialist gate fail**。Shield 设计缺陷暴露 — vs baseline 全 0 ties (3500 ep)，reward `R(T)=+1` 从未触发，policy 退化为 offensive (median win step 35 = 同 spear)。**044C 推荐合并到 043** (drop specialist)。详见 [§11](snapshot-044-adversarial-specialists-league.md#11) | 2026-04-19 | 044A/B 已 verdict / 044C 待决 (合并 043 vs redesign) |
| [SNAPSHOT-045](snapshot-045-architecture-x-learned-reward-combo.md) | **架构 × Learned Reward Combo verdict — combo saturation H3 确认** ([§11](snapshot-045-architecture-x-learned-reward-combo.md#11-045a-首轮结果-2026-04-19-append-only)): 045A 1000ep peak 0.867 @ 180, mean 0.856。vs 031A 0.860 持平 (+0.007/-0.004pp 噪声内)。learned reward 在 Siamese 架构上**没拿到任何增益**。combo saturation H3 confirmed。**031A 上 6 个 reward-axis lanes 全 saturate** (040A-D + 045A + 042A3 range [0.860, 0.867])。045B / 045C 按预注册 §2.4 saturation gate **跳过** | 2026-04-19 | **已 verdict / lane 关闭（saturation）** |
| [SNAPSHOT-046](snapshot-046-cross-train-pair.md) | **Cross-Train Pair — Train vs Frozen 031A, Eval vs Baseline**：测试 "skill transitivity" 假设——如果训一个 P **唯一对手是 frozen 031A@1040 (current SOTA)**，让 P 达到 H2H vs 031A ≥ 0.55，eval P vs baseline 是否 ≥ 0.88 (突破 0.86 ceiling)。直接测 "0.86 是 PPO ceiling 还是 environmental noise floor"。`046B = warmstart + 100% frozen 031A` 主线 (3h 快验证)，`046A scratch` 14h 条件启动。**与 [043 frontier league](snapshot-043-frontier-selfplay-pool.md) 正交** (intensity vs diversity)。`FrozenTeamPolicy` adapter (`cs8803drl/core/frozen_team_policy.py`) 已实施；Phase A plumbing smoke + Phase B 5-iter 训练 smoke 双双 PASS（reward +0.5→+0.9, ~10s/iter on 8K batch, warmstart 正常），200-iter batch 已就绪 (`scripts/batch/experiments/soccerstwos_h100_cpu32_046B_warm_vs_frozen_031A_512x512.batch`) 待启动 | 2026-04-19 | 实现就绪 / 待 200-iter 启动 |
| [SNAPSHOT-047](snapshot-047-deployment-slot-swap-test.md) | **Deployment 0/1 Slot Swap Test**: [SNAPSHOT-022 §R1](snapshot-022-role-differentiated-shaping.md) / [SNAPSHOT-024 §10](snapshot-024-striker-defender-role-binding.md) 提出过 side-swap risk 但从未数据化。当前已完成最小 sanity subset：`031A@1040 normal/swap = 0.851 / 0.839`，`025b@80 normal/swap = 0.824 / 0.832`；**未观察到强 slot-swap 敏感性**，因此已足够解除 [SNAPSHOT-034 ensemble](snapshot-034-deploy-time-ensemble-agent.md) 的前置阻塞 | 2026-04-19 | **已完成最小 verdict** |
| [SNAPSHOT-048](snapshot-048-hybrid-eval-baseline-takeover.md) | **Hybrid Eval — DAGGER-from-baseline 上限测试**: 100ep verdict 已出（节点抢占 + Ray init 累积，1000ep 完成需 6-8h；100ep Δ 量级远超噪声 → kill 节省 GPU）。**4/4 trigger conditions 全部恶化 -7 到 -14pp** (C1_031A_α=0.74, C2_β=0.81, C3_036D_α=0.71, C4_β=0.73 vs C0=0.88/0.80)。swap% 越大恶化越严重。**DAGGER-from-baseline 路径死** — student (0.86) 已 outgrown baseline (0.50 self-play)，所有 "imitate baseline" 类方法上限 ≈ baseline self-play。**DAGGER 框架本身未死**: 031B@1220 (0.882) / 034 ensemble / agent2 作为 teacher 仍可探索（必须另开 snapshot）| 2026-04-19 | **已 verdict / DAGGER-from-baseline 路径关闭** |
| [SNAPSHOT-049](snapshot-049-env-state-restore-investigation.md) | **Soccer-Twos Env State Restore Capability — MCTS 路径可行性验证**：6 个 smoke test (initial + 5 follow-up) 验证 `EnvConfigurationChannel.set_parameters` 实际行为。结论：**ball position/velocity SET 与 player rotation_y SET mid-episode 生效，但 player position/velocity SET 即使 pre-reset 也失败**（C# binary 没实现 PLAYER_POSITION handler）。Full MCTS over agent actions 路径**死**；ball-trajectory oracle 仍可行但 ROI 远低于 ensemble (034) / hybrid (048)。副产物：确认 `info["ball_info"]["position"]` 可拿 absolute world coords，降低 048 R1 风险 | 2026-04-19 | **已完成 verdict** |
| [SNAPSHOT-050](snapshot-050-cross-student-dagger-probe.md) | **Cross-Student DAGGER Probe** — 048 §7.4 strategic 延伸：把 baseline-takeover 替换成 frontier teacher (036D / 031B / 031A 互相 cross)。Phase 1.1 (036D → 031A) 完成: **Δ +1pp NEUTRAL** (vs 048 baseline -7~-14pp 改善 +8~+15pp，验证 H1 teacher quality 单调)。Phase 1.2 (031B → 031A 架构内 stronger teacher) + Phase 1.3 (031A → 036D failure mode 互补的反向 pair) 进行中。Phase 2 (真 DAGGER training) 启动条件 = Phase 1 任一 Δ ≥ +0.03 | 2026-04-19 | Phase 1.1 完成 / 1.2+1.3 进行中 |

## 产物归档

- 官方 evaluator 原始日志：[`artifacts/official-evals/`](artifacts/official-evals/)
- checkpoint 扫描 CSV：[`artifacts/official-scans/`](artifacts/official-scans/)
- SLURM / batch 输出：[`artifacts/slurm-logs/`](artifacts/slurm-logs/)

## 续跑与摘要口径

默认 workflow 已更新为 **同 `RUN_NAME` / 同 canonical run root 的 in-place resume**：

- 首选参数：
  - `RESUME_CHECKPOINT`
  - `RESUME_TIMESTEPS_DELTA`
- 兼容旧参数：
  - `RESTORE_CHECKPOINT`
  - `RESTORE_TIMESTEPS_DELTA`
- warm-start 与续跑明确区分：
  - `WARMSTART_CHECKPOINT` = 换初始化
  - `RESUME_CHECKPOINT` = 沿同一条训练 lineage 继续

默认不再推荐“新建一个空 run root 然后 resume”。如果只是 quota / walltime 中断，续跑应继续写入**原 run root**，这样：

- 原 root 下多个 trial segment 天然构成同一条 lineage
- `progress.csv / checkpoint_eval.csv / checkpoints` 都留在同一棵目录里
- 不需要事后再人工拼接 old/new 顶层 experiment 目录

对于这种 canonical 同 root 续跑，summary 口径仍不是单个 trial 末尾打印的 `Training Summary`，而是 run root 下由 [print_merged_training_summary.py](../../scripts/tools/print_merged_training_summary.py) 生成的两份产物：

- `merged_training_summary.txt`
- `training_loss_curve_merged.png`

其中前者汇总断点前后各段的 best/final/segment 信息，后者提供跨 trial 合并后的训练曲线。

只有在**历史上已经 split 成多个顶层 run root** 的情况下，才使用补救工具：

- [absorb_resume_into_run_root.py](../../scripts/tools/absorb_resume_into_run_root.py)
  - 把 split resume trial 吸回原 canonical root
- [materialize_merged_run_view.py](../../scripts/tools/materialize_merged_run_view.py)
  - 在不能改动原目录时，生成只读 merged view

这两个工具是历史补救，不是默认工作流。

## 命名规范

- `snapshot-NNN-简短标题.md` — 每次实验或分析一个独立文件
- 编号递增，NNN 从 000 开始
- README.md 只做索引，不放具体实验内容
- `ray_results/` 下的 **新实验目录**推荐把 `snapshot/lane` 放在最前面，便于 quota 清理、文档回填和结果 grep
- 推荐格式：`NNNA_<family>_<method>_<base>_<timestamp>` 或 `NNN_<family>_<method>_<timestamp>`
- 例子：
  - `030A_team_field_role_on_028A1060_20260418_...`
  - `030D_team_pbrs_on_028A1060_20260418_...`
  - `031_team_dual_encoder_20260418_...`
- **旧 run 不回改名**：已经写进 snapshot / H2H / failure-capture / official-eval 日志的历史路径保持不动，避免链接漂移

## 评测口径（2026-04-18 更新）

随着 `028/029/030/033` 这些 frontier 点的差距越来越小，单次 official `500` 已经不总是足够稳定。后续默认采用分层评测：

- `official 500`
  - 默认窗口扫描配置
  - 作用：从 `top 5% + ties + window` 中筛主候选
- `failure capture 500`
  - 默认稳定性配套
  - 作用：检查 official 高点是否被高估，并读取失败结构
- `H2H 500`
  - 默认相对强度配套
  - 作用：判断是否真的超过 base、旧冠军或同代对照
- `official 1000`
  - 非默认，只给 finalists
  - 触发条件：
    - official 与 capture 差超过约 `0.03`
    - H2H 与 official 给出不同排序
    - 候选 official 差距在 `0.015~0.020` 以内
    - 该点要被写入 snapshot 作为正式主候选或冠军位判断

这意味着后续我们仍先用 `official 500` 做广覆盖，但像 `030D@320` 这种已经进入前沿比较、且 official/capture 明显分裂的点，应优先升级到 `official 1000`。

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
| [SNAPSHOT-051](snapshot-051-learned-reward-from-strong-vs-strong-failures.md) | **Learned Reward from Strong-vs-Strong Failures**: 把 reward model 训练数据从 "strong-vs-baseline failure" (036/045 当前用) 切换到 "**strong-vs-strong failure**" (031A vs 031B、029B vs 025b 等 H2H 中的输方 trajectory)。理论 motivation: baseline self-play 0.5 WR → 失败 pattern 噪声大；strong-vs-strong 失败是 "高水平对抗中真正决定胜负的 state"，signal 密度高。预期 1000ep peak ≥ 0.890 (vs 045A 0.867)。Stage 0 需采集 5 pairs × 400 ep H2H + 双侧 save losses (~2.8h GPU); Stage 1 重训 reward model; Stage 2 用新 reward 训 student | 2026-04-19 | 预注册 / 待 045B + Phase 1.3b verdict 后启动 |
| [SNAPSHOT-052](snapshot-052-031C-transformer-block-architecture.md) | **031C Transformer Block — 架构 step 3**: full 主线现已可运行：`031B + true MHA + FFN + LayerNorm + residual + transformer-style 512-dim merge`。同时保留两条分解对照：`052A / 031C-min`（只测 FFN/residual/norm）与 `052B / 031C-mha`（在 `031C-min` 上只替换成 true MHA）。当前三条 model / trainer gate / batch 都已就绪，等待首轮 scratch run。 | 2026-04-19 | **lane 关闭** — 052A -8.2pp / 052 -10.8pp REGRESSION (架构 step 3 失败) |
| [SNAPSHOT-053](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md) | **Outcome-PBRS Reward from Calibrated Predictor**: 用 calibrated trajectory predictor (val_acc 93.8% / Brier-calibrated 0.835) 把 per-step state → 胜率，按 PBRS (Ng 1999) potential function 形式 (`r_PBRS = γφ(s')-φ(s)`) 注入 reward。policy invariance 保证理论上不改最优策略，但 dense signal 加速收敛。053A combo (v2 + outcome-PBRS) 200 iter 在 011-23-0 训练中。 | 2026-04-19 | 预注册 / 053A 训练中 (~4h) |
| [SNAPSHOT-054](snapshot-054-mat-min-cross-agent-attention.md) | **MAT-min — Cross-AGENT Attention 残差块（架构 step 3 重试）**: 052 失败后换路径——不再加 transformer block (FFN/LN)，而是在 031B 上插入最小 cross-AGENT attention (Q/K/V projections, V zero-init for graceful degrade to 031B at start)。+98K params (+21%)，规避 052 的 R1 (over-param) 与 R2 (LayerNorm × PPO)。理论支撑: MAT (Wen 2022) on SMAC; ATOC/AC-Atten/TarMAC 系列。1250 iter scratch on 5017695/011-18-0。 | 2026-04-19 | peak 0.880 @ 1100/1230 tied 031B (Δ=-0.002) |
| [SNAPSHOT-055](snapshot-055-distill-from-034e-ensemble.md) | **Distill from 034E ensemble — 🥇 Project SOTA**: 把 034E ensemble (031B@1220 + 045A@180 + 051A@130) 蒸馏进单个 031B-arch student。factor-prob averaging teacher + KL loss + α decay schedule (0.05 → 0)。理论: Hinton 2015 + Rusu 2016 + Czarnecki 2019。 | 2026-04-19 | **peak 0.907 combined 2000ep** (+0.025 vs 031B, z=2.08 `*`); H2H vs 031B *** / 028A *** / 029B *** / 056D marginal |
| [SNAPSHOT-056](snapshot-056-simplified-pbt-lr-sweep.md) | **Simplified PBT LR sweep (4 lanes A/B/C/D)**: 独立 4 LR lanes (1e-4 / 7e-5 / 3e-5 / 1.5e-4 / 3e-4),无 population exchange。目的: LR axis ablation。 | 2026-04-19 | 056D lr=3e-4 peak **0.891** marginal +0.009 vs 031B; 056A/B 小 LR 降，056C 训练中 |
| [SNAPSHOT-057](snapshot-057-rnd-intrinsic-motivation.md) | **RND Intrinsic Motivation**: Random Network Distillation (Burda 2019) 给 policy 加 intrinsic reward 鼓励 exploration。target_net + predictor_net,per-step reward 增量。 | 2026-04-19 | **dead end** — peak 0.56 @50ep, RND intrinsic 干扰 PPO training 基础 |
| [SNAPSHOT-058](snapshot-058-real-curriculum-learning.md) | **Real Curriculum Learning (simplified)**: 4-phase opponent schedule random → mixed-low → mixed-high → baseline,fixed iter boundaries 0/200/500/1000,无 adaptive gate,+v2 shaping。 | 2026-04-19 | peak 0.847 @ 950 §3.5 触发但 50ep/1000ep gap 0.073pp 异常大 → 路径未关闭,062 follow-up |
| [SNAPSHOT-059](snapshot-059-055lr3e4-combo.md) | **055 + LR=3e-4 combo (Tier 1a)**: 055 distill 配置 + 056D 胜出 LR。假设 distill + 高 LR 叠加破 055 combined 0.907 ceiling。 | 2026-04-20 | 预注册 / 训练中 (ETA ~10h) |
| [SNAPSHOT-060](snapshot-060-054M-mat-medium.md) | **054M MAT-medium (Tier 1b)**: 054 + pre-attention LayerNorm + FFN residual block (single-head),graceful degrade init (zero FFN + zero V proj → 起始等价 031B)。介于 054 (0.880) 和失败的 052 (REGRESSION) 之间。 | 2026-04-20 | 预注册 / 训练中 (ETA ~12h) |
| [SNAPSHOT-061](snapshot-061-055v2-recursive-distill.md) | **055v2 recursive distill (Tier 2)**: 5-teacher ensemble = {055@1150, 031B@1220, 045A@180, 051A@130, 056D@1140} + LR=3e-4。self-distillation (Furlanello 2018 BAN)。 | 2026-04-20 | 预注册 / 训练中 (teacher ckpt sanitized per Option B, ETA ~10h) |
| [SNAPSHOT-062](snapshot-062-curriculum-noshape-adaptive.md) | **Curriculum + No-shape + Adaptive Phase Gating (Tier A4.1/A4.2)**: 058 升级 — WR-gated phase transitions + 3-variant boundary sweep (062a/b/c) + remove v2 shaping (reward ≈ 2×(WR-0.5) → direct WR gate signal)。 | 2026-04-20 | 预注册 / 3 variants 训练中 |
| [SNAPSHOT-063](snapshot-063-055-temp-sharpening.md) | **055-temp Temperature Sharpening Distill (Tier 3a)**: 隔离 T=1.0 → T=2.0 单变量,其他与 055 完全相同 (3-teacher 034e ensemble + LR=1e-4 + α schedule)。理论: Hinton 2015 T=2-4 最优 + Rusu 2016 policy distillation;当前 T=1 可能浪费 teacher 不确定性信号。launcher 与标准 H100 batch 已落实。 | 2026-04-20 | 实现完成 / launcher + batch ready |
| [SNAPSHOT-064](snapshot-064-056-pbt-full.md) | **056-PBT-full True PBT (Tier 3b alt) — PAUSED**: 8-member population on 2 GPU + burn-in 500 iter + mini-eval ranking signal + LR/CLIP mutation。Ray `PopulationBasedTraining` built-in 方案 A。user 暂停: 60h GPU + 不确定 yield, ROI 不足,改走 065 + 066 替代。 | 2026-04-20 | **PAUSED** (代替为 snapshot-065 + snapshot-066) |
| [SNAPSHOT-065](snapshot-065-056EF-lr-sweep-upward.md) | **056E/F LR Sweep Upward**: 继续 056 单调趋势 (1e-4 → 3e-4 显示上升),测试 lr=5e-4 (056E) + lr=7e-4 (056F) 是否继续。0 工程,24h GPU 总预算 (vs PBT-full 60h)。 | 2026-04-20 | 预注册 / 训练中 (056E on 5022390, 056F on 5022393) |
| [SNAPSHOT-066](snapshot-066-progressive-distill-BAN.md) | **Progressive Distill BAN (Tier 3b)**: 2-stage 自 distill — 066A pure self-distill from 055@1150 (single teacher), 066B weighted 4-teacher (055 weight 0.5, 034e 3 teachers each 0.166)。代码完成 (teacher_weights env var + _FrozenTeamEnsembleTeacher 加权),smoke PASS。 | 2026-04-20 | 训练中 (066A on 5024106, 066B on 5024108) |
| [SNAPSHOT-067](snapshot-067-temperature-sweep-full.md) | **Temperature Sweep Full Map (Tier 3a 升级)**: 5-point sweep T ∈ {1.0 (anchor=055), 1.5, 2.0 (在跑/063), 3.0, 4.0} → 直接拿 monotonic / peaked / flat 三 pattern 给 temperature 路径定论。0 代码,3 new lanes × 12h = 36 GPU-hours。 | 2026-04-20 | 训练中 (T=3.0 on 5024111, T=1.5/4.0 待节点) |
| [SNAPSHOT-068](snapshot-068-055PBRS-distill.md) | **055+053D Distill with PBRS reward**: 在 055 distill 基础上把 v2 shaping 替换成 053D PBRS reward path,隔离 reward-axis 对 distill SOTA 的增益。2 variants (068_warm / 068_scratch)。 | 2026-04-21 | 预注册 / 训练中 |
| [SNAPSHOT-069](snapshot-069-055v2-plus-043-frontier-pool.md) | **055v2 + 043 Frontier Pool**: 在 055v2 recursive distill 基础上加入 043 frontier self-play pool opponent,测试 recursive distill + diversity curriculum 组合拳。 | 2026-04-21 | 预注册 / 训练中 (5028750) |
| [SNAPSHOT-070](snapshot-070-pool-B-divergent-distill.md) | **Pool B — Divergent-Path Ensemble Distill**: 3-teacher {055 + 053Dmirror (PBRS) + 062a (curriculum no-shape)},最大化 reward-path divergence。 | 2026-04-21 | 预注册 / 训练中 (5028916) |
| [SNAPSHOT-071](snapshot-071-pool-A-homogeneous-distill.md) | **Pool A — Homogeneous-Family Ensemble Distill**: 3-teacher {055 + 055v2@peak + 056D} 同家族组合池,等权 1/3。blocked on 055v2_extend peak ID。 | 2026-04-21 | 预注册 / blocked |
| [SNAPSHOT-072](snapshot-072-pool-C-newcomer-frontier.md) | **Pool C — Newcomer + Frontier (Max-Diversity 4-Teacher)**: teachers {055v2@peak + 056D + 054M@peak + 062a}, weighted 0.3/0.25/0.2/0.25。KL-conflict risk (054M cross-agent-attn vs 3 Siamese) 以保守 weight 处理。 | 2026-04-21 | 预注册 / blocked on 055v2 + 054M peaks |
| [SNAPSHOT-073](snapshot-073-pool-D-cross-reward.md) | **Pool D — Cross-Reward Signal Diversity Distill**: 3-teacher {055 (v2) + 068_* (PBRS) + 062a (no-shape)}, 纯 reward-axis orthogonality。 | 2026-04-21 | 预注册 / blocked on 068 |
| [SNAPSHOT-074](snapshot-074-034-next-deploy-time-ensemble.md) | **074-Family Deploy-Time Ensemble (ZERO training)**: 5 variants (A/B/C/D/E) 探测 member-selection 方法 (blood / arch / H2H / bucket / predictor-rerank)。**All 5 verdicts done 2026-04-21 07:30 EDT** — 4/5 tied at ~0.90 arithmetic-mean plateau, 1/5 regression (074B arch-mix -0.026pp), 1/5 drag (074E predictor -0.010pp)。**Deploy-time prob-avg ensemble paradigm CLOSED for this project**。Grading primary 保持 055@1150。 | 2026-04-21 | **已 verdict / family CLOSED (全 5 variants 结束)** |
| [SNAPSHOT-074B](snapshot-074B-arch-diversity.md) | **074B arch-diversity ensemble {055 + 054M@1230 + 062a}** — 测试 MAT + Siamese 混合 prob-avg。**§3.5 REGRESSION** (0.877, vs 074A Δ-0.026), cross-arch averaging 产生决策冲突,**deploy-time ensemble 需成员同质架构** lesson 入 snapshot-034 meta。 | 2026-04-21 | **已 verdict** |
| [SNAPSHOT-074C](snapshot-074C-h2h-orthogonal.md) | **074C H2H-orthogonal ensemble {055 + 056D + 053Dmirror}** — 按 `055 vs 056D = 0.536 NOT sig` 最平 pair 选员。**§3.3 tied** (0.902); 证明 **H2H orthogonality ≠ actionable decision-space orthogonality**,第 2 次独立复现 Siamese prob-avg tied pattern。 | 2026-04-21 | **已 verdict / 074C 关闭** |
| [SNAPSHOT-074D](snapshot-074D-failure-bucket-orthogonal.md) | **074D failure-bucket-orthogonal ensemble {055 + 055v2@1000 + 053Acont@430}** — 按 bucket distribution 异构选员。**§3.3 tied (borderline)** (0.900); 证明 **bucket label 是 hindsight signal, 不对应 realtime action diff**, 第 3 次独立复现 tied pattern。 | 2026-04-21 | **已 verdict / 074D 关闭** |
| [SNAPSHOT-074E](snapshot-074E-predictor-rerank.md) | **074E Outcome-Predictor Rerank ensemble (074A members + v3 predictor top-K rerank)** — PETS 之外的 value-head ensemble direction,novel 设计点。**§3.4 marginal regression** (0.893, vs 074A Δ-0.010); predictor val_acc 0.835 distribution-shift / α heuristic / averaging-is-correct-uncertainty 三机制合并 drag。**Deploy-time ensemble paradigm CLOSED**。 | 2026-04-21 | **已 verdict / 074E 关闭 / family CLOSED** |
| [SNAPSHOT-075](snapshot-075-strategic-synthesis-toward-0.93.md) | **Strategic Synthesis — 从 0.907 推向 0.93**: 项目全线 reward/arch/distill/ensemble/curriculum 5 轴 big-jump anatomy + 剩余 roadmap; DIR-A/B/C (076/077/078) 三条 student-side distill upgrades 的总设计 rationale。 | 2026-04-21 | 战略综述 (snapshot-only) |
| [SNAPSHOT-076](snapshot-076-wide-student-distill.md) | **DIR-A Wide-Student Distill**: 扩 student capacity (arch env var extension ~2h) 回收 teacher-student gap。distill from 055 teacher + Siamese wider arch。 | 2026-04-21 | 实现完成 / 训练中 (5028920) |
| [SNAPSHOT-077](snapshot-077-per-agent-student-distill.md) | **DIR-B Per-Agent Student Distill**: per-agent shared-policy student 吸收 team-level teacher ensemble (slot-0 convention marginal projection)。`cs8803drl/branches/per_agent_distill.py` engineering complete + smoke passed。 | 2026-04-21 | 实现完成 / 训练中 (5029745) |
| [SNAPSHOT-078](snapshot-078-dagger-distill.md) | **DIR-C DAGGER-Style Two-Stage Distill**: 解决 online-distill state-shift,分阶段 behavior cloning + DAGGER iter。blocked on 2-stage infrastructure (~3-5 days)。 | 2026-04-21 | 预注册 / blocked on infra |
