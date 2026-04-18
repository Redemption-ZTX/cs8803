# 文档中心

## 团队

| 成员 | 角色 | 主要职责 |
|------|------|----------|
| | Co-Lead | |
| | Co-Lead | |

## 文档地图

```
根目录/
├── CLAUDE.md                              # AI 协作入口（行为规则 + 架构摘要 + 保护等级）
├── CHANGELOG.md                           # 版本记录（Keep a Changelog）
├── .claude/settings.json                  # Claude Code 配置（showThinkingSummaries 等）
├── curriculum.yaml                         # 课程学习任务配置
├── requirements.txt                        # 依赖规格
├── sitecustomize.py                        # Python 兼容性补丁
├── cs8803drl/
│   ├── core/
│   │   ├── checkpoint_utils.py            # checkpoint 解析 canonical source
│   │   ├── utils.py                       # 环境工厂、reward shaping、固定队友注入
│   │   └── soccer_info.py                 # 比赛 info 解析与 reward shaping 纯逻辑
│   ├── branches/
│   │   ├── obs_summary.py                 # summary observation 特征扩展
│   │   ├── lstm_transfer.py               # feed-forward → LSTM warm-start 辅助
│   │   ├── imitation_bc.py                # team-level behavior cloning MLP policy 与 checkpoint 格式
│   │   ├── role_specialization.py         # 双 policy 角色分工辅助模块
│   │   ├── shared_role_token.py           # 共享策略 role token 映射与 warm-start 辅助
│   │   ├── shared_central_critic.py       # centralized critic 模型、回调与 warm-start 辅助
│   │   └── expert_coordination.py         # 双专家协调逻辑与场景辅助
│   ├── training/
│   │   ├── train_ray_base_team_vs_random.py
│   │   ├── train_ray_base_team_vs_baseline.py
│   │   ├── train_ray_base_ma_teams.py
│   │   ├── train_ray_team_vs_random_shaping.py
│   │   ├── train_bc_team_policy.py
│   │   ├── train_ray_mappo_vs_baseline.py
│   │   ├── train_ray_team_vs_random_summary_obs.py
│   │   ├── train_ray_team_vs_random_lstm.py
│   │   ├── train_ray_role_specialization.py
│   │   ├── train_ray_shared_policy_role_token.py
│   │   ├── train_ray_shared_central_critic.py
│   │   ├── train_ray_selfplay.py
│   │   └── train_ray_curriculum.py
│   ├── deployment/
│   │   ├── trained_ray_agent.py
│   │   ├── trained_team_ray_agent.py
│   │   ├── trained_ma_team_agent.py
│   │   ├── trained_bc_team_agent.py
│   │   ├── trained_summary_ray_agent.py
│   │   ├── trained_lstm_ray_agent.py
│   │   ├── trained_role_agent.py
│   │   ├── trained_shared_role_agent.py
│   │   ├── trained_shared_cc_agent.py
│   │   ├── trained_fixed_teammate_agent.py
│   │   └── trained_dual_expert_agent.py
│   └── evaluation/
│       ├── evaluate_matches.py
│       └── eval_rllib_checkpoint_vs_baseline.py
├── scripts/
│   ├── README.md                          # scripts 分类与放置规则
│   ├── setup/
│   ├── eval/
│   ├── tools/
│   └── batch/
│       ├── starter/
│       ├── base/
│       ├── adaptation/
│       └── experiments/
│
└── docs/
    ├── README.md                          ← 你在这里
    │
    ├── architecture/                      # 架构与工程
    │   ├── overview.md                    # 架构总览 + 上游差异 + 目录结构
    │   ├── code-audit.md                  # 代码审计索引
    │   ├── code-audit-000.md              # 接手时代码审计 snapshot
    │   ├── engineering-standards.md       # 工程规范（环境搭建/commit流程/实验迭代/环境变量）
    │   └── adr/                           # 架构决策记录
    │       ├── README.md                  # ADR 治理规则 + 模板
    │       ├── topic-map.md               # 主题索引
    │       └── 001-training-framework.md  # Ray RLlib 选型（继承自上游）
    │
    ├── management/                        # 项目管理
    │   ├── ROADMAP.md                     # 路线图与里程碑
    │   ├── WORKLOG.md                     # 工作日志
    │   ├── cleanup-log.md                 # 磁盘清理日志（删除内容与配额变化）
    │   ├── directory-governance.md        # 工作目录治理与放置规则
    │   └── deploy-and-verify.md           # 部署与验证交接文档
    │
    ├── plan/                              # 跨 snapshot 的阶段计划
    │   ├── README.md                      # plan 文档索引与维护规则
    │   ├── plan-001-il-baseline-exploitation.md # imitation learning / baseline exploitation 主线
    │   └── plan-002-il-mappo-dual-mainline.md # 当前总路线：IL / MAPPO 双主线
    │
    ├── experiments/                       # 实验记录
    │   ├── README.md                      # 实验索引 + 命名规范
    │   ├── snapshot-000-prior-team.md     # 前任团队实验结果分析
    │   ├── snapshot-001-reward-shaping-runtime-validation.md # reward shaping 真实环境验证
    │   ├── snapshot-002-h100-overlay-env-validation.md       # H100 overlay 环境与 GPU smoke test
    │   ├── snapshot-003-official-evaluator-realignment.md   # 官方 evaluator 对齐与 warm-start 重选
    │   ├── snapshot-004-role-ppo-and-shared-policy-ablation.md # role/shared-policy 对比实验
    │   ├── snapshot-005-observation-memory-and-centralized-critic-ablation.md # summary/LSTM/centralized critic 复盘
    │   ├── snapshot-006-fixed-teammate-and-dual-expert-rethink.md # dual-expert 与 fixed-teammate 主线重想
    │   ├── snapshot-007-base-lane-reset-and-directory-reorg.md # 目录分层重构与 base/adaptation lane 划分
    │   ├── snapshot-008-starter-aligned-base-model-lane.md # starter 对齐的 scratch base model lane
    │   ├── snapshot-009-base-team-vs-baseline-lane.md # 直接对准 baseline 的 scratch base lane
    │   ├── snapshot-010-shaping-v2-deep-zone-ablation.md # shaping-v2 deep-zone / negative-C A/B
    │   ├── snapshot-011-shaping-v3-progress-gated-ablation.md # v3 progress-gated shaping 首轮负结果收口
    │   ├── snapshot-012-imitation-learning-bc-bootstrap.md # baseline teacher 轨迹采集与 BC bootstrap 预注册
    │   ├── snapshot-013-baseline-weakness-analysis.md # baseline weakness analysis 预注册
    │   ├── snapshot-014-mappo-fair-ablation.md # MAPPO / centralized critic 公平对照首轮结果
    │   ├── snapshot-015-behavior-cloning-team-bootstrap.md # team-level BC 训练器、wrapper 与 batch 落地
    │   ├── snapshot-016-shaping-v4-survival-anti-rush-ablation.md # v4 survival / anti-rush shaping 首轮结果：最佳 PPO 平台约 0.768
    │   ├── snapshot-017-bc-to-mappo-bootstrap.md # BC -> MAPPO bootstrap 进行中：player-level bridge 与 warm-start smoke 已通过
    │   ├── snapshot-018-mappo-v2-opponent-pool-finetune.md # opponent-pool 首轮结果：ckpt290 = 0.812，为当前最强 baseline 主线
    │   ├── snapshot-019-mappo-v2-opponent-pool-anchor-heavy-rebalance.md # opponent-pool 配比消融：baseline30 / anchor30 / v1 20 / bs0 20
    │   ├── snapshot-020-mappo-v4-fair-ablation.md # MAPPO + v4 shaping 公平对照：先看 v4 风格是否值得进入 MAPPO 主干
    │   ├── snapshot-021-actor-teammate-obs-expansion.md # actor teammate-obs expansion：已拆成 021b 保真诊断 / normalized rerun 与 021c auxiliary-head 型 official-aligned 重跑设计
    │   ├── snapshot-022-role-differentiated-shaping.md # agent-id asymmetric shaping：先测“打破 reward 对称性本身”是否有用
    │   ├── snapshot-023-frozenteamcheckpointpolicy-opponent-adapter.md # team-level PPO frozen-opponent adapter：条件式工程预案
    │   ├── snapshot-024-striker-defender-role-binding.md # 严格 striker/defender role binding：按 spawn/field semantics 每局动态绑定真实角色
    │   ├── snapshot-025-bc-champion-field-role-binding.md # 在 017 冠军底座上测试保守版 field-role binding 是否还能继续增益
    │   ├── snapshot-025b-bc-champion-field-role-binding-stability-tune.md # 025 的并行稳定性修复版：首轮结果表明优化收紧可把可信 baseline 成绩抬到冠军竞争区
    │   ├── snapshot-026-reward-liberation-ablation.md # reward liberation A/B/C/D 对照：测试 reward 释放与 exploration 两类解释
    │   ├── snapshot-027-team-level-ppo-coordination.md # team-level PPO scratch：从 team_vs_policy 架构直接训练联合动作策略
    │   ├── snapshot-028-team-level-bc-to-ppo-bootstrap.md # team-policy teacher -> team-level BC -> team-level PPO：在 team policy 语义下完整重建 SOTA bootstrap
    │   └── artifacts/                     # 官方评估日志、扫描 CSV、SLURM 日志归档
    │
    └── references/                        # 参考资料
        ├── papers.md                      # 论文笔记
        ├── links.md                       # 外部链接
        ├── Final Project Instructions Document.md   # 作业要求（Markdown 版）
        ├── Final Project Instructions Document.pdf  # 作业要求（PDF 原件，SSOT）
        ├── DRL_HW1.ipynb                  # HW1: Policy Gradients & PPO
        ├── DRL_HW2_student.ipynb          # HW2: DQN, DDPG, SAC
        └── upstream-README.md             # 上游原版 README（归档）
```

## 文档导航

### 接手项目？从这里开始

1. [architecture/overview.md](architecture/overview.md) — 了解项目全貌、上游差异、目录结构
2. [architecture/code-audit-000.md](architecture/code-audit-000.md) — 接手时逐模块代码分析、问题与改进方向
3. [references/Final Project Instructions Document.md](references/Final%20Project%20Instructions%20Document.md) — 作业要求与评分标准

### 开始开发？看这些

4. [architecture/engineering-standards.md](architecture/engineering-standards.md) — 环境搭建、commit 流程、实验迭代、环境变量速查
5. [architecture/adr/README.md](architecture/adr/README.md) — 架构决策记录与模板
6. [management/ROADMAP.md](management/ROADMAP.md) — 路线图与任务分配
7. [management/directory-governance.md](management/directory-governance.md) — 目录分层、职责边界、兼容保留项
8. [management/cleanup-log.md](management/cleanup-log.md) — 磁盘清理日志，记录每次删除内容与配额变化
9. [management/deploy-and-verify.md](management/deploy-and-verify.md) — 部署与验证交接文档（新人/新环境从这里开始）
   包含 H100 overlay 环境说明与验证入口
10. [plan/plan-002-il-mappo-dual-mainline.md](plan/plan-002-il-mappo-dual-mainline.md) — 当前阶段总计划：IL / MAPPO 双主线

### 跑实验？记录在这里

11. [experiments/README.md](experiments/README.md) — 实验记录模板
12. [management/WORKLOG.md](management/WORKLOG.md) — 每日工作日志
13. [../scripts/README.md](../scripts/README.md) — `scripts/setup` / `scripts/eval` / `scripts/tools` / `scripts/batch` 分类说明

### 需要参考资料？

14. [references/papers.md](references/papers.md) — 论文笔记
15. [references/links.md](references/links.md) — 外部链接与工具
16. [references/DRL_HW1.ipynb](references/DRL_HW1.ipynb) — HW1 实现参考（REINFORCE → VPG → PPO）
17. [references/DRL_HW2_student.ipynb](references/DRL_HW2_student.ipynb) — HW2 实现参考（DQN → DDPG → SAC）

## 文档维护原则

1. 涉及架构变更须同步更新 [overview.md](architecture/overview.md) 或新增 [ADR](architecture/adr/README.md)
2. 每次训练实验在 [experiments/](experiments/README.md) 中记录
3. 每日简写 [WORKLOG](management/WORKLOG.md)，保证双方信息同步
4. 重大决策写 [ADR](architecture/adr/README.md) — 说清选了什么、为什么、放弃了什么
5. 所有文档之间做好交叉引用，可以逐个跳转
6. **索引必须同步更新** — 新增/删除/重命名任何文件时，必须同步更新以下索引：
   - 本文件的文档地图（上方树形图）
   - 根目录 `README.md` 的项目结构和文档表
   - 对应区域的 README（如 [experiments/README.md](experiments/README.md)、[adr/README.md](architecture/adr/README.md)）
   - `CLAUDE.md` 中若有引用也须同步
