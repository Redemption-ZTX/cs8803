# CS8803 DRL — Soccer-Twos

CS8803 深度强化学习课程 Final Project。基于 [soccer-twos-starter](https://github.com/mdas64/soccer-twos-starter) 训练 2v2 足球多智能体。

## 快速开始

```bash
# 一键安装（推荐）
bash scripts/setup/setup.sh

# 或手动安装，见 CLAUDE.md 的 Setup 章节
```

```bash
# H100 / Hopper 节点推荐：在现有 soccertwos 环境上补一个 GPU overlay
bash scripts/setup/setup_h100_overlay.sh
```

```bash
# 看随机 agent 跑比赛
python examples/example_random_players.py

# 训练
python -m cs8803drl.training.train_ray_base_team_vs_random
python -m cs8803drl.training.train_ray_base_team_vs_baseline
python -m cs8803drl.training.train_ray_base_ma_teams
python -m cs8803drl.training.train_ray_team_vs_random_shaping
python -m cs8803drl.training.train_bc_team_policy
python -m cs8803drl.training.train_ray_mappo_vs_baseline
python -m cs8803drl.training.train_ray_role_specialization
python -m cs8803drl.training.train_ray_shared_policy_role_token
python -m cs8803drl.training.train_ray_selfplay
python -m cs8803drl.training.train_ray_curriculum

# 评估
python -m soccer_twos.watch -m example_player_agent
python -m soccer_twos.watch -m cs8803drl.deployment.trained_team_ray_agent
python -m soccer_twos.watch -m cs8803drl.deployment.trained_ma_team_agent
python -m soccer_twos.watch -m cs8803drl.deployment.trained_bc_team_agent
python -m cs8803drl.evaluation.eval_rllib_checkpoint_vs_baseline -c <checkpoint_path>
python -m cs8803drl.evaluation.evaluate_matches -m1 <agent_module> -m2 ceia_baseline_agent
python scripts/eval/evaluate_official_suite.py --team0-module cs8803drl.deployment.trained_ray_agent --opponents baseline -n 200 --checkpoint <checkpoint_path>
```

## 项目结构

```
├── CLAUDE.md                            # AI 协作入口（行为规则 + 架构摘要 + 保护等级）
├── CHANGELOG.md                         # 版本记录
├── curriculum.yaml                      # 课程学习任务配置
├── requirements.txt                     # 依赖规格
├── sitecustomize.py                     # Python 兼容性补丁
│
├── cs8803drl/                           # 主代码包
│   ├── core/                            # 运行时核心：env、checkpoint、reward/info
│   ├── training/                        # 训练入口：所有 train_ray_* 主线与分支
│   ├── deployment/                      # 评估/部署 agent wrapper
│   ├── evaluation/                      # 本地评估入口
│   └── branches/                        # 实验辅助模块（role/lstm/summary/cc 等）
│
├── scripts/                             # 结构化脚本层
│   ├── setup/                           # 环境与 overlay 安装
│   ├── eval/                            # 官方评估、扫描、回填、legacy checkpoint 对战
│   ├── tools/                           # 构建与工具脚本
│   └── batch/                           # SLURM / 直接运行 batch
│       ├── starter/                     # starter 风格入口
│       ├── base/                        # 从零开始的基础模型训练
│       ├── adaptation/                  # 基于 base checkpoint 的下游适配
│       ├── experiments/                 # 实验分支脚本
│       └── ...                          # 其余按职责分组
│
├── agents/                              # submission-ready agent 模块与模板
├── ceia_baseline_agent/                 # 预训练基线 agent
├── example_player_agent/                # 上游模板：单玩家 agent
├── example_team_agent/                  # 上游模板：团队 agent
├── examples/                            # 上游示例脚本（归档参考）
└── docs/                                # 项目文档
```

说明：

- 运行时代码已经从根目录收敛到 `cs8803drl/` 分层包里；根目录只保留项目级元文件、配置和少量兼容入口。
- 目录治理规则见 [docs/management/directory-governance.md](docs/management/directory-governance.md)。
- 工具脚本职责见 [scripts/README.md](scripts/README.md)。

## 文档

详见 [docs/README.md](docs/README.md) — 文档中心。

| 文档 | 说明 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | AI 协作入口，项目架构与约束 |
| [架构总览](docs/architecture/overview.md) | 上游差异、目录结构、前任工作 |
| [代码审计](docs/architecture/code-audit-000.md) | 接手时逐模块分析、问题、改进方向 |
| [工程规范](docs/architecture/engineering-standards.md) | 环境搭建、commit 流程、实验迭代、环境变量速查 |
| [目录治理](docs/management/directory-governance.md) | 根目录保留规则、归档规则、整理边界 |
| [清理日志](docs/management/cleanup-log.md) | 磁盘清理记录、删除内容与配额变化 |
| [阶段计划](docs/plan/plan-002-il-mappo-dual-mainline.md) | 当前主线：IL / BC + baseline exploitation + MAPPO 公平对照 |
| [实验记录](docs/experiments/README.md) | 实验索引与 snapshot |
| [作业要求](docs/references/Final%20Project%20Instructions%20Document.md) | 评分标准（Markdown 版） |

## 上游

- 课程指定 starter: https://github.com/mdas64/soccer-twos-starter
- 环境源码: https://github.com/bryanoliveira/soccer-twos-env
- 原版 README: [docs/references/upstream-README.md](docs/references/upstream-README.md)
