# CS8803 DRL — Soccer-Twos

CS8803 深度强化学习课程 Final Project。基于 [soccer-twos-starter](https://github.com/mdas64/soccer-twos-starter) 训练 2v2 足球多智能体。

## 快速开始

```bash
# 一键安装（推荐）
bash scripts/setup.sh

# 或手动安装，见 CLAUDE.md 的 Setup 章节
```

```bash
# 看随机 agent 跑比赛
python examples/example_random_players.py

# 训练
python train_ray_team_vs_random_shaping.py
python train_ray_selfplay.py
python train_ray_curriculum.py

# 评估
python -m soccer_twos.watch -m example_player_agent
python eval_rllib_checkpoint_vs_baseline.py -c <checkpoint_path>
python evaluate_matches.py -m1 <agent_module> -m2 ceia_baseline_agent
```

## 项目结构

```
├── CLAUDE.md                           # AI 协作入口（行为规则 + 架构摘要 + 保护等级）
├── CHANGELOG.md                        # 版本记录
├── .claude/settings.json               # Claude Code 配置
│
├── train_ray_team_vs_random_shaping.py # 主力训练：team vs policy + reward shaping
├── train_ray_selfplay.py               # 自博弈训练
├── train_ray_curriculum.py             # 课程学习训练
├── utils.py                            # 核心工具（环境工厂、reward shaping、基线加载）
├── checkpoint_utils.py                 # checkpoint 解析公共模块（canonical source）
├── curriculum.yaml                     # 课程学习任务配置
├── sitecustomize.py                    # Python 兼容性补丁
├── eval_rllib_checkpoint_vs_baseline.py # checkpoint 评估
├── evaluate_matches.py                  # 对战评估
├── eval_checkpoints.sh                  # 批量评估 shell 封装
├── trained_ray_agent.py                 # 已训练 agent 加载封装
│
├── agents/                             # 实验 agent 版本（_template/ + vNNN_xxx/）
├── ceia_baseline_agent/                # 预训练基线 agent
├── example_player_agent/               # 上游模板：单玩家 agent
├── example_team_agent/                 # 上游模板：团队 agent
├── examples/                           # 上游示例脚本（归档参考）
├── scripts/                            # setup.sh 一键部署 + PACE 集群作业脚本
├── docs/                               # 项目文档（见下方）
└── report/                             # 最终报告
```

## 文档

详见 [docs/README.md](docs/README.md) — 文档中心。

| 文档 | 说明 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | AI 协作入口，项目架构与约束 |
| [架构总览](docs/architecture/overview.md) | 上游差异、目录结构、前任工作 |
| [代码审计](docs/architecture/code-audit-000.md) | 接手时逐模块分析、问题、改进方向 |
| [工程规范](docs/architecture/engineering-standards.md) | 环境搭建、commit 流程、实验迭代、环境变量速查 |
| [实验记录](docs/experiments/README.md) | 实验索引与 snapshot |
| [作业要求](docs/references/Final%20Project%20Instructions%20Document.md) | 评分标准（Markdown 版） |

## 上游

- 课程指定 starter: https://github.com/mdas64/soccer-twos-starter
- 环境源码: https://github.com/bryanoliveira/soccer-twos-env
- 原版 README: [docs/references/upstream-README.md](docs/references/upstream-README.md)
