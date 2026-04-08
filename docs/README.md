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
    │   └── deploy-and-verify.md           # 部署与验证交接文档
    │
    ├── experiments/                       # 实验记录
    │   ├── README.md                      # 实验索引 + 命名规范
    │   └── snapshot-000-prior-team.md     # 前任团队实验结果分析
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
7. [management/deploy-and-verify.md](management/deploy-and-verify.md) — 部署与验证交接文档（新人/新环境从这里开始）

### 跑实验？记录在这里

7. [experiments/README.md](experiments/README.md) — 实验记录模板
8. [management/WORKLOG.md](management/WORKLOG.md) — 每日工作日志

### 需要参考资料？

9. [references/papers.md](references/papers.md) — 论文笔记
10. [references/links.md](references/links.md) — 外部链接与工具
11. [references/DRL_HW1.ipynb](references/DRL_HW1.ipynb) — HW1 实现参考（REINFORCE → VPG → PPO）
12. [references/DRL_HW2_student.ipynb](references/DRL_HW2_student.ipynb) — HW2 实现参考（DQN → DDPG → SAC）

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
