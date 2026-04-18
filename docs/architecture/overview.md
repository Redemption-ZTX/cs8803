# 架构总览

> 详细的逐模块代码分析见 [代码审计](code-audit.md)（最新: [code-audit-000](code-audit-000.md)）。
> 作业要求见 [Final Project Instructions Document.md](../references/Final%20Project%20Instructions%20Document.md)。

## 上游基础

上游仓库: https://github.com/mdas64/soccer-twos-starter/tree/main

上游 starter 是平铺结构，训练脚本、辅助模块和 agent wrapper 基本都放在仓库根目录。

本项目现阶段已经做了结构化重构：

- 运行时代码迁入 [cs8803drl/](../../cs8803drl)
- 工具脚本迁入 [scripts/](../../scripts)
- 根目录只保留项目级元文件与少量全局配置

当前目录治理规则见 [../management/directory-governance.md](../management/directory-governance.md)。

## 上游 vs 当前项目差异

### 文件状态

| 状态 | 数量 | 说明 |
|------|------|------|
| 未修改 | 多数上游示例 | `examples/`、基线 agent、starter 模板仍作为归档或基线保留 |
| 已修改 | 核心训练与评估链 | `cs8803drl/core/*`、`cs8803drl/training/*`、`cs8803drl/evaluation/*` 等 |
| 新增 | 实验分支与评估工具 | `cs8803drl/branches/*`、`scripts/eval/*`、`scripts/tools/*`、H100 overlay 与 batch 分层 |

### `cs8803drl/core/utils.py` 改动（30 行 → 370+ 行）

- 新增 `RewardShapingWrapper`: 时间惩罚、球推进奖励、对手推进惩罚、控球奖励
- 新增 `_get_baseline_policy()`: 加载基线 checkpoint 为可调用 policy
- 新增 `_make_mixed_opponent_policy()`: 混合对手（baseline + 随机）
- 修改 `create_rllib_env()`: 支持 `reward_shaping` 和 `opponent_mix` 配置注入
- 添加 numpy/cv2 兼容性补丁

### `cs8803drl/training/train_ray_selfplay.py` 改动

- 新增 `FrozenBaselinePolicy`: 不可训练的基线 policy 包装
- 修改 `policy_mapping_fn`: team0 训练策略，team1 基线 + 自博弈池混合
- 新增 `evaluate_vs_baseline()`: 训练中定期对战基线并记录胜率
- env_config 加入 reward_shaping 参数
- 超参通过环境变量控制

### 新增文件

| 文件 | 行数 | 用途 |
|------|------|------|
| `cs8803drl/training/train_ray_team_vs_random_shaping.py` | 462+ | 核心训练脚本: team vs policy + reward shaping + 基线评估回调 |
| `cs8803drl/training/train_ray_base_team_vs_random.py` | ~300 | starter 对齐的 scratch base-model 训练：`team_vs_random`（Base-A / Base-B） |
| `cs8803drl/training/train_ray_base_team_vs_baseline.py` | ~300 | 直接对准评分目标的 scratch base-model 训练：`team_vs_policy` vs `baseline`（Base-D / Base-E） |
| `cs8803drl/training/train_ray_base_ma_teams.py` | ~300 | starter 对齐的 scratch base-model 训练：shared-policy `multiagent_team`（Base-C） |
| `cs8803drl/training/train_ray_role_specialization.py` | ~400 | 实验训练脚本: 双 policy 角色分工 PPO |
| `cs8803drl/training/train_ray_shared_policy_role_token.py` | ~350 | 实验训练脚本: 共享参数 multi-agent PPO + role token |
| `cs8803drl/training/train_ray_shared_central_critic.py` | ~320 | 实验训练脚本: 共享 actor + centralized critic PPO |
| `cs8803drl/training/train_ray_team_vs_random_summary_obs.py` | ~300 | 实验训练脚本: single-player PPO + summary observation |
| `cs8803drl/training/train_ray_team_vs_random_lstm.py` | ~300 | 实验训练脚本: recurrent PPO |
| `cs8803drl/evaluation/eval_rllib_checkpoint_vs_baseline.py` | 452 | checkpoint 对战基线评估 |
| `cs8803drl/evaluation/evaluate_matches.py` | 237 | 通用对战评估框架 |
| `cs8803drl/deployment/trained_ray_agent.py` | 416 | 加载已训练 agent 封装 |
| `cs8803drl/deployment/trained_team_ray_agent.py` | ~170 | team-level base model 部署封装：拼接双人 observation 并拆分 joint action |
| `cs8803drl/deployment/trained_ma_team_agent.py` | ~180 | shared-policy `multiagent_team` base model 部署封装：team obs/team action 对齐 |
| `cs8803drl/deployment/trained_fixed_teammate_agent.py` | ~190 | fixed-teammate single-player agent 封装 |
| `cs8803drl/deployment/trained_dual_expert_agent.py` | ~210 | 双专家协调器 agent 封装 |
| `cs8803drl/deployment/trained_role_agent.py` | ~200 | 双 policy 角色分工 agent 加载封装 |
| `cs8803drl/deployment/trained_shared_role_agent.py` | ~170 | 共享策略 role token agent 加载封装 |
| `cs8803drl/deployment/trained_shared_cc_agent.py` | ~220 | centralized critic agent 加载封装 |
| `sitecustomize.py` | 47 | Python 环境兼容性补丁 |
| `scripts/eval/eval_checkpoints.sh` | 31 | 批量评估 checkpoint 脚本 |
| `cs8803drl/core/soccer_info.py` | ~220 | 共享的比赛 info 解析与 reward shaping 纯逻辑 |
| `cs8803drl/branches/obs_summary.py` | ~150 | summary observation 特征扩展与 warm-start 工具 |
| `cs8803drl/branches/lstm_transfer.py` | ~90 | feed-forward → LSTM warm-start 工具 |
| `cs8803drl/branches/role_specialization.py` | ~200 | role policy mapping、baseline wrapper、warm-start 工具 |
| `cs8803drl/branches/shared_role_token.py` | ~120 | role token 映射与共享策略 warm-start 工具 |
| `cs8803drl/branches/shared_central_critic.py` | ~260 | centralized critic 模型、teammate-action 回调与 warm-start 工具 |
| `cs8803drl/branches/expert_coordination.py` | ~170 | 双专家协调、角色切换、场景重置辅助 |
| `scripts/tools/build_agent_module.py` | ~140 | 从 checkpoint 构建 submission-ready agent 模块 |

## 前任工作总结

已完成:
- reward shaping 体系（`RewardShapingWrapper`），详见 [code-audit-000 § 1.2](code-audit-000.md#12-rewardshapingwrapperl56-225)
- 主力训练脚本 [train_ray_team_vs_random_shaping.py](../../cs8803drl/training/train_ray_team_vs_random_shaping.py)，详见 [code-audit-000 § 2](code-audit-000.md#2-train_ray_team_vs_random_shapingpy--主力训练脚本)
- selfplay 脚本改造（基线对手 + 训练中评估），详见 [code-audit-000 § 3](code-audit-000.md#3-train_ray_selfplaypy--自博弈训练)
- 评估工具链（eval 三件套），详见 [code-audit-000 § 5](code-audit-000.md#5-评估工具链)
- 跑过部分实验（ray_results/ 中有训练结果）

未完成:
- 模仿学习（[作业 bonus 项](../references/Final%20Project%20Instructions%20Document.md#reward-observation-or-architecture-modification-40-points)）
- 提交用的 agent 模块（[提交格式要求](../references/Final%20Project%20Instructions%20Document.md#submission-integrity-10-points)）
- report

## 目录结构

```
cs8803drl/
├── core/                               # env factory / reward / checkpoint core
├── branches/                           # 实验辅助模块
├── training/                           # 所有训练入口
├── deployment/                         # 评估与部署 wrapper
└── evaluation/                         # 本地评估入口

scripts/
├── setup/                              # 环境与 overlay
├── eval/                               # 官方评估 / 扫描 / 回填
├── tools/                              # 构建与工具脚本
└── batch/
    ├── starter/
    ├── base/
    ├── adaptation/
    └── experiments/
```

## 关键约束

- Agent wrapper 现在通过完整模块路径加载，例如 `cs8803drl.deployment.trained_ray_agent`
- Python 3.8 + Ray 1.4.0 锁定（上游兼容性），详见 [工程规范](engineering-standards.md)
- protobuf==3.20.3, pydantic==1.10.13
- ray_results/ 和 checkpoint 不提交 git
