# 架构总览

> 详细的逐模块代码分析见 [代码审计](code-audit.md)（最新: [code-audit-000](code-audit-000.md)）。
> 作业要求见 [Final Project Instructions Document.md](../references/Final%20Project%20Instructions%20Document.md)。

## 上游基础

上游仓库: https://github.com/mdas64/soccer-twos-starter/tree/main

上游提供的是一个平铺结构的 starter kit，所有脚本和模块都在根目录，通过 `from utils import create_rllib_env` 平级 import。Agent 模块必须在根目录才能被 `python -m soccer_twos.watch -m agent_name` 识别。

## 上游 vs 当前项目差异

### 文件状态

| 状态 | 数量 | 说明 |
|------|------|------|
| 未修改 | 17 | 所有 example 脚本、agent 模板、README、requirements.txt、curriculum.yaml、scripts/ |
| 已修改 | 2 | `utils.py`、`train_ray_selfplay.py` |
| 新增 | 6 | 训练脚本、评估工具、兼容性补丁 |

### `utils.py` 改动（30 行 → 370+ 行）

- 新增 `RewardShapingWrapper`: 时间惩罚、球推进奖励、对手推进惩罚、控球奖励
- 新增 `_get_baseline_policy()`: 加载基线 checkpoint 为可调用 policy
- 新增 `_make_mixed_opponent_policy()`: 混合对手（baseline + 随机）
- 修改 `create_rllib_env()`: 支持 `reward_shaping` 和 `opponent_mix` 配置注入
- 添加 numpy/cv2 兼容性补丁

### `train_ray_selfplay.py` 改动

- 新增 `FrozenBaselinePolicy`: 不可训练的基线 policy 包装
- 修改 `policy_mapping_fn`: team0 训练策略，team1 基线 + 自博弈池混合
- 新增 `evaluate_vs_baseline()`: 训练中定期对战基线并记录胜率
- env_config 加入 reward_shaping 参数
- 超参通过环境变量控制

### 新增文件

| 文件 | 行数 | 用途 |
|------|------|------|
| `train_ray_team_vs_random_shaping.py` | 462 | 核心训练脚本: team vs policy + reward shaping + 基线评估回调 |
| `eval_rllib_checkpoint_vs_baseline.py` | 452 | checkpoint 对战基线评估 |
| `evaluate_matches.py` | 237 | 通用对战评估框架 |
| `trained_ray_agent.py` | 416 | 加载已训练 agent 封装 |
| `sitecustomize.py` | 47 | Python 环境兼容性补丁 |
| `eval_checkpoints.sh` | 31 | 批量评估 checkpoint 脚本 |

## 前任工作总结

已完成:
- reward shaping 体系（`RewardShapingWrapper`），详见 [code-audit-000 § 1.2](code-audit-000.md#12-rewardshapingwrapperl56-225)
- 主力训练脚本 `train_ray_team_vs_random_shaping.py`，详见 [code-audit-000 § 2](code-audit-000.md#2-train_ray_team_vs_random_shapingpy--主力训练脚本)
- selfplay 脚本改造（基线对手 + 训练中评估），详见 [code-audit-000 § 3](code-audit-000.md#3-train_ray_selfplaypy--自博弈训练)
- 评估工具链（eval 三件套），详见 [code-audit-000 § 5](code-audit-000.md#5-评估工具链)
- 跑过部分实验（ray_results/ 中有训练结果）

未完成:
- 模仿学习（[作业 bonus 项](../references/Final%20Project%20Instructions%20Document.md#reward-observation-or-architecture-modification-40-points)）
- 提交用的 agent 模块（[提交格式要求](../references/Final%20Project%20Instructions%20Document.md#submission-integrity-10-points)）
- report

## 目录结构

```
cs8803DRL/
├── CLAUDE.md, CHANGELOG.md, README.md, requirements.txt
├── utils.py                            # 核心工具（已大幅扩展）
├── checkpoint_utils.py                 # checkpoint 解析公共模块
├── curriculum.yaml                     # 课程学习配置
├── sitecustomize.py                    # 兼容性补丁
│
├── train_ray_selfplay.py               # 自博弈训练（已修改）
├── train_ray_team_vs_random_shaping.py # 团队 vs 策略 + reward shaping（新增）
├── train_ray_curriculum.py             # 课程学习训练（未修改）
│
├── eval_rllib_checkpoint_vs_baseline.py # checkpoint 评估（新增）
├── evaluate_matches.py                  # 对战评估（新增）
├── eval_checkpoints.sh                  # 批量评估（新增）
├── trained_ray_agent.py                 # agent 加载封装（新增）
│
├── agents/                             # 实验 agent 版本（_template/ + vNNN_xxx/）
├── ceia_baseline_agent/                # 预训练基线
├── example_player_agent/               # 上游模板: 单玩家 agent
├── example_team_agent/                 # 上游模板: 团队 agent
│
├── examples/                           # 上游示例脚本（归档参考）
├── scripts/                            # PACE 集群脚本
├── docs/                               # 文档（见 [docs/README.md](../README.md)）
└── report/                             # 最终报告
```

## 关键约束

- Agent 模块必须在根目录（`soccer_twos.watch -m` 要求），详见 [ADR-001](adr/001-training-framework.md)
- `utils.py` 必须在根目录（所有脚本平级 import）
- Python 3.8 + Ray 1.4.0 锁定（上游兼容性），详见 [工程规范](engineering-standards.md)
- protobuf==3.20.3, pydantic==1.10.13
- ray_results/ 和 checkpoint 不提交 git
