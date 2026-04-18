# 工作目录治理

## 目标

本文件定义当前仓库的目录分层、职责边界和索引同步规则。

目标不是“表面整洁”，而是：

- 让训练、评估、部署入口有清晰边界
- 让 starter 继承代码和本项目实验代码分层共存
- 避免再把实验产物、工具脚本、运行时代码全部堆在一个平面上

## 顶层结构

### 根目录

根目录只保留项目级元文件和少量全局配置：

- `README.md`
- `CLAUDE.md`
- `CHANGELOG.md`
- `requirements.txt`
- `curriculum.yaml`
- `sitecustomize.py`

### `cs8803drl/`

主代码包，所有运行时代码都在这里分层。

#### `cs8803drl/core/`

运行时核心设施：

- `utils.py`
- `checkpoint_utils.py`
- `soccer_info.py`

#### `cs8803drl/training/`

所有训练入口：

- `train_ray_team_vs_random_shaping.py`
- `train_ray_team_vs_random_summary_obs.py`
- `train_ray_team_vs_random_lstm.py`
- `train_ray_role_specialization.py`
- `train_ray_shared_policy_role_token.py`
- `train_ray_shared_central_critic.py`
- `train_ray_selfplay.py`
- `train_ray_curriculum.py`

#### `cs8803drl/deployment/`

部署与评估 wrapper：

- `trained_ray_agent.py`
- `trained_summary_ray_agent.py`
- `trained_lstm_ray_agent.py`
- `trained_role_agent.py`
- `trained_shared_role_agent.py`
- `trained_shared_cc_agent.py`
- `trained_fixed_teammate_agent.py`
- `trained_dual_expert_agent.py`

#### `cs8803drl/evaluation/`

本地评估入口：

- `evaluate_matches.py`
- `eval_rllib_checkpoint_vs_baseline.py`

#### `cs8803drl/branches/`

实验辅助模块：

- `obs_summary.py`
- `lstm_transfer.py`
- `role_specialization.py`
- `shared_role_token.py`
- `shared_central_critic.py`
- `expert_coordination.py`

### `scripts/`

所有工具脚本和运行脚本按层次分组：

- `scripts/setup/`
- `scripts/eval/`
- `scripts/tools/`
- `scripts/batch/starter/`
- `scripts/batch/base/`
- `scripts/batch/adaptation/`
- `scripts/batch/experiments/`

### `agents/`

只放 submission-ready agent 模块与模板，不再混入训练入口或实验基础设施。

### `docs/experiments/artifacts/`

只放原始实验产物：

- 官方 evaluator 原始日志
- checkpoint 扫描 CSV
- SLURM / batch 输出

## 范围与职能分级

### 一级：Runtime Core

职责：环境创建、checkpoint 解析、比赛信息与公共逻辑。  
约束：改动要谨慎，优先保持向后兼容。

### 二级：Training Entry Points

职责：定义训练任务、超参、warm-start/restore 语义、训练内评估。  
约束：所有主训练入口都放在 `cs8803drl/training/`，不再回到根目录。

### 三级：Deployment / Evaluation

职责：官方 evaluator、本地对战、部署 wrapper。  
约束：模块名必须稳定，便于 `python -m ...` 和 `soccer_twos` 加载。

### 四级：Experiment Branch Helpers

职责：实验分支的特征工程、模型结构适配、协调逻辑。  
约束：不得再直接混入 `core/`，除非它确实被多个主线复用。

### 五级：Ops Scripts

职责：安装、评估、构建、batch。  
约束：统一放 `scripts/` 子目录，不允许再单层平铺。

## 必须离开根目录的内容

以下内容不应回到根目录：

- 训练/评估 Python 入口
- `trained_*.py` wrapper
- 实验辅助模块
- setup / eval / batch / tools 脚本
- 原始实验日志和扫描 CSV
- 缓存目录（如 `__pycache__/`）

## 索引同步要求

新增、删除、重命名文件时，至少同步：

- 根目录 `README.md`
- `docs/README.md`
- `docs/architecture/overview.md`
- `scripts/README.md`
- `docs/experiments/README.md`
- `CHANGELOG.md`
- `docs/management/WORKLOG.md`
- `CLAUDE.md`（若入口命令或关键路径变化）

## 当前原则

一句话原则：

**代码进包、脚本进分组、产物进归档，根目录只保留项目级入口与配置。**
