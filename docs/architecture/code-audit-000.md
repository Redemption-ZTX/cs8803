# 代码审计

> 架构总览见 [overview.md](overview.md)。作业要求见 [Final Project Instructions Document.md](../references/Final%20Project%20Instructions%20Document.md)。

逐模块分析现有代码，评估方案、问题、瓶颈、改进方向。

---

## 1. `utils.py` — 核心工具层

### 1.1 兼容性补丁（L1-38）

**现状**: 修补 `np.bool` 废弃和 `cv2` 缺失问题。`sitecustomize.py` 做了同样的事。

**问题**:
- `utils.py` 和 `sitecustomize.py` 有重复的兼容性代码，维护两份容易遗漏
- `cv2` stub 的 `__getattr__` 定义在模块级别而非类上，作用域可能有问题

**结论**: 低优先级，能跑就行。如果要清理，统一到 `sitecustomize.py` 一处即可。

### 1.2 `RewardShapingWrapper`（L56-225）

**现状**: gym.Wrapper，在 `step()` 中叠加额外奖励。4 个 shaping 组件：
1. **time_penalty** (-0.001/step) — 催促 agent 行动
2. **ball_progress** (0.01 * dx) — team0 推球向+x 得奖励，team1 反之
3. **opponent_progress_penalty** (默认关闭=0.0) — 对方推球时惩罚防守方
4. **possession_bonus** (+0.002) — 距球 1.25 以内得奖励

**问题与瓶颈**:
- **info 结构依赖猜测**: `_extract_ball_pos` 和 `_extract_player_positions` 用 best-effort 方式从 info dict 中挖位置信息，字段名靠猜（`ball_info`/`ball`、`player_info`/`player`）。如果 soccer_twos 环境版本不同，info 的 key 可能变，这些函数会静默返回 None，reward shaping 直接失效而不报错
- **只用 x 坐标做 ball_progress**: 只看 dx，不看 dy，无法奖励横向传球或绕过对手。在足球中，横向拉开空间同样重要
- **possession_bonus 过于简单**: 只看距离阈值，不区分谁在控球方向、是否面朝球。两个队友同时靠近球都拿 bonus，鼓励扎堆而非分工
- **opponent_progress_penalty 默认关闭**: 说明前任可能试过但效果不好，或来不及调
- **非 potential-based**: 这些 shaping 不满足 PBRS（potential-based reward shaping）理论保证，可能改变最优策略。对于课程项目可接受，但需在报告中讨论。相关理论见 [papers.md](../references/papers.md)

**改进方向**:
- 验证 `_extract_ball_pos` 是否在当前环境版本下能成功提取，加日志或 assert
- 考虑 potential-based 版本: Φ(s) = -distance(ball, opponent_goal)，shaping = γΦ(s') - Φ(s)
- 添加更多 shaping 信号：agent 朝球方向移动的奖励、传球奖励、防守站位奖励
- possession_bonus 可按距离衰减而非二值阈值

### 1.3 `_get_baseline_policy()`（L231-270）

**现状**: 加载 `ceia_baseline_agent` 的 checkpoint 为 RLlib policy，全局缓存。

**问题**:
- 在 worker 进程中调用时会 `ray.init()`，可能与主进程冲突
- 用 `BaseEnv()` 作 dummy env，如果 RLlib restore 需要真实 space 信息会出问题
- 错误处理粗糙，找不到 `params.pkl` 直接 raise

**结论**: 功能性代码，能用但脆弱。不建议改动除非遇到问题。

### 1.4 `_make_mixed_opponent_policy()`（L273-313）

**现状**: 以 `baseline_prob`（默认 0.9）的概率用基线策略，否则随机。

**问题**:
- 注释写 "10% random" 但实际比例由参数控制，注释误导
- 动作转换逻辑（inv_lookup 等）很复杂，说明 action space 的 flatten/unflatten 是反复踩坑的领域

**结论**: 功能正常，比例可调。

### 1.5 `create_rllib_env()`（L316-378）

**现状**: 环境工厂，支持 reward_shaping、opponent_mix 配置注入。

**问题**:
- 当 `opponent_mix` 启用时，会创建 `tmp_env` 来获取 action_space，但这个 env 之后被直接用作训练 env（L363: `env = tmp_env`）。如果 `soccer_twos.make()` 的环境有状态初始化问题，这可能引入微妙 bug
- 有一行注释掉的 `TransitionRecorderWrapper`，说明前任可能试过数据收集（模仿学习？），但未完成

**改进方向**: 无需大改，但注意 tmp_env 复用问题。

---

## 2. `train_ray_team_vs_random_shaping.py` — 主力训练脚本

### 2.1 整体设计

**现状**: 462 行，单人 agent 训练（team_vs_policy 模式），对手为 90% 基线 + 10% 随机。PPO + reward shaping + 定期对基线评估。

**训练配置**:
- 网络: `[512, 512]` MLP，ReLU，vf_share_layers=True
- PPO 超参: lr=3e-4, gamma=0.99, lambda=0.95, clip=0.2, entropy=0.0
- batch: rollout=1000, train_batch=4000, minibatch=512, sgd_iter=10
- 停止条件: 15M steps 或 2h

PPO 算法理论基础见 [HW1 notebook](../references/DRL_HW1.ipynb)（REINFORCE → VPG → PPO 实现）。

### 2.2 `_sanitize_checkpoint_for_restore()`（L56-103）

**现状**: 清理 checkpoint 中的 optimizer state，解决 Ray 1.4 + 新版 NumPy/PyTorch 的兼容性问题。

**问题**: 这 50 行代码只为解决一个版本兼容问题，说明前任在 checkpoint 恢复上踩了大量坑。

**结论**: 不需要碰，但说明训练中断后恢复是一个痛点。

### 2.3 `BaselineEvalCallbacks`（L106-379）

**现状**: 280 行，占了脚本的 60%。训练中每 N 轮创建一个新的评估环境，用训练中的 policy 对战基线，记录胜率到 custom_metrics。

**问题与瓶颈**:
- **代码量巨大但功能单一**: 280 行中大量是 action 格式转换和 info 解析的防御性代码（`_normalize_single_player_action`、`_coerce_to_discrete_action`、`_extract_winner_from_info` 等），说明 action space 的类型转换是反复出问题的重灾区
- **评估创建新环境**: 每次评估都 `create_rllib_env()` + `env.close()`，Unity 环境启停开销大
- **`_extract_score_from_info` 重复出现**: 同样的函数在 `train_ray_selfplay.py` 和 `eval_rllib_checkpoint_vs_baseline.py` 中都有，3 份拷贝（详见 [§ 6.1](#61-代码重复严重)）
- **entropy_coeff=0.0**: 完全没有探索鼓励，可能导致策略过早收敛到局部最优

**改进方向**:
- 提取公共的 score/winner 解析、action 转换函数到 `utils.py`，消除重复
- 考虑 entropy_coeff > 0（如 0.01）增加探索
- 网络可以试 `[256, 256, 256]` 或加 LSTM 处理时序信息
- `train_batch_size=4000` 偏小，可以增大

---

## 3. `train_ray_selfplay.py` — 自博弈训练

### 3.1 整体设计

**现状**: 2v2 多智能体训练。team0 用 "default" 策略训练，team1 从 baseline（70%）和自博弈池（30%）中采样对手。

**训练配置**:
- 网络: `[256, 256]`（比 team_vs_random 小）
- 5 个 policy: default（训练）、baseline（冻结）、opponent_1/2/3（自博弈池）
- rollout=5000, batch_mode=complete_episodes
- 自博弈更新条件: episode_reward_mean > 0.5 时将 default 权重推入对手池

训练框架选型见 [ADR-001](adr/001-training-framework.md)。

### 3.2 `FrozenBaselinePolicy`（L58-99）

**现状**: 继承 RLlib `Policy`，懒加载基线 checkpoint，不训练。

**结论**: 设计合理，懒加载避免 worker 启动开销。

### 3.3 自博弈更新机制

**问题**:
- **更新阈值硬编码为 0.5**: 这个值是否合理取决于奖励量级。如果 reward shaping 改变了奖励规模，这个阈值可能永远不触发或太容易触发
- **自博弈池只有 3 个**: 历史多样性有限，可能导致策略循环（rock-paper-scissors dynamics）
- **与 team_vs_random_shaping 网络大小不一致**: selfplay 用 [256,256]，shaping 用 [512,512]。如果两者需要共享 checkpoint 会有问题

**改进方向**:
- 自博弈池可扩大到 5-10 个，或用 Elo rating 管理
- 更新阈值改为动态（如相对基线胜率 > 60%）
- 网络大小统一

---

## 4. `train_ray_curriculum.py` — 课程学习（上游未修改）

**现状**: 5 个难度级别（Very Easy → Random Players），reward > 1.5 升级。任务配置见 `curriculum.yaml`。

**问题**:
- **对手是静止的 `lambda *_: 0`**: 课程中只有最后一级才有随机对手，前 4 级对手不动。学到的策略可能在面对真实对手时完全失效
- **全局变量 `current` 管理状态**: 多 worker 下不安全（虽然 RLlib 只在 driver 端调 callback）
- **没有 reward shaping**: 课程学习 + reward shaping 组合可能效果更好
- **没有降级机制**: 只升不降，如果 agent 到新难度后表现骤降，不会回退

**改进方向**:
- 和 reward shaping 结合
- 加入基线对手作为高级别课程
- 考虑双向课程（可升可降）
- 课程学习属于 [作业 bonus 项](../references/Final%20Project%20Instructions%20Document.md#reward-observation-or-architecture-modification-40-points)（+5 分）

---

## 5. 评估工具链

### 5.1 `eval_rllib_checkpoint_vs_baseline.py`（452 行）

**现状**: 从 checkpoint 加载 policy，对战基线，报告胜率。对应 [作业 Policy Performance 评分](../references/Final%20Project%20Instructions%20Document.md#policy-performance-50-points)（打赢 random 25 分 + 打赢 baseline 25 分）。

**问题**:
- **与 `trained_ray_agent.py` 大量重复**: checkpoint 加载、policy 恢复、权重提取逻辑几乎一样，各 400+ 行
- **`_extract_score_from_info` 第 3 份拷贝**

### 5.2 `evaluate_matches.py`（237 行）

**现状**: 通用对战框架，加载两个 agent 模块跑 N 局。

**结论**: 这是最干净的评估脚本。设计合理，可复用。

### 5.3 `eval_checkpoints.sh`（31 行）

**现状**: Shell 封装，`trained_ray_agent` + `evaluate_matches` 组合评估。

**结论**: 简洁实用。

### 5.4 `trained_ray_agent.py`（416 行）

**现状**: 实现 `AgentInterface`，从 checkpoint 加载 RLlib policy 用于评估/提交。

**问题**:
- **这就是提交用的 agent 模块**，但它不是一个独立目录（没有 `__init__.py`、没有 README），不符合 [提交格式](../references/Final%20Project%20Instructions%20Document.md#submission-integrity-10-points)
- 与 `eval_rllib_checkpoint_vs_baseline.py` 中 checkpoint 加载逻辑重复约 200 行

---

## 6. 全局问题

### 6.1 代码重复严重

以下逻辑在多个文件中重复出现：

| 逻辑 | 出现文件 | 次数 |
|------|----------|------|
| `_extract_score_from_info()` | selfplay, team_vs_random, eval_checkpoint | 3 |
| `_unpickle_if_bytes()` | team_vs_random, eval_checkpoint, trained_agent | 3 |
| `_strip_optimizer_state()` | team_vs_random, eval_checkpoint, trained_agent | 3 |
| `_find_torch_state_dict()` | eval_checkpoint, trained_agent | 2 |
| `_looks_like_torch_state_dict()` | eval_checkpoint, trained_agent | 2 |
| checkpoint 加载完整流程 | eval_checkpoint, trained_agent | 2 |

**建议**: 提取到 `utils.py` 或新建 `checkpoint_utils.py`。

### 6.2 没有提交用的 agent 模块

[作业要求](../references/Final%20Project%20Instructions%20Document.md#submission-integrity-10-points)每个 agent 是独立目录 + `__init__.py` + README + zip。目前：
- `trained_ray_agent.py` 是一个单文件，不是模块目录
- 没有任何满足提交格式的自训练 agent

**这是最大的缺口**。

### 6.3 没有 observation space 修改

[作业要求](../references/Final%20Project%20Instructions%20Document.md#reward-observation-or-architecture-modification-40-points) "alter the Observation Space or the Reward Function"。目前只有 reward modification，没有 observation 修改。如果只交 reward modification 也能拿到这 40 分，但多一个维度的实验可以增强报告深度。

### 6.4 缺少模仿学习（[bonus +5 分](../references/Final%20Project%20Instructions%20Document.md#reward-observation-or-architecture-modification-40-points)）

`create_rllib_env` 中注释掉的 `TransitionRecorderWrapper` 说明前任可能想过，但未实现。相关算法理论见 [HW2 notebook](../references/DRL_HW2_student.ipynb)（DQN → DDPG → SAC）。

---

## 7. 优先级排序

> 此排序将作为 [ROADMAP.md](../management/ROADMAP.md) 的输入。

| 优先级 | 任务 | 原因 | 对应评分项 |
|--------|------|------|-----------|
| P0 | 创建符合提交格式的 agent 模块目录 | 不做这个就无法提交 | [Submission Integrity 10 分](../references/Final%20Project%20Instructions%20Document.md#submission-integrity-10-points) |
| P0 | 确认 reward shaping 在当前环境下能正常工作 | `_extract_ball_pos` 可能返回 None | — |
| P1 | 提取重复代码到公共模块 | 后续改动会更安全 | — |
| P1 | 调参优化 — 击败 random 和 baseline 9/10 | 50 分的大头 | [Policy Performance 50 分](../references/Final%20Project%20Instructions%20Document.md#policy-performance-50-points) |
| P2 | 改进 reward shaping 设计 | 40 分中的核心 | [Reward/Obs Modification 40 分](../references/Final%20Project%20Instructions%20Document.md#reward-observation-or-architecture-modification-40-points) |
| P2 | entropy_coeff 和网络架构实验 | 可能显著提升性能 | — |
| P3 | 模仿学习 / observation 修改 | bonus 分 | [Bonus +5 分](../references/Final%20Project%20Instructions%20Document.md#reward-observation-or-architecture-modification-40-points) |
