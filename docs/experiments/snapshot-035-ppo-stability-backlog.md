# SNAPSHOT-035: PPO Stability Backlog (Pre-Registration Only)

- **日期**: 2026-04-18
- **负责人**:
- **状态**: 预注册占位 / 详细设计待启动时补

## 0. 本 snapshot 的性质

这是一个 **pre-registration backlog**——把从 HW1/HW2/HW3 学到的、值得做但暂未排上日程的 PPO-side 稳定性实验集中登记，等手头有 GPU 余量或主线进入瓶颈时再展开。

每个候选只写：**核心改动 + 主要假设 + 工程量估计**。详细设计（具体超参、判据、batch 脚本）等启动时再补成 `035a/035b/035c` 独立 snapshot。

## 1. 候选 #1 — Smaller Entropy Coefficient

### 1.1 改动

仅改一个超参：`ENTROPY_COEFF = 0.003`（约 026-D 的 1/3）

### 1.2 主要假设

[SNAPSHOT-026 D-warm](snapshot-026-reward-liberation-ablation.md) 测了 `entropy_coeff = 0.01`，official 500 = `0.824` < 028A 的 0.844。**0.01 可能太强**——entropy bonus `0.01 × ln(27) ≈ 0.033 / step`，和 v2 shaping 同量级，会持续把 policy 拉向高 entropy。

HW2 SAC 用 **自动调整 α**，初期 `α≈0.1` 后期收敛到 `0.001-0.01`。在 PPO 上手动模拟："**前期高一点，后期降下来**"。

简化版：固定 `entropy_coeff = 0.003`（经验起点），warm-start from 029B@190 / 028A@1220 各跑 100 iter fine-tune。

### 1.3 工程量

零工程——只是新 batch 脚本，改一个 env var。

### 1.4 何时启动

- 如果 030/031/032/033 任一条出 ≥ 0.86 official，先做那条 + 此 entropy 方向叠加
- 如果都失败，直接在 029B@190 上单独测此方向

---

## 2. 候选 #2 — Advantage Normalization

### 2.1 改动

**自定义 PPO policy**，在 loss 里加：

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### 2.2 主要假设

HW1 GAE 实现明确做了 advantage normalization。**RLlib 1.4 的 `ppo_torch_policy.py` 没有自动做**（已确认源码 line 80-81 是 raw advantages × logp_ratio）。

后果：advantages 量级不稳定 → gradient scale 不稳 → 高 advantage 的 (s,a) 对 policy 拉得过猛。

预期收益：**+0.01-0.02 稳定性带来的 WR 提升**，主要在长训中体现（短训影响小）。

### 2.3 工程量

中等——需要：
- 写自定义 policy 类（继承 `PPOTorchPolicy`，override `loss` 方法）
- 注册到自定义 trainer
- 给所有现有 batch 加 `USE_ADVANTAGE_NORM=1` 开关

### 2.4 何时启动

下一个长训实验之前 implement，让长训自然收益。**优先级：中**——稳定性提升不是突破，但 free 的话值得拿。

---

## 3. 候选 #3 — Twin Value Heads (PPO Double Critic)

### 3.1 改动

策略网络架构：从单 value head 改为**两个独立 value head**，target 取 min。

```python
v1, v2 = self.value_heads(features)
v_target = min(v1, v2)
```

### 3.2 主要假设

HW2 Double DQN 关键 inequality：`E[max(C1, C2)] ≥ max(E[C1], E[C2])` → 单 critic 系统性过估计。

PPO 的 value function 也面临同样问题——advantage 估计 = `target - V(s)`，V 过估计会让 advantage 负偏，影响 policy 更新方向。

双 critic 取 min → 更保守的 target → 更少的过估计 → 更稳定的 advantage 估计。

预期收益：value loss 曲线更平稳；可能带来 +0.005-0.015 WR。

### 3.3 工程量

中高——需要：
- 自定义 model（双 value head）
- 自定义 loss（同时训两个 critic）
- 验证 advantage 计算正确路由（用 min 而不是 mean）

### 3.4 何时启动

低优先级。Twin value 在 SAC 上是标配，但 PPO 上的实证收益不大（Schulman 2017 原文没用，PPO2/PPG 也没用）。**只有当 030-033 全部失败、且想榨干每一点 stability 时才做**。

---

## 4. 优先级排序

| # | 候选 | 工程量 | 预期 ROI | 启动条件 |
|---|---|---|---|---|
| 1 | Smaller entropy_coeff | 极低 | 中 | 任何 lane 出 SOTA 后立即叠加 |
| 2 | Advantage normalization | 中 | 中-高 | 下次重要长训之前 |
| 3 | Twin value heads | 中高 | 低-中 | 兜底 |

## 5. 不做的事（已排除）

- **Target networks**：PPO 的 clip 已经解决了类似问题，不需要
- **Replay buffer**：PPO 是 on-policy，不能用
- **Off-policy 算法（DQN/DDPG/SAC）**：作业限定 PPO
- **Model-based RL（PETS-style dynamics learning）**：Soccer-Twos 提供了精确 simulator，model-based 收益小
- **CEM 等 planning**：deploy-time planning 延迟太高，不能 inline 到 `act()`

## 6. 何时升级为完整 snapshot

任一候选实际启动时，开 `035a / 035b / 035c` 独立 snapshot，复制 030/031/032 的模板补完：

- `0` 续跑说明（如适用）
- `§1` 动机
- `§3` 详细设计（超参、网络、shaping）
- `§5` 预声明判据（主判据 / 机制判据 / 失败判据 / gaming 防护）
- `§6` 执行矩阵 / `§9` 工程依赖 / `§10` 执行清单

本 backlog snapshot **不会被 update 成完整版**——它只是登记，详细 verdict 在子 snapshot 里。

## 7. 相关

- [HW1 PPO + GAE notebook](../references/DRL_HW1.ipynb)
- [HW2 SAC + Double Q notebook](../references/DRL_HW2_student.ipynb)
- [HW3 PETS Ensemble notebook](../references/DRL_HW3.ipynb)
- [SNAPSHOT-026 D-warm: entropy=0.01 测试](snapshot-026-reward-liberation-ablation.md)
- [SNAPSHOT-034: deploy-time ensemble](snapshot-034-deploy-time-ensemble-agent.md)
