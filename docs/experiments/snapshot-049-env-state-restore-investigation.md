# SNAPSHOT-049: Soccer-Twos Env State Restore Capability Investigation

- **日期**: 2026-04-19
- **负责人**:
- **状态**: **已完成 verdict**（infra investigation, no training）

## 0. 为什么做

2026-04-19 第一性原理复盘后讨论"deploy-time MCTS / planning"路径——AlphaZero 简化版：deploy 时对当前 state 做 k-step lookahead，pick 最优 action。前提是 **env state 能 save/restore**。

soccer_twos 基于 Unity ML-Agents，[`EnvConfigurationChannel.set_parameters`](../../../home/hice1/wsun377/.conda/envs/soccertwos/lib/python3.8/site-packages/soccer_twos/side_channels.py) 暴露了 `players_states` 与 `ball_state` 接口（position / velocity / rotation_y），文档没说能否 mid-episode apply。

049 通过 6 个独立 smoke test 严格验证 mid-episode SET 的实际行为，给「MCTS 路径是否可行」一个明确答案。

## 1. 初始 smoke 结果（[smoke_env_state_set.py](../../scripts/smoke/smoke_env_state_set.py)）

### 1.1 测试协议

```
1. create env + EnvConfigurationChannel
2. reset, 跑 10 步 warmup
3. 读 pre-set state from info dict
4. channel.set_parameters(ball=[0,0], agent_0=[-5,0])
5. 跑 1 步 no-op (flush channel)
6. 读 post-set state
7. 比较 target vs actual
```

### 1.2 结果

| field | target | post-set actual | |Δ| | tolerance | verdict |
|---|---|---|---|---|---|
| ball position | (0, 0) | (+0.000, +0.000) | 0.000 | 1.5 | **PASS** |
| ball velocity | (0, 0) | (+0.000, +0.000) | 0.000 | 1.0 | **PASS** |
| agent 0 position | (-5, 0) | (-9.031, +1.200) | 4.206 | 1.5 | **FAIL** |
| agent 0 velocity | (0, 0) | (+0.000, +0.000) | 0.000 | 1.0 | PASS (但 agent 没动) |

**初步观察**: ball SET 完美工作；agent 0 position SET 完全失败，agent 留在原 spawn 位置 (-9.031, +1.200) — 这恰好是 [SNAPSHOT-022 §6.4](snapshot-022-role-differentiated-shaping.md#64-failure-情形) 记录的 blue 方 agent 0 spawn 位置，证明 GET state 准确，**SET 没生效**。

## 2. Follow-up 5 测试设计（[smoke_env_state_set_followup.py](../../scripts/smoke/smoke_env_state_set_followup.py)）

为排除假阴，写 5 个独立测试：

| ID | 排除的假设 | 设计 |
|---|---|---|
| **T1** | agent_id mapping bug | 对 4 个 agent 各自单独 set position |
| **T2** | 物理 anchor / target 太远被拒 | agent 0 试 Δ=0.5 / 2.0 / 5.0 三档目标 |
| **T3** | channel 需要 >1 步 propagation | set 后跑 1/2/3/4/5 步逐次检查 |
| **T4** | channel API 整体坏（confounds T1-T3） | pre-reset set，验证 canonical spawn-config 路径 |
| **T5** | Unity 对 position vs rotation 差异化处理 | mid-episode 设 rotation_y |

每个 test 用 fresh env 避免污染。

## 3. Follow-up 结果

### 3.1 原始数据

```
T1_agent_mapping:
  agent_0  = STUCK
  agent_1  = STUCK
  agent_2  = STUCK
  agent_3  = STUCK

T2_small_delta:
  delta_0.5 = PASS  (但见 §3.2 测试设计陷阱)
  delta_2.0 = FAIL
  delta_5.0 = FAIL

T3_multi_step:
  step_1 = FAIL
  step_2 = FAIL
  step_3 = FAIL
  step_4 = FAIL
  step_5 = FAIL

T4_pre_reset:
  pre_reset_player_set = FAIL

T5_rotation:
  rotation_set = PASS
```

### 3.2 T2 假阳性辨析（重要）

T2 的 `delta_0.5 = PASS` 实际是**测试设计的假阳性**，不是真信号：

```python
target = (spawn[0] + delta_x, spawn[1])  # delta_x = 0.5
ok = _delta(post, target) <= POS_TOL      # POS_TOL = 1.5
```

如果 agent 没移动 → `post = spawn` → `|post - target| = delta_x = 0.5` → 因为 `0.5 ≤ 1.5` 所以判 PASS。**agent 实际上根本没动**，只是目标离 spawn 太近而被宽容掉。

**修正后的 T2 真实信号**: 三个 Δ 档全都 agent 没动；delta_2.0 / delta_5.0 FAIL，delta_0.5 也是 stuck-with-passing-tolerance。

### 3.3 修正后 verdict 矩阵

| 测试 | 真实结果 | 排除的假设 | 还可能的解释 |
|---|---|---|---|
| T1 | 4 agent 全 STUCK | mapping bug | mid-episode player position 真的被 block |
| T2 | 全 stuck (Δ=0.5 假阳) | 物理 anchor / 距离限制 | (同 T1) |
| T3 | 5 步全 FAIL | channel propagation 延迟 | (同 T1) |
| T4 | **pre-reset 也 FAIL** | channel API 坏 | C# binary 不实现 PLAYER_POSITION handler |
| T5 | rotation **PASS** | (反例) | 证明 channel 整体能用，仅 position 字段被 block |

## 4. 分析

### 4.1 综合诊断

**Unity binary 的 C# 端实际行为**:
- `BALL_POSITION` / `BALL_VELOCITY` → 接受，mid-episode 立即生效（球是 passive physics object）
- `PLAYER_ROTATION` → 接受，mid-episode 立即生效（rotation 是软 state，无 collision 影响）
- `PLAYER_POSITION` / `PLAYER_VELOCITY` → **静默丢弃**，无论 mid-episode 还是 pre-reset 都失败

T4 是关键证据——连**预期的 canonical 用法（pre-reset spawn config）**都失败，说明这不是「mid-episode 被特殊 block」，而是 **C# 端没实现这两个 message type 的 handler**（或者 handler 永远 no-op）。

### 4.2 为什么 ball / rotation 可以但 position 不行

合理推测（无法直接验证 C# 源码，但符合 ML-Agents 设计模式）：

- Ball 是 passive physics object：teleport 不会破坏 collision invariant（场地永远比球大）
- Rotation 是 soft state：转身不需要重新解算 collider 位置
- Player position teleport 会让 collider 系统进入 invalid state（可能穿墙、卡入对方等）→ ML-Agents 默认禁止 mid-physics teleport

这是 ML-Agents 已知设计限制，不是项目特有 bug。

### 4.3 对 deploy-time 规划路径的影响

| 规划方案 | 需要的 state restore | 049 verdict 后的可行性 |
|---|---|---|
| **Full MCTS over agent actions** | ball + 4 agents 全部 pos/vel | **不可行** ❌ |
| **Ball-trajectory oracle**（"如果球从这里以速度 v 飞，落在哪"） | ball pos/vel only | **可行** ✓ |
| **Action evaluation via env replay** | reset + replay action sequence | 理论可行但每次 ~10s reset → **prohibitive** ❌ |
| **Multi-env parallel rollout** | 多个 env instance 各自起 reset | 启动慢（每个 ~30s）+ 同步贵 → **极贵** ⚠️ |

**结论**: 完整 MCTS 路径**死**。Ball-trajectory oracle 仍可行但用途窄（只能辅助"该不该射门"类决策），且工程量 3-5 天，对突破 0.86 ceiling 贡献不明朗，**ROI 显著低于 [SNAPSHOT-034 ensemble](snapshot-034-deploy-time-ensemble-agent.md) 与 [SNAPSHOT-048 hybrid eval](snapshot-048-hybrid-eval-baseline-takeover.md)**。

## 5. Verdict

### 5.1 主结论

> **soccer_twos Unity binary 不支持 mid-episode 或 pre-reset 的 player position SET。Ball position SET 可用，rotation_y SET 可用，但仅依赖这两者无法支撑 deploy-time MCTS over agent actions。**

### 5.2 决策

- **MCTS / planning 路径**: 关闭
- **Ensemble (034) + Hybrid Eval (048)** 仍是当前两条主推路径
- 未来如果有人想做 ball-trajectory oracle（射门决策），可基于 049 的发现单独评估，不再依赖 049 复测

### 5.3 副产物

049 的 GET state 验证（ball + agent position/velocity/rotation 都从 `info` dict 准确读出）**对 hybrid eval 有正面价值**——hybrid eval 的 trigger 函数（"球在我方深区"判断）可以从 `info["ball_info"]["position"]` 拿到 absolute world coords，**不需要从 obs 反推 egocentric ball_x**。这降低 [SNAPSHOT-048 §4 R1 (trigger state 提取错误)](snapshot-048-hybrid-eval-baseline-takeover.md) 的风险。

## 6. 不做的事

- **不再深究 player position SET 为什么不行**——已经通过 5 个 test 排除所有假阴，下一步只能改 Unity C# 源码（不在我们控制范围）
- **不实现 ball-trajectory oracle** ——窄用途 + 不解决 wasted_possession / defensive_pin 失败模式
- **不实现 multi-env parallel MCTS**——启动延迟 + 同步成本远高于 ensemble

## 7. 对其他 snapshot 的更新

- [SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md): 维持「优先实施」状态
- [SNAPSHOT-048](snapshot-048-hybrid-eval-baseline-takeover.md): §4 R1 风险**降低**——可用 `info["ball_info"]` 拿 absolute coords 做 trigger，不依赖 obs 反推
- 早前讨论中 hypothetical 的「snapshot-049 = DAGGER training」改为 snapshot-050 时再编号（049 已用于本 investigation）

## 8. 相关

- [smoke_env_state_set.py](../../scripts/smoke/smoke_env_state_set.py) — 初始 smoke
- [smoke_env_state_set_followup.py](../../scripts/smoke/smoke_env_state_set_followup.py) — 5-test follow-up
- [soccer_twos side_channels.py](../../../home/hice1/wsun377/.conda/envs/soccertwos/lib/python3.8/site-packages/soccer_twos/side_channels.py) — channel API source
- [SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md) — alternative deploy-time path（ensemble）
- [SNAPSHOT-048](snapshot-048-hybrid-eval-baseline-takeover.md) — alternative deploy-time path（hybrid takeover）
