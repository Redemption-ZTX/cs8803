# SNAPSHOT-001: Reward Shaping Runtime Validation

- **日期**: 2026-04-08
- **环境**: `/home/hice1/wsun377/.conda/envs/soccertwos`
- **目标**: 验证小范围重搭后的 reward shaping 在 `single_player=True` 真实环境中是否真正生效，并确认 submission agent 导出链路是否可用
- **相关**: [ROADMAP.md](../management/ROADMAP.md), [overview.md](../architecture/overview.md), [code-audit-000.md](../architecture/code-audit-000.md)

---

## 验证内容

1. 使用真实 `soccer_twos` 环境启动 `team_vs_policy + single_player + flatten_branched`。
2. 打印单步 `info` 结构，确认 `ball_info` / `player_info` 的真实 payload。
3. 打开 `RewardShapingWrapper(debug_info=True)`，检查 `_reward_shaping` 是否出现。
4. 通过 `env.env_channel.set_parameters()` 人工设置球的位置与速度，验证：
   - `ball_found=True`
   - `player_count=1`
   - `ball_dx` 非空
   - `scalar_reward_delta` 非零并回加到标量 reward
   - `applied_reward` 中出现非时间惩罚项

## 关键发现

- **真实 `info` 结构**：
  - `single_player=True` 下，`info` 顶层直接包含 `player_info` 和 `ball_info`
  - 两者内部的 `position` 是 `np.ndarray`，不是 `list/tuple`
  - 不包含按 agent id 分组的嵌套 dict

- **旧逻辑失败原因**：
  - 坐标解析只接受 `list/tuple`，没有接受 `np.ndarray`
  - `player_info` 在单玩家模式下没有 `agent_id`，旧逻辑无法恢复玩家位置
  - 标量 reward 模式下，就算算出了 shaping，也不会自动回加

- **修复后运行结果**：
  - `_reward_shaping.ball_found = True`
  - `_reward_shaping.player_count = 1`
  - `scalar_reward_delta` 为非零，并直接体现在返回的标量 reward 中
  - 在手动设定球贴近球员并赋予速度后，`applied_reward` 出现明显非零 shaping

## 代表性结果

人工将球放到球员附近并赋予速度后，单步输出中出现：

- `ball_dx = -11.1939`
- `applied_reward[0] = -0.109939...`
- `applied_reward[1] = -0.111939...`
- `scalar_reward_delta = -0.222878...`
- `reward = -0.222878...`

其中：

- `-0.111939 ≈ 0.01 * dx`，说明 ball progress shaping 已生效
- `-0.109939` 比上式大约多 `+0.002`，与 possession bonus 一致
- 标量 reward 精确反映了 team0 shaping 聚合结果，说明 single-player 聚合修复有效

## 结论

- **保留这条小范围重搭路线是合理的**。
- 当前 reward shaping 已从“可能完全没生效”推进到“已在真实环境中确认生效”。
- 下一步可以基于这套修复后的框架做一次短训练 smoke test，再决定是否直接从头重训主线模型。
