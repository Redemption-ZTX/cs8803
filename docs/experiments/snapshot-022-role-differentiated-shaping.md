# SNAPSHOT-022: Agent-ID Asymmetric Shaping (Reward-Symmetry Hypothesis)

> **命名说明**：原名 "Role-Differentiated Shaping"。2026-04-15 P0 spawn 稳定性验证显示 agent_id ↔ field-role 不是干净的 striker/defender 映射（详见 §7 R1），因此本 snapshot 的准确定位是 **"测'打破 agent_id 对称性'本身有没有用"**，而不是"严格 striker/defender role shaping"。配置值不变，叙述收紧。

- **日期**: 2026-04-15
- **负责人**:
- **状态**: 修复后首轮结果已完成（official `baseline 500`、failure capture、head-to-head 已补齐）

## 0. 2026-04-16 关键更正：首轮结果不是有效的 warm-start fine-tune

首轮 `022` 已完成训练并得到训练内高点：

- best eval baseline: `0.820 @ checkpoint-270`
- best eval random: `0.960`

但 2026-04-16 复盘时确认：**这条 lane 实际没有成功吃到预期的 `MAPPO+v2@470` warm-start 权重。**

根因在 [warmstart_shared_cc_policy()](../../cs8803drl/branches/shared_central_critic.py)：

- 该函数原本是为“vanilla PPO 裸 trunk → shared-cc”迁移写的
- `direct_mapping` 期待源 checkpoint key 形如 `_hidden_layers.* / _logits.* / _value_branch.*`
- 而 `MAPPO+v2@470` 作为 shared-cc 源，实际 key 形如 `action_model.* / value_model.*`
- 结果 16 条 direct mapping 全部 `skipped`
- after-init warm-start 路径**被执行了**，但有效权重没有拷入 target model

因此，首轮 `022` 的准确语义不是：

- “`warm470 + agent-id asymmetric shaping` 的 300-iter fine-tune”

而更接近：

- “**近似 scratch + agent-id asymmetric shaping** 的 300-iter 训练”

这条修正对结论的影响是：

- `022` 的**绝对分数**不能再按 warm-start 口径解读
- 但它对 `024` 的**方向性比较仍然有效**
- 在同样没吃到 warm-start 的前提下，`agent-id asymmetry` 比 `spawn-depth field-role asymmetry` 明显更 learnable

本 snapshot 后续的正式 verdict 以“**修复 warm-start bug 后的重跑结果**”为准；首轮结果仅保留方向性参考价值。

### 0.1 2026-04-16 修复与 smoke 验证

上述 bug 现已在 [warmstart_shared_cc_policy()](../../cs8803drl/branches/shared_central_critic.py) 修复：

- 当源 checkpoint 已经是 shared-cc（`action_model.* / value_model.*`）时，直接走 **CC→CC key-for-key copy**
- 不再错误使用“裸 trunk → shared-cc”的 `direct_mapping`

修复后的函数级 smoke 结果：

- source: [MAPPO+v2@470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470)
- result: `{'copied': 16, 'adapted': 0, 'skipped': 0}`
- `before_eq_after = False`
- `after_eq_source = True`

这说明：

- warm-start 不再是“代码路径执行但 0-copy”
- shared-cc 源权重现在**确实被拷入 target model**

因此，本 snapshot 现在已经具备**重新做正式 warm-start 试验**的条件；接下来的重跑结果才是 `022` 的正式判据依据。

## 1. 背景

[SNAPSHOT-013 §12](snapshot-013-baseline-weakness-analysis.md#12-bc%E2%86%92mappo-数据对-11-的进一步强化2026-04-15-后补) 与 [SNAPSHOT-017 §11](snapshot-017-bc-to-mappo-bootstrap.md#11-failure-bucket-深度分析与判据-verdict) 共同确立：`low_possession` 占失败比在 9 条 lane 上稳定在 **22.6%–28.1%**，是跨 intervention 不变量。

初步诊断把根因指向**候选 A (obs 缺失队友信息)**。[SNAPSHOT-021](snapshot-021-actor-teammate-obs-expansion.md) 是对 A 的直接检验。

但还有一个同样 plausible 的根因 **B (reward 对称)**：现有所有 lane 共享 v1 base 的三项 shaping 核心，**全部对两个 team0 agent 对称应用**。对称 reward → 两 agent 学同构 policy → 无角色分化 → `low_possession` state 下两 agent 同时压上或同时回缩，留空另一块。

本 snapshot 是 **B 假设的直接检验**。

## 2. 共享 reward 的对称性

9 条 lane 共享的 shaping 核心（来自 v1 base）：

| shaping 项 | 默认值 | 对称形式 |
|---|---|---|
| `ball_progress_scale` | 0.01 | `+scale·dx` 给 team0 两 agent 都加，`-scale·dx` 给 team1 两 agent |
| `possession_bonus` | 0.002 @ 1.25m | 任一 agent 在 1.25m 内都拿 bonus |
| `time_penalty` | 0.001 | 每步所有 agent 都扣 |
| `opp_progress_penalty_scale` (v2+) | 0.01 | 持球方推进时**防守方全体**都扣 |
| `deep_zone_penalty` (v2+) | 0.003 × 2 层 | ball 进深区时**本方全体** agent 都扣 |

**没有任何一项 shaping 给 agent_id 0 和 agent_id 1 不同的信号**。

即使我们加了 teammate obs（snapshot-021），如果 reward 本身对两个 agent 是完全对称的，共享参数的 shared-actor policy **在 steady-state 下仍然只会学出同构 behavior**——因为对两个 agent optimal 的 action distribution 是一样的。

## 3. 假设

### 主假设

在 agent_id 0 和 agent_id 1 上**施加非对称的 shaping**（shared actor 下的两个 agent 收到不同 reward signal），打破 reward 对称性，让 shared policy 学出**不同 agent_id 下的 position-conditional policy**：

- `low_possession` 占失败比从 22-28% 降至 ≤ 15%
- 整体 WR 从 0.84 推到 ≥ 0.85（可能更高）

**本 snapshot 不严格声称测试 "striker/defender 角色"**。由于 P0 spawn 验证显示 agent_id 与 field role 不是干净映射（§7 R1），我们测的是**对称性破坏本身的价值**——如果打破对称性能修 low_possession，下一轮再按 spawn position 精细化 role 设计。

### 备择假设

如果 role-diff shaping 后 `low_possession` 依然 ≥ 20%，说明对称性不是主因，应回到 A (obs) 路线。

## 4. 角色分化 shaping 方案

[`soccer_info.py`](../../cs8803drl/core/soccer_info.py) 已经支持 per-agent 覆盖（`ball_progress_scale_by_agent`, `opponent_progress_penalty_scale_by_agent`, `possession_bonus_by_agent`），无需改核心代码，只需扩展 env var 控制。

### 4.1 Agent-ID 非对称分配（非严格 role mapping）

- **Agent_id 0** 收到 **"进攻型" 极 shaping 配置**（记作 Pole A）
- **Agent_id 1** 收到 **"防守型" 极 shaping 配置**（记作 Pole B）

**重要实情**（P0 spawn 验证结果，详见 §7 R1）：
- agent_id 0 在 blue 方 spawn = `(-9.03, +1.20)`（更深位置 / 后场）
- agent_id 1 在 blue 方 spawn = `(-6.24, -1.20)`（更前位置 / 中前）
- orange 方 agent 0/1 x 深度几乎相同（6.45 vs 6.66），无清晰前后关系

所以：
- 对 **blue 方**：当前配置实际是**把"进攻 pole"给了 spawn 较深的 agent**——reward signal 会推 agent 0 从后场积极向前；可能有效也可能不匹配
- 对 **orange 方**：两个 agent spawn 深度相同，agent_id 0/1 的"角色"完全由 reward 差异构造，没有 spawn 倾向
- 上路/下路（y 轴）层面：两队都是 agent_id 0 = +y, agent_id 1 = -y，**这个分化是对称一致的**

这也是为什么本 snapshot 不再称"striker / defender role"——**我们实际在测"两个 shared-policy agent 分别训练 Pole A / Pole B 信号"对 policy 行为的影响**，不是在测严格的前锋/后卫分工。

### 4.2 具体 shaping 差分（首轮配置）

| 项 | Pole A (agent_id 0, "进攻型") | Pole B (agent_id 1, "防守型") |
|---|---|---|
| `ball_progress_scale` | **0.015** (1.5×) | **0.005** (0.5×) |
| `opponent_progress_penalty_scale` | **0.005** (0.5×) | **0.020** (2×) |
| `possession_bonus` @ 1.25m | 0.002 | 0.002 |
| `deep_zone_outer_penalty` | 0.003 | **0.006** (2×) |
| `deep_zone_inner_penalty` | 0.003 | **0.006** (2×) |
| `time_penalty` | 0.001 | 0.001 |

**设计 rationale**（放在 agent-id 非对称层面解释，不绑 field role）：
- Pole A 被鼓励推进，对对方推进的惩罚弱——不是因为"它是 striker"，而是它收到的是偏向进攻的 gradient signal
- Pole B 对进攻 progress 只拿半量奖励，对 opponent 推进罚重——偏向防守的 gradient signal
- `possession_bonus / time_penalty` 保持对称——只改进攻/防守轴，不改其他维度
- 如果这套 Pole A/B 非对称 signal 让 shared policy 学出 **"按 agent_id 分化的两套行为"**（§6.3 验证），则证明对称性破坏本身有价值，无论 field role 是否匹配

### 4.3 Warm-start

不像 [SNAPSHOT-021](snapshot-021-actor-teammate-obs-expansion.md) 需要 from scratch（obs 维度变了），本 lane **设计上**支持 warm-start：

| 候选 warm-start 源 | 优势 | 劣势 |
|---|---|---|
| **MAPPO+v2 @ ckpt470** (0.786) | 干净对照"角色分化 shaping 的纯效应" | 起点较低，fine-tune 收益空间大 |
| **BC @ ckpt2100** (0.842) | 在 SOTA 上叠加角色分化 | 如果起点已经学到某种协调模式，fine-tune 可能撞回去 |

**设计上推荐用 MAPPO+v2 @ ckpt470**——对照最干净。如果 BC @2100 上叠的效果更有意思，可以第二轮再试。

> 2026-04-16 更正：首轮 run 虽然 batch 配了 `WARMSTART_CHECKPOINT=...checkpoint-470`，但由于 shared-cc → shared-cc key mapping bug，实际未成功加载该 warm-start。修复后重跑时，才应重新使用本节口径。

## 5. 训练配置

- 训练脚本：[train_ray_mappo_vs_baseline.py](../../cs8803drl/training/train_ray_mappo_vs_baseline.py) + 新环境变量 `SHAPING_ROLE_DIFFERENTIATED=1` 触发 per-agent shaping overrides
- 设计 warm-start: MAPPO+v2 @ ckpt470
- `variation = multiagent_player`, `multiagent = True`
- `FCNET_HIDDENS = 512,512`, `custom_model = shared_cc_model`
- `gamma = 0.99, lambda = 0.95`
- 300 iter fine-tune, ~12M steps, ~5h H100

## 6. 预声明判据

### 6.1 主判据

**500-ep 官方 WR vs baseline ≥ 0.83**

比 [SNAPSHOT-021 §5.1 (0.81)](snapshot-021-actor-teammate-obs-expansion.md#51-主判据) 高一档——因为 **修复后的** 022 设计上有 warm-start 优势（起点应在 `0.786` 左右），纯 reward 改动应该能至少再推几个百分点。

### 6.2 机制判据（本 lane 核心）

**`low_possession` 占失败比 ≤ 15%**（与 021 一致，直接检验假设 B）。

### 6.3 辅助机制判据

**shared policy 出现 agent-id 条件化的行为分化**——通过 failure capture 里每个 agent 的行为统计验证：

- Agent 0 (Pole A) 和 Agent 1 (Pole B) 的 mean 位置 x 出现**系统性差异** — 两个 agent 在 x 方向的平均位置差值 |Δx| ≥ 2 单位（不限方向）
- Agent 0 和 Agent 1 的触球率出现**系统性差异** — 至少一方触球率比另一方高 ≥ 30%
- 在 `ball_x < -4`（我方深区）时，两 agent 的 x 位置分布出现明显差异（例如一个明显后缩、另一个明显前压）

**注意**：由于 P0 spawn 数据显示 blue 方本身就有 agent 0 比 agent 1 深 ~2.8 单位的天然分化，**光看 |Δx| ≥ 2 不足以证明 reward 改动起作用**——那可能就是 spawn 偏移在 episode 平均里的残留。需要对比：

- **scratch MAPPO+v2 (0.786, 对称 shaping)** 的 failure capture 里 agent 0/1 的 |Δx| 基线
- 本 lane 的 |Δx| 必须**显著大于**该基线，才算 reward 真正在分化行为

如果 WR 上升但 |Δx| 和 scratch baseline 相当 → policy 没学到真正的 agent-id 差异化，WR 提升可能是 seed luck 或其他机制。

### 6.4 失败情形

| 触发条件 | 解读 |
|---|---|
| 500-ep WR < 0.76 | role-diff 反而损害性能（角色分化过激）| 
| `low_poss` ≥ 20% 且 WR < 0.80 | B 假设被否决，问题在 obs 或别处 |
| 行为 specialization 未出现（Agent 0 mean x ≈ Agent 1 mean x）| role-diff 信号不够强，需要更激进分化 |

## 7. 风险

### R1 — Agent-ID 到 Field-Role 的映射**不是干净的**（P0 验证关键发现）

#### P0 smoke 完整结果（2026-04-15）

使用 `soccer_twos.make(...)` 连续 20 次 reset，在每局第 1 步施加零动作后读取 `info["player_info"]["position"]`：

| 队 | agent_id | 初始位置 |
|---|---|---|
| Blue (team0) | 0 | `(-9.03, +1.20)` |
| Blue (team0) | 1 | `(-6.24, -1.20)` |
| Orange (team1) | 0 | `( 6.45, +1.20)` |
| Orange (team1) | 1 | `( 6.66, -1.20)` |

20 次 reset 完全稳定（`a0_left_of_a1_count = 20/20`）。

#### 三层对称性拆开看

| 层面 | 对称性结论 |
|---|---|
| **API 本地 agent_id 0/1 重映射** | ✅ 完全对称 — 两队模块内部都只看到本地 `{0,1}` |
| **上路/下路 (y 轴 lane)** | ✅ 对称 — 两队都是 agent_id 0 = `+y` (上路), agent_id 1 = `-y` (下路) |
| **前锋/后卫 (x 轴 depth)** | ❌ **不对称，且 blue 方可能与本 snapshot 配置"反向"** |

#### x 轴 depth 的具体非对称

- **Blue 方**（team0）：
  - agent 0 (x=-9.03) 比 agent 1 (x=-6.24) **更深** ~2.8 单位
  - 按足球直觉，agent 0 的 spawn 位置更像 defender，agent 1 更像 striker
  - 本 snapshot 的 Pole A ("进攻型") 分配给了 agent 0——对 blue 方而言**和 spawn 直觉相反**
- **Orange 方**（team1）：
  - agent 0 (x=6.45) 和 agent 1 (x=6.66) 几乎同深
  - spawn 层面没有"谁是 striker 谁是 defender"的先验——**完全由 reward 差异构造**

#### 对 022 的直接影响

- 本 snapshot **无法**严格声称在测 "striker / defender role shaping"
- 但 **可以**声称在测 "两个 agent 的 reward 对称性破坏是否带来 policy 行为分化"——这是更弱但更诚实的 claim
- 首轮 022 结果的解读范围：
  - 如果 `low_poss ≤ 15%` + 行为出现 |Δx| 显著分化 → **对称性破坏本身有价值**，下一轮可以做"按 spawn 精确绑定 role"的严格版
  - 如果 `low_poss ≤ 15%` 但行为 |Δx| 没有超过 scratch baseline → policy 靠其他机制间接修 low_poss，非 agent-id 分化贡献
  - 如果 `low_poss` 不降 → 对称性破坏不够，或者配置方向对 blue/orange 的 spawn 刚好相反导致抵消

#### 严格版的后续工作

若 022 证明"对称性破坏有价值"且团队想做严格 role shaping，需要：
- 在 env wrapper 里按每局第 1 步的 spawn `x` 符号和大小给每个 agent 打 "role tag"
- Shaping `*_by_agent` dict 按 role tag 分配，而非按 agent_id
- 这需要 runtime 的 role tagging 机制，比当前的 env-var 硬编码 agent_id 复杂
- **不在 022 范围内**，后续若需要可新开 snapshot

#### 遗留风险

- **Side-swap eval**：本 P0 只验证了训练环境的 spawn 稳定性，**不能**替代 eval 时 team0 ↔ team1 交换后 agent_id 映射的独立检查
- **Inference 层的 agent_id 语义一致性**：如果 deployment wrapper 在某些路径里把 team0 slot 0/1 反过来塞给本地 `{0,1}`，训练时学到的 agent-id 行为差异会被打乱

### R2 — 角色分化强度选对了吗

首轮用 1.5× / 0.5× 分化。可能太弱（两个 agent 仍然学 almost identical policy）或太强（defender 不敢过半场，offense coverage 崩盘）。

缓解：
- 首轮出结果后，若 WR 持平或微降，尝试更激进分化（2× / 0.3×）
- 若 WR 崩塌，尝试更弱分化（1.2× / 0.8×）

### R3 — warm-start 源已经学到对称 policy

MAPPO+v2 @ ckpt470 是用对称 shaping 训出来的，policy 本身已经"向中立协调"收敛。在它上面加 role-diff shaping，可能需要较长 fine-tune 才能让 policy 真正分化。

缓解：
- 300 iter 若不够，可延长到 500 iter
- 若仍未看到行为 specialization，考虑 from-scratch 做 role-diff shaping（接近 021 的成本）

### R4 — 可能只修 `low_poss`，不提 WR

如果 B 假设对但收益机制是"解决了 low_poss bucket 但没有全局提升"（因为 low_poss 只占 4.5% of all episodes），最终 WR 增量可能只有 +0.03-0.05，达不到 6.1 的 0.83 阈值。

缓解：即使主判据失败但机制判据（low_poss ≤ 15%）过，仍视为 B 假设成立，回填判据 §6.1 的阈值设计过高。

## 8. 不做的事（明确边界）

- **不改 obs**（那是 SNAPSHOT-021 的职责）
- **不加 rolling self / opponent pool** （不混入 SNAPSHOT-018/019 的变量）
- **不改模型结构 / γ / λ / batch**
- **不叠 BC 作为 warm-start**（MAPPO+v2 @470 干净对照）
- **不尝试三 agent 角色**（我们只有 2 agent，保持 striker / defender 分化）

## 9. 与 [SNAPSHOT-021](snapshot-021-actor-teammate-obs-expansion.md) 的关系

两 snapshot **独立验证根因 A / B**。决策矩阵见 [SNAPSHOT-021 §8](snapshot-021-actor-teammate-obs-expansion.md#8-与-snapshot-022-的关系)。

**推荐执行顺序**：022 先 (5h GPU, 便宜)，021 后 (16h+ GPU, 贵)。若 022 已修复 low_poss → 021 可能不必做；若 022 失败 → 021 值得开。如果 GPU 资源充足，并行启动更快拿到诊断答案。

## 10. 相关

- [SNAPSHOT-013: baseline weakness analysis](snapshot-013-baseline-weakness-analysis.md) §11 / §12（诊断来源）
- [SNAPSHOT-014: MAPPO 公平对照](snapshot-014-mappo-fair-ablation.md)（warm-start 源）
- [SNAPSHOT-017: BC→MAPPO bootstrap](snapshot-017-bc-to-mappo-bootstrap.md) §11（low_poss 不变量完整证据）
- [SNAPSHOT-021: actor teammate-obs expansion](snapshot-021-actor-teammate-obs-expansion.md)（平行 A-假设验证）
- [cs8803drl/core/soccer_info.py](../../cs8803drl/core/soccer_info.py) — per-agent shaping override 已实现

## 11. 下一步执行清单

1. 修复 [warmstart_shared_cc_policy()](../../cs8803drl/branches/shared_central_critic.py) 的 shared-cc → shared-cc key 映射，使 `MAPPO+v2@470` 真正加载到 `022`。
2. 1-iter smoke：
   - 打印每个 agent 的 shaping reward，确认 agent_id 0 和 agent_id 1 收到**不同**的信号
   - 打印 `[warmstart] copied ...` 与 `warmstart_summary.txt`，确认 `copied > 0`
   - 打印 agent spawn position，确认 role 绑定稳定
3. 重跑 batch `scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_role_diff_shaping_v2_warm470_512x512.batch`
4. 启动 300 iter fine-tune from MAPPO+v2 @ ckpt470（~5h H100）
5. `top 5% + ties → baseline 500` 选模
6. best ckpt 做 save-all failure capture，重点看：
   - `low_possession` 占比
   - Agent_id 0 / 1 的平均 x 位置分离度
   - Agent_id 0 / 1 的触球率差异
7. 将“首轮无效 warm-start 结果”与“修复后正式结果”分开记录，按 §6 判据 + [SNAPSHOT-021 §8](snapshot-021-actor-teammate-obs-expansion.md#8-与-snapshot-022-的关系) 决策矩阵落 verdict 到本文件 §12+（append-only）

## 12. 修复后正式结果（2026-04-17 后补，append-only）

修复 [warmstart_shared_cc_policy()](../../cs8803drl/branches/shared_central_critic.py) 的 shared-cc → shared-cc 迁移 bug 后，`022` 的正式 rerun 为：

- run_dir: [PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430)
- warm-start 证据: [warmstart_summary.txt](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/warmstart_summary.txt)

### 12.1 official `baseline 500` 复核

按 `top 5% + ties + 前后 2 点`，正式复核窗口为 `10/20/30/40/270/280/290`：

| checkpoint | official `baseline 500` |
|---|---:|
| [checkpoint-10](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000010/checkpoint-10) | 0.742 |
| [checkpoint-20](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000020/checkpoint-20) | 0.792 |
| [checkpoint-30](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000030/checkpoint-30) | 0.786 |
| [checkpoint-40](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000040/checkpoint-40) | 0.790 |
| [checkpoint-270](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000270/checkpoint-270) | 0.810 |
| [checkpoint-280](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000280/checkpoint-280) | **0.818** |
| [checkpoint-290](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000290/checkpoint-290) | 0.792 |

正式 best point 为：

- 主候选: [checkpoint-280](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000280/checkpoint-280) = `0.818`
- 强备选: [checkpoint-270](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000270/checkpoint-270) = `0.810`

这说明修复 warm-start 后，`022` 的真实强度明显高于“近似 scratch 旧版”；`agent-id asymmetric shaping` 本身是成立的。

### 12.2 failure capture

对 `270/280` 两个高点做 `baseline 500` failure capture：

| checkpoint | capture WR | saved losses |
|---|---:|---:|
| [checkpoint-270](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000270/checkpoint-270) | **0.840** | 80 |
| [checkpoint-280](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000280/checkpoint-280) | 0.822 | 89 |

primary failure bucket（按保存目录重算）：

- [checkpoint-270 failure dir](artifacts/failure-cases/mappo_role_diff_checkpoint270_baseline_500)
  - `late_defensive_collapse = 48/80`
  - `low_possession = 17/80`
  - `unclear_loss = 8/80`
  - `poor_conversion = 6/80`
- [checkpoint-280 failure dir](artifacts/failure-cases/mappo_role_diff_checkpoint280_baseline_500)
  - `late_defensive_collapse = 46/89`
  - `low_possession = 23/89`
  - `unclear_loss = 14/89`
  - `poor_conversion = 4/89`

failure structure 的关键信号是：

- `022` 的最大优点是 **`low_possession` 压得很低**
- 同时失败会更多回流到 `late_defensive_collapse`
- 因而它更像“先把控球/站位问题修掉”的线，而不是最平衡的整局结构线

### 12.3 head-to-head

修复后 `022` 的 head-to-head 已归档：

- [022_270_vs_017_2100.log](artifacts/official-evals/headtohead/022_270_vs_017_2100.log)

结果：

- `022@270` vs `017@2100` = `225W-275L-0T`, `win_rate = 0.450`

这说明：

- `022` 已经是明确正结果
- 但在真正 head-to-head 下，仍然明显弱于当前冠军 [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100)

### 12.4 append-only verdict

修复后 `022` 的正式结论是：

- **`agent-id asymmetry` 本身有价值**
- 最佳 official 点为 [checkpoint-280](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430/MAPPOVsBaselineTrainer_Soccer_ee781_00000_0_2026-04-16_04-54-52/checkpoint_000280/checkpoint-280) = `0.818`
- failure capture 显示这条线确实明显压低了 `low_possession`
- 但整体仍低于 `024` 与 `017`

当前排序中，`022` 应归入：

- **强正结果 / 次一级主线**
- 不是当前冠军线

### 12.5 旧版“近似 scratch”结果的补充价值（2026-04-17 后补）

首轮错误版本虽然不能作为正式 warm-start 结论，但仍然有信息价值。对旧版 `022` 做的 official `baseline 500` 复核窗口为 `140/150/160/170/180/250/260/270/280/290`：

| checkpoint | old official `baseline 500` |
|---|---:|
| [checkpoint-140](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439/MAPPOVsBaselineTrainer_Soccer_ace80_00000_0_2026-04-15_22-55-07/checkpoint_000140/checkpoint-140) | 0.518 |
| [checkpoint-150](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439/MAPPOVsBaselineTrainer_Soccer_ace80_00000_0_2026-04-15_22-55-07/checkpoint_000150/checkpoint-150) | 0.528 |
| [checkpoint-160](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439/MAPPOVsBaselineTrainer_Soccer_ace80_00000_0_2026-04-15_22-55-07/checkpoint_000160/checkpoint-160) | 0.536 |
| [checkpoint-170](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439/MAPPOVsBaselineTrainer_Soccer_ace80_00000_0_2026-04-15_22-55-07/checkpoint_000170/checkpoint-170) | 0.574 |
| [checkpoint-180](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439/MAPPOVsBaselineTrainer_Soccer_ace80_00000_0_2026-04-15_22-55-07/checkpoint_000180/checkpoint-180) | 0.572 |
| [checkpoint-250](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439/MAPPOVsBaselineTrainer_Soccer_ace80_00000_0_2026-04-15_22-55-07/checkpoint_000250/checkpoint-250) | 0.646 |
| [checkpoint-260](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439/MAPPOVsBaselineTrainer_Soccer_ace80_00000_0_2026-04-15_22-55-07/checkpoint_000260/checkpoint-260) | 0.652 |
| [checkpoint-270](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439/MAPPOVsBaselineTrainer_Soccer_ace80_00000_0_2026-04-15_22-55-07/checkpoint_000270/checkpoint-270) | 0.664 |
| [checkpoint-280](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439/MAPPOVsBaselineTrainer_Soccer_ace80_00000_0_2026-04-15_22-55-07/checkpoint_000280/checkpoint-280) | **0.700** |
| [checkpoint-290](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439/MAPPOVsBaselineTrainer_Soccer_ace80_00000_0_2026-04-15_22-55-07/checkpoint_000290/checkpoint-290) | 0.670 |

代表性 failure capture 点为 [checkpoint-270](../../ray_results/PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439/MAPPOVsBaselineTrainer_Soccer_ace80_00000_0_2026-04-15_22-55-07/checkpoint_000270/checkpoint-270)：

- old capture WR = `0.650`
- [old failure dir](artifacts/failure-cases/mappo_role_diff_old_checkpoint270_baseline_500)
  - `late_defensive_collapse = 89/175`
  - `low_possession = 42/175`
  - `poor_conversion = 23/175`
  - `unclear_loss = 15/175`

和修复版相比：

- old best official `0.700` → repaired best official `0.818`
- old representative capture `0.650` → repaired representative capture `0.840 @ checkpoint-270`
- 主要失败桶几乎都被明显压下去，尤其 `low_possession` 与 `poor_conversion`

所以旧版 `022` 提供的最有价值信息是：

- 即使在“近似 scratch”语义下，`agent-id asymmetry` 也不是完全无效，已经能看到方向性收益
- 但它更像一条**建立在协调底座之上的 fine-tune 机制**
- 真正吃到 `warm470` 后，这条线才从 `0.65~0.70` 档抬到 `0.81~0.84` 档

## 13. §6 预声明判据 explicit verdict + A/B 矩阵定位

### 13.1 §6.1 主判据 verdict

**500-ep 官方 WR vs baseline ≥ 0.83**

- 实测 best = **0.818** @ ckpt280
- ❌ **FAIL**（差 0.012）

### 13.2 §6.2 机制判据 verdict — `low_possession` ≤ 15%

从 §12.2 failure capture 数据计算 low_poss 占失败比：

| ckpt | total losses | low_poss | **low_poss %** | 对照 MAPPO+v2 @470 (24.0%) |
|---|---|---|---|---|
| 270 | 80 | 17 | **21.2%** | −2.8 pp |
| 280 | 89 | 23 | **25.8%** | +1.8 pp |

- ❌ **FAIL**（21.2-25.8%，都在跨 lane 22-28% 基线带内）
- ckpt270 的 21.2% 是**所有 lane 的最低单点**，但 ckpt280 立刻回到 25.8%——在 80 vs 89 total losses 的 binomial 噪声下，这不是 structural shift
- **low_poss 绝对数** 从 MAPPO+v2 的 30 降到 17（ckpt270）是显著的，但它来自**总失败减少**（125→80, −36%），不是 low_poss 被选择性修复

### 13.3 关于 §12.2 "low_possession 压得很低"的修正

§12.2 原描述"022 的最大优点是 low_possession 压得很低"需要收紧：

- **绝对数**确实低（17/500 = 3.4% of all episodes vs MAPPO+v2 的 30/500 = 6.0%）——从"每 500 局里发生几次 low_poss 失败"看，降幅 43%
- **但占失败比**仍是 21-26%（和所有其他 lane 的 22-28% 一致）
- 正确表述："022 通过降低**总失败率**间接减少了 low_poss **绝对频率**，但没有改变它在**失败桶中的占比**"

### 13.4 §6.3 辅助机制判据 — 行为分化

agent_id 0 / 1 的行为是否出现系统性差异，需要 per-agent position trace 分析（§12.2 未做此步骤）。当前仅有 bucket-level 数据，**无法 pass 或 fail** 这条判据。

**建议**：不为此追做额外分析——§6.2 的 low_poss 判据已经给出 FAIL，主问题已清楚。

### 13.5 §6.4 失败情形检查

| 条件 | 实测 | 触发？ |
|---|---|---|
| WR < 0.76 | 0.818 | ❌ 未触发 |
| low_poss ≥ 20% 且 WR < 0.80 | low_poss 21-26%, WR 0.818 | ❌ 未触发（WR 不 < 0.80）|
| 行为未分化 | 未验证 | ⏳ 未判定 |

**无失败情形触发**。022 是一个**WR 正结果但机制判据未通过**的 lane。

### 13.6 与 024 的对比结论

修复 warm-start 后：
- **022 (agent-id binding) 0.818 < 024 (spawn-depth binding) 0.842**
- 这完全翻转了 broken 版的"022 > 024"
- spawn-aligned role binding 在 warm-start 正确时**显著优于** agent-id-only 非对称

这与 §7 R1 的预判一致：024 的 "信号加强 spawn 已有倾向" 比 022 的 "信号对抗 spawn inertia" 更有效——前提是 warm-start 保留了 spawn-aware 的 policy 结构。

### 13.7 [SNAPSHOT-021 §8](snapshot-021-actor-teammate-obs-expansion.md#8-与-snapshot-022-的关系) A/B 矩阵定位

结合 021c-B（aux head, low_poss **恶化** 到 40%）和 022/024（role-diff, low_poss **不变** 21-33%）：

> **A 单独有害，B 单独中性，两者都不修 low_poss 占比**

这落在矩阵第四行：

> 都未降 → 两假设都不单独成立

**补充修正**：A 不只是"不成立"，是**反向有害**（aux head 恶化 low_poss）；B 不只是"不成立"，是**对 WR 正向但对 low_poss 中性**。

**项目级诊断终局**：`low_possession ≈ 22-28% of failures` 是 Soccer-Twos + shared-policy + own-obs-only 框架的**结构性常量**，在 12 种 intervention 下从未被选择性修复。

### 13.8 Scratch vs Warm-start 2×2 对照（022 + 024 合并视角）

修复版正式 run 和首轮 broken（等同 scratch）run 的 500-ep failure capture 合并后，得到项目里**最干净的 warm-start 效应分解表**：

| 条件 | WR | losses | late_col % | **low_poss %** | poor_conv % |
|---|---|---|---|---|---|
| MAPPO+v2 @470 **（对称 shaping 对照）** | 0.750 | 125 | 49.6% | **24.0%** | 8.8% |
| **022 scratch** (agent-id asym, 无 WS) | 0.650 | 175 | 50.9% | **24.0%** | 13.1% |
| **022 warm-start** (agent-id asym, WS from v2@470) | **0.840** | 80 | **60.0%** | 21.2% | 7.5% |
| **024 scratch** (spawn-depth, 无 WS) | 0.666 | 167 | 52.7% | **20.4%** | 10.2% |
| **024 warm-start** (spawn-depth, WS from v2@470) | **0.830** | 85 | 42.4% | **32.9%** | 12.9% |

四个关键读法：

**1. Warm-start 是压倒性主效应**

| 路径 | scratch → warm-start | Δ WR |
|---|---|---|
| 022 | 0.650 → 0.840 | **+0.190** |
| 024 | 0.666 → 0.830 | **+0.164** |

单纯 warm-start 贡献 +0.16-0.19 WR，是 reward asymmetry 自身贡献的 4-5 倍。之前分析把 WR 功劳给 role-diff shaping——实际大部分来自 MAPPO+v2 @470 warm-start 的 policy 基础。

**2. Scratch 版的 `low_poss` 和对称 MAPPO+v2 完全一致**

- 022 scratch: **24.0%** = MAPPO+v2 baseline 的 24.0%
- 024 scratch: **20.4%**（在 binomial 噪声内）

从零训，非对称 shaping 对 low_poss **毫无效果**。B 假设在 scratch 条件下完全不成立。

**3. Warm-start 改变了桶分布，但方向取决于 binding 类型**

- 022 warm-start: low_poss **微降** (24.0% → 21.2%)，late_collapse **飙升** (50.9% → 60.0%)
- 024 warm-start: low_poss **上升** (20.4% → 32.9%)，late_collapse **下降** (52.7% → 42.4%)

warm-start + asymmetry 不是在修桶，是在**桶间重新分配**：022 把 low_poss 搬到 late_collapse，024 把 late_collapse 搬到 low_poss。两者 WR 都高但失败结构**镜像相反**。

**4. Scratch 训 role-diff 比对称 shaping 更慢**

- MAPPO+v2 scratch @ ~300 iter: 约 0.71（snapshot-014 推算）
- 022 scratch @ 280: 0.700
- 024 scratch @ 280: 0.688

非对称 reward 给 shared-policy 引入额外学习负担（同一网络需要条件化出两套行为），early-training 收敛比对称版慢。
