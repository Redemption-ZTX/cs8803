# SNAPSHOT-024: Striker/Defender Role Binding

- **日期**: 2026-04-15
- **负责人**:
- **状态**: 修复后首轮结果已完成（official `baseline 500`、failure capture、head-to-head 已补齐）

## 0. 2026-04-16 关键更正：首轮结果不是有效的 warm-start field-role fine-tune

首轮 `024` 已完成训练并得到训练内最佳点：

- best eval baseline: `0.680 @ checkpoint-250/260/280/290`
- best eval random: `1.000`

但与 [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md) 相同，2026-04-16 复盘时确认：

- batch 虽然配置了 `WARMSTART_CHECKPOINT=...MAPPO+v2@470`
- 训练入口 after-init warm-start 路径也确实被调用
- 但 [warmstart_shared_cc_policy()](../../cs8803drl/branches/shared_central_critic.py) 的 key mapping 只适配“裸 trunk → shared-cc”
- 对 shared-cc 源（`action_model.* / value_model.*`）会造成 `copied=0, adapted=0, skipped=16` 的无效迁移

因此，首轮 `024` 的准确语义不是：

- “`warm470 + field-role binding` 的 300-iter fine-tune”

而更接近：

- “**近似 scratch + field-role binding** 的 300-iter 训练”

这意味着：

- 首轮 `024` 的低分不能直接解释成“warm-start 后 strict role binding 伤害了已学到的强策略”
- 但它与 `022` 的**方向性比较仍然有效**：在同样没吃到 warm-start 的前提下，`spawn-depth field-role asymmetry` 明显比 `agent-id asymmetry` 更难学

本 snapshot 的正式 verdict 同样以“**修复 warm-start bug 后的重跑结果**”为准；首轮结果只保留为方向性参考。

### 0.1 2026-04-16 修复与 smoke 验证

与 [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md) 共用的 warm-start bug 现已修复：

- [warmstart_shared_cc_policy()](../../cs8803drl/branches/shared_central_critic.py) 新增 shared-cc → shared-cc 的直接 key match 迁移分支
- 不再把 `action_model.* / value_model.*` 源误判成“裸 trunk”

修复后的函数级 smoke 结果：

- source: [MAPPO+v2@470](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/MAPPOVsBaselineTrainer_Soccer_a245f_00000_0_2026-04-13_04-06-11/checkpoint_000470/checkpoint-470)
- result: `{'copied': 16, 'adapted': 0, 'skipped': 0}`
- `before_eq_after = False`
- `after_eq_source = True`

因此，`024` 现在也已恢复到原设计语义：

- **真正的 `warm470 + field-role binding` fine-tune**

后续正式重跑的结果，才是本 snapshot 的有效 verdict。

## 1. 为什么需要 024

[SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md) 现在已经被明确收紧为：

- **agent-id asymmetric shaping**
- 也就是先测“**打破 reward 对称性本身**有没有用”

这个定义更诚实，但它也留下了一个明确边界：

- `022` 成功，最多说明 **reward asymmetry 有价值**
- `022` 失败，也不能直接否掉 **真实 striker/defender role specialization**

所以 `024` 的职责，是把“agent-id 非对称”和“field-role 非对称”彻底拆开。

## 2. 024 要回答的问题

**如果 role 不是绑定在 `agent_id`，而是绑定在每局真实的 spawn / field semantics 上，是否会比 `022` 更有效？**

也就是：

- `022` 测的是 **agent-id-based asymmetry**
- `024` 测的是 **field-role-based asymmetry**

## 3. 和 022 的关系

| Snapshot | 测什么 | 绑定方式 |
|---|---|---|
| [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md) | 打破 reward 对称性本身是否有用 | `agent_id 0 / 1` |
| **SNAPSHOT-024** | 真实 striker/defender specialization 是否更强 | `spawn / field role` |

最重要的一句是：

**`024` 不是 `022` 的重跑，而是更强、更干净的后续实验。**

## 4. 核心假设

### 主假设

如果 role 绑定在**每局真实场上语义**而不是固定 `agent_id`，那么：

- `low_possession` 占失败比会比 `022` 更明显下降
- 行为 specialization 会更容易解释
- official `baseline 500` 可能超过纯 `022`

### 备择假设

如果 `024` 相比 `022` 没有额外提升，说明：

- reward asymmetry 的主要收益已经在 `022` 里被吃到
- 明确的 field-role binding 不是关键增益来源

## 5. 设计原则

### 5.1 role 必须依赖运行时 field semantics

候选信号包括：

- 第 1 步读取每个 agent 的 spawn `x/y`
- 按 `x` 深度划分“更前 / 更后”
- 必要时再结合 `y` lane（上路 / 下路）

目标是把每局的两个 team0 agent 映射成：

- `striker`
- `defender`

而不是固定成：

- `agent_id 0`
- `agent_id 1`

### 5.2 shaping 差分先沿用 022 的一阶版本

首轮不重新搜索系数，只把 `022` 当前的两极配置搬过来：

| 项 | striker | defender |
|---|---|---|
| `ball_progress_scale` | 高 | 低 |
| `opponent_progress_penalty_scale` | 低 | 高 |
| `deep_zone_penalty` | 轻 | 重 |
| `possession_bonus` | 相同 | 相同 |
| `time_penalty` | 相同 | 相同 |

这样做的目的，是让 `024` 和 `022` 的唯一区别尽量收敛到：

- **role binding 方式**

## 6. 工程需求

这条线比 `022` 多出的真正工程量在于：

1. env wrapper 需要在 reset / episode 开始时识别 role
2. role 信息要能稳定传到 shaping config 或 runtime reward path
3. eval / failure capture 时要能复原“谁是 striker / 谁是 defender”

可能的实现方式：

- 在 wrapper 内为每局生成 `role_by_agent`
- shaping 逻辑不再直接查 `agent_id -> scale`
- 改成查 `role_by_agent[agent_id] -> scale`

### 6.1 当前实现状态

本轮已经完成 024 的最小 runtime plumbing：

- [train_ray_mappo_vs_baseline.py](../../cs8803drl/training/train_ray_mappo_vs_baseline.py) 新增 `SHAPING_FIELD_ROLE_BINDING`
- [role_specialization.py](../../cs8803drl/branches/role_specialization.py) 新增 `build_field_role_reward_shaping_config()`
- [utils.py](../../cs8803drl/core/utils.py) 的 [RewardShapingWrapper](../../cs8803drl/core/utils.py) 已支持：
  - `role_binding_mode=spawn_depth`
  - 每局动态生成 `role_by_agent`
  - 再按 `striker / defender` 把 role-specific shaping 路由到当前 agent
- batch 已新增：
  - [soccerstwos_h100_cpu32_mappo_field_role_binding_v2_warm470_512x512.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_mappo_field_role_binding_v2_warm470_512x512.batch)

最小 smoke 已通过：

- env debug 输出成功打印 `role_binding_mode = spawn_depth`
- 首轮 smoke 观测到 `role_by_agent = {1: 'striker', 0: 'defender'}`

所以 024 现在不是纯预注册，而是**已实现并完成首轮试跑**；但首轮结果因为 warm-start 失效，只能作为方向性参考，仍需修复后重跑。

## 7. 判据

### 7.1 主判据

如果 `024` 开训，主判据不是单看分数，而是看：

- 是否**严格优于** `022` 最佳点
- 或者在相近分数下，failure structure 是否更干净

### 7.2 机制判据

`024` 的关键机制判据比 `022` 更强：

- `low_possession` 占失败比是否进一步下降
- 行为分化是否能和 `striker / defender` 语义对应起来
- 这种分化在 side-swap 下是否仍然成立

## 8. 风险

### R1 — role 判定规则本身可能引入噪声

如果 spawn 深度差很小，或者某些局面前后关系并不稳定，role 绑定会抖动。

### R2 — 需要比 022 更多的 runtime plumbing

这条线不是简单改 env var，需要真正的 role tagging / routing。

### R3 — 如果 022 已经足够好，024 的收益可能很小

也就是说，024 的价值更多在于**解释力**，不一定总能转成很大的性能提升。

## 9. 推荐执行顺序

1. 先跑 [SNAPSHOT-022](snapshot-022-role-differentiated-shaping.md)
2. 用 022 的结果判断 reward asymmetry 本身是否值得继续
3. 若值得，再开 024 做严格 role-binding 版本

## 10. 下一步执行清单

1. 修复 [warmstart_shared_cc_policy()](../../cs8803drl/branches/shared_central_critic.py) 的 shared-cc → shared-cc 迁移 bug。
2. 用 `1-iter` smoke 验证 `024` 的 `warmstart_summary.txt` 显示 `copied > 0`，并继续打印 `role_by_agent`。
3. 重跑 024 首轮训练。
4. 对最佳点做 `baseline 500` 复核。
5. 若有正信号，再补 failure capture 与 side-swap 分析。

## 11. 修复后正式结果（2026-04-17 后补，append-only）

修复 [warmstart_shared_cc_policy()](../../cs8803drl/branches/shared_central_critic.py) 的 shared-cc → shared-cc 迁移 bug 后，`024` 的正式 rerun 为：

- run_dir: [PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516)
- warm-start 证据: [warmstart_summary.txt](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/warmstart_summary.txt)

### 11.1 official `baseline 500` 复核

按 `top 5% + ties + 前后 2 点`，正式复核窗口为 `10/20/30/40/50/60/70/80/130/140/150/160/170/230/240/250/260/270`：

| checkpoint | official `baseline 500` |
|---|---:|
| [checkpoint-10](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000010/checkpoint-10) | 0.784 |
| [checkpoint-20](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000020/checkpoint-20) | 0.760 |
| [checkpoint-30](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000030/checkpoint-30) | 0.758 |
| [checkpoint-40](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000040/checkpoint-40) | 0.800 |
| [checkpoint-50](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000050/checkpoint-50) | 0.786 |
| [checkpoint-60](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000060/checkpoint-60) | 0.792 |
| [checkpoint-70](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000070/checkpoint-70) | 0.782 |
| [checkpoint-80](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000080/checkpoint-80) | 0.750 |
| [checkpoint-130](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000130/checkpoint-130) | 0.762 |
| [checkpoint-140](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000140/checkpoint-140) | 0.772 |
| [checkpoint-150](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000150/checkpoint-150) | 0.780 |
| [checkpoint-160](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000160/checkpoint-160) | 0.806 |
| [checkpoint-170](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000170/checkpoint-170) | 0.800 |
| [checkpoint-230](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000230/checkpoint-230) | 0.814 |
| [checkpoint-240](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000240/checkpoint-240) | 0.780 |
| [checkpoint-250](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000250/checkpoint-250) | 0.804 |
| [checkpoint-260](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000260/checkpoint-260) | 0.798 |
| [checkpoint-270](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000270/checkpoint-270) | **0.842** |

正式 best point 为：

- 主候选: [checkpoint-270](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000270/checkpoint-270) = `0.842`
- 强备选: [checkpoint-230](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000230/checkpoint-230) = `0.814`
- 放弃点: [checkpoint-160](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000160/checkpoint-160)

修复后，`024` 直接推翻了首轮“`022 > 024`”的旧印象：在真实 warm-start 下，**`024 > 022`**。

### 11.2 failure capture

对 `160/230/270` 三个代表点做 `baseline 500` failure capture：

| checkpoint | capture WR | saved losses |
|---|---:|---:|
| [checkpoint-160](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000160/checkpoint-160) | 0.758 | 121 |
| [checkpoint-230](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000230/checkpoint-230) | 0.802 | 99 |
| [checkpoint-270](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000270/checkpoint-270) | **0.830** | 85 |

primary failure bucket（按保存目录重算）：

- [checkpoint-160 failure dir](artifacts/failure-cases/mappo_field_role_checkpoint160_baseline_500)
  - `late_defensive_collapse = 54/121`
  - `low_possession = 37/121`
  - `unclear_loss = 12/121`
  - `poor_conversion = 10/121`
- [checkpoint-230 failure dir](artifacts/failure-cases/mappo_field_role_checkpoint230_baseline_500)
  - `late_defensive_collapse = 45/99`
  - `low_possession = 26/99`
  - `unclear_loss = 17/99`
  - `poor_conversion = 8/99`
- [checkpoint-270 failure dir](artifacts/failure-cases/mappo_field_role_checkpoint270_baseline_500)
  - `late_defensive_collapse = 36/85`
  - `low_possession = 28/85`
  - `poor_conversion = 11/85`
  - `unclear_loss = 7/85`

failure structure 显示：

- `270` 不是“高分脆点”，而是已经相当平衡的成熟候选
- `230` 是稳健次优点
- `160` 可以放掉

### 11.3 head-to-head

修复后 `024` 的 head-to-head 已归档：

- [024_270_vs_017_2100.log](artifacts/official-evals/headtohead/024_270_vs_017_2100.log)
- [024_270_vs_018_290.log](artifacts/official-evals/headtohead/024_270_vs_018_290.log)

结果：

- `024@270` vs `017@2100` = `229W-271L-0T`, `win_rate = 0.458`
- `024@270` vs `018@290` = `259W-241L-0T`, `win_rate = 0.518`

这说明：

- `024@270` 已经能在 head-to-head 中小优 [checkpoint-290](../../ray_results/PPO_mappo_v2_opponent_pool_512x512_20260414_212239/MAPPOVsOpponentPoolTrainer_Soccer_a6f40_00000_0_2026-04-14_21-23-05/checkpoint_000290/checkpoint-290)
- 但面对 [checkpoint-2100](../../ray_results/PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/MAPPOVsBaselineTrainer_Soccer_e53af_00000_0_2026-04-14_21-46-18/checkpoint_002100/checkpoint-2100) 仍然明确下风

### 11.4 append-only verdict

修复后 `024` 的正式结论是：

- **真实 field-role binding 在强 warm-start 底座上是成立的**
- best official 点 [checkpoint-270](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516/MAPPOVsBaselineTrainer_Soccer_0aab8_00000_0_2026-04-16_04-55-39/checkpoint_000270/checkpoint-270) = `0.842`
- 它已经进入冠军竞争区，并在 head-to-head 中小优 `018`
- 但 `017@2100` 仍然是更强的当前冠军

当前全局排序中，`024` 应归入：

- **第一梯队挑战者**
- 位置大致为 `017 > 024 > 018 > 022`

### 11.5 旧版“近似 scratch”结果的补充价值（2026-04-17 后补）

首轮错误版本虽然不能作为正式 warm-start 结论，但它帮助我们识别了 `024` 对底座的依赖程度。对旧版 `024` 做的 official `baseline 500` 复核窗口为 `230/240/250/260/270/280/290`：

| checkpoint | old official `baseline 500` |
|---|---:|
| [checkpoint-230](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260415_230033/MAPPOVsBaselineTrainer_Soccer_7dc3d_00000_0_2026-04-15_23-00-57/checkpoint_000230/checkpoint-230) | 0.634 |
| [checkpoint-240](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260415_230033/MAPPOVsBaselineTrainer_Soccer_7dc3d_00000_0_2026-04-15_23-00-57/checkpoint_000240/checkpoint-240) | 0.656 |
| [checkpoint-250](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260415_230033/MAPPOVsBaselineTrainer_Soccer_7dc3d_00000_0_2026-04-15_23-00-57/checkpoint_000250/checkpoint-250) | 0.666 |
| [checkpoint-260](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260415_230033/MAPPOVsBaselineTrainer_Soccer_7dc3d_00000_0_2026-04-15_23-00-57/checkpoint_000260/checkpoint-260) | 0.670 |
| [checkpoint-270](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260415_230033/MAPPOVsBaselineTrainer_Soccer_7dc3d_00000_0_2026-04-15_23-00-57/checkpoint_000270/checkpoint-270) | 0.676 |
| [checkpoint-280](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260415_230033/MAPPOVsBaselineTrainer_Soccer_7dc3d_00000_0_2026-04-15_23-00-57/checkpoint_000280/checkpoint-280) | **0.688** |
| [checkpoint-290](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260415_230033/MAPPOVsBaselineTrainer_Soccer_7dc3d_00000_0_2026-04-15_23-00-57/checkpoint_000290/checkpoint-290) | 0.672 |

代表性 failure capture 点为 [checkpoint-290](../../ray_results/PPO_mappo_field_role_binding_v2_warm470_512x512_20260415_230033/MAPPOVsBaselineTrainer_Soccer_7dc3d_00000_0_2026-04-15_23-00-57/checkpoint_000290/checkpoint-290)：

- old capture WR = `0.666`
- [old failure dir](artifacts/failure-cases/mappo_field_role_old_checkpoint290_baseline_500)
  - `late_defensive_collapse = 88/167`
  - `low_possession = 34/167`
  - `unclear_loss = 20/167`
  - `poor_conversion = 17/167`

和修复版相比：

- old best official `0.688` → repaired best official `0.842`
- old representative capture `0.666` → repaired representative capture `0.830 @ checkpoint-270`
- `late_defensive_collapse`、`unclear_loss`、`poor_conversion` 都大幅下降

这说明旧版 `024` 的真正信息不是“field-role binding 不行”，而是：

- **`024` 比 `022` 更依赖 warm-start 底座**
- 在“近似 scratch”语义下，`024` 只比 `022 old` 稍弱一档，且两条都停留在 `0.65~0.70`
- 一旦吃到真实 `warm470`，`024` 会比 `022` 有更强的兑现幅度，并直接进入冠军挑战区

## 12. §7 预声明判据 explicit verdict + B 假设诊断

### 12.1 §7.1 主判据 verdict

预声明："是否严格优于 022 最佳点，或 failure structure 更干净"

| 比较 | 022 best | 024 best | 结果 |
|---|---|---|---|
| 500-ep official | 0.818 @ ckpt280 | **0.842 @ ckpt270** | ✅ **024 > 022** (+0.024) |
| head-to-head vs BC @2100 | 0.450 | 0.458 | ≈ 等，都输 BC |
| failure structure | late_collapse 60% / low_poss 21% | late_collapse 42% / low_poss 33% | 024 更平衡 |

✅ **PASS**——024 在 WR 和 failure 平衡性上都严格优于 022。

### 12.2 §7.2 机制判据 — `low_possession` 占失败比

从 §11.2 failure capture 计算：

| ckpt | losses | low_poss | **low_poss %** | 对照 MAPPO+v2 (24.0%) |
|---|---|---|---|---|
| 160 | 121 | 37 | **30.6%** | +6.6 pp |
| 230 | 99 | 26 | **26.3%** | +2.3 pp |
| **270** | 85 | 28 | **32.9%** | **+8.9 pp** |

❌ **FAIL**——非但没降到 ≤ 15%，ckpt270 反升到 32.9%。spawn-aligned role binding **不修 low_possession**。

### 12.3 与 022 的机制对比

| 维度 | 022 @270 | 024 @270 |
|---|---|---|
| total losses | 80 | 85 |
| late_collapse / % | 48 / **60.0%** | 36 / **42.4%** |
| low_poss / % | 17 / 21.2% | 28 / **32.9%** |
| poor_conv / % | 6 / 7.5% | 11 / 12.9% |

- 022 把失败集中到 late_collapse（60%）——agent-id 非对称"强迫"policy 往一个方向偏，其他桶被压
- 024 更平衡（late 42%, low_poss 33%）——spawn-aligned binding 让 policy 在各桶间更均匀失败
- **两者的 low_poss 绝对数都低于 MAPPO+v2（17/28 vs 30），但这来自总失败减少，不是 low_poss 被选择性修**

### 12.4 A/B 假设的最终诊断

结合 [021c-B](snapshot-021-actor-teammate-obs-expansion.md#13-021c-b-5-预声明判据-explicit-verdict--a-假设诊断)（A 假设, low_poss 恶化到 40%）和本 lane + 022（B 假设, low_poss 21-33%）：

| 干预 | low_poss 占比 | WR | 结论 |
|---|---|---|---|
| A 单独（aux head）| 40% ← 恶化 | 0.794 | A **有害** |
| B agent-id（022）| 21-26%（基线内）| 0.818 | B **WR+, low_poss 中性** |
| B spawn-depth（024）| 26-33%（偏高）| **0.842** | B **WR 强+, low_poss 偏负** |

> **`low_possession ≈ 25% of failures` 是 shared-policy + own-obs-only CTDE 架构在 2v2 coordination 上的结构性 floor。12 种 intervention 中没有任何一种选择性降低 low_poss 占比。**

### 12.5 024 的真正贡献

024 没修 low_poss，但做到了：

1. **WR 0.842 追平 BC @2100**——用 3h GPU（300 iter warm-start fine-tune）达到 BC 16h 的同等水平
2. **确认 spawn-aligned > agent-id-only**——024 > 022 (+0.024) 在修复 warm-start 后清晰
3. **成为项目第一梯队 submission 候选**——即使 head-to-head 不如 BC，WR/cost ratio 最高
4. **给 report 贡献完整 B 假设 ablation**：对称 (0.786) → agent-id 非对称 (0.818) → spawn-aligned (0.842)，递进清晰

### 12.6 submission 位置

| 角色 | 候选 | 500-ep | 理由 |
|---|---|---|---|
| **Performance agent** | **BC @2100** | **0.842** | head-to-head 最强 |
| Performance backup | **024 @270** | 0.842 (tied) | WR 同级, h2h 微弱 (0.458 vs BC) |
| Modification agent | v4 PPO @400 | 0.768 | shaping 故事最丰富 |
| Novel concept (+5) | BC 自身 | — | imitation learning |

### 12.7 Scratch vs Warm-start 2×2 对照（022 + 024 合并视角）

首轮 broken warm-start run（等同 scratch）和修复版 run 的 500-ep failure capture 合并：

| 条件 | WR | losses | late_col % | **low_poss %** | poor_conv % |
|---|---|---|---|---|---|
| MAPPO+v2 @470 **（对称 shaping 对照）** | 0.750 | 125 | 49.6% | **24.0%** | 8.8% |
| **022 scratch** (agent-id asym, 无 WS) | 0.650 | 175 | 50.9% | **24.0%** | 13.1% |
| **022 warm-start** (agent-id asym, WS from v2@470) | **0.840** | 80 | **60.0%** | 21.2% | 7.5% |
| **024 scratch** (spawn-depth, 无 WS) | 0.666 | 167 | 52.7% | **20.4%** | 10.2% |
| **024 warm-start** (spawn-depth, WS from v2@470) | **0.830** | 85 | 42.4% | **32.9%** | 12.9% |

四个关键读法：

**1. Warm-start 是压倒性主效应（+0.16 to +0.19 WR）**

| 路径 | scratch → warm-start | Δ WR |
|---|---|---|
| 022 | 0.650 → 0.840 | **+0.190** |
| 024 | 0.666 → 0.830 | **+0.164** |

单纯 warm-start 贡献远大于 reward asymmetry 自身贡献（+0.02-0.04）。之前分析把 WR 功劳给 role-diff shaping——实际**大部分来自 MAPPO+v2 @470 warm-start 的 policy 基础**。

**2. Scratch 版 `low_poss` 和对称 MAPPO+v2 完全一致**

- 022 scratch: **24.0%**（和 MAPPO+v2 的 24.0% 精确相同）
- 024 scratch: **20.4%**（略低，在 binomial 噪声内）

从零训，非对称 shaping 对 low_poss **毫无效果**。

**3. Warm-start 改变桶分布，方向取决于 binding 类型**

- 022 warm-start: low_poss **微降** (24→21%)，**late_collapse 飙升** (51→60%)
- 024 warm-start: low_poss **上升** (20→33%)，**late_collapse 下降** (53→42%)

warm-start + asymmetry 不修桶，**在桶间重新分配**。022 和 024 的分配方向**镜像相反**。

**4. Scratch 训 role-diff 比对称 shaping 更慢**

- MAPPO+v2 scratch @ ~300 iter: 约 0.71
- 022 scratch @ 280: 0.700
- 024 scratch @ 280: 0.688

非对称 reward 给 shared-policy 引入额外学习负担——early-training 收敛比对称版慢。

**这张表对 report 的价值**：它是项目里唯一一个**分解了 warm-start 效应和 shaping 效应**的对照，适合作为 report 的核心 Figure / Table 之一。
