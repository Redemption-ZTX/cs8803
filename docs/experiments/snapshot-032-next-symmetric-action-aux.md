# SNAPSHOT-032-next: Symmetric Team Action Aux Refinement

- **日期**: 2026-04-18
- **负责人**:
- **状态**: 已完成首轮结果

## 0. 立项原因

[SNAPSHOT-032](snapshot-032-team-level-native-coordination-aux-loss.md) 的首轮结果没有给出一个“干净的大胜”：

- `032-A-control@200` 在 official `baseline 500` 上略高：`0.836 official / 0.828 capture`
- `032-A@170` 在 direct H2H 上反而更强：
  - `vs 032-A-control@200 = 0.528`
  - `vs 028A@1060 = 0.536`

这说明首轮结果**更像“当前 aux 写法还不够好”**，而不是“aux 方向本身错误”。

因此需要一个最小增量、但信息量很高的 follow-up：保留 `032` 的基本训练框架不变，只修正当前最可疑的两个实现问题：

1. **one-sided aux target**
   - 现实现只预测 joint action 的后 3 维（第二名 agent 的动作因子）
   - 它更像单边 regularizer，而不是真正的双向 coordination signal
2. **aux 机制不可观测**
   - 现实现里有 `metrics()`，但没有成功把 `aux_loss / aux_acc` 变成训练期稳定可读的 `progress.csv` 列
   - 导致我们只能看 WR/H2H，不能判断 aux 到底有没有学到东西

`032-next` 的目标，就是先把这两个问题拆掉。

## 1. 核心假设

如果 `032-A` 首轮没有把 baseline ceiling 明显抬高，主要是因为 aux 信号当前过于单边、且不可观测，那么：

1. 把 action-prediction aux 改成**双向 / 对称**
2. 把 `aux_loss / aux_acc` 稳定写入训练日志

之后我们应该能更清楚地区分三种情况：

- **情况 A**：aux 真的有用
  - `baseline 500` 上开始超过 zero-weight continuation
  - `aux_acc` 稳定上升
- **情况 B**：aux 只改善策略质量，不改善 baseline WR
  - baseline official 仍与 control 接近
  - 但 H2H 继续稳定小优
  - `aux_acc` 同时上升
- **情况 C**：aux 其实只是噪声 / regularizer 幻觉
  - baseline official 与 H2H 都不再优于 control
  - `aux_acc` 也学不起来

## 2. 032 首轮问题定位

当前 [team_action_aux.py](../../cs8803drl/branches/team_action_aux.py) 的写法有两个工程风险：

### 2.1 One-sided target

当前 target 只取：

```python
_target_start = nvec.size // 2
target = actions[:, _target_start : _target_start + _target_dims]
```

这相当于只预测 second local agent 的 action factors。

风险：

- 信号不对称
- 更像“让 trunk 对后半边 action slot 更可预测”
- 不够像“team-level mutual modeling”

### 2.2 Metrics 不落盘

当前 model 内部虽然维护了：

- `aux_action_loss`
- `aux_action_acc_mean`
- `aux_action_acc_dim*`

但首轮 [progress.csv](../../ray_results/032A_team_action_aux_on_028A1060_512x512_20260418_053238/TeamVsBaselineShapingPPOTrainer_Soccer_96926_00000_0_2026-04-18_05-32-59/progress.csv) 没有稳定出现这些列。

风险：

- 无法区分“aux 没学到” vs “aux 学到了但没转化”
- 无法做 weight / gap 的第二轮调参

## 3. 032-next 最小修改范围

`032-next` 不改 base trainer、不改 warm-start、不改 reward。

只做两个改动：

### 3.1 `032-next-A`: Symmetric bidirectional action aux

把当前 one-sided aux 改成：

- 预测 agent-0 的 3 个 action factors
- 预测 agent-1 的 3 个 action factors
- 总 aux loss 为两侧 CE 的均值或加权和

示意：

```text
shared trunk feat
  ├── policy head -> joint action logits
  ├── aux_head_01 -> predict agent1 action factors
  └── aux_head_10 -> predict agent0 action factors

aux_loss = 0.5 * CE(pred_01, a1) + 0.5 * CE(pred_10, a0)
total_loss = ppo_loss + λ * aux_loss
```

这一步仍然保留“最小改动”：

- 不引入新的 encoder 结构
- 不改 warm-start checkpoint 格式
- 仍然兼容 `028A@1060` 的 base

### 3.2 `032-next-control`: Logging-only continuation control

为了避免把“指标可见性变化”误当成“aux 变好”：

- control 也走同一套 metrics logging 路径
- 但 `aux_weight = 0`

这样我们可以确保：

- 两条线的日志口径一致
- 差异主要来自 symmetric aux，不是 logging side-effect

### 3.3 稳定落盘的 metrics

必须把以下指标稳定写进训练日志：

- `aux_action_loss`
- `aux_action_acc_mean`
- `aux_action_acc_agent0`
- `aux_action_acc_agent1`
- 如有必要，再细分 `dim0/1/2`

要求：

- `progress.csv` 可直接读
- `result.json` / TensorBoard 也能看到

## 4. 首轮执行口径

### 4.1 Base

继续沿用：

- `028A@1060`

原因：

- 它仍然是当前最稳的 team-level base
- `032` 首轮已经证明它足够强，且 continuation 本身有效

### 4.2 训练预算

与 `032` 首轮保持一致，避免引入新混淆：

- `300 iter`
- `~12M timesteps`
- 其余 PPO/shaping 参数尽量不动

### 4.3 Lane

| lane | 改动 | 目的 |
|---|---|---|
| `032-next-A` | 双向 symmetric action aux + metrics | 测“修正写法后 aux 是否更干净” |
| `032-next-control` | 相同 logging 路径，但 `aux_weight=0` | 对照 continuation 本身 |

### 4.4 已落实的首轮实现

- 新增 symmetric model path：`TeamActionAuxSymmetricTorchModel`
- trainer 已支持：
  - `AUX_TEAM_ACTION_SYMMETRIC=1`
  - model-side `aux_*` metrics 注入 `learner_stats`
- batch 已就绪：
  - [032-next-A batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_032nextA_symmetric_action_aux_on_028A1060_512x512.batch)
  - [032-next-control batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_032nextA_control_symmetric_action_aux0_on_028A1060_512x512.batch)

## 5. 判据

### 5.1 主判据

| 项 | 阈值 | 解读 |
|---|---|---|
| official `baseline 500` vs control | 至少不低，最好 `+0.01` 以上 | aux 是否终于把收益转移到 baseline score |
| H2H vs `032-A-control@200` | `>= 0.53` | 是否延续 `032-A` 首轮的 direct-H2H 优势 |
| H2H vs `028A@1060` | `>= 0.55` | 是否比 `032` 首轮更干净地超过 base |

### 5.2 机制判据

| 项 | 期望 |
|---|---|
| `aux_action_acc_mean` | 明显高于随机并持续上升 |
| `aux_action_acc_agent0/1` | 两侧都上升，不出现单边塌缩 |
| baseline official 与 H2H | 至少有一项明显优于 control |

### 5.3 失败判据

| 条件 | 解读 |
|---|---|
| aux acc 不升 | 写法仍然没学到东西 |
| H2H 不再优于 control | `032` 首轮优势可能只是噪声 |
| baseline official 与 control 继续打平，且机制指标也弱 | aux 本身价值存疑 |

## 6. 如果 032-next 仍然不够

如果 `032-next` 仍然只给出“微弱 H2H 正信号”，那么下一步才值得考虑更结构化的版本：

- **masked/local-only aux path**
  - 让某一侧的 aux head 不能直接走“joint action reconstruction”捷径
- 或者引入更明确的 team-level native architecture（和 [031](snapshot-031-team-level-native-dual-encoder-attention.md) 结合）

也就是说：

- `032-next` 先测“是不是 one-sided + 不可观测的问题”
- 如果还不够，再测“是不是 full shared trunk 本身就太容易抄答案”

## 7. 相关

- [SNAPSHOT-032](snapshot-032-team-level-native-coordination-aux-loss.md)
- [SNAPSHOT-028](snapshot-028-team-level-bc-to-ppo-bootstrap.md)
- [SNAPSHOT-031](snapshot-031-team-level-native-dual-encoder-attention.md)

## 8. 首轮结果

### 8.1 训练完成情况

- `032-next-A` 已完成：
  - run root: [032nextA_symmetric_action_aux_on_028A1060_formal](../../ray_results/032nextA_symmetric_action_aux_on_028A1060_formal)
  - internal best baseline: [checkpoint-50](../../ray_results/032nextA_symmetric_action_aux_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_87553_00000_0_2026-04-18_08-24-21/checkpoint_000050/checkpoint-50) `= 0.880`
- `032-next-control` 已完成：
  - run root: [032nextControl_symmetric_action_aux0_on_028A1060_formal](../../ray_results/032nextControl_symmetric_action_aux0_on_028A1060_formal)
  - internal best baseline: [checkpoint-120](../../ray_results/032nextControl_symmetric_action_aux0_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_88fdf_00000_0_2026-04-18_08-24-24/checkpoint_000120/checkpoint-120) `= 0.920`

两条线的 [checkpoint_eval.csv](../../ray_results/032nextA_symmetric_action_aux_on_028A1060_formal/checkpoint_eval.csv) / [checkpoint_eval.csv](../../ray_results/032nextControl_symmetric_action_aux0_on_028A1060_formal/checkpoint_eval.csv) 都从 `10 -> 290` 完整，无需 backfill。

### 8.2 official baseline 1000

这轮按 finalist 窗口直接打了 `baseline 1000`，不再先做大窗口 `500ep` 扫描。

`032-next-A`

| checkpoint | official `baseline 1000` |
|---|---|
| [40](../../ray_results/032nextA_symmetric_action_aux_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_87553_00000_0_2026-04-18_08-24-21/checkpoint_000040/checkpoint-40) | `0.786` |
| [50](../../ray_results/032nextA_symmetric_action_aux_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_87553_00000_0_2026-04-18_08-24-21/checkpoint_000050/checkpoint-50) | `0.774` |
| [60](../../ray_results/032nextA_symmetric_action_aux_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_87553_00000_0_2026-04-18_08-24-21/checkpoint_000060/checkpoint-60) | `0.779` |
| [100](../../ray_results/032nextA_symmetric_action_aux_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_87553_00000_0_2026-04-18_08-24-21/checkpoint_000100/checkpoint-100) | `0.784` |
| [110](../../ray_results/032nextA_symmetric_action_aux_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_87553_00000_0_2026-04-18_08-24-21/checkpoint_000110/checkpoint-110) | `0.793` |
| [120](../../ray_results/032nextA_symmetric_action_aux_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_87553_00000_0_2026-04-18_08-24-21/checkpoint_000120/checkpoint-120) | `0.767` |

`032-next-control`

| checkpoint | official `baseline 1000` |
|---|---|
| [60](../../ray_results/032nextControl_symmetric_action_aux0_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_88fdf_00000_0_2026-04-18_08-24-24/checkpoint_000060/checkpoint-60) | `0.765` |
| [70](../../ray_results/032nextControl_symmetric_action_aux0_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_88fdf_00000_0_2026-04-18_08-24-24/checkpoint_000070/checkpoint-70) | `0.798` |
| [80](../../ray_results/032nextControl_symmetric_action_aux0_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_88fdf_00000_0_2026-04-18_08-24-24/checkpoint_000080/checkpoint-80) | `0.779` |
| [110](../../ray_results/032nextControl_symmetric_action_aux0_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_88fdf_00000_0_2026-04-18_08-24-24/checkpoint_000110/checkpoint-110) | `0.781` |
| [120](../../ray_results/032nextControl_symmetric_action_aux0_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_88fdf_00000_0_2026-04-18_08-24-24/checkpoint_000120/checkpoint-120) | `0.782` |
| [130](../../ray_results/032nextControl_symmetric_action_aux0_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_88fdf_00000_0_2026-04-18_08-24-24/checkpoint_000130/checkpoint-130) | `0.822` |

首轮 official `1000ep` 结果显示：

- `032-next-A` best official 仅为 [checkpoint-110](../../ray_results/032nextA_symmetric_action_aux_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_87553_00000_0_2026-04-18_08-24-21/checkpoint_000110/checkpoint-110) `= 0.793`
- `032-next-control` best official 为 [checkpoint-130](../../ray_results/032nextControl_symmetric_action_aux0_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_88fdf_00000_0_2026-04-18_08-24-24/checkpoint_000130/checkpoint-130) `= 0.822`

### 8.3 aux metrics 可观测性

这轮最大的工程收获是：`aux_*` 指标已经稳定写进 [progress.csv](../../ray_results/032nextA_symmetric_action_aux_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_87553_00000_0_2026-04-18_08-24-21/progress.csv) / [progress.csv](../../ray_results/032nextControl_symmetric_action_aux0_on_028A1060_formal/TeamVsBaselineShapingPPOTrainer_Soccer_88fdf_00000_0_2026-04-18_08-24-24/progress.csv)。

`032-next-A`

- `aux_metrics/aux_action_loss`: `0.4636 -> 0.1764`
- `aux_metrics/aux_action_acc_mean`: `0.8432 -> 0.9291`

`032-next-control`

- `aux_metrics/aux_action_loss`: 基本横盘在 `~1.09`
- `aux_metrics/aux_action_acc_mean`: 基本横盘在 `~0.35`

这说明：

- symmetric bidirectional aux 的 plumbing 是通的
- `032-next-A` 的 aux head 确实学到了可预测结构
- `032-next-control` 也确实是一个有效对照，而不是“日志没接上”的伪 control

### 8.4 当前读法

和 `032` 首轮相比，`032-next` 把问题定位推进了一步：

- 现在已经不太像“单边 target / metrics 不可观测”导致的假阴性
- 更像是**当前 action-prediction aux 目标本身，与主任务收益不够对齐**

更直白一点说：

- 这版 aux **会学**
- 但它学到的东西，至少在 `vs baseline` 主指标上，并没有转化成更强的策略
- 甚至在 `official baseline 1000` 上，已经被 zero-weight continuation control 明显压过

因此 `032-next` 的首轮结论应收为：

- `032-next-control` 是首轮 winner
- `032-next-A` 在机制上成功，但在结果上偏负面
- 当前证据更支持“**aux 目标问题**”而不是“**aux plumbing 问题**”

### 8.5 对 032 主线的影响

这轮结果反过来也帮助解释 `032` 首轮：

- 老 `032-A@170` 之所以还能在 H2H 上给出方向性正号，未必说明“action-prediction aux 本身非常好”
- 它更可能来自旧写法中的 regularization / optimization side-effect
- 当我们把 symmetric 写法和 logging 都修正之后，baseline `1000ep` 结果反而没有提升

因此目前不建议继续沿“预测当前 teammate action factors”这一路追加更多算力。

如果要继续做 `032` 系列，更值得的方向会是：

- 预测更难、也更有任务语义的量
  - 例如 teammate future occupancy / lane intent / ball-contest intent
- 或者让 aux 输入更难通过 shared trunk 直接“抄答案”

## 9. 当前结论

`032-next` 的价值已经成立，但成立方式与最初希望的不同：

- 它没有把 `032` 推成更强的 winner lane
- 它把“问题到底在写法，还是在目标”这件事区分得更清楚了

当前最稳的结论是：

**`032-next` 基本排除了 plumbing 解释，并把主要怀疑点推进到了 aux target/task alignment 本身。**
