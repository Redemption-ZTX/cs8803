# SNAPSHOT-037: Architecture-Level Dominance in Peer Play — **RETRACTED**

- **日期**: 2026-04-18（写作 + retract 同日）
- **状态**: **⚠️ RETRACTED** — 本文件的所有核心 claim 都建立在 H2H log 读取错误上。保留此 stub 供后续归档查阅，**任何实验决策和 report 写作都不应引用本文件的原初主张**。
- **作者错误原因**: 读 `docs/experiments/artifacts/official-evals/headtohead/*.log` 时假设"`policies:` 下第一段 = 文件名中第一个 agent"，实际上是**按模块名字母序排**。对所有 cross-architecture H2H (team-level vs per-agent)，`trained_shared_cc_opponent_agent` (per-agent) 字母序先于 `trained_team_ray_agent` (team-level)，所以第一段是**对手**的战绩，不是主体。

## 0. 撤回前的错误 claim（仅供存档）

本文件原本 claim 的内容（**均为错误**）:

- ❌ 028A@1060 (team-level raw) 击败 029B@190 (per-agent SOTA) at 0.538
- ❌ 030A@360 击败 029B@190 at 0.550
- ❌ 030D@320 击败 029B@190 at 0.559 (1000ep combined)
- ❌ "架构层优势"假设被验证
- ❌ 双轴 SOTA 非传递性

**真实方向全部相反**：

## 1. 修正后的 H2H 真相

| A (team-level) vs B (per-agent) | A rate | 真实结论 |
|---|---:|---|
| 028A@1060 vs 029B@190 | **0.462** | 028A **输给** 029B (z=1.70, p=0.045) |
| 030A@360 vs 029B@190 | **0.450** | 030A **输给** 029B (z=2.24) |
| 030D@320 vs 029B@190 (1st) | 0.438 | 030D **输给** 029B |
| 030D@320 vs 029B@190 (rerun) | 0.444 | 030D **输给** 029B |
| 030D@320 vs 029B@190 (1000ep) | **0.441** | 030D **输给** 029B (z=3.73, p<0.0001) |
| 030D@320 vs 025b@080 (1st) | 0.468 | 030D **输给** 025b |
| 030D@320 vs 025b@080 (rerun) | 0.450 | 030D **输给** 025b |
| 030D@320 vs 025b@080 (1000ep) | **0.459** | 030D **输给** 025b (z=2.59) |

Team-level 内部 H2H（均为 team-level，字母序的 first-printed 确实是 m1）:

| A vs B | A rate | 结论 |
|---|---:|---|
| 030D@320 vs 028A@1060 | 0.536 | 030D 赢 028A 自己 warmstart |
| 030D@320 vs 030A@360 | 0.504 | tie |

## 2. 修正后的真相叙述

**029B@190 在两个判据下都是冠军**：
1. **baseline 500 WR**: 0.868（全项目最高）
2. **peer H2H**：击败所有受测 team-level 变体（0.538-0.562）

**025b@080 (per-agent 冠军 field-role) 也击败 team-level 030D**。所以 per-agent 家族 (017 → 025b → 029B) 对 team-level 家族 (028A → 030A/030D/032A/033A) 在 peer play 上**系统性占优**。

**没有非传递性**。**没有架构层翻转优势**。[snapshot-030 §0.3](snapshot-030-team-level-advanced-shaping-chain.md) 的"team-level 信息论上限"假设目前**缺乏 peer-play 支持证据**。

## 3. 对后续实验决策的影响（正确版）

- **036C (训练中) 的 warmstart 选 029B@190 是正确的**（最强点，baseline + peer 双轴）
- **snapshot-036 §5.3 原本规划的"考虑 036-team-level warmstart"** ← 目前**没有支持证据**。team-level 在 peer play 也弱于 per-agent，切过去只会同时损失两个判据
- **无需做 028A vs 025b / 028A vs 017 的补充 H2H** 来"完成架构矩阵" —— 方向已定
- **也无需做 030D-control 来分离 shaping vs architecture** —— team-level 没有我们以为的那个 architecture 层优势

## 4. 我应该如何避免重犯

写在备忘里供后续参考:
- 任何 H2H 结论前，**直接 grep log 的 `^  cs8803drl.deployment` 和其下 `policy_win_rate`**，不要从混合 grep 结果推测
- 更安全：写一个 helper 函数，接受 log 文件 + agent 模块名，返回该模块的 win_rate。避免"第一段是谁"的假设
- 跨 architecture 的 H2H 尤其容易读反，因为 deploy module 名完全不同

## 5. 本文件的归属

本文件已**撤回**。snapshot-030 §12-§13 的 local 分析（用户编写的）是**正确**的，是唯一需要的 H2H 归档。本文件仅保留为**错误反面教材** + 读 log 格式说明。

所有**正确的 H2H 归档**请看：
- [SNAPSHOT-030 §12.2](snapshot-030-team-level-advanced-shaping-chain.md)（030D 的全部 H2H 数据，方向正确）
- [SNAPSHOT-032 §14](snapshot-032-team-level-native-coordination-aux-loss.md)（032A/032Ac 的 H2H，这部分我早些时候的解读也可能有方向错误，需要用户二次核对）
- [SNAPSHOT-033 §13.5](snapshot-033-team-level-native-coordination-reward.md)（033A 的 H2H）
