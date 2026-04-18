# SNAPSHOT-013: Baseline Weakness Analysis

- **日期**: 2026-04-13
- **负责人**:
- **状态**: 已完成（首轮 baseline weakness analysis）

## 1. 动机

到目前为止，我们已经反复分析了：

- 我们自己的 checkpoint 为什么输给 baseline

但还没有系统分析：

- baseline 自己在什么局面下会犯错
- baseline 在什么局面下会丢球
- baseline 是否存在稳定可利用的模式

这会导致一个问题：

- 我们只知道“自己哪里不好”
- 却不知道“对手哪里能被打”

## 2. 假设

如果把 baseline 当作可分析的系统，而不是只当固定对手，那么：

1. 可以找出 baseline 的稳定弱点；
2. 这些弱点可以直接指导：
   - RL fine-tune 目标
   - wrapper 设计
   - ensemble 设计
   - 后续 reward shaping / curriculum 设计

## 3. 第一阶段范围

本 snapshot 的首轮 baseline weakness analysis 已完成；下文保留原始设计、分析工具和首轮结论。

第一阶段交付物：

1. episode 记录聚合分析脚本；
2. 三组推荐分析对局；
3. 一个统一 summary 输出格式。

## 4. 工具链

### 4.1 现有记录工具

- [capture_failure_cases.py](../../scripts/eval/capture_failure_cases.py)
- [evaluate_matches.py](../../cs8803drl/evaluation/evaluate_matches.py)
- [failure_cases.py](../../cs8803drl/evaluation/failure_cases.py)

### 4.2 新聚合工具

- [analyze_episode_records.py](../../scripts/eval/analyze_episode_records.py)

它用于把保存下来的 episode JSON 聚合成：

- outcome 分布
- primary label 分布
- label 计数
- step 统计
- metrics 均值（整体 / 按 label）

当前已落地：

- [analyze_episode_records.py](../../scripts/eval/analyze_episode_records.py)
- smoke 聚合结果：
  - [v2 analysis summary](artifacts/failure-cases/v2_checkpoint440_baseline_500_rerun/analysis_summary.json)
  - [baseline_vs_random smoke summary](artifacts/baseline-weakness/smoke_baseline_vs_random_1/summary.json)
- 标准批处理入口：
  - [baseline_vs_baseline.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_baseline_weakness_baseline_vs_baseline.batch)
  - [baseline_vs_random.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_baseline_weakness_baseline_vs_random.batch)

## 5. 第一阶段推荐分析对局

### 5.1 baseline vs baseline

目的：

- 看 baseline 在镜像强对手下最常见的 non-win / draw / loss 模式

### 5.2 baseline vs random

目的：

- 找 baseline 在明显优势局里为什么仍会掉球

### 5.3 baseline vs current best agent

目的：

- 找 baseline 被我们惩罚时的局面分布

## 6. 第一阶段命令范式

### 6.1 先采 episode

```bash
python scripts/eval/capture_failure_cases.py \
  --team0-module ceia_baseline_agent \
  --opponent example_player_agent.agent_random \
  -n 100 \
  --base-port 61205 \
  --save-dir docs/experiments/artifacts/baseline-analysis/baseline_vs_random_100 \
  --save-mode nonwins
```

### 6.2 再聚合分析

```bash
python scripts/eval/analyze_episode_records.py \
  --episodes-dir docs/experiments/artifacts/baseline-analysis/baseline_vs_random_100
```

## 7. 第一阶段成功判据

通过标准：

1. episode 保存目录可被脚本稳定聚合；
2. summary 输出包含：
   - outcome 分布
   - primary label 分布
   - step 统计
   - per-label metrics
3. 可以据此提出至少一条具体的 exploitable hypothesis。

## 8. 当前判断

这条线的价值不在于立刻提高一个 checkpoint 的分数，而在于：

- 给后续 `IL -> RL fine-tune`
- 以及可能的 wrapper / ensemble 设计

提供更像“对手建模”的依据，而不是继续盲调。

## 9. 第一轮结果

已完成：

- [baseline_vs_baseline_20260413_022238](artifacts/baseline-weakness/baseline_vs_baseline_20260413_022238)
- [baseline_vs_random_20260413_022305](artifacts/baseline-weakness/baseline_vs_random_20260413_022305)

对应聚合：

- [bvb analysis summary](artifacts/baseline-weakness/baseline_vs_baseline_20260413_022238/analysis_summary.json)
- [bvr analysis summary](artifacts/baseline-weakness/baseline_vs_random_20260413_022305/analysis_summary.json)

### 9.1 baseline vs baseline

核心结果：

- `255W-245L`
- overall `steps median = 46`, `p75 = 78`
- primary label：
  - `late_defensive_collapse = 113`
  - `unclear_loss = 114`
  - `territory_loss = 18`

更细一点：

- `late_defensive_collapse`
  - `steps median = 49`
  - `tail_mean_ball_x = -9.31`
  - 说明 baseline 自己最典型的输法，是比赛进入中后段后，球持续回到我方深位并最终崩掉
- `unclear_loss`
  - `steps median = 42.5`
  - `tail_mean_ball_x = +8.69`
  - 说明 baseline 还有一类非典型输法：球并不在自己深位，甚至经常在前场或中前场，但最后仍然没有转成胜利
- `territory_loss`
  - primary 只有 `18`
  - 但 `steps mean = 122.5`
  - 说明 baseline 也存在少量非常长、非常拖的“场面漂移型”输局

第一条明确结论：

- baseline 对 baseline 的主要问题不是“早期会被快速打穿”
- 而是“长局里会漂、会晚崩、也会在并不明显劣势的局面下没能收掉比赛”

### 9.2 baseline vs random

核心结果：

- `490W-10L`
- overall `steps median = 39`
- baseline 胜 random 很稳，但 random 的少数赢局非常长：
  - baseline 赢局 `steps median = 38.5`
  - random 赢局 `steps median = 84`
  - random 赢局 `steps mean = 140.9`

失败模式：

- `late_defensive_collapse = 6`
- `unclear_loss = 4`

更细一点：

- `late_defensive_collapse`
  - `steps mean = 150.2`
  - `tail_mean_ball_x = -11.31`
  - 说明 baseline 不是被 random 快速打死，而是在极少数长局里自己拖进深位危险区后崩掉
- `unclear_loss`
  - `steps mean = 127.0`
  - `tail_mean_ball_x = +11.52`
  - 说明还有少量“球在前场但就是没终结掉，最后反而输”的局

第二条明确结论：

- baseline 对弱对手并非总是高效终结
- 它偶尔会把本该轻松收掉的比赛拖成长局
- 而这些长局恰恰是它更容易犯错的窗口

### 9.3 当前 exploitable hypothesis

这一轮数据支持的 hypothesis 不是“开局抢一波把 baseline 打死”，而是：

1. **把比赛拖进 baseline 不舒服的长局窗口**
   - baseline 的主要失误发生在中后段，而不是开局阶段
2. **一旦 baseline 没能在前中期终结比赛，它的后段稳定性会下降**
   - `baseline vs baseline` 和 `baseline vs random` 都指向这一点
3. **baseline 不只是防线会崩，它也会出现前场/中前场不收比赛的局**
   - `unclear_loss` 在 `baseline vs baseline` 中占比不低

### 9.4 对下一阶段的意义

对 `IL / BC`：

- baseline 依然是值得模仿的 teacher
- 但不是完美 teacher
- BC 的意义更像是先学会它的强项，再用 RL fine-tune 去修它的长局漂移和终结不稳

对 `baseline exploitation`：

- 值得优先设计的不是“纯开场强攻”
- 而是“如何把比赛稳定拖进 baseline 不舒服的长局窗口，并在后段放大它的稳定性缺陷”

### 9.5 当前限制

这两轮 weakness analysis 没有打开 reward-shaping debug，因此：

- `team*_progress_toward_goal`
- `team*_shaping_reward_sum`

在这两批 summary 里基本没有信息量。

但这不影响本轮主结论，因为这里关注的是：

- step 结构
- failure label 分布
- `mean_ball_x / tail_mean_ball_x`

也就是 baseline 本身的行为模式，而不是 shaping 诊断。

## 10. 和 v2 agent 的失败桶交叉对比

上面 9.x 只看 baseline 自身的失败模式。把它和我们 v2@440 vs baseline 的失败数据 (500-ep, 见 [v2_checkpoint440_baseline_500_rerun](artifacts/failure-cases/v2_checkpoint440_baseline_500_rerun)) 并排看，结论更锋利。

### 10.1 失败桶分布对比

| primary_label | v2 输给 baseline (n=134) | baseline 输给 baseline (n=245) |
|---|---|---|
| late_defensive_collapse | 52% | 46% |
| **low_possession** | **22%** | **0%** |
| **poor_conversion** | **8%** | **0%** |
| unclear_loss | 13% | 47% |
| territory_loss | 3% | 7% |
| opponent_forward_progress | 2% | 0% |

关键非对称：

- **`low_possession` 和 `poor_conversion` 合计 30% 的我方失败，在 baseline 自对弈里完全不出现**。这两类不是"Soccer-Twos 结构性难题"，是**我们这个 agent 特有的病**。
- `late_defensive_collapse` 占比两边都约一半 — 这是 Soccer-Twos 的**结构性**失败模式，不是我们专属缺陷。
- `unclear_loss` 在 BvB 里高达 47%，在 v2 只有 13% — 更像是分类器对"旗鼓相当"局面的 artifact，不值得专门治。

### 10.2 失败局时长对比

| | median 步数 | 含义 |
|---|---|---|
| v2 输给 baseline | **32** | 我们输得最快 |
| BvB 失败局 | 47 | 对称对抗下的基准 |
| v2 赢下 baseline | 43 | 赢局和 BvB 量级相当 |

v2 的输球局比 BvB 的失败局**快 15 步（−32%）**。baseline 攻我们时的效率**明显高于**它攻自己时。这不是"我们防守略弱"，而是**我们有可预测的防守空当被 baseline 稳定触发**——specific pattern，不是 general skill gap。

### 10.3 天花板重新估计

BvR 结果显示 baseline 真实胜率 **≈ 0.98 而非 1.00**。这意味着：

- baseline 自己也有 ~2% 的结构性漏洞（late_collapse × 6, unclear × 4 / 500）
- 9/10 提交门槛要求的真实胜率 ~0.95，**等于 baseline 自己对 random 的水平**，不是"超越 baseline"
- 这是一个**存在的、而非神话的**目标

### 10.4 重新校准的优先级

之前默认框架是"从 0.73 追到 0.95 = 要变得比 baseline 强 25%"——军备竞赛视角。

交叉对比后的正确框架是**补漏洞工程**：

| 优先级 | 动作 | 预期增量 | 依据 |
|---|---|---|---|
| P0 | 修 `low_possession` (22% → ≤5%) | +0.05~0.10 WR | baseline 不犯这错；信号指向 credit-assignment / 协调问题 |
| P1 | 修 `poor_conversion` (8% → ≤3%) | +0.02~0.05 WR | baseline 不犯这错；信号指向 shot / finishing 决策 |
| P2 | 改善 `late_defensive_collapse` (52% → 46%) | +0.02~0.05 WR | 让我们"持平" baseline；这是结构性问题，增量有上限 |
| P3 | 让我们的失败局拖到 ≥47 步 median | +0.01~0.03 WR | 修"可预测空当"；attack patterns 标准化 |

全部达成约能把 0.73 推到 **0.85~0.93**，基本进入 9/10 的可触达区间。

### 10.5 对 MAPPO 成功判据的直接影响

[snapshot-014-mappo-fair-ablation.md](snapshot-014-mappo-fair-ablation.md) 的 success criterion 不应该只定在 win rate。基于本节的交叉对比，建议 MAPPO 的机制判据写成：

- **主判据**: baseline 胜率 ≥ v2 同样本 +0.05（沿用 snapshot-010 口径）
- **机制判据 A（关键）**: `low_possession` 在失败桶里占比从 22% → ≤ 10%
  - 这是 centralized critic 修复 credit-assignment 病的最直接信号
  - 如果只有 WR 涨但 `low_possession` 没降 → MAPPO 只是表面赢球，没修根本协调问题
- **机制判据 B**: 失败局 median 步数从 32 → ≥ 40
  - 意味着我们"可预测空当"被压缩，baseline 不能再 32 步解决我们

如果 MAPPO 过主判据 + 机制判据 A，这条线就是**真正的突破**而非增量调参。

### 10.6 对 IL/BC pipeline 的校准

[snapshot-012-imitation-learning-bc-bootstrap.md](snapshot-012-imitation-learning-bc-bootstrap.md) 里 BC 的 teacher 就是 baseline。本节数据改变了一些假设：

- baseline 在 **中前期**表现稳（38.5 median 赢 random），BC 学这部分**有价值**
- baseline 在 **长局/后段** 会漂会崩（BvB 失败 median 49 步, tail_ball_x −9.3）——BC 学这部分**会继承 teacher 的缺陷**
- 因此 BC → RL fine-tune 的 **RL 阶段**需要专门提供长局、后段、深位这些 teacher 也处理不好的 state，用 RL 去超越 teacher 的短板

这意味着 BC warm-start 不是简单"先像 baseline"然后"再用 RL 硬训"，而是**需要有针对性的 curriculum**:

1. BC 阶段尽量广覆盖（让 agent 学到 baseline 的中前期行为）
2. RL fine-tune 阶段的训练分布**偏向长局 / 中后段 / 深位** —— 这些是 teacher 的弱点，也是我们赢 baseline 的唯一窗口

### 10.7 一个值得追加的对局设计

当前 weakness analysis 的第三组（5.3 `baseline vs current best agent`）还没跑。建议优先级**提前**到第一：

- 跑 `baseline (team0)` vs `v2@440 (team1)` 500 局，保存全部 episode
- 对比 `baseline 赢的那 366 局` 里 **baseline 自己是怎么赢的**
- 和 BvB 里 baseline 赢的那 255 局对比
- 如果 baseline 打我们时有**额外**的攻击 pattern（BvB 里不用但打我们时触发），那就是我们被 exploit 的具体空当
- 这些 pattern 直接指导：
  - MAPPO 是否真的修了它们
  - 将来 wrapper/ensemble 侧要补的防守 repertoire

## 11. MAPPO 数据对 §10 推论的反馈（2026-04-13 后补）

[SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md) §9 完成后，§10 里的几条关键推论得到了**直接验证或否决**。

### 11.1 §10.5 预声明的机制判据 A：**FAIL**

预声明原文（§10.5）：
> "机制判据 A（关键）: `low_possession` 在失败桶里占比从 22% → ≤ 10%——这是 centralized critic 修复 credit-assignment 病的最直接信号"

实测 500-ep 失败桶占比：

| Lane | low_possession % |
|---|---|
| v1 PPO | 24% |
| v2 PPO | 22% |
| MAPPO no-shape | **26%** |
| MAPPO v1 | 24% |
| MAPPO v2 | 24% |

**五条 lane 全在 22-26% 的极窄区间**，MAPPO no-shape 甚至高于 PPO。**credit-assignment 假设被明确否决。**

### 11.2 §10.5 机制判据 B：**PASS (但部分)**

预声明原文（§10.5）：
> "机制判据 B: 失败局 median 步数从 32 → ≥ 40"

实测：

| Lane | loss median steps |
|---|---|
| v2 PPO | 35 |
| MAPPO v2 | 34 |
| MAPPO no-shape | 34 |
| MAPPO v1 | 27 |

**MAPPO v1 反而比 PPO v1 更短**。"延迟失败"效应在 MAPPO 下不稳定——这说明 §10.5 当时把"`low_possession` 下降"和"失败延后"当成同一机制的表现是**错的**，二者彼此独立。

### 11.3 §10.1 的诊断被**强化**：`low_possession` 是"我们特有、所有干预都修不到"的 bug

§10.1 表格指出 BvB 的 `low_possession = 0%`，我们的 v2 PPO 为 22%。加入 MAPPO 三条 lane 后，这个对比变得**更坚固**：

| 对象 | low_possession % (of losses) |
|---|---|
| BvB (baseline 自对弈) | **0%** |
| v1 PPO | 24% |
| v2 PPO | 22% |
| MAPPO no-shape | 26% |
| MAPPO v1 | 24% |
| MAPPO v2 | 24% |

**跨五种干预（+/−shaping, +/−CC）的占比恒定在 22-26%。** 这不是 reward 设计、不是 critic 结构、不是 shaping 方向问题——是**更底层**的东西。三个可能的根因：

1. **obs 层面**：single_player 模式下 policy 看不到队友的具体位置和意图
2. **环境层面**：2v2 设置下单策略控双人时偶尔被两个 baseline 在空间上"压扁"
3. **训练分布层面**：低占球率的起始状态在 on-policy rollout 里罕见，policy 没学过怎么抢回球

无论哪个根因，**PPO/MAPPO/shaping 内部调参都修不到它**。

### 11.4 §10.6 的预判被**验证**：BC 获得明确的针对性判据

§10.6 原本就推测 BC 的价值不只是"先学会像 baseline"，而是"先吸收 teacher 的强项"。MAPPO 数据让这一条变得**可量化**：

- BC 的 teacher (baseline) 在 BvB 里 `low_possession = 0%`
- 我们所有 RL 路线（PPO/MAPPO/± shaping）都卡在 22-26%
- **BC warm-start 是目前唯一能直接把"不放弃抢球"行为传染给 student 的武器**

因此 [SNAPSHOT-015](snapshot-015-behavior-cloning-team-bootstrap.md) 的核心 success criterion 应该写成：

- **主判据**: BC→RL 路线的 500-ep 胜率 ≥ MAPPO+v2 的 0.786 + 0.03 = **0.816**
- **机制判据（必过）**: BC→RL 路线的 `low_possession` 占失败比 ≤ **10%**
  - 这是 §10.5 当时对 MAPPO 预期但未实现的效果
  - 若 BC→RL 也过不了这条，说明 22-26% 的 `low_possession` 不是 teacher 的行为问题，而是环境/obs 的结构问题，需要下一步改 obs

### 11.5 §10.4 的优先级表修订

§10.4 原表里 P0 是"修 low_possession"，推算收益 +0.05~0.10 WR。**MAPPO 数据证明 reward/critic 路线修不到**，但也**没有否定"这个桶值得修"**——只是**修法要换**。修订后的 P0-P3：

| 原优先级 | 修订后 |
|---|---|
| P0 修 low_possession via shaping/CC | **P0 修 low_possession via BC / obs 改造** |
| P1 修 poor_conversion | P1 不变（MAPPO v2 已经降到 8%，接近 baseline 水平） |
| P2 改善 late_collapse | **P2' late_collapse 绝对数已降 23%，不用专门治** |
| P3 压失败局时长 | P3 取消（MAPPO 证明这不是独立维度） |

**新的主战场是 BC/DAgger 路线能否修 `low_possession`。** 这是 §11.3 诊断落地后的直接推论。

## 12. BC→MAPPO 数据对 §11 的进一步强化（2026-04-15 后补）

[SNAPSHOT-017 §11](snapshot-017-bc-to-mappo-bootstrap.md#11-failure-bucket-深度分析与判据-verdict) 完成 BC→MAPPO 首轮 + 4-点 500-ep failure capture 后，§11 预声明的 P0 方向（"BC 修 low_possession"）**被实证否决**。

### 12.1 9-lane low_possession 不变量现在有完整证据

| Lane | intervention | `low_poss` % of losses |
|---|---|---|
| v1 PPO @430 | shaping v1 | 23.6% |
| v2 PPO @440 | shaping v2 | 23.6% |
| v4 PPO @400 | shaping v4 (survival + anti-rush) | 28.1% |
| MAPPO+v2 @470 | MAPPO + shaping v2 | 24.0% |
| Pool 018 @290 | MAPPO + shaping v2 + 4-成员 pool | 26.5% |
| **BC @1870** | BC warm-start → MAPPO + shaping v2 | **23.3%** |
| **BC @2100** | 同上 (SOTA) | **26.7%** |
| **BC @2170** | 同上 | **23.9%** |
| **BC @2250** | 同上 | **22.6%** |

**9 条 lane 占比全在 22.6% - 28.1% 区间**，跨越：

- 2 种 critic 结构（vanilla / centralized）
- 5 种 shaping 配置（v1/v2/v3/v4/no-shape）
- 2 种训练对手（100% baseline / opponent pool）
- 2 种初始化（scratch / BC teacher warm-start）
- 多种训练长度（300 → 2335 iter）

**没有任何一种 intervention 把这个数字推出 22-28% 带**。

### 12.2 §11.3 根因诊断被更新

§11.3 当时列了三个候选根因：

| 候选 | BC 数据后的状态 |
|---|---|
| (1) **obs 层面**：policy 看不到足够的队友/时间信息 | ✅ **强化到最高可信度**——BC 见过 teacher 所有轨迹（包括 baseline 在 low_poss state 的正确行为）仍然修不到 |
| (2) 环境层面：2v2 下单策略控双人被压扁 | 可能但不是主因（Pool 018 / BC 都是 multiagent_player，不是 single_player 模式，依然卡在 22-28%） |
| (3) **训练分布层面**：低占球起始罕见 | ❌ **被 BC 否决**——BC 数据分布包含 teacher 在低占球 state 的全部决策 |
| (4) **reward 层面**：两个 team0 agent 长期收到对称信号 | ✅ **证据被加强**——即使去掉 dense shaping，保留的 sparse goal reward 仍然是 team-symmetric；而现有代码又已确认支持 per-agent override 但历史 lane 从未使用 |

**更新后的结论**：根因 **(1) obs 表征瓶颈** 仍是最高可信度假设，但 **(4) reward 对称性** 的证据并没有被 BC 否掉，反而因代码审计而被加强。更准确的说法是：**A（obs）更像主因，B（reward symmetry）仍是并行可检验的强候选，而不是被排除的备选。**

关键逻辑：BC 在训练时看到 teacher 在低占球 state 如何行动；如果 student 有足够的 obs 信息来区分这些 state 并模仿 teacher 的反应，low_poss 就该降。**它没降**说明 student 的 obs space 中**缺失**让 teacher 能正确反应的信号——很可能是**队友位置/速度/意图**，也可能包括**比赛时间上下文**。

这意味着：**BC 有一个结构性局限——只能传递 teacher 能从 student 的 obs space 里推断出的行为**。如果 teacher 的决策依赖 obs 里缺失的信息，BC 无法传递那部分能力。与此同时，若 reward 本身对两个 agent 长期保持对称，student 即使学到局部 teacher 行为，也可能继续收敛回同构 policy。

### 12.3 §11.5 P0 修订 v2

原修订 P0：**修 low_possession via BC / obs 改造**

再次修订（2026-04-15）：**修 low_possession via obs 改造 + reward 对称性检验**——BC 已经实证不够，而 code audit 进一步加强了 reward-symmetry 这条证据链。

| 当前地位 | 修订后 P0 |
|---|---|
| BC 路线 | 已完成；证明"teacher behavior 传递"这条路径**失败** |
| **obs 改造** | 直接武器；必须从 env wrapper 层动 |
| **reward 对称性检验** | 并行直接武器；无需大改核心代码，已可通过 per-agent override 验证 |
| **PSRO / league** | 间接武器，不直接修 low_poss，但可能通过 robustness 提升整体 WR |

**下一 snapshot 候选**（如果项目时间允许）：
- **obs 空间扩展**——把 teammate 的 position / velocity，再加 1 维 normalized episode time，拼进每个 agent 的 obs。代价：
- env wrapper 改造（~0.5-1 天）
- warm-start 不可用（obs 维度变了）→ from-scratch 重训（~8h GPU）
- 不确定回报（如果 obs 假设错，白花一天）
- **reward 对称性检验**——使用现有 per-agent shaping override 做 striker/defender 分化。代价显著更低，但若 shared actor 无显式 role cue，负结果解释要更谨慎。

### 12.4 对 0.95 目标的重新评估

- 9-lane 天花板 ≈ **0.84**（BC @2100 = 0.842 是当前最强点）
- 9/10 稳过门槛 ≈ 真实 WR **0.95**
- 剩余 gap：**0.108**
- 单 low_possession 桶占所有失败的 25%，占 500 eps 的 ~4.5%——**完全修好可提升 ~0.045 WR**
- 即使 low_poss 降到 10%，从 9-lane 普适天花板外推**最多到 0.88-0.89**，仍不到 0.95

**坦率判断**：**0.95 在当前 obs + arch 框架下大概率不可达**。项目策略应该调整为：
- 冲 **0.85-0.88** 而不是 0.95
- Performance 部分分（15-20/25 pts for baseline, 25/25 for random）
- 其他分项（Modification 40 + Novel +5 + Report 100）打满
- 整体分数目标 ~75-85 / 100 instead of chasing 90+
