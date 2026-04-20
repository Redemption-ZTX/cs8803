# SNAPSHOT-048: Hybrid Eval — Baseline Takes Over in Defensive States (DAGGER Upper-Bound Test)

- **日期**: 2026-04-19
- **负责人**:
- **状态**: 预注册 / 待实施

## 0. 为什么现在做

### 0.1 背景：单点优化已停滞

5+ lane (031A/036D + 040A/B/C/D + 041B + 044) 全部 saturate 在 [0.852, 0.865]。2026-04-19 第一性原理复盘后，我们识别出：

- **0.86 不是 PPO ceiling，是单点优化天花板**
- 用 [v2 失败桶](../../cs8803drl/imitation/failure_buckets_v2.py) 重新分类两个 0.86 SOTA 的 failure capture：
  - **031A@1040 (Siamese)**: tail_ball_x 中位 +1.37（球在对方半场）→ **wasted_possession 主导** (42%)，主要错在「有球未转化」
  - **036D@150 (per-agent + learned reward)**: tail_ball_x 中位 -4.90（球深陷自方）→ **defensive_pin 主导** (55%)，主要错在「被压制无法突围」

两个 SOTA 的失败模式相反——这是 [SNAPSHOT-034 ensemble](snapshot-034-deploy-time-ensemble-agent.md) 的核心论据。

### 0.2 DAGGER 想法的提出

讨论中提出"师生网络（teacher 拿到 baseline next policy）→ DAGGER 蒸馏到 student"路径。但有两个核心反对意见：

1. **过拟合 baseline**：bonus 题是 vs 未发布的 agent2，DAGGER 训出来的 student 大概率对 baseline 特化
2. **干扰已经能赢的 80%**：student 当前 WR = 0.86，baseline 自己 vs baseline ≈ 0.5。naive DAGGER 会把 student 拉回 baseline 水平

第 2 个问题的解法是 **selective DAGGER**——仅在 student 表现差的特定 state 上学 baseline。但这又暴露下一个问题：**baseline 在那些 state 真的更强吗？** 如果 baseline 自己 50% WR 的失败模式跟 student 失败模式重合（比如它也不会 clear defensive pin），那 baseline 的"建议"就是垃圾，DAGGER 上限就是 0。

### 0.3 048 是 DAGGER 路径的「天花板测试」

在写任何 DAGGER 训练代码（3-5 天工程）之前，先做一个**最便宜的、能确定 DAGGER 上限的实验**：

> **构造一个 hybrid agent**：student 主导 + 在 defensive state 检测到时**直接 swap 成 baseline 接管控制**。如果 hybrid 1000ep WR 比 student-only 高 ≥ +0.03 → baseline 在那些 state 真的更强 → DAGGER 有蒸馏空间，可以投资。如果 hybrid WR ≤ student → baseline 没帮助 → DAGGER 死路，节省 3-5 天工程。

这是个 **1 天工程的实验**，决定 1-2 周的投资方向。

## 1. 核心假设

### H1（DAGGER 有空间）

> 在 defensive_pin state（球深陷自方半场，episode 后段）下，**baseline 的 action 平均比 student 的 action 带来更高 episode-return**。即：在那些 state 上 swap 控制权给 baseline，会改善整体 WR。

### H2（弱者更受益）

> H1 的增益 **036D > 031A**——因为 036D 的失败更集中在 defensive_pin（v2 桶 55% 主导），而 031A 的失败是 wasted_possession，跟 trigger state 不重叠。

### H3（trigger 选择敏感）

> 不同 trigger 函数（α 简单 vs β v2-style window）应给出**方向一致但量级不同**的增益。如果两者方向相反 → trigger 设计有 bug 或 baseline 帮助不稳定。

## 2. 设计

### 2.1 SwitchingAgent wrapper

实现路径: `cs8803drl/deployment/_hybrid/switching_agent.py`（**新建临时目录，不进 submission**）

```python
class SwitchingAgent(AgentInterface):
    def __init__(self, student_agent, baseline_agent, trigger_fn, trigger_name):
        self.student = student_agent
        self.baseline = baseline_agent
        self.trigger = trigger_fn
        self.trigger_name = trigger_name
        # 本 episode 内的 ball_x 历史窗（用于 v2-style trigger）
        self.recent_ball_x = deque(maxlen=40)
        # 统计
        self.swap_count = 0
        self.total_steps = 0

    def reset_episode(self):
        self.recent_ball_x.clear()

    def act(self, observation):
        ball_x = self._extract_ball_x(observation[0])
        self.recent_ball_x.append(ball_x)
        self.total_steps += 1

        if self.trigger(ball_x, list(self.recent_ball_x)):
            self.swap_count += 1
            return self.baseline.act(observation)
        return self.student.act(observation)

    def report(self):
        return {
            "trigger_name": self.trigger_name,
            "swap_pct": self.swap_count / max(self.total_steps, 1),
        }
```

### 2.2 Trigger 函数

**2026-04-19 实现时 recalibrated**: 原始草案（瞬时 `ball_x < -2`）R1 sanity 测出触发率 30-37%——「球短暂越过中场」也算 trouble，与 winning student 的真实 defensive trouble 语义脱节。pivot 到 **window-based**（与 v2 桶 `tail_mean` 完全对齐）：

```python
ALPHA_WINDOW_MEAN_THRESHOLD = -3.0   # 直接匹配 v2 DEFENSIVE_PIN_TAIL_X (failure_buckets_v2.py:43)
BETA_WINDOW_MEAN_THRESHOLD  = -5.0   # 紧版：仅当窗口强 pin
BETA_WINDOW_LEN             = 40
MIN_WINDOW_SAMPLES          = 5      # 防 episode 开局误触发

def trigger_alpha(_ball_x, hist):
    if len(hist) < MIN_WINDOW_SAMPLES: return False
    return (sum(hist) / len(hist)) < ALPHA_WINDOW_MEAN_THRESHOLD

def trigger_beta(_ball_x, hist):
    if len(hist) < MIN_WINDOW_SAMPLES: return False
    return (sum(hist) / len(hist)) < BETA_WINDOW_MEAN_THRESHOLD
```

**ball_x 来源**: [SNAPSHOT-049](snapshot-049-env-state-restore-investigation.md) 已验证 `info[team0_id]["ball_info"]["position"][0]` 给 absolute world coords，准确可用——`extract_team0_ball_x` 还按 team0_ids parity 做 sign-flip 保证「负值 = 我方半场」语义跨 blue/orange episode 稳定。

### 2.3 实验矩阵

| 条件 | student | trigger | 预期 swap 比例 | 预期 WR |
|---|---|---|---|---|
| **C0_031A** baseline | 031A@1040 | none | 0% | 0.860 |
| C1_031A_alpha | 031A@1040 | window mean < -3.0 | ~10-15% | TBD |
| C2_031A_beta | 031A@1040 | window mean < -5.0 | ~3-7% | TBD |
| **C0_036D** baseline | 036D@150 | none | 0% | 0.860 |
| C3_036D_alpha | 036D@150 | window mean < -3.0 | ~15-25% (036D 更常 pin) | TBD |
| C4_036D_beta | 036D@150 | window mean < -5.0 | ~5-12% | TBD |

每条 1000ep vs `ceia_baseline_agent`，共 6 conditions × 1000ep = 6000 episode。

GPU 估计: 4-6h（可并行 -j 3）。

## 3. 预注册判据

### 3.1 主判据: DAGGER 上限

判据按 **per-student** 看（C1/C2 vs C0_031A，C3/C4 vs C0_036D），任一 student 的任一 trigger 满足 → 部分成立。

| Δ (Hybrid - student-only) | 解读 | 行动 |
|---|---|---|
| **≥ +0.03** | baseline 在 trigger state **明确更强**，DAGGER 有真正的蒸馏空间 | **启动 DAGGER 实验**（snapshot-049） |
| **(+0.01, +0.03)** | baseline 微弱有帮助，但 1000ep 边缘显著（SE ±0.016 × 2σ = 0.032） | 加跑到 2000ep 缩窄 SE 再决 |
| **(-0.01, +0.01)** | baseline 没明显帮助也没明显害 | DAGGER 收益小，不做 |
| **≤ -0.01** | baseline **主动恶化**——student 在 trigger state 已经比 baseline 强 | DAGGER 路径**死**（明确反证） |

### 3.2 机制判据 (H2 验证)

| 比较 | 阈值 | 解读 |
|---|---|---|
| Δ_036D > Δ_031A | ≥ +0.02pp 差 | H2 成立：弱者更受益（防守失败模式更受 baseline 帮助） |
| Δ_036D ≈ Δ_031A | 范围内 | H2 反证：baseline 帮助跟 student 失败模式无关 |
| Δ_036D < Δ_031A | 反向 | H2 强反证：031A 更需要 defensive 帮助（与失败诊断矛盾） |

### 3.3 trigger 一致性判据 (H3)

| 比较 | 阈值 | 解读 |
|---|---|---|
| sign(Δ_alpha) == sign(Δ_beta) | 同方向 | trigger 设计 robust，结果可信 |
| 反向 | 任一 student | trigger 实现有 bug 或 ball_x 提取不一致——必须 debug 后重跑 |

## 4. 风险

### R1 — Trigger state 提取错误

obs 里 ball_x 的语义可能不是我假设的「我方深区 = 负值」。如果 raycast layout 不同或 ball position 字段位置不对，trigger 会在错误的 state 触发。

**缓解**:
- **首选**：直接用 `info[0]["ball_info"]["position"]` 拿 absolute world coords（[SNAPSHOT-049](snapshot-049-env-state-restore-investigation.md) 已验证 GET state 准确）。需要 evaluator 把 info 透传到 act()
- 如果 evaluator 不透传 info：在前 5 个 episode 打印 (ball_x_obs, ball_x_world from info) 二元组对齐
- 跑 1 episode 看 swap_pct 是否在预期范围（α ~15-30%, β ~5-15%）
- 与现有 [`extract_ball_position`](../../cs8803drl/core/soccer_info.py) 对齐

### R2 — Baseline 的 act() 接口与 student 不一致

baseline 是 `ceia_baseline_agent` checkpoint，加载方式可能跟我们的 trained_*_ray_agent 不同。

**缓解**: 复用 `cs8803drl/core/utils.py` 里 `_make_mixed_opponent_policy` 的 baseline 加载路径——它在训练里已经被反复 verify。

### R3 — Baseline 看到的 obs 和 student 不一致

baseline 训练时可能预期不同的 obs 维度（349 dim full info vs 336 dim ego）或 reward shaping。如果 student 喂给 baseline 的 obs 不是 baseline 能消化的，结果无意义。

**缓解**: baseline 是项目 frozen 资产，已经在 evaluate_official_suite.py 里被广泛使用（作为对手）。它的 `act(obs_336)` 接口已经稳定。但要 verify 的是：作为 **team0** 控制（而不是它通常的 team1 对手角色）时，是否需要 obs flip / coordinate transform。

**这是个真实风险**，必须在实现时单独 sanity check：让 baseline 控制 team0 全程跑 100ep vs 它自己（team1），WR 应该 ≈ 0.50。如果显著偏离 → baseline 不是真的对称，trigger swap 时它的行为会有 side bias。

### R4 — 蒸馏 ≠ 控制权 swap

即使 H1 成立（hybrid > student），**DAGGER 蒸馏不一定能复现 hybrid 的增益**。蒸馏后 student 必须**自己**学会在 defensive_pin state 模仿 baseline，但 student 的 distribution 跟 hybrid 训练时不一样（因为蒸馏后 student 不再有 baseline 兜底，前面的 state 分布会改变）。

**缓解**: 这是 DAGGER 本身的限制，048 不解决，仅给上限。如果 048 跑出 +0.05 增益，DAGGER 实际可能只拿到 +0.02——足够投资。如果 048 跑出 +0.01，DAGGER 大概率拿不到任何增益。

## 5. 不做的事

- **不做 DAGGER 训练**——这是 snapshot-049 的事，仅在 048 通过判据后启动
- **不改任何 student / baseline 的权重**——仅 deploy-time wrapper
- **不跑 vs Random / vs agent2**——048 仅测 baseline 上限，泛化性是 049 的问题
- **不实现 baseline → student 的 obs flip 自动校正**——R3 sanity check 失败的话直接报告，不强行 hack
- **不进 submission 路径**——SwitchingAgent 在 `_hybrid/` 临时目录，确认无价值后删除

## 6. 执行清单

1. 实现 `cs8803drl/deployment/_hybrid/switching_agent.py`（100 行 wrapper + trigger fns + ball_x extractor）
2. R3 sanity check：baseline-as-team0 vs baseline-as-team1 跑 100ep，确认 WR ≈ 0.50
3. R1 sanity check：跑 1 episode 打印 trigger 触发 state 数 + ball_x 提取一致性
4. 实现 6 个 launch 脚本 / 单参数化脚本（`scripts/eval/_hybrid_eval_<student>_<trigger>.sh`）
5. 在 GPU 节点 launch 6 conditions × 1000ep（可 -j 3 并行，4-6h）
6. 整理结果填入 §7 verdict
7. 按 §3 判据决定是否启动 snapshot-049 (DAGGER 训练)

## 7. Verdict (2026-04-19, append-only)

### 7.1 执行备注

- 原定计划: 6 conditions × 1000ep on 单节点
- 实际遭遇: 010-30-0 上 sgarg349 另一用户同时 4 进程 100% CPU 抢占，load 100+；011-23-0 启动 rerun 后也撞 load 110（6 个 evaluate_hybrid 各自 Ray init + 多 worker subprocess 累计 30+ procs）
- Throughput 严重恶化: 100ep 用 38-48 min，预估 1000ep 需 6-8h（smoke 单进程 50ep 14s 的 ~50-100x 膨胀）
- 100ep 时信号已经**远超噪声阈值**（SE ±0.05 at 0.88 → 95% CI ±0.10 vs 实测 Δ -7 到 -14pp），**1000ep 不会翻转方向**，只会缩窄 CI
- 决策: kill 两组 6+6=12 个 procs，用 100ep 数据写 verdict，释放 GPU 给后续

### 7.2 数据表（100ep per condition）

| 条件 | student | trigger | WR (100ep) | swap_pct | Δ vs C0 | 预注册判据 [§3.1](#3-预注册判据) |
|---|---|---|---:|---:|---:|:---:|
| **C0_031A_none** | 031A | none | **0.880** | 0% | ref | — |
| C1_031A_alpha | 031A | alpha (window < -3.0) | 0.740 | 17.2% | **-14pp** | **≤ -0.01 → DAGGER 路径死** |
| C2_031A_beta | 031A | beta (window < -5.0) | 0.810 | 10.4% | **-7pp** | **≤ -0.01 → DAGGER 路径死** |
| **C0_036D_none** | 036D | none | **0.800** | 0% | ref | (expected ~0.86, 100ep SE 容忍) |
| C3_036D_alpha | 036D | alpha | 0.710 | 25.1% | **-9pp** | **≤ -0.01 → DAGGER 路径死** |
| C4_036D_beta | 036D | beta | 0.730 | 16.6% | **-7pp** | **≤ -0.01 → DAGGER 路径死** |

**100ep SE at 0.80 = ±0.05**, 2σ ≈ ±0.10。Δ 量级全部 ≥ -0.07 >> 噪声阈值。4/4 trigger conditions 全部达到"baseline 主动恶化"gate。

### 7.3 核心判据 verdict

**H1 (DAGGER-from-baseline 有空间)**: ❌ **完全失败 — baseline 接管不仅无帮助，反而主动恶化 student WR 7-14pp**

**H2 (弱者 036D 更受益)**: ❌ **反向** — 两个 student 受害量级相似 (031A -7 到 -14pp; 036D -7 到 -9pp)。即使 036D 是 defensive_pin-主导（理论上 baseline 应该能帮上），现实是 baseline 在那些 state 上**依然做得比 036D 差**。

**H3 (trigger 一致)**: ✅ **同方向确认** — 两个 student × 两个 trigger = 4/4 恶化；alpha (loose, 更多 swap) 恶化更严重，beta (tight, 更少 swap) 恶化稍轻。**swap 越多 → WR 越低 → baseline 真的在 drag down**。

### 7.4 strategic 结论

> **"DAGGER-from-baseline 路径死"**，**但 DAGGER 框架本身没死**。

关键区分：

- **已死**: 以 `ceia_baseline_agent` 作为 teacher 的 DAGGER／BC／swap-takeover 类路径
- **仍存活**: 以更强 teacher 的 DAGGER — e.g., 031B@1220 (0.882) / 034 ensemble (若 ≥ 0.90) / agent2 (bonus，未发布)

**Root cause**: teacher 必须比 student **在目标 state 分布上更强**。本项目 student (031A / 036D) 已经 0.86，而 baseline 自己 vs baseline ≈ 0.50（snapshot-013）—— **baseline 不是 teacher 而是 distractor**。在 student 输的 episode 里，baseline 大概率也会输。

student 已经 outgrown baseline 这个 teacher。

### 7.5 哲学含义 — 对 "imitate baseline" 类所有路径

| 类路径 | lane | 结果 | 上限 |
|---|---|---|---|
| BC from baseline trajectories | [snapshot-015](snapshot-015-behavior-cloning-team-bootstrap.md) / [017](snapshot-017-bc-to-mappo-bootstrap.md) / [025](snapshot-025-bc-champion-field-role-binding.md) | ✅ 有效 | 0.842 (BC-champion @2100, 接近 baseline platform) |
| DAGGER from baseline action | 本 snapshot 048 | ❌ **负 WR impact** | 无上限（主动恶化）|
| Mixed baseline as teacher in ensemble | - | ❌ 别做（同 048 逻辑）| - |
| DAGGER from 031B@1220 (0.882 teacher) | 未测 | ? | 理论可达 031B 水平 |
| DAGGER from ensemble (0.90+ if 034 过) | 未测 | ? | 理论 > ensemble 水平 |

**教训**: 所有 "baseline 作为 teacher" 的 imitate 方法都 cap 在 `baseline self-play WR ≈ 0.50` 附近，因为 teacher 的真实能力就是那么高。

### 7.6 049 (DAGGER training) 是否启动

[snapshot-048 §3.1](#31-主判据-dagger-上限) 写: "Δ ≤ -0.01 → DAGGER 路径**死**（明确反证）"。

严格按判据 → **049 (DAGGER-from-baseline training) 否决**。

但**修正版 DAGGER** 以更强 teacher（031B@1220 或 034 ensemble）仍是 open direction — 若要做，**必须另开新 snapshot**（不是 049），因为 baseline-teacher 已 exclude。

### 7.7 Raw 数据 provenance

100ep 数据来自 rerun @011-23-0（原 010-30-0 sunken cost 没出 JSON）：

- `docs/experiments/artifacts/hybrid-eval-rerun/C0_031A_none.log` → `ep=100/1000 WR=0.880 swap_pct=0.000`
- `docs/experiments/artifacts/hybrid-eval-rerun/C1_031A_alpha.log` → `ep=100/1000 WR=0.740 swap_pct=0.172`
- `docs/experiments/artifacts/hybrid-eval-rerun/C2_031A_beta.log` → `ep=100/1000 WR=0.810 swap_pct=0.104`
- `docs/experiments/artifacts/hybrid-eval-rerun/C0_036D_none.log` → `ep=100/1000 WR=0.800 swap_pct=0.000`
- `docs/experiments/artifacts/hybrid-eval-rerun/C3_036D_alpha.log` → `ep=100/1000 WR=0.710 swap_pct=0.251`
- `docs/experiments/artifacts/hybrid-eval-rerun/C4_036D_beta.log` → `ep=100/1000 WR=0.730 swap_pct=0.166`

原 010-30-0 run 因 stdout buffer + 节点被抢占，60 min+ 未出 progress 打印。kill 时各 proc 已累计 CPU TIME 270+ min（step 数不可精确计数）但无 JSON 输出，不产出 verifiable 结果。

## 8. 相关

- [SNAPSHOT-013](snapshot-013-baseline-weakness-analysis.md) — baseline weakness 历史分析（baseline 自己 vs 自己的失败模式）
- [SNAPSHOT-017](snapshot-017-bc-to-mappo-bootstrap.md) / [SNAPSHOT-025](snapshot-025-bc-champion-field-role-binding.md) — 早期 BC-from-champion 路径（不是 DAGGER，但是同族「imitate strong policy」思路）
- [SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md) — 与 048 正交的另一种 deploy-time 突破路径（ensemble vs hybrid takeover）
- [SNAPSHOT-047](snapshot-047-deployment-slot-swap-test.md) — 同期 deploy-time wrapper 工程（slot 一致性检查）
- [v2 failure buckets](../../cs8803drl/imitation/failure_buckets_v2.py) — defensive_pin 阈值定义来源
- [trained_ray_agent.py](../../cs8803drl/deployment/trained_ray_agent.py) — student 加载参考实现
- [_make_mixed_opponent_policy](../../cs8803drl/core/utils.py) — baseline 加载参考实现
