# Model Ranking & H2H Registry

- **用途**: 项目所有已训练 policy 的 **单一真相源** (single source of truth) — baseline WR / random WR / peer H2H / 排名。
- **规则**: 每次做完 official 500+ eval 或 H2H，必须更新本文件。不允许在 snapshot 里写数字而不同步到这里。
- **最后更新**: 2026-04-19

---

## 0. 读法与写法（**必读**）

### 0.1 读 H2H log 的陷阱（2026-04-18 血的教训）

`docs/experiments/artifacts/official-evals/headtohead/*.log` 中 `policies:` 下**按模块名字母序排**，**不按文件名顺序**。

跨架构 H2H 里:
- `trained_shared_cc_opponent_agent` (**per-agent**，字母序在前) 先打印
- `trained_team_ray_agent` (**team-level**，字母序在后) 后打印

所以 `030D_vs_029B.log` 的**第一段**其实是 **029B** 的成绩，不是 030D。曾经把**所有**跨架构 H2H 读反过 → 见 [SNAPSHOT-037 RETRACTED](snapshot-037-architecture-dominance-peer-play.md)。

### 0.2 正确读 H2H 的办法

```bash
# 旧 log（只看 raw evaluator 输出）
grep -E "^  cs8803drl.deployment|^    policy_win_rate:|^    policy_wins:" <file>.log

# 新 log（如果是 scripts/eval/evaluate_headtohead.py 生成）
grep -E "^team0_overall_|^team0_blue_|^team0_orange_|^team0_side_gap|^reading_note:" <file>.log
```

旧 log 中每组 `cs8803drl.deployment.XXX` 下紧跟其**自己的**胜率。按**模块名**查，不按"第一段"查。
新 log 中优先读 `---- H2H Recap ----` 里的 `team0_overall_*`；`blue/orange_*` 只是分边诊断。

### 0.3 更新本文件的流程

新做了一次 H2H / official eval:

1. 把结果加到对应表（§3 baseline / §4 random / §5 peer H2H）
2. 一定**标明样本大小 n** 和**置信区间或 z / p**
3. 如果新结果和旧结果冲突（同 checkpoint 两次 500ep 差 > 3pp），保留两个结果，不要改旧的
4. 更新 §6 per-criterion rankings 如果有变动
5. 更新顶部"最后更新"日期
6. 在 §8 Changelog 加一行

---

## 1. Model Registry

表头说明:
- **Architecture**: per-agent (336-dim obs, one policy per agent) / team-level (672-dim joint obs, one policy outputting both actions)
- **Status**: 🟢 active (近期在用) / 🟡 reference (保留 baseline 用) / 🔴 retired

| Model ID | Architecture | Warmstart 源 | 关键 shaping | Status | 笔记 |
|---|---|---|---|---|---|
| `017@2100` | per-agent | BC+v2 scratch | v2 | 🟡 reference | 早期 BC 冠军 |
| `025b@080` | per-agent | 017 → field-role binding | v2 + field-role | 🟢 active | per-agent top-tier；直连 H2H 对 029B 仍接近 |
| `028A@1060` | team-level | scratch BC+v2 | v2 | 🟢 active | team-level raw warmstart |
| `028A@1220` | team-level | scratch BC+v2 | v2 | 🟡 reference | 早期峰值，后来不稳 |
| `029B@190` | per-agent | 029A B-warm handoff | v2 + PBRS handoff | 🟢 active | per-agent official 最强点；直连 H2H 对 `025b` 仍未拉开 |
| `030A@360` | team-level | 028A@1060 + field-role | v2 + field-role | 🟢 active | field-role port 到 team-level；当前更稳 |
| `030D@320` | team-level | 028A@1060 + PBRS | v2 + PBRS goal-prox | 🟢 active | PBRS port 到 team-level；当前更高 ceiling |
| `030D@360` | team-level | 028A@1060 + PBRS | v2 + PBRS | 🟡 reference | 030D 另一候选，略弱于 @320 |
| `032A@170` | team-level | 028A@1060 + aux loss λ=0.05 | v2 + team-action aux | 🟡 reference | aux loss 实验 |
| `032Ac@200` | team-level | 028A@1060, aux λ=0.0 | v2 (aux 关) | 🟡 reference | 032 control |
| `032nextC@130` | team-level | 028A@1060, symmetric aux λ=0.0 | v2 (symmetric aux path, aux 关) | 🟡 reference | 032-next control；当前 baseline `1000ep` 最好点 |
| `032nextA@110` | team-level | 028A@1060 + symmetric aux | v2 + symmetric team-action aux | 🟡 reference | 032-next aux 主线 |
| `033A@80` | team-level | 028A@1060 + team PBRS (spacing+coverage) | v2 + team PBRS | 🟡 reference | 早期高点 |
| `033A@130` | team-level | 028A@1060 + team PBRS | v2 + team PBRS | 🟡 reference | 另一 candidate |
| `036C@150` | per-agent | 029B@190 + learned reward | v2 + learned W/L reward | 🟢 active | 当前 baseline-oriented 主候选；训练期有 `inf`，需先修数值稳定性 |
| `036C@270` | per-agent | 029B@190 + learned reward | v2 + learned W/L reward | 🟡 reference | official `1000ep` 最佳点；peer H2H 参考点 |
| `036D@150` | per-agent | 029B@190 + learned reward (stable) | v2 + λ=0.003 + warmup_steps=10000 + finite-check | 🟢 active | 036D 1000ep 最强点 (0.860)，1000ep H2H vs 029B = 0.507 (tied); failure capture: wasted_poss -9.2pp vs 029B, no gaming, L mean 26 步 |
| `036D@250` | per-agent | 同 @150 | 同 @150 | 🟢 active | 036D 第二高 (0.856) |
| `036D@130` | per-agent | 同 @150 | 同 @150 | 🟡 reference | 036D 第三高 (0.848) |
| `031A@1040` | team-level | scratch BC+v2 + Siamese dual encoder | v2 only | 🟢 active | **项目新 SOTA 候选**：1000ep avg 0.860 (2000 game) AND H2H vs 029B = 0.552 (z=3.29, p<0.001) **首个双判据击败 029B** |
| `031A@1170` | team-level | 同 @1040 | 同 @1040 | 🟢 active | 031A second peak (1000ep avg 0.858); 待 H2H |
| `031A@1000` | team-level | 同 @1040 | 同 @1040 | 🟡 reference | 031A late window peak (0.852) |
| `031B@1220` | team-level | scratch BC+v2 + cross-attention | v2 only | 🟢 active | 当前 strongest single model；official `1000ep = 0.882`，H2H 已显著胜 `029B / 025b / 036D` |
| `043A'@080` | team-level | `031B@1220` warm-start | v2 + diversity curriculum (`baseline50 + 031A20 + 029B15 + 036D15`) | 🟡 reference | `043` 第一代 curriculum 主候选；official `1000ep = 0.901`，但 peer 轴只小优 `031B` 且输 `034E` |
| `043B'@440` | team-level | `043A'@080` resume | v2 + harder diversity curriculum (`baseline40 + 031B20 + 029B20 + 036D20`) | 🟢 active | `043` 双主候选之一；official `1000ep = 0.904` 是当前最高 baseline 尖峰，H2H `vs 031B = 0.600`、`vs 034E = 0.554`，但 `capture 500 = 0.852` 说明仍偏脆 |
| `043C'@480` | team-level | `043A'@080` resume | v2 + heavier diversity curriculum (`baseline35 + 031B25 + 029B20 + 036D20`) | 🟢 active | `043` 双主候选之一；official `1000ep = 0.895` 略低于 `043B`，但 `capture 500 = 0.880`、H2H `vs 031B = 0.614`、`vs 034E = 0.576`，且 direct H2H 对 `043B` 为 `0.532`（未显著），当前更像更 rounded 的 overall variant |
| `034C-frontier` | mixed ensemble | `031A@1040 + 036D@150 + 029B@190` | deploy-time probability averaging | 🟢 active | 第一代 frontier ensemble；peer 轴很强但 baseline `1000ep = 0.843` 回落明显 |
| `034E-frontier` | mixed ensemble | `031B@1220 + 036D@150 + 029B@190` | deploy-time probability averaging | 🟢 active | 当前 ensemble 主候选；official `500/1000 = 0.904 / 0.890`，H2H 胜 `031B / 029B / 034C` |
| `038A@160` | team-level | 028A@1060 + depenalized v2 | ball_progress + possession | 🟢 active | 038 Stage 1 @ lane A 1000ep 最好点 |
| `038B@110` | team-level | 028A@1060 + goal-prox PBRS | v2 + goal-prox PBRS | 🟢 active | 038 Stage 1 @ lane B 1000ep 最好点 |
| `038C@50` | team-level | 028A@1060 + event lane | v2 + shot/tackle/clearance events | 🟢 active | 038 Stage 1 @ lane C 1000ep 最好点 |
| `038D@40` | team-level | 028A@1060 + v2 + entropy=0.01 | v2 + entropy coeff 提升 | 🟢 active | 038 Stage 1 最强点 (1000ep 0.806)；待 failure capture |
| `039@230` | per-agent | 029B@190 + (intended) adaptive learned reward | v2 + λ=0.003 + warmup_steps=10000 + (broken) refresh callback | 🟢 active | 039 1000ep 最强点 (0.843)；refresh 因 wrapper-discovery bug 未生效，等价 036D-style 稳定化 |

---

## 2. 术语 / 缩写

| 术语 | 含义 |
|---|---|
| `official 500` | 单次 official evaluator 跑 500 episode 的 baseline WR |
| `official 1000` | 单次 official evaluator 跑 1000 episode 的 baseline WR |
| `capture 500` | 500 episode failure capture；稳定性代理，不是 official |
| `combined 1000` | 两次独立 `500ep` 结果按总局数合并后的读法；常用于 H2H rerun，不等于 `official 1000` |
| `SE` | Standard error. 500ep ≈ 0.017；1000ep ≈ 0.012 |
| `95% CI` | ±1.96 × SE ≈ ±3.3pp (500ep) / ±2.4pp (1000ep) |
| `z` | z-score 对 50% null (H2H) 或 对某参考值 |
| `p` | p-value (one-tailed unless noted) |
| `*` | p < 0.05；`**` p < 0.01；`***` p < 0.001 |

---

## 3. vs `ceia_baseline_agent` (assignment grade criterion)

**assignment 目标 = 9/10 = 0.900**。

### 3.1 Official `baseline 500`（只记 official evaluator）

| Model | WR | W-L | SE | 95% CI | 来源 / 备注 |
|---|---:|---:|---:|---|---|
| **`034E-frontier`** | **0.904** | **452-48** | **0.013** | **[0.878, 0.930]** | 当前最强 ensemble `500ep` 点 |
| **`034C-frontier`** | **0.890** | **445-55** | **0.014** | **[0.863, 0.917]** | 第一代 frontier ensemble 峰值 |
| **029B@190** | **0.868** | 434-66 | 0.015 | [0.835, 0.897] | per-agent SOTA |
| 030D@320 (1st) | 0.862 | 431-69 | 0.015 | [0.829, 0.891] | — |
| 030D@360 (1st) | 0.856 | 428-72 | 0.016 | [0.823, 0.885] | — |
| 028A@1220 | 0.844 | 422-78 | 0.016 | [0.811, 0.873] | official 峰值；capture 明显回落，见 §3.2 |
| 017@2100 | 0.842 | 421-79 | 0.016 | [0.809, 0.872] | 历史 per-agent 冠军 |
| 025b@080 | 0.842 | 421-79 | 0.016 | [0.809, 0.872] | per-agent field-role 冠军 |
| 030D@330 | 0.840 | 420-80 | 0.016 | [0.807, 0.870] | — |
| 032Ac@200 | 0.836 | 418-82 | 0.017 | [0.803, 0.866] | peak |
| 030A@360 | 0.832 | 416-84 | 0.017 | [0.799, 0.862] | — |
| 032A@170 | 0.826 | 413-87 | 0.017 | [0.793, 0.859] | peak |
| 033A@80 | 0.826 | 413-87 | 0.017 | — | early peak |
| 033A@130 | 0.826 | 413-87 | 0.017 | — | mid peak |
| 030A@300 | 0.830 | 415-85 | 0.017 | — | — |
| 032A@260 | 0.820 | 410-90 | 0.017 | — | internal `0.92` 未在 official 站住 |
| 030D@290 | 0.812 | 406-94 | 0.017 | — | official 中后段普通平台，不是 `0.880` |
| 028A@1060 | 0.810 | 405-95 | 0.018 | [0.774, 0.841] | warmstart ref；主候选 |

### 3.2 `capture 500`（稳定性代理；不是 official，也不是 1000ep）

| Model | official 500 | capture 500 | Δ(capture-official) | 读法 |
|---|---:|---:|---:|---|
| 030A@360 | 0.832 | 0.842 | +0.010 | 对齐最好，可信度高 |
| 030A@300 | 0.830 | 0.796 | -0.034 | 不如 `@360` 稳 |
| 030D@320 | 0.862 | 0.820 | -0.042 | ceiling 高，但 official 可能偏高 |
| 030D@360 | 0.856 | 0.808 | -0.048 | 同上，且略弱于 `@320` |
| 028A@1060 | 0.810 | 0.806 | -0.004 | 对齐稳定 |
| 028A@1220 | 0.844 | 0.796 | -0.048 | official 峰值明显偏高 |
| 032A@170 | 0.826 | 0.826 | +0.000 | 很稳 |
| 032Ac@200 | 0.836 | 0.828 | -0.008 | 稳 |
| 033A@80 | 0.826 | 0.786 | -0.040 | baseline 高点但 H2H 不占优 |
| 033A@130 | 0.826 | 0.778 | -0.048 | H2H 略好，但 baseline 稳定性更差 |

**注意**:
- 本节只用于判断“official 高点是否被高估”，**不能**当作 `1000ep WR`。
- `029B@190` 目前有两次 capture（`0.876` 与 `0.840`），应保留为区间信息，不应和 official `0.868` 硬平均。

### 3.3 Official `baseline 1000`（frontier / active points only）

本节只列**目前已经补了 `official 1000` 的模型**。
它是更高置信度的 baseline 轴，但**不等于 global rank**，也不能覆盖 H2H。

| Model | official 500 | official 1000 | W-L | SE | 95% CI | Δ(1000-500) | 备注 |
|---|---:|---:|---:|---:|---|---:|---|
| **🥇 `055@1000`** ✅2k+capture+H2H×3 | — | **0.902** | **1804-196** (n=2000) | **0.007** | **[0.888, 0.916]** | — | **🏆 NEW PROJECT SOTA**: distillation from 034E ensemble scratch; 1000ep orig 0.904 + rerun 0.900 = 0.902 combined; capture 0.858. **Stage 3 H2H all `***`**: vs 034E teacher 0.590 z=4.03 (**student > teacher +9pp!**), vs 031B base 0.638 z=6.17, vs 043A' peer SOTA 0.622 z=5.46. Cut at 1020/1250 iter by mass kill event. See [snapshot-055 §7](snapshot-055-distill-from-034e-ensemble.md#7-verdict--055-distill-034e--project-sota-2026-04-20-append-only) |
| **`043A'@080`** ✅2k | — | **0.900** | **1800-200** (n=2000) | **0.007** | **[0.886, 0.914]** | — | previous SOTA; now decisively beaten by 055 in H2H (+12.2pp `***`) but still baseline-tied; 2000ep combined (orig 0.901 + rerun 0.899 = 0.900) |
| **`043C'@480`** ✅2k | — | **0.8965** | **1793-207** (n=2000) | **0.007** | **[0.882, 0.911]** | — | 2000ep combined (orig 0.895 + rerun 0.898 = 0.8965); 更重 frontier 的 35% baseline 对照; H2H vs 031B = 0.614, vs 034E = 0.576, vs 043B' = 0.532 |
| **`043B'@440`** ✅2k | — | **0.893** | **1786-214** (n=2000) | **0.007** | **[0.879, 0.907]** | — | **rerun 揭示 single-shot 0.904 高估了 +1.1pp**; 真值 0.893 (orig 0.904 + rerun 0.882 = 0.893); 仍是 baseline 高 candidate 但不是 leaderboard 第一; H2H vs 031B = 0.600, vs 034E = 0.554 |
| **`053Acont@310`** ✅2k | — | **0.896** | **1792-208** (n=2000) | **0.007** | **[0.882, 0.910]** | — | combined 2000ep (orig 0.901 + rerun 0.891 = 0.896); 053A continue iter 200→500 (PBRS) plateau; **不同于过往 single-point peaks (rerun -1pp 是 SE 内, 不是 luck collapse)** |
| **`053Acont@430`** ✅2k+capture+H2H | — | **0.898** | **2245-255** (n=2500: 1000+1000+500) | **0.006** | **[0.886, 0.910]** | — | combined n=2500 (1000ep orig 0.903 + rerun 0.888 + 500ep capture 0.908 = 0.898); H2H **vs 053A@190 = 0.506 TIED** (continue 仅 stabilize plateau, PBRS saturate ~200 iter); H2H **vs 031B@1220 = 0.582 z=3.67 `***`** (+8.2pp PBRS gain reproduce); plateau-shaped (8 ckpts 跨 130 iter 全 ≥ 0.872) — 项目 first stability advantage |
| **`053A@190`** ✅2k+capture | — | **0.891** | **2228-272** (n=2500: 1000+1000+500) | **0.006** | **[0.879, 0.903]** | — | **031B warmstart + 200 iter combo (v2 + outcome PBRS λ=0.01)**, ckpt 190 (last); 三向交叉验证收敛: 1000ep orig 0.907 + rerun 0.873 + capture 500ep 0.894 → 0.891; **= 034E ensemble (0.890) 等价水平 → single-model 达 ensemble 等价 (H_055 stretch confirmed)**; ckpt 190 是 last ckpt 暗示 training 提前结束, 053A continue (200→500) → 053Acont peak 0.903 |
| **`034E-frontier`** ✅2k | **0.904** | **0.892** | **1784-216** (n=2000) | **0.007** | **[0.878, 0.906]** | -0.012 | ensemble 主候选; 034E-control = 0.869; combined 2000ep (orig 0.890 + rerun 0.894 = 0.892); 跟 053A@190 single-model 在 noise 内 tied |
| **`034E-frontier`** | **0.904** | **0.890** | **890-110** | **0.010** | **[0.871, 0.909]** | **-0.014** | 当前 ensemble 主候选；`034E-control = 0.869` |
| **`051A@130`** | — | **0.888** | **888-112** | **0.010** | **[0.868, 0.908]** | — | **single-model 1000ep 历史最高**, warmstart 031B@1220 + combo (v2 + 0.003·learned_051), early peak +130 iter; tied with 031B (+0.6pp marginal) 和 ensemble 034E (-0.2pp), 都在 ±1σ 内; 距 0.90 grading 仅 -1.2pp |
| **`051D@740`** ✅2k | — | **0.889** | **1778-222** (n=2000) | **0.007** | **[0.875, 0.903]** | — | warmstart **031B@80 (weak)** + learned-only (no v2) + 800 iter; combined (orig 0.900 + rerun 0.878 = 0.889); 13 ckpt late-window mean 0.888; 用 4× budget 把弱 base 拉到 ~0.89, 但仍未突破 PBRS path; 相对 051A combo 0.888 持平 → learned-only ≈ combo @ 4× budget |
| **`031Bnoshape@1030`** ✅2k | — | **0.875** | **1750-250** (n=2000) | **0.007** | **[0.861, 0.889]** | — | **v2 ablation**: 031B arch + sparse-only (无 v2) + 1250 iter scratch; combined (orig 0.879 + rerun 0.871 = 0.875); peak vs 031B-with-v2 0.880 = **-0.5pp within 1σ SE** → **v2 shaping 在 1250 iter scratch 上 NOT statistically significant**; v2 主要价值是 convergence accel, 不是 peak elevation |
| **`031B@1220`** ✅2k | **0.882** | **0.8795** | **1759-241** (n=2000) | **0.007** | **[0.866, 0.894]** | -0.002 | combined 2000ep (orig 0.882 + rerun 0.877 = 0.8795); **真值 0.880**, single-shot 0.882 高估 +0.2pp; 仍是 cross-arch SOTA scratch base |
| `051A@180` | — | 0.882 | 882-118 | 0.010 | [0.862, 0.902] | — | 051A 第二高点 |
| `051A@150` | — | 0.875 | 875-125 | 0.010 | [0.855, 0.895] | — | 051A mid window |
| `051B@170` | — | 0.872 | 872-128 | 0.011 | [0.851, 0.893] | — | 051B (learned-only) peak; **vs 031B base = -1.0pp 退化**, 跟 045B 在 031A base 上 +1.0pp 反向 |
| `051A@170` | — | 0.870 | 870-130 | 0.011 | [0.849, 0.891] | — | — |
| `051A@190` | — | 0.870 | 870-130 | 0.011 | [0.849, 0.891] | — | — |
| `051A@160` | — | 0.869 | 869-131 | 0.011 | [0.848, 0.890] | — | — |
| `051A@140` | — | 0.866 | 866-134 | 0.011 | [0.845, 0.887] | — | — |
| `051B@140` | — | 0.865 | 865-135 | 0.011 | [0.844, 0.886] | — | — |
| `051B@190` | — | 0.862 | 862-138 | 0.011 | [0.841, 0.883] | — | — |
| `051B@160` | — | 0.861 | 861-139 | 0.011 | [0.840, 0.882] | — | — |
| `051B@180` | — | 0.854 | 854-146 | 0.011 | [0.832, 0.876] | — | — |
| `051B@150` | — | 0.849 | 849-151 | 0.011 | [0.827, 0.871] | — | — |
| `051B@130` | — | 0.843 | 843-157 | 0.011 | [0.821, 0.865] | — | 051B 最低点 |
| **`034C-frontier`** | **0.890** | **0.843** | **843-157** | **0.012** | **[0.820, 0.866]** | **-0.047** | 第一代 frontier ensemble；peer 轴很强但 baseline 回落明显 |
| **029B@190** | 0.868 | **0.846** | 846-154 | 0.011 | [0.824, 0.868] | -0.022 | 当前最稳的 baseline 头名 |
| `036C@270` | — | 0.833 | 833-167 | 0.012 | [0.810, 0.856] | — | official `1000ep` 最佳点 |
| `036C@150` | — | 0.832 | 832-168 | 0.012 | [0.809, 0.855] | — | capture 更强，当前更稳的 baseline-oriented 候选 |
| `032nextC@130` | — | 0.822 | 822-178 | 0.012 | [0.798, 0.846] | — | `032-next` control；仅代表 baseline 轴 |
| `030D@320` | 0.862 | 0.816 | 816-184 | 0.012 | [0.792, 0.840] | -0.046 | official `500` 高点收缩明显 |
| `017@2100` | 0.842 | 0.810 | 810-190 | 0.012 | [0.786, 0.834] | -0.032 | 经典 per-agent 强基线 |
| `030A@360` | 0.832 | 0.809 | 809-191 | 0.012 | [0.785, 0.833] | -0.023 | 比 `030D` 更稳 |
| `025b@080` | 0.842 | 0.804 | 804-196 | 0.013 | [0.779, 0.829] | -0.038 | baseline 轴收缩，但 H2H 仍强 |
| `032nextA@110` | — | 0.793 | 793-207 | 0.013 | [0.768, 0.818] | — | plumbing 已通，但目标未转化为 baseline 优势 |
| `032Ac@200` | 0.836 | 0.789 | 789-211 | 0.013 | [0.764, 0.814] | -0.047 | `032` control |
| `032A@170` | 0.826 | 0.787 | 787-213 | 0.013 | [0.762, 0.812] | -0.039 | `032` aux 主线 |
| `033A@80` | 0.826 | 0.786 | 786-214 | 0.013 | [0.761, 0.811] | -0.040 | baseline-oriented candidate |
| `028A@1060` | 0.810 | 0.783 | 783-217 | 0.013 | [0.757, 0.809] | -0.027 | team-level base |
| `033A@130` | 0.826 | 0.777 | 777-223 | 0.013 | [0.751, 0.803] | -0.049 | H2H-oriented candidate |
| `038D@40` | — | **0.806** | 806-194 | 0.013 | [0.781, 0.831] | — | [038 Stage 1](snapshot-038-team-level-liberation-handoff-placeholder.md#10-2026-04-18-1550-edt-stage-1-四条-lane-首轮结果) 最强点；v2+entropy=0.01 |
| `038D@60` | — | 0.806 | 806-194 | 0.013 | [0.781, 0.831] | — | 038D 另一 tied 点 |
| `038D@20` | — | 0.803 | 803-197 | 0.013 | [0.778, 0.828] | — | 038D 早期强 |
| `038C@50` | — | 0.800 | 800-200 | 0.013 | [0.775, 0.825] | — | 038 event-lane 最强点 |
| `038C@90` | — | 0.799 | 799-201 | 0.013 | [0.774, 0.824] | — | — |
| `038C@80` | — | 0.798 | 798-202 | 0.013 | [0.773, 0.823] | — | — |
| `038B@110` | — | 0.797 | 797-203 | 0.013 | [0.772, 0.822] | — | 038 goal-prox PBRS 最强点 |
| `038A@160` | — | 0.796 | 796-204 | 0.013 | [0.771, 0.821] | — | 038 depenalized-v2 最强点 |
| `038B@170` | — | 0.796 | 796-204 | 0.013 | — | — | — |
| `038A@180` | — | 0.793 | 793-207 | 0.013 | — | — | — |
| `038D@90` | — | 0.793 | 793-207 | 0.013 | — | — | — |
| `036D@150` | — | **0.860** | 860-140 | 0.011 | [0.838, 0.882] | — | [snapshot-036D §10.3](snapshot-036d-learned-reward-stability-fix.md#103-official-1000ep关键结果)；**首次 1000ep max 上超 029B@190 warmstart 0.846**（marginal, single 1000ep）|
| `036D@250` | — | 0.856 | 856-144 | 0.011 | [0.834, 0.878] | — | 036D late peak |
| `036D@130` | — | 0.848 | 848-152 | 0.011 | [0.826, 0.870] | — | 036D mid peak |
| `036D@10` | — | 0.846 | 846-154 | 0.011 | [0.824, 0.868] | — | warmstart-near (036D 早期)|
| `036D@70` | — | 0.836 | 836-164 | 0.012 | [0.812, 0.860] | — | — |
| `036D@160` | — | 0.835 | 835-165 | 0.012 | [0.811, 0.859] | — | — |
| `036D@40` | — | 0.817 | 817-183 | 0.012 | [0.793, 0.841] | — | — |
| `039@230` | — | 0.843 | 843-157 | 0.012 | [0.819, 0.867] | — | [snapshot-039 §9.4](snapshot-039-airl-adaptive-reward-learning.md#94-official-1000ep已出)；refresh 实际未生效，等价 036D-style 稳定化 |
| `039@40` | — | 0.836 | 836-164 | 0.012 | [0.812, 0.860] | — | early peak |
| `039@190` | — | 0.833 | 833-167 | 0.012 | [0.809, 0.857] | — | — |
| `039@10` | — | 0.830 | 830-170 | 0.012 | [0.806, 0.854] | — | warmstart-near |
| `039@150` | — | 0.823 | 823-177 | 0.012 | [0.799, 0.847] | — | — |
| `039@140` | — | 0.820 | 820-180 | 0.012 | [0.796, 0.844] | — | — |
| `039@170` | — | 0.820 | 820-180 | 0.012 | — | — | — |
| **`031A@1040`** | — | **0.867** | 867-133 | 0.011 | [0.846, 0.888] | — | **项目 1000ep max 最高**（[snapshot-031 §11.3](snapshot-031-team-level-native-dual-encoder-attention.md#113-1000ep-eval已完成)）；2000-game avg 0.860 |
| `031A@1040 (rerun)` | — | 0.853 | 853-147 | 0.011 | [0.831, 0.875] | — | 1st run 0.867 + 2nd run 0.853 = 2000-game avg **0.860**, 真实 SOTA 在此 |
| `031A@1170` | — | 0.865 | 865-135 | 0.011 | [0.844, 0.886] | — | 031A second peak |
| `031A@1170 (rerun)` | — | 0.850 | 850-150 | 0.011 | [0.828, 0.872] | — | 2000-game avg 0.858 |
| `031A@1000` | — | 0.852 | 852-148 | 0.011 | [0.830, 0.874] | — | — |
| `031A@800` | — | 0.850 | 850-150 | 0.011 | [0.828, 0.872] | — | — |
| `031A@930` | — | 0.841 | 841-159 | 0.012 | [0.818, 0.864] | — | — |
| `031A@770` | — | 0.840 | 840-160 | 0.012 | [0.817, 0.863] | — | — |
| `031A@580` | — | 0.812 | 812-188 | 0.012 | [0.788, 0.836] | — | early peak |
| `040A@60` | — | 0.863 | 863-137 | 0.011 | [0.841, 0.885] | — | [snapshot-040 §11.2](snapshot-040-team-level-stage2-on-031A.md#112-1000ep-官方-baseline-evalsaturation-实证)；depenalized v2 handoff peak |
| `040A@50` | — | 0.853 | 853-147 | 0.011 | [0.831, 0.875] | — | — |
| `040A@40` | — | 0.852 | 852-148 | 0.011 | [0.830, 0.874] | — | — |
| `040B@190` | — | 0.863 | 863-137 | 0.011 | [0.841, 0.885] | — | PBRS handoff peak |
| `040B@170` | — | 0.862 | 862-138 | 0.011 | [0.840, 0.884] | — | — |
| `040B@140` | — | 0.855 | 855-145 | 0.011 | [0.833, 0.877] | — | — |
| `040B@130` | — | 0.855 | 855-145 | 0.011 | [0.833, 0.877] | — | — |
| `040B@180` | — | 0.845 | 845-155 | 0.011 | [0.823, 0.867] | — | — |
| `040B@150` | — | 0.837 | 837-163 | 0.012 | [0.814, 0.860] | — | — |
| `040C@50` | — | 0.865 | 865-135 | 0.011 | [0.844, 0.886] | — | event lane handoff peak |
| `040C@40` | — | 0.852 | 852-148 | 0.011 | [0.830, 0.874] | — | — |
| `040C@60` | — | 0.836 | 836-164 | 0.012 | [0.812, 0.860] | — | — |
| `040D@140` | — | 0.863 | 863-137 | 0.011 | [0.841, 0.885] | — | v2+entropy=0.01 peak |
| `040D@150` | — | 0.852 | 852-148 | 0.011 | [0.830, 0.874] | — | — |
| `040D@130` | — | 0.841 | 841-159 | 0.012 | [0.818, 0.864] | — | — |
| `041B@60` | — | **0.852** | 852-148 | 0.011 | [0.830, 0.874] | — | [snapshot-041 §11.2](snapshot-041-per-agent-stage2-pbrs-on-036D.md#112-1000ep-官方-baseline-eval)；per-agent Stage 2 PBRS handoff peak, **退化 vs 036D base 0.860 (-0.008)** |
| `041B@50` | — | 0.844 | 844-156 | 0.011 | [0.822, 0.866] | — | — |
| `041B@20` | — | 0.835 | 835-165 | 0.012 | [0.812, 0.858] | — | — |
| `041B@10` | — | 0.829 | 829-171 | 0.012 | [0.806, 0.852] | — | — |
| `041B@30` | — | 0.823 | 823-177 | 0.012 | [0.800, 0.846] | — | — |
| `041B@40` | — | 0.822 | 822-178 | 0.012 | [0.799, 0.845] | — | — |

**040 全 4 lane 关键观察**: peak 全部在 [0.863, 0.865]，与 [031A@1040 (0.860 2000-game avg)](snapshot-031-team-level-native-dual-encoder-attention.md#11) 在 1000ep SE ±0.016 内——**Stage 2 shaping (PBRS / event / depenalized / entropy) 在 031A high base 上无统计意义增益**。snapshot-040 §6.2 R2 saturation 风险确认。详见 [snapshot-040 §11](snapshot-040-team-level-stage2-on-031A.md#11-首轮结果2026-04-194-lane-全部完成)。

**041B 唯一退化 lane**: peak 0.852 < 036D@150 base 0.860 (-0.008)。Stage 2 PBRS + 036D 双 shaping 叠加反而冲掉 036D 学到的优势。详见 [snapshot-041 §11.3](snapshot-041-per-agent-stage2-pbrs-on-036D.md#113-verdict--路径否决轻微退化)。

**跨 lane 观察 (040 + 041 合计)**: 所有 v2-derivative lane 1000ep peak 落在 [0.852, 0.865]；跨架构 (per-agent / team-level Siamese)、跨 reward 修改 (PBRS / event / depenalized / entropy / learned reward / 双 shaping) **全部打不破 ~0.86 ceiling**。v2 shaping 本身可能是 bottleneck，需要考虑 MaxEnt / pure sparse 路径。

**当前读法**:

- `official 1000` 已经明确收紧了不少 `500ep` 高点，尤其是 `030D / 032 / 033 / 025b`
- `029B@190` 仍然是当前最稳的 baseline 头名
- `036C@150/270` 已进入 baseline frontier，但它们当前更像 baseline-oriented 强挑战者，而不是 peer 轴已坐实的顶线
- `032nextC@130` 在 baseline 轴上很亮，但在没有 peer H2H 前，不能直接把它写成全局跃升

### 3.4 当前 baseline WR 排名（按**当前主候选 checkpoint** 的 official 500）

这里只按 **official 500** 排，不把 capture 当作第二次 official；稳定性请结合 §3.2 一起读。

1. 🥇 **029B@190**: 0.868 (500ep) — **最强**，距 0.900 目标 -3.2pp
2. 🥈 030D@320: 0.862 (500ep) — strongest team-level official point，但 capture 回落较大
3. 🥉 025b@080 = 017@2100: 0.842 (500ep)
4. 032Ac@200: 0.836 (500ep)
5. 030A@360: 0.832 (500ep) — official 略低于 `030D`，但 stability 更好
6. 032A@170 = 033A@80 = 033A@130: 0.826 (500ep)
7. 028A@1060: 0.810 (500ep)

### 3.5 当前 baseline WR 排名（按**目前已有数据的 official 1000**）

这里只按 **official 1000** 排，而且**只比较已经测过 `1000ep` 的点**。

1. 🥇 **029B@190** — `0.846`
2. 🥈 `036C@270` — `0.833`
3. 🥉 `036C@150` — `0.832`
4. `032nextC@130` — `0.822`（baseline 轴高，但尚无 strong peer H2H）
5. `030D@320` — `0.816`
6. `017@2100` — `0.810`
7. `030A@360` — `0.809`
8. `025b@080` — `0.804`
9. `032nextA@110` — `0.793`
10. `032Ac@200` — `0.789`
11. `032A@170` — `0.787`
12. `033A@80` — `0.786`
13. `028A@1060` — `0.783`
14. `033A@130` — `0.777`

---

## 4. vs `random` (assignment grade criterion 2)

目标 9/10 = 0.900。大多数 trained policy 已经压到 ≥ 0.96，不是主要 bottleneck。

| Model | 50ep vs random | 500ep vs random | 笔记 |
|---|---:|---:|---|
| 032A@260 | 1.000 (50-0-0) | 未测 | 完美 |
| 029B@190 | 未记录 | 未测 | 期望 ≥ 0.96 |
| 030A@20 | 0.980 | 未测 | internal peak |
| 030D@320 | 0.980 | 未测 | — |
| 030A@360 | 0.980 | 未测 | — |
| 032Ac@200 | 0.980 | 未测 | — |
| 033A@240 | 0.960 | 未测 | — |

**TODO**: 补几条 500ep vs random 把 "≥ 0.9" 做成硬事实。

---

## 5. Peer H2H Matrix

**每格单元含义**: `行模型` 对 `列模型` 的 win_rate （样本 n=500 除非另标）。**空 = 未测**。

⚠️ 所有数字均已按 §0.2 流程**方向验证**。

### 5.1 核心 H2H 矩阵（frontier models）

| ↓ 行 \ 列 → | 029B@190 | 025b@080 | 030A@360 | 030D@320 | 036C@270 | 028A@1060 | 032A@170 | 032Ac@200 | 033A@130 | 033A@80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **031B@1220** | **0.584*** | **0.566** ** | — | — | — | — | — | — | — | — |
| **031A@1040** | **0.552*** (1000ep) | **0.532\* (1000ep)** | — | — | — | **0.568*** (1000ep) | — | — | — | — |
| **029B@190** | — | 0.492 | **0.550*** | **0.559** *** (1000ep) | **0.578*** | **0.538*** | — | — | — | — |
| **025b@080** | 0.508 | — | — | **0.541** ** (1000ep) | — | — | — | — | — | — |
| **030A@360** | 0.450 | — | — | 0.496 | — | — | — | — | — | — |
| **030D@320** | 0.441 (1000ep) | 0.459 (1000ep) | 0.504 | — | — | 0.536 | — | — | — | — |
| **036C@270** | **0.422*** | — | — | — | — | — | — | — | — | — |
| **028A@1060** | 0.462 | — | — | 0.464 | — | — | 0.464 | 0.442 | 0.482 | 0.518 |
| **032A@170** | — | — | — | — | — | 0.536 | — | 0.528 | — | — |
| **032Ac@200** | — | — | — | — | — | **0.558** ** | 0.472 | — | — | — |
| **033A@130** | — | — | — | — | — | 0.518 | — | — | — | — |
| **033A@80** | — | — | — | — | — | 0.482 | — | — | — | — |

符号: `***` p<0.001, `**` p<0.01, `*` p<0.05, `(1000ep)` = 两次 500ep 合并。未标即 p ≥ 0.05 或边缘。

### 5.2 H2H 样本大小补充

所有 500ep 单次 H2H 样本大小 = 500（250 blue + 250 orange 求和）。z-score 对 50% null:
- `|z| > 1.96` → p < 0.05（`*`）
- `|z| > 2.58` → p < 0.01（`**`）
- `|z| > 3.29` → p < 0.001（`***`）

具体 log 位置: `docs/experiments/artifacts/official-evals/headtohead/<A>_vs_<B>.log`

### 5.3 已测 H2H 的 `n / z / p` 明细（只列目前有数据的）

| Pairing（行 vs 列） | WR | n | z | p(one-tailed) | 备注 |
|---|---:|---:|---:|---:|---|
| **`034E-frontier vs 031B@1220`** | **0.596** | **500** | **4.293** | **<0.0001** | **`***`**；当前 ensemble 主候选对 strongest single model 的直连正优势 |
| **`034E-frontier vs 029B@190`** | **0.590** | **500** | **4.025** | **<0.0001** | **`***`**；对 per-agent frontier 维持硬正号 |
| **`034E-frontier vs 034C-frontier`** | **0.544** | **500** | **1.968** | **0.0245** | **`*`**；方向上支持 `034E > 034C`，但优势属中等幅度 |
| **`043A'@080 vs 031B@1220`** | **0.524** | **500** | **1.073** | **0.142** | **NOT sig**；方向上小优 `031B`，但更像窄优势而非硬突破 |
| **`043A'@080 vs 034E-frontier`** | **0.442** | **500** | **-2.594** | **0.0048 (toward <0.5)** | **`**` negative**；baseline 轴虽过 `0.90`，但 peer 轴仍明确低于 `034E` |
| **`043B'@440 vs 031B@1220`** | **0.600** | **500** | **4.472** | **<0.0001** | **`***`**；`043` 第二代 harder curriculum 已不再只是 baseline specialist，而是对当前 strongest single model 给出硬正号 |
| **`043B'@440 vs 034E-frontier`** | **0.554** | **500** | **2.415** | **0.0079** | **`**`**；当前 `043` 主候选已对 ensemble 主线给出正优势，但仍需和 `capture 500 = 0.852` 的脆点一起读 |
| **`043C'@480 vs 031B@1220`** | **0.614** | **500** | **5.098** | **<0.0001** | **`***`**；比 `043B'` 还略高的 peer 正号，支持“更 rounded 的重-frontier 版本”这一读法 |
| **`043C'@480 vs 034E-frontier`** | **0.576** | **500** | **3.399** | **0.0003** | **`***`**；对当前 ensemble 主线给出更强正号，同时 `capture 500 = 0.880` 也明显高于 `043B'` |
| **`043B'@440 vs 043C'@480`** | **0.468** | **500** | **-1.431** | **0.0763 (toward <0.5)** | **NOT sig**；等价地说 `043C` 对 `043B` 是 `0.532` 的窄优势。方向上支持 `043C` 更 rounded，但幅度还不够写成硬替代 |
| `029B@190 vs 025b@080` | 0.492 | 500 | -0.358 | 0.3603 | 直连未拉开 |
| `029B@190 vs 030A@360` | 0.550 | 500 | 2.236 | 0.0127 | `*` |
| `029B@190 vs 030D@320` | 0.559 | 1000 | 3.732 | 0.0001 | 两次 500ep 合并，`***` |
| `029B@190 vs 036C@270` | 0.578 | 500 | 3.488 | 0.0002 | `***`；`036C` baseline 轴很强，但 peer 轴仍明显低于 `029B` |
| `029B@190 vs 028A@1060` | 0.538 | 500 | 1.699 | 0.0446 | 边缘显著，`*` |
| `025b@080 vs 030D@320` | 0.541 | 1000 | 2.594 | 0.0047 | 两次 500ep 合并，`**` |
| `030D@320 vs 030A@360` | 0.504 | 500 | 0.179 | 0.4290 | 基本打平 |
| `030D@320 vs 028A@1060` | 0.536 | 500 | 1.610 | 0.0537 | 方向性正号，尚未过 `0.05` |
| `032A@170 vs 032Ac@200` | 0.528 | 500 | 1.252 | 0.1052 | 方向性正号，尚未显著 |
| `032Ac@200 vs 028A@1060` | 0.558 | 500 | 2.594 | 0.0047 | `**` |
| `032A@170 vs 028A@1060` | 0.536 | 500 | 1.610 | 0.0537 | 方向性正号，尚未过 `0.05` |
| `033A@130 vs 028A@1060` | 0.518 | 500 | 0.805 | 0.2104 | 未显著 |
| `033A@80 vs 028A@1060` | 0.482 | 500 | -0.805 | 0.2104 | 未显著 |
| **`036D@150 vs 029B@190`** | **0.507** | **1000** | **0.443** | **0.329** | **完全平局**；036D baseline +1.4pp 不能转化 peer 优势；vs 037 教训方向不冲突（baseline 微优 + peer tie，不是 baseline 弱 + peer 赢）|
| **`031A@1040 vs 029B@190`** | **0.552** | **1000** | **3.290** | **0.0005** | **`***`**；031A 同时 baseline +1.4pp **AND** peer +10.4pp，**项目首个双判据击败 029B 的 lane**；blue 0.580 / orange 0.524 都 >0.5（无侧别运气） |
| **`031A@1040 vs 025b@080`** | **0.532** | **1000** | **2.024** | **0.0215** | **`*`**；031A vs per-agent peer-axis 隐性头名 025b，边缘显著但方向稳定；blue 0.538 / orange 0.526 都 >0.5（无侧别运气）。优势 +0.032 比 vs 029B (+0.052) 弱 -2pp，符合 "025b ≥ 029B in peer axis" (025b vs 029B = 0.508) 的预期。031A 是首个**同时**对 029B 和 025b 都赢的 lane |
| **`031A@1040 vs 028A@1060`** | **0.568** | **1000** | **4.301** | **<0.0001** | **`***` 三连最强**；同 reward (v2) 同 base 类型 (team-level) 不同架构（Siamese vs flat MLP）。031A 此战是 baseline +7.7pp 的 H2H 镜像（0.860 vs 0.783）。blue 0.606 / orange 0.530 都 >0.5（侧别不对称 +7.6pp，但方向一致）。**架构改造（dual encoder）真实 H2H 优势已坐实**；caveat 是训练时长 1250 vs 1060 iter，不能完全分离架构 vs 训练时长贡献 |
| **`031B@1220 vs 031A@1040`** | **0.516** | **500** | **0.715** | **0.237** | **NOT sig**；架构 axis (cross-attention vs Siamese)。即使 baseline 上 031B +2.2pp (0.882 vs 0.860)，**直接对决 H2H 几乎平手**——peer-axis 上架构 step 2 (Siamese → cross-attention) **效果远小于** step 1 (flat → Siamese, 031A vs 028A = 0.568 ***)。这是 031B 在 baseline 上多学的「**部分是 baseline-specialization**」的关键 caveat。blue 0.500 / orange 0.532 |
| **`031B@1220 vs 029B@190`** | **0.584** | **500** | **3.755** | **<0.001** | **`***`**；031B vs per-agent v2 SOTA。比 031A vs 029B (0.552 ***) 还强 +3.2pp。跨架构优势 robust。blue 0.604 / orange 0.564 |
| **`031B@1220 vs 025b@080`** | **0.566** | **500** | **2.951** | **0.0016** | **`**`**；031B vs per-agent BC champion 025b。比 031A vs 025b (0.532 *) 强 +3.4pp。blue 0.568 / orange 0.564（无侧别 gap）|
| **`031B@1220 vs 036D@150`** | **0.574** | **500** | **3.308** | **0.00047** | **`***`**；031B vs per-agent + learned reward。**031B vs 036D (cross-reward) ≈ vs 029B (same v2)** (0.574 vs 0.584) → 031B 优势主来自架构 (cross-attention)，不依赖特定 reward path |

---

## 6. Per-Criterion Rankings

### 6.1 By `vs baseline`（official 500；广覆盖历史轴）

1. 🥇 **`034E-frontier`** — `0.904`
2. 🥈 **`034C-frontier`** — `0.890`
3. 🥉 **`029B@190`** — `0.868`
4. `030D@320` — `0.862`
5. `028A@1220` — `0.844`
6. `017@2100` = `025b@080` — `0.842`
7. `032Ac@200` — `0.836`
8. `030A@360` — `0.832`
9. `032A@170` = `033A@80` = `033A@130` — `0.826`
10. `028A@1060` — `0.810`

### 6.2 By `vs baseline`（official 1000；高置信度 baseline 轴，仅限已测点）

1. 🥇 **`043B'@440`** — **0.904**
2. 🥈 **`043A'@080`** — **0.901**
3. 🥉 **`043C'@480`** — **0.895**
4. **`034E-frontier`** — **0.890**
5. **`051A@130`** — **0.888**
6. **`031B@1220`** — **0.882**
7. **`031A@1040`** — **0.860** (2000-game avg, 1st 0.867 + 2nd 0.853)
8. **`036D@150`** — **0.860** (single 1000ep)
9. `031A@1170` — `0.858` (2000-game avg)
10. `036D@250` — `0.856`
11. `031A@1000` — `0.852`
12. `031A@800` — `0.850`
13. `036D@130` — `0.848`
14. **`029B@190`** — `0.846`
15. `039@230` — `0.843`
16. `034C-frontier` — `0.843`
17. `036C@270` — `0.833`
18. `036C@150` — `0.832`
19. `032nextC@130` — `0.822`
20. `030D@320` — `0.816`
21. `017@2100` — `0.810`
22. `030A@360` — `0.809`

**注意**:
- 这一轴现在最适合回答"谁对 `ceia_baseline_agent` 更稳"
- 对当前 frontier 顶层候选来说，`043B' / 043A' / 043C' / 034E` 这四个点都落在一个很窄的高平台里；**只靠 1000ep baseline 上约 1pp 的差距，不足以做硬排序**
- 031A 和 036D 在 baseline 轴上**完全 tied** (~0.860)，但 §5.3 H2H 显示 031A 在 peer 轴上**显著强于** 036D (后者 vs 029B 是 tied)
- 它**不覆盖** peer H2H，因此不能单独拿来做全局冠军排序
- 尤其 `036C / 025b / 030A / 030D / 032nextC` 的位置，必须和 §5 H2H 一起读

### 6.3 By peer H2H strength（**partial order / 分层**，不强行线性总排）

**目前 hard evidence only**（只写有直连 H2H 支撑的关系）:

0. **Current top training-distribution pair: `043B'@440` and `043C'@480`**
   - `043B'` 保留当前最高 `official baseline 1000 = 0.904`
   - `043C'` 则给出 `capture 500 = 0.880`、`H2H vs 031B = 0.614`、`vs 034E = 0.576`
   - direct H2H `043B vs 043C = 0.468`（等价 `043C vs 043B = 0.532`），方向上支持 `043C` 更 rounded，但 `n=500` 下仍未显著
   - 当前最稳的口径不是“`043C` 已正式取代 `043B`”，而是：**`043B` 是 baseline 尖峰更高的版本，`043C` 是更 rounded 的 overall 版本**
1. **New ensemble top candidate: `034E-frontier`**
   - baseline `500 / 1000 = 0.904 / 0.890`
   - 直连 H2H: `vs 031B@1220 = 0.596`（n=500, z=4.29, p<0.0001, `***`）
   - 直连 H2H: `vs 029B@190 = 0.590`（n=500, z=4.03, p<0.0001, `***`）
   - 直连 H2H: `vs 034C-frontier = 0.544`（n=500, z=1.97, p=0.0245, `*`）
   - 当前证据已经足够把它写成 **ensemble 主候选**
   - caveat: 仍缺 `vs 031A@1040` 的直连 H2H，因此不应把它武断写成“对所有旧 frontier 全部已坐实碾压”
1. **Earlier-generation baseline-oriented challenger: `043A'@080`**
   - `official baseline 1000 = 0.901`
   - 直连 H2H: `vs 031B@1220 = 0.524`（n=500, not sig）
   - 直连 H2H: `vs 034E-frontier = 0.442`（n=500, 显著负号）
   - 当前最稳的读法是：它验证了 `training-distribution` 路线本身有效，但已被 `043B'` 取代为 `043` 家族主候选
3. **Previous single-policy top tier: `031A@1040`**
   - 直连 H2H: `031A@1040 vs 029B@190 = 0.552`（n=1000, z=3.29, p=0.0005, `***`）
   - 直连 H2H: `031A@1040 vs 025b@080 = 0.532`（n=1000, z=2.02, p=0.022, `*` 边缘显著但方向稳定）
   - 直连 H2H: **`031A@1040 vs 028A@1060 = 0.568`**（n=1000, z=4.30, p<0.0001, **`***` 三连最强**；同 reward 同 base 类型，干净架构对照）
   - 同时 baseline 1000ep avg `0.860` > 029B `0.846` (+1.4pp)
   - **首个同时在 baseline 和 peer 两个判据上都击败 frontier 三个候选 (029B + 025b + 028A) 的 lane**
   - **架构改造（Siamese dual encoder）真实 H2H 优势已坐实**：vs 028A_1060 同 reward 同 base 类型只差架构，+0.068 H2H 增益 ≈ baseline 0.860 vs 0.783 = +7.7pp 的镜像
   - caveat: 031A 跑 1250 iter, 028A 1060 iter；架构 vs 训练时长贡献分离仍待 028A 长跑对照
4. **`031B@1220`**
   - `official 1000 = 0.882`
   - 直连 H2H: `vs 029B@190 = 0.584`、`vs 025b@080 = 0.566`、`vs 036D@150 = 0.574`
   - `vs 031A@1040 = 0.516` 未显著，说明它的额外 baseline 增益有一部分是 baseline-specialization
5. **Previous top tier: `029B@190 ≈ 025b@080`**
   - 直连 H2H: `029B@190 vs 025b@080 = 0.492`（n=500，未拉开）
   - `029B` 已击败所有受测 team-level **except 031A** (`028A 0.538`, `030A 0.550`, `030D 0.559`)
   - `029B` 也已显著击败当前 learned-reward 参考点 `036C@270` (`0.578`)
   - `025b` 已在 1000ep H2H 上击败 `030D@320` (`0.541`)
6. **Baseline-strong + peer-tied: `036D@150`**
   - baseline 1000ep `0.860` (与 031A tied)
   - 但 H2H vs `029B@190 = 0.507`（n=1000, z=0.44, p=0.33, **平局**）
   - 与 031A 形成对比：同 baseline 高度但 peer 没突破 029B
7. **Baseline-strong but peer-unresolved: `036C@150/270`**
   - baseline `1000ep` 上，`036C@270 = 0.833`, `036C@150 = 0.832`
   - 但目前唯一 peer 直连结果是 `036C@270 vs 029B@190 = 0.422`
   - 因此它更适合暂放在“baseline-oriented 强挑战者”，而不是 peer 轴 top tier
8. **Next tier: `030D@320 ≈ 030A@360`**
   - 直连 H2H: `030D@320 vs 030A@360 = 0.504`（n=500，基本打平）
   - 两者都低于 `029B`
   - `030D` 对 `028A` 有方向性优势（`0.536`），但在当前 `n=500` 下尚未显著
9. **Middle tier: `032A@170 ≈ 032Ac@200`, and `032Ac@200 > 028A@1060`**
   - `032A@170 vs 032Ac@200 = 0.528`，方向性偏向 `032A`，但未显著
   - `032Ac@200 vs 028A@1060 = 0.558`，当前是这组三者里唯一显著结果
   - `032A@170 vs 028A@1060 = 0.536` 也偏正，但尚未过 `0.05`
   - 这三者和 `030A/030D` 的直连 H2H 仍缺
10. **Narrow-signal tier: `033A@130` 与 `033A@80` 目前都未与 `028A@1060` 拉开**
   - `033A@130 vs 028A@1060 = 0.518`
   - `033A@80 vs 028A@1060 = 0.482`

**当前最稳的读法**:
- baseline WR 与 peer H2H 大方向一致，但 frontier 处不能再武断写成单一总排名。
- 尤其 `029B` 与 `025b`、`030D` 与 `030A`、`032A` 与 `032Ac`，都更适合写成“近邻层级 + 直连结果”。

---

## 7. 待测 H2H (Open Questions)

按优先级排，完成后移到上面的 matrix:

1. ~~028A@1060 vs 025b@080~~ (低优先；方向已 obvious)
2. ~~028A@1060 vs 017@2100~~ (低优先)
3. **029B@190 vs 028A@1220** — 旧 team-level 峰值对 per-agent SOTA
4. `036C@150 vs 029B@190` — **暂缓**；`036C` 首轮已确认 baseline 轴很强，但训练日志存在大量 `kl/total_loss=inf`，当前优先级是先修数值稳定性
5. 500ep vs random 补齐所有 frontier model
6. 032/033 系列对 029B@190 的 H2H（预期同向输，但量级未测）
7. **032nextC@130 vs 028A@1060** — 新 control 是否只是 baseline 轴更高，还是也真的超过 base
8. **032nextC@130 vs 030A@360 / 030D@320** — `032-next` control 在 baseline 轴亮眼，但 peer 强度仍未知

---

## 8. Changelog

| 日期 | 事件 | 作者 |
|---|---|---|
| 2026-04-20 ~01:55 | **🚨 031B-noshape v2 ablation verdict**: 031B arch + sparse-only + 1250 iter scratch on 5015755. Stage 1 12 ckpts mean 0.866, peak 1030 single 0.879. Rerun: 1030 combined **0.875** (n=2000, CI [0.861, 0.889]), 1040 combined 0.870. **vs 031B-with-v2 combined 0.880 = -0.5pp 在 1σ SE 内, NOT statistically significant**. **v2 shaping 在 1250 iter scratch 上无显著真增益** — 主要价值是 convergence speed 不是 peak. 间接支持 053D-mirror "v2 as ceiling" hypothesis (至少证 v2 不 help). 053Acont 0.898 vs 031B 0.880 的 +1.8pp **几乎全来自 PBRS**, 不是 v2. | Claude |
| 2026-04-20 ~06:15 | **🏆🏆🏆 055 distill = PROJECT SOTA breakthrough + H2H 决定性击败所有 prior candidates**: 055 = 031B-arch scratch + distill from 034E ensemble (KL teacher probs + PPO env reward), 1020/1250 iter (cut by mass kill event). **Stage 1**: 10 ckpts mean 0.887, peak **0.904 @ iter 1000**. **Rerun verify**: ckpt 1000 orig 0.904 + rerun 0.900 = **combined 0.902** (n=2000, CI [0.888, 0.916]). 严格 verified 越过 0.900 grading threshold。**Stage 2 capture**: 0.858。**Stage 3 H2H**: (1) **vs 034E teacher = 0.590 z=4.03 `***`** — **Student decisively beats teacher +9pp** (Hinton 2015 pattern), (2) vs 031B base = 0.638 z=6.17 `***` (+13.8pp), (3) **vs 043A' peer SOTA = 0.622 z=5.46 `***`** (+12.2pp decisively). **Distillation paradigm validated**: compress ensemble knowledge → single-model beats all prior candidates on both baseline + peer axes. Deploy cost = 1/3 of ensemble. 详见 [snapshot-055 §7](snapshot-055-distill-from-034e-ensemble.md#7-verdict--055-distill-034e--project-sota-2026-04-20-append-only) | Claude |
| 2026-04-20 ~04:27 | **🚨 Mass kill event**: 8 training lanes (054/055/056ABCD/057/058) simultaneously killed by presumed Claude session disconnect → srun dies → training dies. Ckpts saved via symlink to scratch archive (except 056D whose archive is corrupted/empty). Recovery via post-train 1000ep eval of saved ckpts. 054/055 高 completion (92%/82%), 056A-D 50-90% (056C DOA, 056D corrupted), 057 (62%) + 058 (67%) partial. 055 结果 despite early cut 仍是 breakthrough. | Claude |
| 2026-04-19 ~23:25 | **051D Stage 1 verdict + 051 family 横向对比**: 051D peak ckpt 740 = 0.900 single → rerun 0.878 → **combined 0.889** (n=2000)。13 ckpts late-window mean 0.888。**051 family 横向**: 051A 0.888 (combo, 200 iter) ≈ 051D 0.889 (learned-only, 800 iter, weak base) → **learned-only 用 4× budget 才追上 combo**。**跨 family**: 053 PBRS path > 051 learned-bucket path 在同 base + budget (053Acont 0.898 vs 051D 0.889 = +0.9pp)。**整体 verdict**: learned-bucket reward family saturate ~0.89, 后续投资转 053 PBRS / 055 distill / 057 RND / 058 curriculum 路径。详见 [snapshot-051 §9](snapshot-051-learned-reward-from-strong-vs-strong-failures.md#9-051d-stage-1-verdict--051-family-横向对比-2026-04-19) | Claude |
| 2026-04-19 ~22:35 | **053Acont 完整 verdict — Stage 2 + Stage 3 done**: Stage 2 capture ckpt 430 = **0.908 (n=500)** (project 最高 single 500ep capture); Combined n=2500 (903+888+454)/2500 = **0.898**。Stage 3 H2H portfolio: **vs 053A@190 = 0.506 z=0.27 TIED** (continue ≠ stronger, just stable plateau, PBRS signal saturate at ~200 iter); **vs 031B@1220 = 0.582 z=3.67 `***`** (PBRS combo gain +8.2pp 复现 053A@190 的 +7pp pattern). **结论**: 053Acont 是当前 single-model 主候选, 跟 043A' 0.900 / 043C' 0.8965 在 1σ 内 tied, 但具备**多 ckpt plateau stability** (跨 130 iter, 8 ckpts 都 ≥ 0.872) 这是过往 SOTAs 不具备的稳定性。详见 [snapshot-053 §11](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md#11-053a-continue-iter-200500--plateau-style-stability--verdict-2026-04-19) | Claude |
| 2026-04-19 ~22:30 | **053Acont rerun verify → plateau confirmed**: ckpt 310 rerun 0.891 (orig 0.901) → combined 0.896; ckpt 430 rerun 0.888 (orig 0.903) → combined 0.8955。**两个 ckpt 跨 120 iter 都 ≈ 0.896 verified** — plateau 是 real 不是 single-shot artifact。Combined 0.896 vs 043C' 0.8965 / 043B' 0.893 / 034E 0.892 全部 within 1σ tied，但 053Acont 的优势是**两个独立 ckpt 都达此水平** (不是 single-point peak)。仍未 decisively 越过 0.900 (CI 上界 0.910)，但比 053A@190 (0.891) 真有 +0.5pp。Stage 2 capture peak ckpt 430 在 plan。 | Claude |
| 2026-04-19 ~22:22 | **🚨 053A continue Stage 1 1000ep — plateau-style breakthrough (PRELIM)**: 053A@190 续训 200→500 iter (PBRS combo) 完成; top 8 ckpts 1000ep 全部 mean 0.8886, **2/8 ≥ 0.900** (310=0.901 / 430=0.903), 7/8 ≥ 0.880, 1/8 (320=0.872 dip)。**质变**: 不再是过去那种 single-point lucky peak (043B 0.904 single → rerun 0.882; 053A@190 0.907 → 0.873), 而是跨 130 iter 的稳定 plateau cluster. PBRS combo 续训证明 reward signal compound 有效。Rerun for 310/430 + Stage 2 capture 已 launch on 5015751 port 65405。详见 [snapshot-053](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md) §11 (待 verdict 写入)。 | Claude |
| 2026-04-19 ~20:18 | **031B@1220 fairness rerun 完成**: rerun 0.877 (orig 0.882) → combined 0.8795 (n=2000, SE 0.007, CI [0.866, 0.894]); 真值 = **0.880** (single-shot 高估 +0.2pp)。完整 combined 2000ep leaderboard: 🥇 043A' 0.900 / 🥈 043C' 0.8965 / 🥉 043B' 0.893 / 034E 0.892 / 053A 0.891 / 031B 0.880。Δ from 031B base: 053A +1.1pp, 034E +1.2pp, 043B' +1.3pp, 043C' +1.7pp, 043A' +2.0pp。Self-play (043) 与 reward (053) 与 ensemble (034) 三 path 顶端在 [0.891, 0.900] 紧排, **没有 path 独占巨大优势** — 主路径全部 saturate 在 ensemble level。突破 0.90 需要更激进的方向 (continue training / distillation / RND / curriculum)。 | Claude |
| 2026-04-19 ~20:14 | **053A H2H 三轴 verify 完整**: (1) vs 034E ensemble n=500: **0.492 z=-0.36 TIED**. (2) vs 031B@1220 n=500: **0.570 z=3.13 `**`** (+7pp PBRS gain). (3) vs 029B@190 n=500: **0.634 z=6.0 `***`** (+13.4pp 决定性击败 per-agent SOTA)。**053A 现成为项目 H2H validation 最强 single-model**: TIED ensemble, 显著 > warmstart base, 决定性 > per-agent SOTA。**H_055 "single = ensemble" CONFIRMED on triple axes**, PBRS reward path 是 valid 突破方向 (同档 self-play 043 系, 但成本 200 iter vs multi-lane)。详见 [snapshot-053 §10.5](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md#105-053a-h2h-portfolio-总结-单-model-在三个判据维度的-verification) | Claude |
| 2026-04-19 ~20:10 | **053A H2H 双轴 verify 完成**: (1) 053A@190 vs 034E ensemble n=500: **0.492 z=-0.36 TIED** → 单网络 PBRS = 3-way ensemble within noise. (2) 053A@190 vs 031B@1220 n=500: **0.570 z=3.13 `**` significant** → PBRS combo 在 200 iter 内给 031B 带来真实 +7pp H2H 增益 (远超 baseline 1000ep 上看到的 +0.9pp marginal)。**整合**: 053A 在 baseline (combined 0.891 ≈ 034E 0.892) + peer H2H (vs 034E TIED) 双轴都达到 ensemble 等效水平 → **H_055 "single = ensemble" stretch goal 双轴 confirmed**。详见 [snapshot-053 §10](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md#10-stage-3-h2h--single-model-pbrs-vs-ensemble-2026-04-19) | Claude |
| 2026-04-19 ~19:55 | **034E ensemble rerun 完成 → 公平排名 final**: 034E rerun 0.894 (orig 0.890) → combined 0.892。**Final 2000ep ranking**: 🥇 043A'@080 0.900 / 🥈 043C'@480 0.8965 / 🥉 043B'@440 0.893 ≈ 034E 0.892 ≈ 053A@190 0.891。**关键事实**: 只有 043A'@080 严格 ≥ 0.900 grading (但 CI 下界 0.886 仍跨阈值, 不算 decisive); 其他 4 lane 在 [0.891, 0.897] tied cluster。**053A single-model PBRS = 034E ensemble equivalent (within noise) — H_055 stretch confirmed**; 在 2000ep 标准下 self-play (043 系) 与 PBRS reward path (053) 同档, 没有路径独占优势。 | Claude |
| 2026-04-19 ~19:53 | **公平 rerun 完成 + 全 top candidate 重排**: user 提醒不公平比较, launch 043A'/B'/C' rerun。结果: **043A'@080 = 0.900** (orig 0.901 + rerun 0.899, 最稳定), **043C'@480 = 0.8965** (orig 0.895 + rerun 0.898), **043B'@440 = 0.893** (orig 0.904 + rerun 0.882, **rerun 揭示原 0.904 高估 +1.1pp**), **053A@190 = 0.891** (n=2500), **034E rerun in-flight**。**关键修正**: combined 2000ep 下唯一 ≥ 0.900 的是 043A'@080 (CI [0.886, 0.914])。**single-shot result 系统性偏高约 1pp** — 这是 1000ep SE ±0.016 + 单 shot 选择偏差的结合。后续所有 1000ep 应当至少 rerun 一次再写 verdict。 | Claude |
| 2026-04-19 ~19:50 | **公平 rerun 校准 (in-flight)**: user 提醒 053A 用 2000ep 标准而其他 top candidate 都是 1000ep single-shot, 不公平。Launch 043A'@080 / 043B'@440 / 043C'@480 各 1000ep rerun (port 64005), 034E ensemble rerun pending (separate module). 结果待入库后所有 top candidates 都用 combined 2000ep 重新排名。 | Claude |
| 2026-04-19 ~19:43 | **`053A@190` Stage 4 rerun + capture 三向交叉验证**: orig 1000ep 0.907 + rerun 1000ep 0.873 + capture 500ep 0.894 → 真值收敛到 ~**0.890** (combined n=2500, SE ~0.006, 95% CI [0.878, 0.902])。**Verdict**: 单 shot 0.907 是 +1.5σ luck on true 0.890; 053A 真实 peak 与 034E ensemble (0.890 single shot) 等价 = **H_055 stretch goal confirmed: 单网络 PBRS 达到 ensemble 等价水平**。0.890 仍在 1000ep SE 范围内不能严格 declare > 031B 0.882, 但 053A 用 ckpt 190 (last ckpt, 50ep 0.96 上升中) 暗示 training 提前结束 — Continue training 200→500 iter 已 launch on 5015751 测试 peak 上限。详见 [snapshot-053 §9.5](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md#95-stage-4-rerun-结果-完成-1943-edt) | Claude |
| 2026-04-19 ~19:38 | **`053A@190` single-shot read 0.907**: 053A (031B@1220 warmstart + 200 iter combo: v2 shaping + outcome PBRS λ=0.01) ckpt 190 1000ep = **0.907 (907W-93L)**。其他 ckpts (130/140/150/180) 都在 [0.869, 0.877]。**Stage 4 rerun 决定真值** — 见上一行 entry。 | Claude |
| 2026-04-19 ~23:15 | **`043C'@480` follow-up 入库**：更重 frontier 的 `043C'` 现已补完 `failure capture + peer H2H + direct compare`。结果为：`capture 500 = 0.880`、`H2H vs 031B@1220 = 0.614 ***`、`vs 034E-frontier = 0.576 ***`，并在 direct H2H 中对 `043B'@440` 给出 `0.532` 的窄优势（从 `043B vs 043C = 0.468` 转写，未显著）。当前口径收紧为：`043B / 043C` 组成 `043` 家族的双主候选，其中 `043B` baseline 尖峰更高，`043C` 更 rounded。 | Claude |
| 2026-04-19 ~22:20 | **`043B'@440` 入库**：`043A'@80` 续跑的 harder curriculum (`baseline 40% + frozen 031B/029B/036D`) 经 `official baseline 1000` shortlist 复核后，best 为 **`043B'@440 = 0.904`**。follow-up 进一步给出 `failure capture 500 = 0.852`、`H2H vs 031B@1220 = 0.600 ***`、`H2H vs 034E-frontier = 0.554 **`。当前口径：`043B'` 已把 `043` 从 baseline-oriented curriculum 推进成 overall frontier candidate，但仍带着 `late_defensive_collapse + low_possession` 型脆点。 | Claude |
| 2026-04-19 ~17:30 | **SNAPSHOT-043A' 入库**：`031B@1220` warm-start + diversity curriculum (`baseline 50% + 031A 20% + 029B 15% + 036D 15%`) formal 已完成。`official baseline 1000` 并行 shortlist 中 best 为 **`043A'@080 = 0.901`**，另有 `440=0.895 / 280=0.894 / 190=0.891`，说明不是单点孤峰。follow-up：failure capture `500 = 0.922`、`H2H vs 031B@1220 = 0.524`（窄优势、未显著）、`H2H vs 034E-frontier = 0.442`（显著负号）。当前口径：`043A'` 是 **baseline-oriented frontier candidate**，但 overall 主线仍应保留给 `034E-frontier`。 | Claude |
| 2026-04-19 ~09:40 | **SNAPSHOT-034 第二代 frontier ensemble 入库**：新增 `034C-frontier / 034E-frontier` registry 与 baseline / H2H 记录。关键结果：`034E-frontier official 500/1000 = 0.904 / 0.890`，`vs 031B@1220 = 0.596 ***`，`vs 029B@190 = 0.590 ***`，`vs 034C-frontier = 0.544 *`。当前最稳口径：`034E-frontier` 是 **ensemble 主候选**，但仍缺 `vs 031A@1040` 直连 H2H，暂不把它写成“对所有旧 frontier 全部坐实碾压”。同时补入 `031B@1220 official 1000 = 0.882` 到 baseline 高置信度轴，明确它是当前 strongest single model。 | Claude |
| 2026-04-18 | 初始创建；汇总 029B / 030A / 030D / 032 / 033 系列所有 H2H；标注方向验证规则（§0.2） | Claude |
| 2026-04-18 | 撤回 SNAPSHOT-037 的"架构层翻转"错读；恢复 frontier H2H 的保守层级读法（不再把 `029B` 写成已坐实压过 `025b`） | Claude |
| 2026-04-18 | 严格区分 official 500 / capture 500 / H2H；删除把 official 与 capture 混成“1000ep”的错误写法；修正 `032A vs 032Ac` 排名方向，补入 `029B vs 025b` 直连 H2H，并为所有已测 H2H 补齐 `n / z / p` | Claude |
| 2026-04-18 | 新增 `official baseline 1000` 专节与 ranking，明确区分 `official 1000`、`capture 500` 与 `combined 1000`；补入 `029B / 025b / 017 / 030A / 030D / 028A / 032 / 033 / 032-next` 的当前 `1000ep` baseline 结果，并把 `032nextC@130 / 032nextA@110` 纳入 registry | Claude |
| 2026-04-18 ~15:50 | [SNAPSHOT-038 Stage 1](snapshot-038-team-level-liberation-handoff-placeholder.md#10-2026-04-18-1550-edt-stage-1-四条-lane-首轮结果) 四条 lane 1000ep 首轮结果入库：`038A/B/C/D` 每条 top 5 checkpoint 均测完。1000ep max 从 0.796 (038A) 到 **0.806 (038D @40 & @60)**，全部在 `028A@1060` warmstart (0.783) 的 ±CI 内。`038D` 相对其他 3 条有 +0.6–1.0pp 方向性优势但单次 1000ep SE ~0.013 内。所有 038 点均明显低于 `029B@190` (0.846) 4–6pp。按预注册 §8.3 Stage 2 激活条件，`038D` 位于边界，暂缓 handoff，先做 failure capture 再决 | Claude |
| 2026-04-18 ~16:55 | [SNAPSHOT-039 §9](snapshot-039-airl-adaptive-reward-learning.md#9-首轮结果2026-04-18-1647-edt) 首轮结果入库：训练 300 iter 完成无 inf；但 callback 中 `trainer.workers.local_worker().env` 没找到 `LearnedRewardShapingWrapper`，**10 次 refresh 全部跳过 broadcast** → 实际等价 036D-style 静态稳定化（λ=0.003 + warmup + finite-check）。1000ep max **0.843 (ckpt 230)**, mean 0.829，比 036C 略好 +0.5–1.0pp（来自稳定化，不是 adaptive），与 029B@190 warmstart 0.846 在 SE 内追平但**未超越**。`H_039` 实际未被测试 | Claude |
| 2026-04-18 ~17:00 | [SNAPSHOT-036D §10](snapshot-036d-learned-reward-stability-fix.md#10-首轮结果2026-04-18-1700-edt) 首轮结果入库：稳定性 fix **完全成功** —— 0/300 iter inf（vs 036C 的 16%）。1000ep mean **0.843** / max **0.860 (ckpt 150)**。**首次 learned reward fine-tune 在 1000ep 上 max 超 029B@190 warmstart (0.846)** by +1.4pp，单点 marginal；3 个 ckpt (130/150/250) 都 ≥ 0.848 不是单峰。还未达到 §3 的 ≥0.87 阈值（"真实增益"），更未达 9/10 (0.90)。**待**: H2H vs 029B@190、failure capture、1000ep 重测确认 0.860 不是抽样上限 | Claude |
| 2026-04-18 ~20:35 | **更正**：036D inf 率不是 0/300。重新扫 progress.csv 实测 **036D 95/300 (31.7%) inf**，**039 85/300 (28.3%) inf**。`λ↓ + warmup + finite-check` **没消除 KL 爆炸**，反而比 036C 的 16% 还高。WR 更好的实际机制是 PPO grad_clip 把 inf gradient 变成无效 update + finite-check 防 NaN 流到 advantage，**不是阻止 inf 发生**。[snapshot-036D §10.1/§10.5/§10.7](snapshot-036d-learned-reward-stability-fix.md) 和 [snapshot-039 §9.1](snapshot-039-airl-adaptive-reward-learning.md) 已修正 | Claude |
| 2026-04-18 ~20:50 | **031A 入库** ([snapshot-031 §11](snapshot-031-team-level-native-dual-encoder-attention.md#11-首轮结果031a2026-04-18))：scratch dual-encoder team-level，1250 iter。1000ep mean **0.847** / max **0.867 (ckpt 1040)** —— **项目首个 1000ep max 上明确超 029B@190 warmstart (0.846) 的非 learned-reward 结果**，超 +2.1pp on max。两 ckpt (1040+1170) 都 ≥ 0.865。但**未做 H2H**, 单次 1000ep；不能武断说 SOTA。同时 **036D@150 vs 029B@190 H2H 1000ep = 0.507 (z=0.44, p=0.33, 平局)**，确认 036D 的 +1.4pp baseline 优势不转化为 peer 优势。**036D failure capture**: `wasted_possession 38.6%` (vs 029B 47.8%, **-9.2pp**), `unclear_loss 10.2%`(无 gaming), L mean 26 步(无 turtle)。结构上 036D 真改善了 wasted_possession 这个一开始的 bottleneck | Claude |
| 2026-04-18 ~21:55 | **031A 三连验证完成 → 项目新 SOTA**: (1) 1000ep rerun: ckpt 1040 0.867→0.853 (2000-game avg **0.860**), ckpt 1170 0.865→0.850 (avg 0.858)；真实 baseline ~0.860。(2) **H2H vs 029B@190 (n=1000): 552-448 = 0.552, z=3.29, p=0.0005 `***`** —— 显著击败 029B；blue 0.580/orange 0.524 双侧>0.5。(3) Failure capture: `progress_deficit 15.3%` (frontier 最低), `wasted_possession 42.4%` (vs 029B 47.8%), `unclear_loss 12.9%` (无 gaming), L mean 34.4 步无 turtle。**031A 是项目首个同时在 baseline 和 peer 两个判据上都击败 029B@190 的 lane**。caveat: scratch dual-encoder + team-level，方法栈 vs 029B 完全不同；vs 025b/036D 等 H2H 仍未测 | Claude |
| 2026-04-18 | 补入 `036C` 首轮结果：把 registry 从泛化的 `036C-warm` 收紧为 `036C@150 / 036C@270`，新增 `official baseline 1000` (`0.832 / 0.833`) 与 `029B@190 vs 036C@270 = 0.578` 的 peer H2H，并把 open question 更新为“先修数值 `inf` 再决定是否继续 peer rerun” | Claude |
| 2026-04-18 ~22:30 | **031A@1040 vs 025b@080 H2H** ([snapshot-031 §11.6](snapshot-031-team-level-native-dual-encoder-attention.md#116-h2h-1000ep-vs-029b190--025b080关键验证已完成)): `532-468 = 0.532` (n=1000, z=2.024, p=0.0215, `*`)。边缘显著但方向稳定（blue 0.538 / orange 0.526 双侧 >0.5）。这是**031A 对 frontier 的双 SOTA 同向赢**——既击败 029B@190（baseline-axis 头名）也击败 025b@080（peer-axis 隐性头名）。优势 vs 025b (+0.032) 比 vs 029B (+0.052) 弱 -2pp，符合 "025b 在 peer 轴上略强 029B" (直连 0.508) 的既有读法。031A SOTA 地位坐实 | Claude |
| 2026-04-18 ~23:00 | **031A@1040 vs 028A@1060 H2H** ([snapshot-031 §11.6.1](snapshot-031-team-level-native-dual-encoder-attention.md#1161-vs-028a1060-的特殊意义同-reward-同-base-直接架构对比)): `568-432 = 0.568` (n=1000, z=4.301, p<0.0001, **`***` 三连最强**)。同 reward (v2) 同 base 类型 (team-level) 不同架构（Siamese dual encoder vs flat MLP）的最干净对照。+6.8pp 的 H2H 增益和 baseline `0.860 vs 0.783 = +7.7pp` 数量级一致，**架构改造真实 H2H 优势已坐实**。blue 0.606 / orange 0.530 都 >0.5（侧别不对称 +7.6pp 但方向一致）。caveat: 训练时长 1250 vs 1060 iter，架构 vs 训练时长贡献分离仍待 028A 长跑对照 | Claude |
| 2026-04-18 ~23:05 | [SNAPSHOT-040](snapshot-040-team-level-stage2-on-031A.md) 已开 — team-level Stage 2 handoff on 031A SOTA base：把 026/038 体系内的 advanced shaping (PBRS / event / depenalized v2 / entropy-only) 叠加在 `031A@1040 (1000ep 0.860)` 这个**项目首个对齐 per-agent 平台的 team-level base** 上，复现 per-agent `026 → 029` 路径上的 +2.6pp 增益。四条 lane (`040A/B/C/D`) 镜像 026/038；首轮先跑 `040B (PBRS handoff)`。同时 [SNAPSHOT-031 §12](snapshot-031-team-level-native-dual-encoder-attention.md#12-031-b-激活cross-attention2026-04-18-预注册) `031-B` 预注册激活（cross-attention 设计/超参/判据已就位） | Claude |
| 2026-04-19 ~03:00 | **SNAPSHOT-040 verdict — saturation 假设确认** ([snapshot-040 §11](snapshot-040-team-level-stage2-on-031A.md#11-首轮结果2026-04-194-lane-全部完成))：4 条 lane (`040A/B/C/D`) 全部完成 200 iter + 1000ep eval。peak WR 全部在 [0.863, 0.865]，**与 031A@1040 base (0.860) 在 1000ep SE ±0.016 内** —— shaping (PBRS / event / depenalized / entropy) 在 031A high base 上 **无统计意义增益** (+0.003 到 +0.005pp)。预注册 §5.1/§5.2 全部未达 (0.875 / 0.88)。**§6.2 R2 saturation 风险预言成真**。040 路径终结，跳过 capture/H2H 节省 GPU。下一步: [041 per-agent 镜像](snapshot-041-per-agent-stage2-pbrs-on-036D.md) (036D 学到的 +1.4pp 可能没饱和) / [042 cross-arch transfer](snapshot-042-cross-architecture-knowledge-transfer.md) / [044 specialist league](snapshot-044-adversarial-specialists-league.md) | Claude |
| 2026-04-19 ~03:30 | **SNAPSHOT-041 verdict — 路径否决，唯一退化** ([snapshot-041 §11](snapshot-041-per-agent-stage2-pbrs-on-036D.md#11-首轮结果2026-04-19041b-完成))：041B (per-agent Stage 2 PBRS handoff on 036D@150) 完成 200 iter + 1000ep eval。6 个 ckpts (10/20/30/40/50/60), peak **0.852 @ ckpt 60**, mean 0.834。**全部低于 036D base 0.860 (-0.008 退化)**。Stage 2 PBRS + 036D 双 shaping 叠加反而冲掉 036D 学到的 +1.4pp 优势。预注册 §5.1/§5.2 全部未达。**041 路径终结**。**关键观察 (跨 040 + 041 合计 5 个 lane)**：所有 v2-derivative lane peak 落在 [0.852, 0.865]，跨架构跨 reward 路径**全部打不破 ~0.86 ceiling**。v2 shaping 本身可能是 bottleneck，下一步需考虑 MaxEnt / pure sparse / learned reward only 路径绕开 v2 prior | Claude |
| 2026-04-19 ~03:30 | **SNAPSHOT-044 verdict — specialist gate fail, 设计缺陷暴露** ([snapshot-044 §11](snapshot-044-adversarial-specialists-league.md#11))：044A spear 1000ep peak win_rate 0.766 < gate 0.85 (fast_win_rate 0.716)。044B shield 1000ep peak non_loss_rate 0.790 < gate 0.85，**3500 episode 全 0 ties** (baseline 太 aggressive 不允许 stalemate) → shield reward `R(T)=+1` 从未触发 → policy 退化为 offensive (median win step 35 = 同 spear)。两 specialist gate 双 fail。**根因诊断**：(a) baseline env 不支持 ties；(b) shield "defensive" dense shaping 实际是隐藏 offensive shaping；(c) FAST_WIN_THRESHOLD=100 太宽松，绝大多数 vanilla wins 已经 ≤100 步。**044C league 推荐合并到 043** (drop specialist, pool ≈ 043 frontier-only) | Claude |
| 2026-04-18 ~22:00 | **038 vs 026 跨架构对比** ([snapshot-038 §10.11](snapshot-038-team-level-liberation-handoff-placeholder.md#1011-errata-2026-04-18-同-shaping-跨架构对比-038-team-level-vs-026-per-agent))：user 指出 038 是 026 的 team-level 镜像，正确比较应是同字母跨架构。**关键 caveat: 026 起点 BC@2100 = 0.842（接近 per-agent 平台天花板），038 起点 028A@1060 = 0.783（团队架构 base 差 5.9pp 起跑）**。绝对 WR: 038 落后 1-7pp，但大部分是 base 差距继承。**per-base 增量 (Δ over own warmstart base) 对比反而翻转**: A 026 -0.032 vs 038 +0.013 (+4.5pp ↑team), B 026 +0.022 (peak) vs 038 +0.014, C 026 +0.004 vs 038 +0.017 (+1.3pp ↑team), D 026 -0.018 vs 038 +0.023 (+4.1pp ↑team)。**修正读法: shaping 在 team-level 上边际效用至少不差于 per-agent**（也可能是 026 base 已饱和的 artifact，要分清需要在 0.84+ 的 team-level base 上重测）。修正前的"team-level shaping 失效"结论收回 | Claude |
| 2026-04-19 ~04:30 | **SNAPSHOT-044 verdict — specialists 双 fail** ([snapshot-044 §11](snapshot-044-adversarial-specialists-league.md#11-首轮结果044a-矛--044b-盾2026-04-19))：200 iter scratch 训练完成，post-train eval 1000ep + 500ep specialist-objective capture。**044A 矛 fail**: fast_win_rate 上界 = 1000ep peak win_rate = 0.766 < 0.85 阈值 (-13.4pp)，500ep capture 实测 0.716；mean win L=44.9 步 ≤ 80 ✓。**044B 盾 fail**: 1000ep peak non_loss_rate = win_rate = 0.790 < 0.85 (-6.0pp)；3000+500 ep 累计 **ties = 0**（baseline 对手在 1500-step cap 内基本不和棋，shield 任务环境层面 infeasible）；0 ties 与 044A 高度一致的 win 步数 (median 35 vs 37) + 控球分布 (mean 49% vs 50%) 表明 shield reward `R(T)=+1` 没有产生差异化行为。按 §6.1 gating，044C league pool fallback 到"specialists fail"列（baseline 40% + 4 frontier 各 15%），与 SNAPSHOT-043 几乎等价。建议合并到 043 lane 不重复 GPU。Stage 0 infra 改动: evaluate_matches.py / capture_failure_cases.py 增加 `wins / fast_wins / ties` save-mode + atomic edit；pick_top_ckpts.py 新增 `--metric` flag 支持 specialist lane 选模 | Claude |
| 2026-04-19 | **第一性原理复盘 + v2 失败桶证据 → SNAPSHOT-034 优先级提升 + 新增 SNAPSHOT-047**：5+ lane saturate 在 [0.852, 0.865] 后复盘——0.86 不是 PPO ceiling 而是单点优化天花板。用 [v2 桶](../../cs8803drl/imitation/failure_buckets_v2.py)（修正 v1 阈值偏见）重分类 SOTA failure capture: **031A@1040 wasted_possession 主导 (tail_x +1.37, poss 0.50)** vs **036D@150 defensive_pin 主导 (tail_x -4.9, poss 0.45)** —— 两个 0.86 SOTA 失败模式真正正交，强化 PETS ensemble 期望增益（v1 旧证据基础上的 +1）。[SNAPSHOT-034](snapshot-034-deploy-time-ensemble-agent.md) 从「预注册」推到「优先实施」（与 045/046/039-fix 并行不冲突 GPU）。新增 [SNAPSHOT-047](snapshot-047-deployment-slot-swap-test.md) 作为 ensemble 前置 sanity check —— [SNAPSHOT-022 §R1](snapshot-022-role-differentiated-shaping.md) 早提出 0/1 deployment slot 一致性 risk 但从未数据化，047 在 4 个 SOTA (031A/036D/029B/025b) 上跑 normal vs swap 1000ep，1-2h 解锁 ensemble 1 天工程 | Claude |
| 2026-04-19 | **新增 SNAPSHOT-048 (DAGGER 上限测试) + SNAPSHOT-049 (env state restore verdict)**：MCTS / planning 路径经 6 个 smoke test 验证后**关闭** ([SNAPSHOT-049](snapshot-049-env-state-restore-investigation.md)) — Unity binary 接受 ball position/rotation SET 但 player position SET 即使 pre-reset 也失败（C# 端没实现 handler）。Full MCTS over agent actions 不可行。Pivot 到 [SNAPSHOT-048](snapshot-048-hybrid-eval-baseline-takeover.md) hybrid eval：1 天工程 (SwitchingAgent + 6 conditions × 1000ep) 给 DAGGER 路径的**上限**。判据 ≥+0.03 启动 DAGGER；≤+0.01 路径死。049 副产物：`info["ball_info"]["position"]` 可拿 absolute coords，降低 048 trigger state 提取风险 | Claude |
| 2026-04-19 ~07:23 | **SNAPSHOT-031B Stage 1 verdict — 项目新 SOTA 0.882** ([snapshot-031 §13](snapshot-031-team-level-native-dual-encoder-attention.md#13-031b-首轮结果-2026-04-19append-only)): 031B (cross-attention scratch + v2, 1250 iter scratch) Stage 1 完成，24 ckpt × 1000ep eval (top 5%+ties+±1)。**peak ckpt-1220 = 0.882 (882W-118L, n=1000, SE ±0.010, 95% CI [0.862, 0.902])**，1230=0.881, 1240=0.872 三个 late-window 全 ≥ 0.872 — 真稳定 peak。**vs 031A (0.860): max +2.2pp, mean +0.5-1pp**。**首次 1000ep cross 0.88+，距 0.90 grading 仅 -1.8pp**。验证 [snapshot-027 §9](snapshot-027-team-level-ppo-coordination.md) "late-window > 50ep spike" doctrine: 内部 0.96 @ 590 → 1000ep 0.856 (-0.104 drift)。**0.86 ceiling 论被反驳**: 040(4)+041+044 saturate 在 [0.852, 0.865] 是 31A 架构上限不是 PPO 上限，cross-attention 推到 0.88。数值健康: 1203 iter 0 inf (vs 036/039 16-32% inf, 因 v2 解析 shaping vs learned reward NN). Stage 2 (failure capture @1220) + Stage 3 (H2H vs 031A@1040 / 029B@190) pending | Claude |
| 2026-04-19 ~07:11 | **SNAPSHOT-042A3 Stage 1 verdict — KL distill 稳但未突破** ([snapshot-042 §11](snapshot-042-cross-architecture-knowledge-transfer.md#11-042a3-首轮结果-2026-04-19append-only)): 042A3 (KL distill from 029B@190 onto 031A@1040, 200 iter) Stage 1 完成，5 ckpt × 1000ep。**peak ckpt-80 = 0.863, mean across 5 ckpt = 0.857, 极窄 ±0.008**。**vs 031A (0.860) max +0.003pp 持平**。trajectory 极稳（5 ckpt range 0.846-0.863 vs 031B 同 budget 范围 0.06）。max\|kl\|=2.32 (vs 031B 10.71) — KL distill anchor 把 policy 拽在 029B 周围不大 drift。verdict: KL distill 适合做 stability regularizer 不是 SOTA breakthrough lane。Stage 2/3 skip（结果已定型）。可叠 031B + KL distill 看能否同时拿稳定 + breakthrough（snapshot-051 候选）| Claude |
| 2026-04-19 ~09:08 | **031B vs Random 1000ep verify + snapshot-039 §11 Fix-B 落地**: (1) 031B@{1220,1230,1240} × 1000ep vs Random on 013-19-0 port 58005, 149s elapsed: **0.990 / 0.994 / 0.994** ([snapshot-031 §13.11](snapshot-031-team-level-native-dual-encoder-attention.md#1311-vs-random-1000ep-verify-2026-04-19-tier-12-落地))。Grading 第二条 (Random 9/10=0.90) **robust pass** — 10ep submission P(≥9wins|p=0.99) ≈ 0.996. 031B 双判据 single-run P ≈ 0.65; multi-seed (N=3) P 提升到 0.957。(2) [snapshot-039 §13](snapshot-039-airl-adaptive-reward-learning.md#13-fix-b-落地2026-04-19append-only) Fix-B (callback sanitize learner_stats inf/nan + 写 custom_metrics inf 计数器) 实施 + smoke PASS。未来 039 重训可直接 plot `airl_inf_total_kl` 观测 inf rate，不需 post-hoc CSV scan。Fix-A (KL inf 根因 fix) 仍 backlog | Claude |
| 2026-04-19 ~08:50 | **新增 SNAPSHOT-050: Cross-Student DAGGER Probe** ([snapshot-050](snapshot-050-cross-student-dagger-probe.md)): 048 §7.4 strategic 延伸——把 baseline-takeover 换成 frontier teacher。**Phase 1.1 (036D → 031A) 100ep 完成**: Δ +1pp (sub-noise), vs 048 baseline -10pp 改善 +11pp，验证 H1 (teacher quality 与 takeover Δ 单调正相关)。但 +1pp 也 sub-noise → 036D-as-teacher 对 031A NEUTRAL，没拿到 cross-student DAGGER 的有效证据。Phase 1.2 (031B → 031A 架构内 stronger teacher, 011-23-0 port 64401) + Phase 1.3 (031A → 036D 失败模式互补的反向 pair, 015-30-0 port 65401) 进行中。**Phase 2 (真 DAGGER training) 启动条件 = Phase 1 任一 Δ ≥ +0.03**。复用 048 evaluator (已加 `--takeover-{module,checkpoint}` 参数化) | Claude |
| 2026-04-19 ~18:30 | **SNAPSHOT-052 verdict — 架构 step 3 决定性 REGRESSION, lane 关闭** ([snapshot-052 §7](snapshot-052-031C-transformer-block-architecture.md#7-verdict--架构-step-3-决定性-regression-2026-04-19-append-only)): 两条 sub-line 都 1250 iter scratch + cross-attention 架构上加 transformer block 元素。**052A (031C-min, FFN+LayerNorm+residual only) 1000ep peak = 0.800 @ ckpt 1080 (vs 031B 0.882 = -8.2pp, REGRESSION)**; **052 (full 031C, +true MHA + 1024→512 merge) peak = 0.774 @ ckpt 870 (vs 031B = -10.8pp, REGRESSION)**。Decomposition: refinement block 净 -8.2pp + MHA/merge 额外 -2.6pp。**架构 diminishing returns 翻转**: step 1 +7.7pp (Siamese) → step 2 +2.2pp (cross-attn) → step 3 **-8 ~ -11pp** (transformer block)。Stage 2 capture: **052A failure = possession_stolen +9.6pp (aggressive-but-fragile)**, **052 failure = wasted_possession +12.5pp (possession-heavy-but-wasteful)**。两个都不是「学不会」, 是「学到不同的弱 policy」。**架构 step 3 lane 关闭, 0.90 突破路径转 053 PBRS / 046E ablation / self-play league** | Claude |
| 2026-04-19 ~17:25 | **SNAPSHOT-053A 启动 — outcome-PBRS 取代 multi-head bucket reward** ([snapshot-053](snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md)): paradigm shift 从 episode-summary bucket 多头分类 → per-step trajectory transformer outcome predictor + PBRS ΔV(s) reward。Direction 1.b path: prototype 78.8% (2000ep) → v2 retrain 93.8% full seq (15000ep, 但 prefix 短时 miscalibrated overconfident 0.84) → A3 calibrated retrain (random prefix truncation) val_acc 0.835 + per-prefix gap 0.015→0.240 evolves correctly → wire 进 PBRS wrapper (cs8803drl/imitation/outcome_pbrs_shaping.py) + utils.create_rllib_env integration。**053A combo (v2 + outcome PBRS λ=0.01) on 031B@1220 warmstart**, 200 iter, launched on 011-23-0 PORT_SEED=23. **判据 vs 051A 0.888**: ≥0.892 BREAKTHROUGH / 0.880-0.892 MARGINAL / <0.880 REGRESSION | Claude |
| 2026-04-19 ~16:15 | **SNAPSHOT-046E verdict — sample-inefficient, hypothesis 部分否定** ([snapshot-046 §13](snapshot-046-cross-train-pair.md#13-046e-verdict--sample-inefficiency-confirmed-hypothesis-部分支持-2026-04-19-append-only)): 760 iter scratch + cross-attn arch + vs frozen 031B@1220 完成。100ep peak baseline = **0.810 @ ckpt 620** (vs 031B 0.882 = -7.2pp)。H2H vs 031B trajectory **0.148→0.294→0.372→0.382→0.420→0.476** across iter 50/150/300/450/600/750 — 单调上升, 没 over-specialization, 但 **750 iter 才到 0.476 vs 031B**。**Sample-eff 比 vs-baseline 训练慢 4-5×** (031B 自己 ~200 iter 到 0.86 baseline)。原因: vs 强对手 → student 弱时大量 sparse 负 reward → 学习极慢。**Hypothesis "vs 强 opp 加快 generalization" 部分否定**, 但路径不死: 需要 curriculum / pool / dense reward (Direction 1.b option A) 任一来 fix sample efficiency。**046 单一固定强对手 lane 关闭, 等 031B-noshape verdict 后决定是否走 1.b + 强 opp 组合** | Claude |
| 2026-04-19 ~13:10 | **SNAPSHOT-034 §11 + SNAPSHOT-045 §13 双重 verdict — ensemble 突破路径耗尽, 045A 是 noisy clone** ([snapshot-034 §11](snapshot-034-deploy-time-ensemble-agent.md#11-034ea--034eb-反向验证--架构-diversity--v2-桶-fingerprint-2026-04-19append-only) / [snapshot-045 §13](snapshot-045-architecture-x-learned-reward-combo.md#13-045a-h2h-stage-3-verdict--045a-是-031a-noisy-clone-不是新-model-2026-04-19append-only)): (1) **034ea {031B+045A+051A} = 0.878 anti-lift -0.1pp (个体均值 0.879)** vs 034E 0.890; **034eb 4-way 加回 036D = 0.882 (+0.8pp lift)** 仍低于 034E。证明 ensemble lift 来源是**架构 family diversity** (team-level + per-agent), 不是 v2 桶 fingerprint。(2) **045A H2H ×4 (n=1000)**: vs 031A=0.492 NS, vs 031B=0.491 NS, vs 036D=0.570 ★★★, vs 029B=0.575 ★★★。**vs 031A 平 = 045A 是 031A noisy clone, learned reward 没产生新 policy direction**; vs per-agent +7pp 是继承 031A 架构优势, 不是 045A 本身价值。(3) **v2 桶 fingerprint ≠ policy diversity** — episode-level statistic 跟 step-level decision boundary 解耦。**Meta-lesson (用户反复强调)**: ensemble = stability/cost optimization, 不是智力提升; 突破 0.90 必须靠**单 model 架构突破** (052A 031C in-flight), 不能靠继续找 ensemble candidate。**034 / 045 lane 全部 freeze**, 不再 launch H2H/capture/新组合 | Claude |
| 2026-04-19 ~12:50 | **SNAPSHOT-039 lane 关闭 — Fix-B 实证为纯 cosmetic, KL inf 频率不变** ([snapshot-039 §14](snapshot-039-airl-adaptive-reward-learning.md#14-039fixb-stage-1-verdict--fix-b-是纯-cosmetic-kl-behavior-不变-lane-关闭-2026-04-19append-only)): 039fixB 300 iter formal 训练完成。progress.csv 直接对比 — Trial1 KL_inf% 10.5%, Trial2 KL_inf% 38.1%, **fixB KL_zero% (sanitized) 35.7%**, 三组 max\|KL\|finite 完全一致 (~3.8e-03), max\|TL\|finite 完全一致 (~8.5e-02)。Fix-B 只把 inf 在 logging 阶段抹成 0, **真实 inf 事件率 = sanitize 出的 0% 频率 ≈ 35.7% / 38.1%, 跟 inf 版本本质相同**。AIRL adaptive reward 真信号仍 disentangle 不开 — 35.7% iter 是 PPO grad_clip 兜底吃的「空 update」。50ep 内部 peak 0.92 @ 80, 预测 1000ep peak ≈ 0.85 (跟 039fix Trial2 0.852 同档), 不进 SOTA / ensemble candidate。**跳过 1000ep eval 释放节点**, 关 lane。Fix-A 下沉为 "031B / 034ea/eb / 052 / 046 全部用尽且仍 < 0.90" 才回头 | Claude |
| 2026-04-19 ~12:30 | **SNAPSHOT-051 Stage 2 verdict — 051A combo edge new SOTA single, reward source NOT leverage** ([snapshot-051 §8](snapshot-051-learned-reward-from-strong-vs-strong-failures.md#8-stage-2-verdict--051a051b-1000ep-2026-04-19append-only)): 051A/051B 各 7 ckpt × 1000ep on 011-28-0 / 013-19-0 (port 56355/56455, 271s)。**051A peak 0.888 @ ckpt 130 (early-peak, +130 iter from warmstart 031B@1220)** = +0.6pp marginal vs 031B base 0.882, statistically tied with ensemble 034E 0.890 (1000ep SE ±0.010, 都在 ±1σ 内)。**051B learned-only peak 0.872 @ ckpt 170 = -1.0pp vs 031B base 反而退化**（vs 045B 在 031A base 上 +1.0pp 反向, 暗示 strong base 已 internalize learned reward 的协调 signal, 叠加反而 drift）。**关键 falsification — reward model 数据源不是 leverage 点**：045 (baseline failures) 和 051 (strong-vs-strong) reward model 在 combo lane 上各给 +0.7pp / +0.6pp marginal, 几乎相同 → 重新训 reward model 的边际效益接近零。**051A 是 single-model 1000ep 历史最高 (0.888)**, 距 0.90 grading 仅 -1.2pp。早期 peak @ 130 暗示 saturation lock-in 在 architecture 而非 reward。051C/D/annealed 不再启动。**对 ensemble 034F: 051A 是新候选**（peak 高 + 大概率继承 045A 的 wasted_possession-主导 fingerprint, capture in-flight on 010-20-0 验证）| Claude |
| 2026-04-19 ~11:35 | **SNAPSHOT-050 Phase 1.3b 500ep verdict — borderline, Phase 2 暂缓** ([snapshot-050 §7](snapshot-050-cross-student-dagger-probe.md#7-phase-13b-verdict--500ep-rerun-on-free-node-2026-04-19append-only)): Phase 1.3 100ep 报 Δ +10/+12pp 在 500ep rerun 中缩水到 **C1_α=+0.032 (1.9σ 边缘) / C2_β=+0.016 (sub-noise)**。C0_036D_solo=0.830 比 036D 真 1000ep=0.860 低 1.5σ, 用 036D 1000ep 当 reference 时 Δ=+0.002/-0.014 sub-noise。结合 Phase 1.2 (031B→031A 100ep) **C2_β=-8pp** + Phase 1.1 (036D→031A 100ep) **+1pp**, 三个 phase1 lane 都没给 ≥+0.03 robust signal。**H3 弱化**（031B baseline +2.2pp 没 transfer 到 takeover, 与 [snapshot-031 §13.8](snapshot-031-team-level-native-dual-encoder-attention.md#138-stage-3--h2h-matrix-4-matchups-n500-each-2026-04-19-0742) peer H2H 0.516 NS 一致）。**H2 互补失败模式**仅 1.3b 给 borderline+ 方向证据。**Phase 2 (真 DAGGER training) 暂不启动**, lane 标记 borderline 不关闭, 等 045 capture / 051 1000ep verdict 后再排优先级 | Claude |
| 2026-04-19 ~11:30 | **SNAPSHOT-045B Stage 1 verdict — H_replace falsified, learned-only ≈ combo ≈ base** ([snapshot-045 §12](snapshot-045-architecture-x-learned-reward-combo.md#12-045b-首轮结果--learned-only-on-031a1040-2026-04-19append-only)): 6 ckpt × 1000ep on atl1-1-03-014-23-0 (port 56305, top 10% fallback per skill, 253s)。**peak 0.870 @ 150, mean 0.853**。vs 045A combo (0.867 / 0.856) **statistically tied** (\|Δ\| < 0.5σ)；vs 031A base (0.860) +1.0pp marginal **不显著** (1000ep SE ±0.016 内)。**关键意义**: 把 v2 完全去掉 (USE_REWARD_SHAPING=0)、learned reward 单独驱动，结果几乎不变 → falsify "v2 淹没 learned signal" 假设。**学习 reward 在 031A architecture 0.86 ceiling 上 marginal value ≈ 0pp**，跟 v2 在不在场无关，是 architecture-imposed saturation。reward-axis evidence 累积到 **7 个 lane (040A-D + 042A3 + 045A + 045B) 全 sub-noise**，"架构 > reward" 路径继续被加强。implication for 051 (in-flight): 若 reward model 数据源 (strong-vs-strong) 是 leverage 点 → 应明显 > 045；若 architecture ceiling 才是真正 bottleneck → 051 在 031B 0.882 上也会 saturate | Claude |
| 2026-04-19 ~08:33 | **SNAPSHOT-045A Stage 1 verdict — combo saturation H3 确认** ([snapshot-045 §11](snapshot-045-architecture-x-learned-reward-combo.md#11-045a-首轮结果-2026-04-19-append-only)): 8 ckpt × 1000ep on atl1-1-03-013-19-0 (port 62005, top 10% fallback per skill, 469s)。**peak 0.867 @ 180, mean 0.856**。vs 031A 0.860 = +0.7pp marginal gain，**1000ep SE ±0.016 内统计上不可区分**。对比 per-agent learned reward gain (036D 0.860 vs 029B 1000ep 0.846 = +1.4pp 边缘 z≈1.2) — **team-level 拿到约一半 gain**，可能因 Siamese 共享 encoder 已内化部分协调 signal。**031A architecture 上 6 个 reward-axis lanes 全 sub-noise**: 040A-D + 045A + 042A3 in range [0.860, 0.867]。**1 个 architecture-axis lane (031B cross-attention) 突破 0.882 (+2.2pp 显著)**。evidence ratio 6:1 for "架构 > reward" 路径。045B/045C 按预注册 §2.4 saturation gate **跳过**。clean (0 inf) | Claude |
| 2026-04-19 ~08:30 | **SNAPSHOT-039fix Stage 1 verdict (open, lane not closed)** ([snapshot-039 §12](snapshot-039-airl-adaptive-reward-learning.md#12-039fix-1000ep-结果2026-04-19append-only--数据不足以收口)): 16 ckpt × 1000ep on atl1-1-03-014-23-0 (port 61005, 962s elapsed)。**peak 0.852 @ ckpt 230, mean 0.833**。vs 036D 0.860 = -0.008pp 持平边缘；vs 031B 0.882 = -0.030pp 落后。**KL/total_loss 25.1% inf** (Trial 1 10.5%, Trial 2 38.1%)，policy_loss/vf_loss CLEAN — §10 reward-side fix 工作但 KL-side 未修。**两轮 AIRL 测试都不"干净"** (39 broken: callback 没 fire; 39fix: callback fire 但 KL inf)，因此 lane 暂不关 — Fix-A 重训 (12h) 是唯一 disentangle 方法但 ROI 低，作为 backlog 等 031B/034 路径用尽再回头 | Claude |
| 2026-04-19 ~08:14 | **SNAPSHOT-048 verdict — DAGGER-from-baseline 路径死**（DAGGER 框架本身仍可救）([snapshot-048 §7](snapshot-048-hybrid-eval-baseline-takeover.md#7-verdict-2026-04-19append-only)): 6 conditions 100ep 数据足以判定（节点被另一用户抢占, 1000ep 完成需 6-8h 但 100ep 信号已远超噪声，kill 节省 GPU）。**4/4 trigger conditions 全部恶化 -7 到 -14pp**: C1_031A_alpha 0.74 (Δ-14pp), C2_031A_beta 0.81 (-7pp), C3_036D_alpha 0.71 (-9pp), C4_036D_beta 0.73 (-7pp)。swap% 越大恶化越严重 → baseline 不是 teacher 而是 distractor。**Root cause**: student WR ≈ 0.86, baseline self-play WR ≈ 0.50 ([snapshot-013](snapshot-013-baseline-weakness-analysis.md))，**student 已 outgrown baseline teacher**。哲学含义: 所有 "imitate baseline" 类方法（BC / DAGGER / mixed teacher）上限 ≈ baseline self-play (0.50)。BC@2100=0.842 已经接近平台。**DAGGER 框架未死** — 以 031B@1220 (0.882) / 034 ensemble (≥ 0.90 if pass) / agent2 作为 teacher 仍可探索（必须另开 snapshot）。049 (DAGGER-from-baseline training) 否决 | Claude |
| 2026-04-19 ~07:42 | **SNAPSHOT-031B Stage 2 + 3 完成 — peer axis 与 031A 平手** ([snapshot-031 §13.7-13.10](snapshot-031-team-level-native-dual-encoder-attention.md#137-stage-2--failure-capture-1220-n500-2026-04-19-0742)): Stage 2 capture 1220@500ep WR=0.876 (vs 1000ep 0.882), 62 saved losses。**v2 bucket 跟 031A 几乎一样** (defensive_pin 44% vs 47%, wasted_possession 37% vs 42%)，**31B 不修特定 failure mode，是 uniform -27% loss 总数减少**。Stage 3 H2H ×4 (n=500): **vs 031A=0.516 NOT sig (z=0.715, p=0.24)** ⚠️ 架构 axis 上 step 2 (Siamese→cross-attention) peer-axis 增益**远小于** step 1 (031A vs 028A = 0.568 ***)；vs 029B=0.584 *** / vs 025b=0.566 ** / vs 036D=0.574 *** (cross-reward, 三个 per-agent frontier 全部显著)。**关键 caveat**: 031B baseline +2.2pp 但 vs 031A H2H 平 → **多余 baseline 增益部分是 baseline-specialization**，对 agent2 (bonus +5) 不必然继承。**031B 是当前 best 提交候选 (0.882 距 0.90 仅 -1.8pp)**，但下一步突破从「同架构变种」转向 [046](snapshot-046-cross-train-pair.md) cross-train 或 [034](snapshot-034-deploy-time-ensemble-agent.md) ensemble | Claude |

---

## 9. 相关 snapshot

- [SNAPSHOT-029](snapshot-029-post-025b-sota-extension.md) — 029B 来源
- [SNAPSHOT-030](snapshot-030-team-level-advanced-shaping-chain.md) — 030A / 030D 实验
- [SNAPSHOT-032](snapshot-032-team-level-native-coordination-aux-loss.md) — 032 系列
- [SNAPSHOT-033](snapshot-033-team-level-native-coordination-reward.md) — 033 系列
- [SNAPSHOT-036](snapshot-036-learned-reward-shaping-from-demonstrations.md) — 036C 首轮结果已出；当前主结论是 baseline 强正号，但数值稳定性待修
- [SNAPSHOT-037 RETRACTED](snapshot-037-architecture-dominance-peer-play.md) — 错误读图的反面教材
