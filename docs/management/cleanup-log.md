# 磁盘清理日志

> 目标：对 quota / 大目录清理做 append-only 记录，明确写出“计划删什么 / 实际删了什么 / 没删什么 / quota 变化”。

## 2026-04-19

### 2026-04-19 19:23 EDT — 删除 `038 / 040 / 041` 已收口 lane 的中间 checkpoint

用户确认：
- 可以继续做下一轮 quota 清理。
- 口径仍然是：只删除**已闭环 / 已收口 / 非当前主线**实验的中间 checkpoint，保留文档仍会回看的锚点与尾部 checkpoint。

本轮选择原则：
- 先不碰 `030 / 032`，因为这些 run 在历史 snapshot 里直接挂了较多 checkpoint 链接，先避免误伤文档证据链。
- 本轮只处理已经明确收口为：
  - `038` Stage 1 历史线
  - `040` Stage 2 饱和/否决线
  - `041B` 退化线

执行前 quota：
- `24649M / 30720M`

本轮处理的 run 与保留 checkpoint：

1. `040A_team_depenalized_v2_handoff_on_031A1040_formal_rerun2`
- 保留：
  - `40, 50, 60, 200`
- 删除：
  - `10, 20, 30, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190`

2. `040B_team_pbrs_handoff_on_031A1040_formal`
- 保留：
  - `130, 140, 150, 170, 180, 190, 200`
- 删除：
  - `10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 160`

3. `040C_team_event_lane_handoff_on_031A1040_formal_rerun`
- 保留：
  - `40, 50, 60, 200`
- 删除：
  - `10, 20, 30, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190`

4. `040D_team_v2_entropy_handoff_on_031A1040_formal`
- 保留：
  - `130, 140, 150, 200`
- 删除：
  - 主 trial 与后续 segment 中除上述锚点外的中间 `checkpoint_*`
  - 实际删除总数：
    - `31`

5. `041B_mappo_pbrs_on_036D150_512x512_20260419_004530`
- 保留：
  - `10, 20, 30, 40, 50, 60, 150, 200`
- 删除：
  - `70, 80, 90, 100, 110, 120, 130, 140, 160, 170, 180, 190`

6. `038A_team_depenalized_v2_stage1_on_028A1060_512x512_20260418_121346`
- 保留：
  - `40, 80, 90, 100, 120, 160, 180, 190, 200`
- 删除：
  - `10, 20, 30, 50, 60, 70, 110, 130, 140, 150, 170`

7. `038B_team_goal_prox_stage1_on_028A1060_512x512_20260418_120728`
- 保留：
  - `40, 80, 90, 100, 110, 120, 170, 180, 190, 200`
- 删除：
  - `10, 20, 30, 50, 60, 70, 130, 140, 150, 160`

8. `038C_team_event_lane_stage1_on_028A1060_512x512_20260418_120730`
- 保留：
  - `40, 50, 80, 90, 100, 120, 180, 190, 200`
- 删除：
  - `10, 20, 30, 60, 70, 110, 130, 140, 150, 160, 170`

9. `038D_team_v2_entropy_stage1_on_028A1060_512x512_20260418_120734`
- 保留：
  - `20, 40, 60, 80, 90, 100, 120, 180, 190, 200`
- 删除：
  - `10, 30, 50, 70, 110, 130, 140, 150, 160, 170`

执行结果：
- 实际处理 run 数：
  - `9`
- 实际删除 checkpoint 目录数：
  - `130`

quota 结果：
- 删除后：
  - `23965M / 30720M`
- 本轮净回收：
  - `684M`

说明：
- 这轮继续遵守“只删中间 checkpoint，不删 run root 元数据”的原则。
- `030 / 032` 暂时故意没动，因为它们在历史 snapshot 中存在较多直接 checkpoint 链接；如后续要继续清理，需先把文档锚点自动化提取得更稳。

### 2026-04-19 19:11 EDT — 删除已 commit、已文档化、非当前主线 run 的中间 checkpoint

用户确认：
- 可以继续做一轮更激进但仍可控的 quota 清理。
- 目标是删除那些**已经 commit 过、但仍保留大量中间 checkpoint、且不属于当前重要主线**的 run。
- 同时要求把“subtree / vendor 目录禁止推远端”写入工程规范。

执行口径：
- 仅处理**已闭环 / 非当前主线 / 当前文档已明确收口**的 run。
- 每个 trial 只保留：
  - 文档里已经引用或后续仍可能回看的 anchor checkpoint
  - 该 trial 的最后一个 checkpoint
- 只删除 trial 内部的中间 `checkpoint_*` 目录。
- 不删除：
  - run root 下的 `progress.csv / result.json / checkpoint_eval.csv / training_loss_curve.png / summary`
  - 当前仍在推进中的主线 run
  - 文档仍以其为冠军/主候选的 lane

执行前 quota：
- `27014M / 30720M`

本轮处理的 run 与保留 checkpoint：

1. `042A3_team_kl_distill_from_029B_on_031A1040_formal`
- 保留：
  - `80, 90, 100`
- 删除：
  - `10, 20, 30, 40, 50, 60, 70, 110, 120, 130, 140, 150, 160, 170`

2. `042A3_team_kl_distill_from_029B_on_031A1040_resume170`
- 保留：
  - `180, 190, 200`
- 删除：
  - 无（该 resume 段本身只保留了尾部 checkpoint）

3. `044A_team_spear_scratch_512x512_20260419_004717`
- 保留：
  - `160, 170, 180, 200`
- 删除：
  - `10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 190`

4. `044B_team_shield_scratch_512x512_20260419_005130`
- 保留：
  - `180, 190, 200`
- 删除：
  - `10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170`

5. `045A_team_combo_on_031A1040_formal_rerun1`
- 保留：
  - `10, 20, 130, 140, 150, 170, 180, 190, 200`
- 删除：
  - `30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 160`

6. `045B_learned_only_on_031A1040_512x512_20260419_095729`
- 保留：
  - `100, 110, 120, 140, 150, 160, 200`
- 删除：
  - `10, 20, 30, 40, 50, 60, 70, 80, 90, 130, 170, 180, 190`

7. `046E_scratch_vs_frozen_031B_cross_attention_512x512_20260419_095358`
- 保留：
  - `50, 150, 300, 450, 570, 600, 620, 680, 750, 760`
- 删除：
  - 除上述保留点外的全部中间 `checkpoint_*`

8. `052A_team_transformer_min_scratch_v2_512x512_20260419_102129`
- 保留：
  - `890, 970, 1050, 1060, 1080, 1090, 1100, 1140, 1150, 1160, 1250`
- 删除：
  - 除上述保留点外的全部中间 `checkpoint_*`

9. `052_team_transformer_full_scratch_v2_512x512_20260419_103340`
- 保留：
  - `490, 520, 530, 800, 810, 830, 870, 1140, 1150, 1160, 1250`
- 删除：
  - 除上述保留点外的全部中间 `checkpoint_*`

说明：
- `045B_learned_only_on_031A1040_512x512_20260419_095450` 这条 lane 没有可删的 `checkpoint_*`，因此本轮未实际处理。
- 这轮清理的判断依据是：上述 run 在当前 `snapshot / README / rank` 体系下均已收口为已知负结果、已知非主线、已知饱和线，或仅保留少量文档 anchor 即足够支撑后续阅读。

执行结果：
- 删除对象：
  - `9` 个 run 的中间 checkpoint
- 预计回收：
  - `2365M`
- 实际 quota 变化：
  - 删除后：`24642M / 30720M`
  - 净回收：`2372M`

补充说明：
- 这轮删除的主力空间，仍然是 `ray_results/.../checkpoint_*/checkpoint-*` 这类**无扩展名** checkpoint 本体文件。
- 这些 checkpoint 文件名没有 `.pt / .json / .csv` 等后缀，因此在磁盘统计里会被归类为 `[no-ext]`。
- run root 下的训练/eval 元数据与文档引用点没有动。

### 2026-04-19 19:05 EDT — 删除 `rank.md` 未引用的 pre-030 run 中间 checkpoint

用户确认：
- `rank.md` 里**没提到**的、并且属于 `snapshot 29` 以前的历史 run，其中间 `checkpoint_*` 可以删除。

执行口径：
- 仅处理 **pre-030** 历史 run。
- 仅处理 **未被 `docs/experiments/rank.md` 引用** 的 run。
- 每个 trial **保留最后一个 checkpoint**，只删除其中间 `checkpoint_*` 目录。
- 不删除：
  - run root 下的 `progress.csv / result.json / checkpoint_eval.csv / plot / summary`
  - `rank.md` 已引用的冠军/主候选 run
  - `*_merged` 及其 source segments

执行前估算：
- 计划删除 checkpoint 目录数：
  - `108`
- 预计回收：
  - `1371.5M`

实际删除明细：

1. `PPO_mappo_field_role_binding_v2_warm470_512x512_20260416_045516`
- 删除 iteration：
  - `10, 20, 30, 40, 50, 60, 70, 80, 130, 140, 150, 160, 170, 230, 240, 250, 260`

2. `PPO_mappo_aux_teammate_scratch_512x512_20260416_065850`
- 删除 iteration：
  - `400, 410, 420, 430, 440, 450, 460, 470, 480, 490`

3. `PPO_mappo_field_role_binding_bc2100_512x512_20260417_050955`
- 删除 iteration：
  - `20, 70, 80, 110, 120, 130, 140, 150, 160, 170`

4. `PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260415_225439`
- 删除 iteration：
  - `140, 150, 160, 170, 180, 250, 260, 270, 280`

5. `PPO_mappo_v2_opponent_pool_512x512_20260414_212239`
- 删除 iteration：
  - `240, 270, 280, 290`

6. `PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337`
- 删除 iteration：
  - `210, 230, 260`

7. `PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616`
- 删除 iteration：
  - `320, 360, 410, 430, 470, 480`

8. `PPO_mappo_field_role_binding_v2_warm470_512x512_20260415_230033`
- 删除 iteration：
  - `230, 240, 250, 260, 270, 280`

9. `PPO_mappo_role_diff_shaping_v2_warm470_512x512_20260416_045430`
- 删除 iteration：
  - `10, 20, 30, 40, 270, 280`

10. `PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2`
- 删除 iteration：
  - `1410, 1870, 2100, 2170, 2240`

11. `PPO_team_level_v2_scratch_768x512_20260417_095059`
- 删除 iteration：
  - `650, 690, 830, 1130, 1210`

12. `PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221`
- 删除 iteration：
  - `40, 60`

13. `PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113`
- 删除 iteration：
  - `350, 360, 450`

14. `PPO_team_vs_baseline_shaping_v2_deepzone_scratch_512x512_20260412_210755`
- 删除 iteration：
  - `350, 370, 390, 440, 460`

15. `PPO_team_vs_baseline_shaping_v1_scratch_512x512_20260412_210902`
- 删除 iteration：
  - `180, 340, 410, 430, 440`

16. `PPO_mappo_teammate_obs_norm_scratch_512x512_20260416_172518`
- 删除 iteration：
  - `440, 450`

17. `PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251`
- 删除 iteration：
  - `360, 420`

18. `PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545`
- 删除 iteration：
  - `290, 460`

19. `PPO_mappo_liberation_B_warm_bc2100_512x512_20260417_062526`
- 删除 iteration：
  - `170, 250`

20. `PPO_team_vs_baseline_shaping_v4_survival_scratch_512x512_20260413_053133`
- 删除 iteration：
  - `400, 430`

21. `PPO_mappo_vs_baseline_shaping_v4_512x512_resume_20260415_155350`
- 删除 iteration：
  - `480`

22. `PPO_team_vs_baseline_shaping_scratch_512x512_continue_20260412_061353`
- 删除 iteration：
  - `400`

结果汇总：
- 实际删除 run 数：
  - `22`
- 实际删除 checkpoint 目录数：
  - `108`

quota 结果：
- 删除前：
  - `28325M / 30720M`
- 删除后：
  - `26946M / 30720M`
- 本轮净回收：
  - `1379M`

说明：
- 这轮清理只删“中间 checkpoint”，不删 run root 元数据。
- 这轮清理的目的，是在不动 `rank.md` 主引用 run 的前提下，优先回收 `ray_results` 中最肥的无扩展名 checkpoint 文件。
- 当前 quota 仍主要由 `ray_results` 占据，后续若继续清理，应优先考虑已完成大 run 的中间 checkpoint，而不是文档产物。

### 2026-04-19 18:45 EDT — smoke runs 批量删除

用户确认：
- `ray_results` 下所有顶层 `*smoke*` 目录可以删除。

实际删除范围：
- 删除对象为 `ray_results/` 顶层所有名字匹配 `*smoke*` 的目录与文件。
- 包括但不限于：
  - `ray_results/039fix_smoke_20260419_035821`
  - `ray_results/042A3_smoke_20260419_042534`
  - `ray_results/043Aprime_smoke_20260419_112305`
  - `ray_results/045A_smoke_20260419_051839`
  - 以及早期 `PPO_*smoke* / BC_*smoke* / smoke_029*` 等 smoke 目录
  - `ray_results/043Aprime_smoke_launch.log`

结果：
- 删除后，`ray_results` 顶层已无 `*smoke*` 匹配项。

quota 结果：
- 删除前：
  - `28694M / 30720M`
- 删除后：
  - `28636M / 30720M`
- 本轮净回收：
  - `58M`

说明：
- 本轮只删除 smoke 运行目录/文件。
- 不涉及任何 formal / rerun 主结果目录。

### 2026-04-19 18:49 EDT — 删除历史 H2H 样本包 `h2h_v3` / `h2h_v3_all`

用户确认：
- `docs/experiments/artifacts/failure-cases/h2h_v3`
- `docs/experiments/artifacts/failure-cases/h2h_v3_all`
可以删除。

删除前大小：
- `h2h_v3 = 271M`
- `h2h_v3_all = 88M`

实际删除：
- 已删除：
  - `docs/experiments/artifacts/failure-cases/h2h_v3`
  - `docs/experiments/artifacts/failure-cases/h2h_v3_all`
- 明确保留：
  - `docs/experiments/artifacts/failure-cases/h2h_v3_all_v2`

与 `053` 的关系：
- 用户明确要求保留说明：上述两包历史 H2H 样本与 `053` 路线相关。
- 当前项目中与该路线对应的**正式沉淀内容**以：
  - [SNAPSHOT-053](../experiments/snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md)
  - `docs/experiments/artifacts/trajectories/v3_all_30pair/`
为准。
- 因此本次删除的是早期 failure-case 形态的大样本包，而不是 `053` 的主数据产物。

quota 结果：
- 删除前：
  - `28637M / 30720M`
- 删除后：
  - `28278M / 30720M`
- 本轮净回收：
  - `359M`

### 2026-04-19 18:55 EDT — 删除 `h2h_v3_all_v2` 残余

背景：
- 在 18:37 EDT 的中断删除中，`h2h_v3_all_v2` 已从约 `0.93G / 15030 files` 缩到约 `11.4M / 182 files`。
- 用户随后明确要求：既然这包已经被部分删掉，就不要再保留残余，以免误导后续判断。

实际删除：
- 已删除：
  - `docs/experiments/artifacts/failure-cases/h2h_v3_all_v2`

quota 结果：
- 删除前：
  - `28291M / 30720M`
- 删除后：
  - `28279M / 30720M`
- 本轮净回收：
  - `12M`

与 `053` 的关系：
- 与前一条相同，`053` 路线的正式保留内容仍以
  - [SNAPSHOT-053](../experiments/snapshot-053-outcome-pbrs-reward-from-calibrated-predictor.md)
  - `docs/experiments/artifacts/trajectories/v3_all_30pair/`
为准。

### 2026-04-19 18:37 EDT — quota 应急清理（中途叫停）

背景：
- 用户 home quota 在清理前已接近上限。
- 清理前 quota：
  - `29674M / 30720M`

前置核查：
- `ray_results` 约 `16.09G`
- `docs/experiments/artifacts/failure-cases` 约 `1.86G`
- 识别出 3 个超大历史 H2H 样本目录：
  - `docs/experiments/artifacts/failure-cases/h2h_v3_all_v2`
  - `docs/experiments/artifacts/failure-cases/h2h_v3`
  - `docs/experiments/artifacts/failure-cases/h2h_v3_all`
- 同时核查发现 `*_merged` 目录不是独立副本，而是带 symlink 的 merged view，因此**没有删除任何 merged source segments**。

原始尝试删除目标：
- `docs/experiments/artifacts/failure-cases/h2h_v3_all_v2`
- `docs/experiments/artifacts/failure-cases/h2h_v3`
- `docs/experiments/artifacts/failure-cases/h2h_v3_all`
- `ray_results/039fix_smoke_20260419_035821`
- `ray_results/042A3_smoke_20260419_042534`
- `ray_results/043Aprime_smoke_20260419_112305`
- `ray_results/045A_smoke_20260419_051839`
- `ray_results/043Bprime_resume080_vs_harder_frontier_pool_031B_cross_attention_512x512_20260419_151436`
- `ray_results/043Cprime_resume080_vs_heavier_frontier_pool_031B_cross_attention_512x512_20260419_151441`
- `ray_results/045A_team_combo_on_031A1040_formal`

中断原因：
- 用户明确指出“这三个没法删”，指向上面的 3 个 `h2h_v3*` 目录。
- 因此立即终止 `rm` 进程，并在终止后重新核查目录存在性与 quota。

终止后实际状态：

1. `h2h_v3_all_v2`
- 目录仍存在
- 但在叫停前发生了**部分删除**
- 清理前：
  - 约 `0.93G`
  - `15030 files`
- 清理后：
  - 约 `11.4M`
  - `182 files`

2. `h2h_v3`
- 目录存在
- 未被实际删除
- 当前大小：
  - `249.9M`
  - `3919 files`

3. `h2h_v3_all`
- 目录存在
- 未被实际删除
- 当前大小：
  - `81.0M`
  - `1287 files`

4. 其余 7 个 `ray_results` 目标
- 全部仍存在
- 本轮未被实际删除：
  - `ray_results/039fix_smoke_20260419_035821`
  - `ray_results/042A3_smoke_20260419_042534`
  - `ray_results/043Aprime_smoke_20260419_112305`
  - `ray_results/045A_smoke_20260419_051839`
  - `ray_results/043Bprime_resume080_vs_harder_frontier_pool_031B_cross_attention_512x512_20260419_151436`
  - `ray_results/043Cprime_resume080_vs_heavier_frontier_pool_031B_cross_attention_512x512_20260419_151441`
  - `ray_results/045A_team_combo_on_031A1040_formal`

quota 结果：
- 清理后 quota：
  - `28681M / 30720M`
- 本轮净回收：
  - `993M`

结论：
- 本轮 quota 缓解**确实发生了**，但来源不是计划中的整批删除，而是：
  - `h2h_v3_all_v2` 在用户叫停前已被部分清理
- `h2h_v3`、`h2h_v3_all`、smoke 目录、失败首发目录、`045A` 原始失败 formal 目录都**仍然保留**
- `*_merged` 目录及其 source segments **未动**

后续约束：
- 将 `h2h_v3` / `h2h_v3_all` / `h2h_v3_all_v2` 视为“未经再次确认不得删除”的目录。

---

## 2026-04-20

### 04:17-04:30 EDT — 🚨 紧急 home quota 救援 (symlink-to-SCRATCH 批量移动)

#### 事故
- 04:17 EDT 发生 **home quota exceeded crash cascade**:
  - home quota 30716M / 30720M (99.99% full)
  - 触发 `OSError: [Errno 122] Disk quota exceeded` in Ray ckpt write
  - **7 lanes 同时死亡**, 全部 EXIT_CODE=1:
    | Lane | 死亡 iter | ckpts saved | 进度 |
    |---|---:|---:|---:|
    | 054 MAT-min | 478/1250 | 115 | 38% |
    | 055 distill 034E | 230/1250 | 102 | 18% |
    | 056A PBT LR=3e-5 | ~240/1250 | 109 | 19% |
    | 056B PBT LR=7e-5 | ~240/1250 | — | 19% |
    | 056C PBT LR=1.5e-4 | ~240/1250 | — | 19% |
    | 056D PBT LR=3e-4 | ~240/1250 | — | 19% |
    | 057 RND | 77/1250 | 77 | 6% |
    | 058 curriculum | 84/1250 | 84 | 7% |
    | 053Dmirror | 0 (Unity init fail x2) | 0 | — |

- 11 个训练 + multiple eval lanes 同一时刻 GPU usage 变 0 MiB

#### 救援策略

**策略**: **move 到 SCRATCH + 从 home symlink** — 保留 agent code 中 hardcoded 路径的 resolvability. NOT 删除任何东西.

**Archive 目的地**: `/storage/ice1/5/1/wsun377/ray_results_archive/` (PACE SCRATCH, 4.1P 可用 / 已用 4%)

#### 详细迁移清单 (65 个 lanes, ~16GB 总量, 全部 symlink 可用)

**Phase 1** (~5GB freed) — agent code 引用的历史 lanes:
- `031A_team_dual_encoder_scratch_v2_512x512_20260418_054948` (563M) — 034b/c ensemble members
- `PPO_mappo_036C_learned_reward_on_029B190_512x512_20260418_075657` (433M)
- `PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107` (435M) — 034b/c/d/e/f/eb members
- `PPO_mappo_039_airl_adaptive_on_029B190_512x512_20260418_132607` (435M)
- `039fixB_with_sanitize_20260419_092209` (436M)
- `030A_team_field_role_on_028A1060_512x512_20260418_051107` (398M)
- `030D_team_pbrs_on_028A1060_512x512_20260418_051114` (398M)
- `032A_team_action_aux_on_028A1060_512x512_20260418_053238` (307M)
- `032Acontrol_team_action_aux0_on_028A1060_512x512_20260418_053246` (307M)
- `032nextA_symmetric_action_aux_on_028A1060_formal` (378M)
- `032nextControl_symmetric_action_aux0_on_028A1060_formal` (378M)
- `033A_team_coord_pbrs_on_028A1060_512x512_20260418_055015` (239M)
- `PPO_team_level_bc_bootstrap_028A_512x512_formal` (127M)
- `PPO_team_level_v2_scratch_768x512_20260417_095059` (188M)
- `PPO_mappo_029C_025b80_oppool_peers_512x512_formal` (120M)
- `PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2` (192M)

**Phase 2** — 043/044/045/046 families:
- `043Aprime_warm_vs_frontier_pool_031B_formal` (291M)
- `043Bprime_resume080_vs_harder_frontier_pool_031B_cross_attention_rerun1` (243M)
- `043Cprime_resume080_vs_heavier_frontier_pool_031B_cross_attention_rerun1` (243M)
- `043Bprime_resume080_vs_harder_frontier_pool_031B_cross_attention_512x512_20260419_151436` (60K stub)
- `043Cprime_resume080_vs_heavier_frontier_pool_031B_cross_attention_512x512_20260419_151441` (64K stub)
- `044A_team_spear_scratch_512x512_20260419_004717` (26M)
- `044B_team_shield_scratch_512x512_20260419_005130` (24M)
- `045A_team_combo_on_031A1040_formal_rerun1` (53M)
- `045B_learned_only_on_031A1040_512x512_20260419_095729` (45M)
- `046B_warm_vs_frozen_031A_512x512_20260419_071550` (87M)
- `046D_warm_vs_frozen_031B_cross_attention_512x512_20260419_093605` (18M)
- `046E_scratch_vs_frozen_031B_cross_attention_512x512_20260419_095358` (105M)
- `046F_scratch_vs_frozen_051A_cross_attention_512x512_20260419_141732` (13M)

**Phase 3** — 038/040/041/042/051(A-C)/052 + 死亡训练 lanes:
- 死亡训练 (可能 resume): 054/055/056A-D/057/058/053Dmirror (见上)
- 038 family: 038A(×3), 038B, 038C, 038D (82-89M each)
- 040 family: 040A(×3), 040B, 040C(×2), 040D (35-62M each)
- 041B_mappo_pbrs_on_036D150 (144M)
- 042A3_team_kl_distill_from_029B: formal (33M), resume170 (20M), merged (108K)
- 051A_combo_on_031B_with_051reward (127M)
- 051B_learned_only_on_031B_with_051reward (126M)
- 051C_learned_only_warm031B80_cross_attention (118M)
- 052_team_transformer_full_scratch (180M)
- 052A_team_transformer_min_scratch (197M)
- 039fix_mappo_airl_adaptive_on_029B190: formal (416M), resume140 (465M), merged (200K)

#### 校验 (symlinks 可用)

所有 archived lanes 现在 home 端是 **symbolic link** 到 SCRATCH archive:

```bash
$ ls -la ray_results/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948
lrwxrwxrwx ... → /storage/ice1/5/1/wsun377/ray_results_archive/031A_team_dual_encoder_scratch_v2_512x512_20260418_054948
```

5 个 sample 验证:
- ✓ `031A_team_dual_encoder_scratch_v2_512x512_20260418_054948` → target accessible
- ✓ `PPO_mappo_036D_stable_learned_reward_on_029B190_512x512_20260418_124107` → target accessible
- ✓ `043Aprime_warm_vs_frontier_pool_031B_formal` → target accessible
- ✓ `055_distill_034e_ensemble_to_031B_scratch_20260419_193252` → target accessible
- ✓ `054_mat_min_scratch_v2_512x512_20260419_185019` → target accessible

**所有 agent code (agents/v034*) 中 hardcoded 的 `ray_results/...` 路径仍然 resolve** — 不影响 deploy / eval.

#### Quota 结果

```
救援前: 30716M / 30720M (99.99% full, 4MB headroom)
救援后: 16536M / 30720M (54% full, 14GB headroom)
净释放: ~14GB
```

#### 保留在 home (未 archive) 的 lanes + 理由

- `031B_team_cross_attention_scratch_v2_512x512_20260418_233553` (642M) — 包含 **031B@80 (weak base 源)** + 031B@1080 (project SOTA base); 多个 active lanes 直接依赖
- `031B_team_cross_attention_scratch_v2_resume1080` (108M) — 031B@1220 base container, critical
- `031B_noshape_ablation_cross_attention_512x512_20260419_155110` (742M) — 刚完成 verdict (2026-04-20 01:55), 可能要做 capture/H2H
- `051D_learned_only_warm031B80_cross_attention_512x512_20260419_164416` (475M) — 刚完成 verdict, 可能做 Stage 2/3
- `053A_outcome_pbrs_combo_on_031B_512x512_20260419_172337` (123M) — 053Acont 的 warmstart 源
- `053Acont_iter200_to_500_20260419_194712` (182M) — 当前 single-model 主候选, verdict 已入 rank.md
- 若干小型 PPO_mappo_* (87-118M each) — legacy, 低优先级

#### 遗留动作

- [ ] **Structural fix**: 所有未来 launch script 加 `export LOCAL_DIR=/storage/ice1/5/1/wsun377/ray_results_scratch` → Ray 直接写 SCRATCH, 不再占用 home. 防再次 quota crash.
- [ ] **Dead lanes resume / restart decision**: 054/055/056A-D/057/058/053Dmirror — 取决于是否有节点 + 投入值不值.
- [ ] **Stale .running flags**: 已手动清理 (054/055/056A-D/057/058/053Dmirror).

#### 教训

1. **Ray 默认 local_dir 指向 project-local ray_results** — 在 home quota constrained 的 PACE 环境上必然撞墙. 应一开始就 redirect 到 SCRATCH.
2. `.running` trap 用 `$?` 捕 tee 的 exit code, tee 总是 0 — 实际 python 错误被 mask. 应用 `${PIPESTATUS[0]}`.
3. **Symlink-archive > delete**: 保 agent code resolvability, 低风险 reversible — 可随时 `mv` 回来.
