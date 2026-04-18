# scripts/

脚本目录按用途分层，不再用单层平铺。

## 结构

### `setup/`

- `setup.sh`：基础环境安装与校验
- `setup_h100_overlay.sh`：H100 / Hopper overlay 环境

### `eval/`

- `evaluate_official_suite.py`：官方 evaluator 批量评估
- `evaluate_official_scan.py`：官方 evaluator checkpoint 粗扫/精扫
- `evaluate_checkpoint_suite.py`：本地评估辅助
- `backfill_run_eval.py`：给已完成 run 回填 checkpoint 评估
- `capture_failure_cases.py`：为单个 checkpoint 采集失败 episode JSON，用于 reward shaping 诊断
- `analyze_episode_records.py`：聚合已保存的 episode JSON，输出 failure label / outcome / steps / metrics summary
- `eval_checkpoints.sh`：面向 legacy `trained_ray_agent` wrapper 的快速批量 checkpoint 对战脚本

### `tools/`

- `build_agent_module.py`：从 checkpoint 构建 submission-ready agent 模块
- `filter_train_output.py`：训练输出过滤
- `collect_baseline_trajectories.py`：采集 baseline/self-play teacher 轨迹，用于 imitation learning / behavior cloning 起步

### `batch/experiments/`

- `soccerstwos_h100_cpu32_team_vs_baseline_shaping_v1_scratch_512x512.batch`：显式锁定 shaping-v1 默认值的 scratch 对照线
- `soccerstwos_h100_cpu32_team_vs_baseline_shaping_v2_deepzone_scratch_512x512.batch`：deep-zone + negative-C 的 shaping-v2 对照线
- `soccerstwos_h100_cpu32_team_vs_baseline_shaping_v3_progress_gated_scratch_512x512.batch`：在 v2 基础上把正向 progress reward 改为 possession-gated 的 v3 对照线
- `soccerstwos_h100_cpu32_team_vs_baseline_shaping_v4_survival_scratch_512x512.batch`：回到 v2 基座，在 baseline-targeted scratch 线上叠加 defensive survival bonus 与 fast-loss penalty 的 v4 对照线
- `soccerstwos_h100_cpu32_collect_baseline_selfplay_team.batch`：baseline-vs-baseline team-level teacher 轨迹采集标准入口
- `soccerstwos_h100_cpu32_bc_team_baseline_selfplay_512x512.batch`：基于 baseline self-play teacher dataset 的 team-level BC 标准训练入口
- `soccerstwos_h100_cpu32_baseline_weakness_baseline_vs_baseline.batch`：baseline-vs-baseline episode 全量采样与聚合分析入口
- `soccerstwos_h100_cpu32_baseline_weakness_baseline_vs_random.batch`：baseline-vs-random episode 全量采样与聚合分析入口
- `soccerstwos_h100_cpu32_mappo_vs_baseline_noshaping_512x512.batch`：公平 MAPPO / centralized-critic 首轮对照，固定 baseline 对手、无 shaping
- `soccerstwos_h100_cpu32_mappo_vs_baseline_shaping_v1_512x512.batch`：公平 MAPPO / centralized-critic 首轮对照，叠加 shaping-v1
- `soccerstwos_h100_cpu32_mappo_vs_baseline_shaping_v2_512x512.batch`：公平 MAPPO / centralized-critic 首轮对照，叠加 shaping-v2

### `batch/`

- `starter/`：starter 风格与课程提供的入口
- `base/`：从零开始的基础模型训练脚本
- `adaptation/`：基于已有 base checkpoint 的下游适配训练脚本
- `experiments/`：实验分支和已归档探索脚本

## 当前主线

- `batch/adaptation/soccerstwos_h100_cpu32_singleplayer_fixed_teammate.batch`
- `batch/adaptation/soccerstwos_h100_cpu32_singleplayer_fixed_teammate_scratch.batch`

## 当前 Base Lane

- `batch/base/soccerstwos_h100_cpu32_base_team_vs_random_512x512.batch`
- `batch/base/soccerstwos_h100_cpu32_base_team_vs_random_512x256.batch`
- `batch/base/soccerstwos_h100_cpu32_base_team_vs_baseline_512x512.batch`
- `batch/base/soccerstwos_h100_cpu32_base_team_vs_baseline_512x256.batch`
- `batch/base/soccerstwos_h100_cpu32_base_ma_teams.batch`

## 当前约束

- 新脚本必须放进对应子目录，不再回到 `scripts/` 根层平铺
- starter 对齐脚本放 `batch/starter/`
- 基础模型脚本放 `batch/base/`
- 下游适配脚本放 `batch/adaptation/`
- 探索性实验放 `batch/experiments/`
- 原始日志仍统一归档到 `docs/experiments/artifacts/slurm-logs/`
