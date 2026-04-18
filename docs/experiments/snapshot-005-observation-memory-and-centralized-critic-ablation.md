# SNAPSHOT-005: Observation / Memory / Centralized-Critic 方向复盘

- **日期**: 2026-04-09
- **负责人**:
- **目标**:
  - 验证 `summary observation`、`LSTM`、`shared actor + centralized critic` 是否能突破现有 `~0.80` baseline 胜率平台
  - 判断问题是否来自表征/记忆不足，还是训练目标与最终比赛目标错位
- **配置**:
  - summary-observation: [train_ray_team_vs_random_summary_obs.py](../../cs8803drl/training/train_ray_team_vs_random_summary_obs.py)
  - recurrent PPO: [train_ray_team_vs_random_lstm.py](../../cs8803drl/training/train_ray_team_vs_random_lstm.py)
  - centralized critic: [train_ray_shared_central_critic.py](../../cs8803drl/training/train_ray_shared_central_critic.py)
  - 评估: [scripts/eval/evaluate_official_suite.py](../../scripts/eval/evaluate_official_suite.py)
  - 选模口径: 训练内 `50` 局筛选，官方 `200` 局确认
- **结果**:
  - summary-observation warm225
    - 训练内 best: [checkpoint_90](../../ray_results/PPO_summary_obs_warm225_20260409_011316/SummaryObsPPOTrainer_Soccer_dda21_00000_0_2026-04-09_01-13-38/checkpoint_000090/checkpoint-90)
    - 训练内筛选: `43/50 = 0.86`
    - 结论: reward / loss 曲线更平，但与最终胜率仍然脱钩
  - LSTM warm225
    - 训练内 best: [checkpoint_50](../../ray_results/PPO_lstm_warm225_20260409_115548/LSTMPPOTrainer_Soccer_a6a2b_00000_0_2026-04-09_11-56-20/checkpoint_000050/checkpoint-50)
    - 训练内筛选: `3/50 = 0.06`
    - 结论: 当前 evaluator / reset 约束下，这条线可直接排除
  - centralized critic warm225
    - 训练内 best: [checkpoint_80](../../ray_results/PPO_shared_cc_warm225_20260409_123447/SharedCCPPOTrainer_Soccer_11dec_00000_0_2026-04-09_12-35-08/checkpoint_000080/checkpoint-80)
    - 官方确认:
      - `vs baseline`: `160/200 = 0.800`
      - `vs random`: `199/200 = 0.995`
- **结论**:
  - summary-observation 与 LSTM 两条线没有提供可靠增益。
  - 当天对 centralized critic 的记录只说明：在当时那套 `warm225 + 256x128 + 3M steps` 设定下，官方 `200` 局结果约为 `0.80`。
  - 这一结果本身并不能证明 centralized critic / MAPPO 无效，更不能单独支持“该方向已经被否定”。
- **相关**:
  - [snapshot-003-official-evaluator-realignment.md](snapshot-003-official-evaluator-realignment.md)
  - [snapshot-004-role-ppo-and-shared-policy-ablation.md](snapshot-004-role-ppo-and-shared-policy-ablation.md)
  - [training_loss_curve.svg](../../ray_results/PPO_shared_cc_warm225_20260409_123447/training_loss_curve.svg)

## 2026-04-13 修订说明

在重新核对 [PPO_shared_cc_warm225_20260409_123447](../../ray_results/PPO_shared_cc_warm225_20260409_123447) 的 [params.json](../../ray_results/PPO_shared_cc_warm225_20260409_123447/SharedCCPPOTrainer_Soccer_11dec_00000_0_2026-04-09_12-35-08/params.json)、[progress.csv](../../ray_results/PPO_shared_cc_warm225_20260409_123447/SharedCCPPOTrainer_Soccer_11dec_00000_0_2026-04-09_12-35-08/progress.csv) 和 [checkpoint_eval.csv](../../ray_results/PPO_shared_cc_warm225_20260409_123447/checkpoint_eval.csv) 后，需要对 centralized critic 的结论收紧：

### 修订 1：旧 shared_cc run 不能当作“MAPPO 已被证伪”

旧 run 的真实条件是：

- `variation = multiagent_player`
- 对手是固定 baseline（`FrozenBaselinePolicy`）
- `reward_shaping = disabled`
- `fcnet_hiddens = [256, 128]`
- warm-start 来源是 [checkpoint-225](../../ray_results/PPO_continue_ckpt160_cpu32_20260408_183648/PPO_Soccer_79ad0_00000_0_2026-04-08_18-37-08/checkpoint_000225/checkpoint-225)
- 总训练长度仅 `125 iter / 3M steps`

因此它和后来 `team_vs_baseline + shaping + 512x512 + 500 iter` 的主线并不是公平对照。

### 修订 2：旧结果更像“方向有信号”，不是“方向没用”

旧 shared_cc run 的 baseline 训练内轨迹是：

- `it 5: 0.86`
- `it 30: 0.66`
- `it 80: 0.86`
- `it 125: 0.78`

这说明 centralized critic 至少没有在当时设定下直接训坏，但也不能说明已经被充分验证。

### 修订 3：真正需要的是公平 MAPPO 对照

因此，本 snapshot 对 centralized critic 的最终修订结论是：

- `summary observation` 与 `LSTM` 的负结论仍然成立；
- `centralized critic / MAPPO` 的负结论不成立；
- 该方向需要在当前阶段用更公平的配置重新测试，而不是基于这次 orthogonal run 直接放弃。

后续重新测试已单独预注册为：

- [SNAPSHOT-014](snapshot-014-mappo-fair-ablation.md)
