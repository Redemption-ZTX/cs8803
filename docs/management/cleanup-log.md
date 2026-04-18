# 磁盘清理日志

## 目的

本文件记录每次磁盘清理实际删除了什么、为什么删、删前删后配额变化如何。

目标不是“证明清理过”，而是：

- 避免误删正式实验资产
- 让后续清理能复用同一套标准
- 让“哪些 run 已被清掉”有明确历史

## 记录规则

每次清理至少记录：

- 日期
- 清理前后配额
- 删除对象
- 删除理由
- 是否涉及正式实验 run

推荐优先级：

1. 明显临时目录  
   例如 `~/ray_results/*DummyEnv*`
2. 缓存 / 包缓存  
   例如 `~/.cache/pip`、`~/.conda/pkgs`、`~/.npm`
3. 仓库内未完成、未形成结果链、且未被文档引用的 run
4. 已被 rerun 替代的旧 run  
   这类必须单独确认后再删

## 2026-04-16

### 清理前配额

- `30716M / 30720M`

### 已删除

#### Home 临时与缓存

- `~/ray_results/*DummyEnv*`
  - 共删除 `5857` 个 dummy-env 临时 Ray 目录
  - 删除理由：deployment / checkpoint-load / smoke 产生的短命目录，不属于正式实验资产
- `~/.cache/pip`
  - 删除理由：pip 下载缓存，不影响现有环境使用
- `~/.conda/pkgs`
  - 删除理由：conda 包缓存，不影响现有 env
- `~/.npm`
  - 删除理由：npm 缓存

#### 仓库 `ray_results/` 内已删除 run

以下目录满足“未形成结果链，且未在文档中形成持续引用”的清理条件：

- `ray_results/PPO_base_ma_teams_20260411_190724`
- `ray_results/PPO_team_vs_random_h100_20260408_133641`
- `ray_results/PPO_base_team_vs_random_512x512_20260411_163047`
- `ray_results/PPO_base_ma_teams_20260411_185845`
- `ray_results/PPO_base_team_vs_random_512x256_20260411_163826`
- `ray_results/PPO_benchmark_fast_h100_20260408_151622`
- `ray_results/PPO_mappo_aux_teammate_scratch_512x512_20260416_065223`
- `ray_results/PPO_base_team_vs_random_512x512_20260411_164603`
- `ray_results/PPO_base_team_vs_random_512x256_20260411_164607`

### 明确保留

以下类型本次明确未动：

- 已写入 snapshot / README / CHANGELOG / WORKLOG 的正式实验 run
- 当前主线相关 MAPPO / BC / opponent-pool / 021c run
- `~/.conda/envs`
- `~/.venvs`

### 清理后配额

- `26524M / 30720M`

### 结果

- 本轮共回收约 `4.2G`
- 当前配额已从“接近硬上限”恢复到“可继续训练 / 评估”的安全区间

### 后续建议

- 下次优先检查“已被 rerun 替代的旧 run”
- 对已写入 snapshot 的 run，不在没有明确替代方案时直接删除
- 若需要进一步回收空间，优先考虑“只保留 best checkpoint / eval / summary，裁剪冗余 checkpoint”这类精细化清理

## 2026-04-17

### 清理前配额

- `28274M / 30720M`

### 已删除

#### Home 临时目录

- `~/ray_results/*DummyEnv*`
  - 删除约 `449` 个 dummy-env / dummy-agent 临时 Ray 目录
  - 删除理由：agent 构建、checkpoint 加载、official/local eval 过程中自动生成的短命目录，不属于正式实验资产

#### VS Code Server 缓存与旧版本

- `~/.vscode-server/data/CachedExtensionVSIXs/*`
  - 删除理由：扩展安装缓存，可随时重建
- `~/.vscode-server/data/logs/*`
  - 删除理由：运行日志缓存
  - 说明：由于 NFS `.nfs*` 占用文件，残留约 `35M`，未强删
- `~/.vscode-server/cli/servers/Stable-c9d77990917f3102ada88be140d28b038d1dd7c7`
- `~/.vscode-server/cli/servers/Stable-e7fb5e96c0730b9deb70b33781f98e2f35975036`
  - 删除理由：旧版 VS Code server，可按需自动重装
- `~/.vscode-server/extensions/github.copilot-chat-0.42.3`
  - 删除理由：旧版重复扩展
- 对以下旧版扩展做了部分清理，成功回收大部分空间，但仍保留被当前进程占用的少量残留文件：
  - `~/.vscode-server/extensions/github.copilot-chat-0.43.0`
  - `~/.vscode-server/extensions/anthropic.claude-code-2.1.109-linux-x64`
  - `~/.vscode-server/extensions/openai.chatgpt-26.409.20454-linux-x64`

### 明确保留

- 仓库内 `ray_results/` 的正式实验 run
- 当前仍在运行的 `025 / 025b / 026A / 026B / 026C` 等任务目录
- `~/.conda/envs`
- `~/.venvs`
- 当前最新 VS Code server 与最新扩展版本

### 清理后配额

- `26545M / 30720M`

### 结果

- 本轮共回收约 `1.73G`
- `~/ray_results` 从大量 dummy-env 临时目录收缩到约 `872K`
- `~/.vscode-server` 从约 `3.1G` 收缩到约 `2.2G`

### 备注

- 本轮没有触碰仓库里的正式训练结果目录
- 若后续还需继续回收空间，可优先处理：
  - `.vscode-server` 中仍残留的旧扩展占用
  - 被新 rerun 明确替代、且文档里已有结论承接的旧 run

## 2026-04-17（第二轮紧急清理）

### 清理前配额

- `30720M / 30720M`

### 已删除

#### Home 临时目录

- `~/ray_results/PPO_Dummy*`
- `~/ray_results/PPO_Soccer_2026-04-16_04-17-102ls7vjh_`
  - 删除理由：均为 dummy-env / smoke / evaluator 自动产生的 home 侧临时 Ray 目录，不属于仓库内正式实验资产

#### VS Code Server 日志与旧版本

- `~/.vscode-server/data/logs/*`
  - 删除理由：运行日志缓存
  - 说明：仍有少量 `.nfs*` 忙文件残留，未强删
- `~/.vscode-server/code-c9d77990917f3102ada88be140d28b038d1dd7c7`
- `~/.vscode-server/code-e7fb5e96c0730b9deb70b33781f98e2f35975036`
  - 删除理由：旧版 VS Code server 代码目录；当前活动版本仍保留

### 明确保留

- 仓库内 `ray_results/` 正式实验结果
- `~/.vscode-server/cli/servers/Stable-41dd792b5e652393e7787322889ed5fdc58bd75b`
- 当前 VS Code 扩展主版本
- `~/.conda/envs`
- `~/.venvs`

### 清理后配额

- `30551M / 30720M`

### 结果

- 本轮额外回收约 `169M`
- `~/ray_results` 收缩到约 `752K`
- `~/.vscode-server` 收缩到约 `923M`

### 备注

- 本轮目标是把“已顶满”的 quota 先拉回安全线，因此只做了最保守的 home/cache 侧清理
- 若后续还需继续回收空间，下一优先级应是：
  - `.vscode-server/data/logs` 中残留的 `.nfs*` 忙文件
  - 被新 rerun 明确替代的旧实验 run

## 2026-04-17（第三轮：SNAPSHOT-001~009 精细 checkpoint 清理）

### 清理前配额

- `30632M / 30720M`

### 清理策略

- 目标范围限定为 `SNAPSHOT-001` 到 `SNAPSHOT-009` 文档中实际引用到的旧 `ray_results/` run
- 仅删除这些旧 run 下的冗余 `checkpoint_*` 目录
- 明确保留：
  - snapshot 文档中直接引用的关键 checkpoint
  - `progress.csv`
  - `params.json`
  - `checkpoint_eval.csv`
  - `training_loss_curve.*`
  - trial 元数据与其余记录文件

### 已删除

- 共删除 `307` 个旧 checkpoint / eval log 目录
- 重点清理对象包括：
  - `PPO_base_team_vs_baseline_512x512_20260411_192140`
  - `PPO_base_team_vs_baseline_512x256_20260411_192339`
  - `PPO_shared_cc_warm225_20260409_123447`
  - `PPO_summary_obs_warm225_20260409_011316`
  - `PPO_lstm_warm225_20260409_115548`
  - `PPO_role_cpu32_20260408_193132`
  - `PPO_role_warm225_aggressive_20260408_224536`
  - `PPO_attack_expert_warm225_20260409_141445`
  - `PPO_singleplayer_fixed_teammate_*`
  - `PPO_continue_ckpt160_cpu32_20260408_183648`

### 明确保留的关键 checkpoint

- `PPO_smoke_rewardfix_gpu_h100_overlay_20260408/.../checkpoint_000001`
- `PPO_continue_ckpt160_cpu32_20260408_183648/.../checkpoint_000225`
- `PPO_role_cpu32_20260408_193132/.../checkpoint_000030`
- `PPO_role_warm225_aggressive_20260408_224536/.../checkpoint_000040`
- `PPO_summary_obs_warm225_20260409_011316/.../checkpoint_000090`
- `PPO_lstm_warm225_20260409_115548/.../checkpoint_000050`
- `PPO_shared_cc_warm225_20260409_123447/.../checkpoint_000080`
- `PPO_attack_expert_warm225_20260409_141445/.../checkpoint_000060`

### 清理后配额

- `29527M / 30720M`

### 结果

- 本轮额外回收约 `1.25G`
- 总体剩余空间恢复到约 `1.19G`
- 做到了“保留早期 snapshot 的关键证据链，同时移除大多数冗余 checkpoint”

### 备注

- 本轮没有触碰 `SNAPSHOT-010+` 的实验目录
- 若后续还需继续回收空间，下一优先级可考虑：
  - 各类 `smoke_*` / `*smoke*` run
  - 已被后继 lane 明确替代的完整旧 run

## 2026-04-17（第四轮：SNAPSHOT-010~020 精细 checkpoint 清理）

### 清理前配额

- `29526M / 30720M`

### 清理策略

- 目标范围限定为 `SNAPSHOT-010` 到 `SNAPSHOT-020` 文档中实际引用到的 `ray_results/` run
- 仅删除这些 run 下的冗余 `checkpoint_*` 目录
- 明确保留：
  - snapshot 文档中直接引用的关键 checkpoint
  - `progress.csv`
  - `params.json`
  - `checkpoint_eval.csv`
  - `training_loss_curve.*`
  - trial 元数据与其余记录文件

### 已删除

- 共删除 `765` 个非关键 checkpoint 目录
- 重点清理对象包括：
  - `PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616`
  - `PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545`
  - `PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113`
  - `PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2`
  - `PPO_mappo_v2_opponent_pool_512x512_20260414_212239`
  - `PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221`
  - `PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337`
  - `PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251`

### 明确保留的关键 checkpoint

- `PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/.../checkpoint_002100`
- `PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/.../checkpoint_001410`
- `PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/.../checkpoint_001870`
- `PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/.../checkpoint_002170`
- `PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/.../checkpoint_002240`
- `PPO_mappo_vs_baseline_shaping_v2_bc_player_512x512_main_rerun2/.../checkpoint_002250`
- `PPO_mappo_v2_opponent_pool_512x512_20260414_212239/.../checkpoint_000240`
- `PPO_mappo_v2_opponent_pool_512x512_20260414_212239/.../checkpoint_000270`
- `PPO_mappo_v2_opponent_pool_512x512_20260414_212239/.../checkpoint_000280`
- `PPO_mappo_v2_opponent_pool_512x512_20260414_212239/.../checkpoint_000290`
- `PPO_mappo_v2_opponent_pool_512x512_20260414_212239/.../checkpoint_000300`
- `PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/.../checkpoint_000040`
- `PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/.../checkpoint_000060`
- `PPO_mappo_v2_opponent_pool_anchor30_512x512_20260415_034221/.../checkpoint_000140`
- `PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337/.../checkpoint_000210`
- `PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337/.../checkpoint_000230`
- `PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337/.../checkpoint_000260`
- `PPO_mappo_v2_opponent_pool_anchor30_512x512_resume_20260415_155337/.../checkpoint_000280`
- `PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/.../checkpoint_000320`
- `PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/.../checkpoint_000360`
- `PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/.../checkpoint_000410`
- `PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/.../checkpoint_000430`
- `PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/.../checkpoint_000470`
- `PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/.../checkpoint_000480`
- `PPO_mappo_vs_baseline_shaping_v1_512x512_20260413_034616/.../checkpoint_000490`
- `PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/.../checkpoint_000290`
- `PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/.../checkpoint_000460`
- `PPO_mappo_vs_baseline_shaping_v2_512x512_20260413_040545/.../checkpoint_000470`
- `PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/.../checkpoint_000350`
- `PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/.../checkpoint_000360`
- `PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/.../checkpoint_000450`
- `PPO_mappo_vs_baseline_noshaping_512x512_20260413_030113/.../checkpoint_000490`
- `PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/.../checkpoint_000360`
- `PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/.../checkpoint_000420`
- `PPO_mappo_vs_baseline_shaping_v4_512x512_20260415_035251/.../checkpoint_000430`

### 清理后配额

- `21384M / 30720M`

### 结果

- 本轮预计回收约 `8.7G`
- `ray_results` 收缩到约 `12G`
- 当前剩余空间恢复到约 `9.1G`

### 备注

- 本轮没有触碰 `SNAPSHOT-021+` 的实验目录
- 本轮目标是“只精简旧实验的大体积 checkpoint，不破坏后续复核所需的证据链”
- 若后续仍需继续回收空间，下一优先级可考虑：
  - `021` 之后但已经被新 lane 明确替代的完整旧 run
  - 当前 formal 续跑完成后的中间 checkpoint 再压缩
