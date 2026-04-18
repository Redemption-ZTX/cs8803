# SNAPSHOT-002: H100 Overlay Environment Validation

- **日期**: 2026-04-08
- **负责人**: Codex
- **目标**: 在不破坏既有 `soccertwos` 环境的前提下，补一个 H100 可用的新环境，并完成一次真实 GPU smoke test。

## 背景

旧环境 `/home/hice1/wsun377/.conda/envs/soccertwos` 的关键版本是：

- `torch 1.8.1+cu102`
- `ray 1.4.0`
- `soccer-twos 0.1.14`
- `mlagents 0.27.0`

在 H100 节点上，旧环境虽然 `torch.cuda.is_available() == True`，但最小 CUDA 张量运算会报：

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

同时 PyTorch 运行时警告当前安装只支持较老的 CUDA 架构，而 H100 是 `sm_90`。因此问题不是训练脚本，而是 GPU 栈过旧。

PyTorch 官方的历史版本安装页提供了 CUDA 12.1 / 11.8 等新 wheel 的安装方式：

- https://pytorch.org/get-started/previous-versions/

## 方案

不修改旧 `soccertwos` 环境，改为在其上创建 overlay venv：

- 基础解释器：`/home/hice1/wsun377/.conda/envs/soccertwos/bin/python`
- overlay 路径：`/home/hice1/wsun377/.venvs/soccertwos_h100`
- 复用旧环境中的 `ray / soccer_twos / mlagents`
- 仅覆盖：
  - `torch==2.1.2+cu121`
  - `typing-extensions==4.13.2`

对应脚本：

- [setup_h100_overlay.sh](/home/hice1/wsun377/Desktop/cs8803drl/scripts/setup/setup_h100_overlay.sh)

## 运行结果

### 1. 最小 CUDA 验证

在 H100 节点 `atl1-1-03-012-28-0.pace.gatech.edu` 上执行 overlay env 验证，结果：

- `torch 2.1.2+cu121`
- `torch_file` 来自 overlay env
- `ray / soccer_twos / mlagents` 继续来自旧 `soccertwos` 环境
- `cuda_tensor 1.0`
- `cuda_device NVIDIA H100 80GB HBM3`

这说明 overlay 的核心目标达成：旧环境不动，Torch 成功切到 H100 可用版本。

### 2. GPU smoke train

运行命令：

```bash
srun --jobid=4627343 --overlap bash -lc '
cd /home/hice1/wsun377/Desktop/cs8803drl
export RUN_NAME=PPO_smoke_rewardfix_gpu_h100_overlay_20260408
export TIMESTEPS_TOTAL=4000
export TIME_TOTAL_S=1800
export CHECKPOINT_FREQ=1
export NUM_GPUS=1
export FRAMEWORK=torch
/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python -m cs8803drl.training.train_ray_team_vs_random_shaping
'
```

关键信号：

- 本地 learner 明确打印 `TorchPolicy (worker=local) running on 1 GPU(s).`
- rollout workers 仍在 CPU 上，这对当前 Ray 1.4 + PPO 配置是可接受的
- 训练正常完成 1 个 iteration，未再出现 `no kernel image` 错误

结果摘要：

- `training_iteration: 1`
- `timesteps_total: 4000`
- `episode_reward_mean: -0.8671302614808083`
- `episode_reward_max: -0.5399766609668738`
- `episode_reward_min: -1.0780933265686028`
- `time_total_s: 32.156121253967285`
- `num_healthy_workers: 4`

checkpoint：

- [checkpoint-1](/home/hice1/wsun377/Desktop/cs8803drl/ray_results/PPO_smoke_rewardfix_gpu_h100_overlay_20260408/PPO_Soccer_5e103_00000_0_2026-04-08_13-21-24/checkpoint_000001/checkpoint-1)

## 已知问题

### 1. `pip check` 仍有一个 metadata mismatch

overlay env 中会保留这条已知冲突：

```text
mlagents 0.27.0 has requirement torch<1.9.0,>=1.6.0
```

这是包 metadata 的旧约束，不代表当前运行一定失败。至少在这次验证里：

- `mlagents` 可 import
- Unity 环境可正常连接
- RLlib 可用 GPU 完成 1 iteration smoke train

### 2. Ray 1.4 在 PACE 上仍有 dashboard / metrics 噪声

训练过程中仍会看到类似：

```text
socket.gaierror: [Errno -2] Name or service not known
```

这来自 Ray dashboard/metrics agent 的 hostname 解析，在当前集群节点上是已知噪声。本次 smoke train 仍成功结束，因此暂记为非阻塞问题。

## 结论

当前最合理的 GPU 方案不是推倒旧环境重装，而是保留旧 `soccertwos` 作为 CPU / 兼容基线，再额外维护一个 H100 overlay env。

这条路线已经完成了：

1. H100 最小 CUDA 验证
2. `ray + soccer_twos + mlagents + torch` 联合 import 验证
3. 真实 GPU smoke train

因此后续正式训练建议优先使用：

- `/home/hice1/wsun377/.venvs/soccertwos_h100/bin/python`
