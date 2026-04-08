# 部署与验证交接文档

> 工程规范见 [engineering-standards.md](../architecture/engineering-standards.md)。
> 前任实验分析见 [snapshot-000](../experiments/snapshot-000-prior-team.md)。
> 代码审计见 [code-audit-000](../architecture/code-audit-000.md)。

本文档覆盖：从 git clone 到确认环境可用、验证现有 checkpoint、确认 reward shaping 是否生效的完整流程。

---

## Step 1: 部署环境

### 1.1 PACE 集群

```bash
# 连接（先开 GT VPN）
ssh YOUR_GT_USERNAME@login-ice.pace.gatech.edu

# 建议在 scratch 目录操作（home 只有 15GB）
cd $SCRATCH

# 克隆仓库
git clone https://github.com/Redemption-ZTX/cs8803.git
cd cs8803

# 一键安装
bash scripts/setup.sh
```

如果 conda 不在默认路径，先 source：
```bash
source ~/miniconda3/etc/profile.d/conda.sh
# 或者按 PACE 文档加载 module
```

### 1.2 本地（Windows/macOS/Linux）

```bash
git clone https://github.com/Redemption-ZTX/cs8803.git
cd cs8803
bash scripts/setup.sh
```

### 1.3 验证环境

```bash
bash scripts/setup.sh --verify
```

预期输出：所有项显示 ✓。如果 baseline checkpoint 显示 ✗，检查 `ceia_baseline_agent/` 是否完整。

### 1.4 快速烟雾测试

```bash
conda activate soccertwos
python examples/example_random_players.py
```

如果 Unity 窗口弹出（本地）或无报错退出（headless），环境正常。

---

## Step 2: 测试现有最佳 checkpoint

前任最佳 checkpoint: `ray_results/PPO_team_vs_mix_baseline90_random10_cont_eval_cont2/PPO_Soccer_e71e6_00000_0_2026-03-31_15-57-53/checkpoint_000712/`

对应 reward 峰值 +1.91，详见 [snapshot-000 § 可用 Checkpoints](../experiments/snapshot-000-prior-team.md#可用-checkpoints)。

### 2.1 用 eval_rllib_checkpoint_vs_baseline.py 评估

```bash
conda activate soccertwos

# 找到 checkpoint 文件
CKPT="ray_results/PPO_team_vs_mix_baseline90_random10_cont_eval_cont2/PPO_Soccer_e71e6_00000_0_2026-03-31_15-57-53/checkpoint_000712/checkpoint-712"

# 对战基线，50 局
python eval_rllib_checkpoint_vs_baseline.py -c "$CKPT" -n 50
```

记录输出的 `win_rate`。

### 2.2 用 evaluate_matches.py 评估（通过 agent 模块）

```bash
# 先把 checkpoint 放到 agent_performance 目录
cp "$CKPT" agent_performance/checkpoint
# params.pkl 也需要
cp "$(dirname $CKPT)/../params.pkl" agent_performance/params.pkl

# 对战基线
python evaluate_matches.py -m1 agent_performance -m2 ceia_baseline_agent -n 10

# 对战随机
python evaluate_matches.py -m1 agent_performance -m2 example_player_agent -n 10
```

### 2.3 需要记录的指标

| 指标 | 结果 | 作业要求 |
|------|------|---------|
| vs Random Agent 胜率 | _待填_ | 9/10 = 25 分 |
| vs Baseline Agent 胜率 | _待填_ | 9/10 = 25 分 |
| 评估局数 | _待填_ | — |
| 使用的 checkpoint | _待填_ | — |
| commit hash | _待填_ | — |

**将结果记录到新的 experiment snapshot**（`docs/experiments/snapshot-001-baseline-eval.md`）。

---

## Step 3: 验证 Reward Shaping 是否生效

这是 [code-audit-000 § 1.2](../architecture/code-audit-000.md#12-rewardshapingwrapperl56-225) 中标记的 P0 风险：`_extract_ball_pos()` 依赖 info dict 的字段名，可能静默返回 None。

### 3.1 运行验证脚本

在项目根目录创建并运行：

```python
# verify_reward_shaping.py
"""One-off script to verify reward shaping extracts ball/player positions."""
import soccer_twos
from utils import create_rllib_env, RewardShapingWrapper

env = create_rllib_env({
    "variation": soccer_twos.EnvType.team_vs_policy,
    "multiagent": False,
    "single_player": True,
    "flatten_branched": True,
    "reward_shaping": True,
})

# Unwrap to find RewardShapingWrapper
cur = env
found_shaping = False
while hasattr(cur, 'env'):
    if isinstance(cur, RewardShapingWrapper):
        found_shaping = True
        break
    cur = cur.env

print(f"RewardShapingWrapper present: {found_shaping}")

obs = env.reset()
for step in range(5):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    # Test extraction
    ball_pos = RewardShapingWrapper._extract_ball_pos(info)
    player_pos = RewardShapingWrapper._extract_player_positions(info)

    print(f"\nStep {step}:")
    print(f"  info type: {type(info)}")
    if isinstance(info, dict):
        print(f"  info keys: {list(info.keys())[:5]}")
        for k, v in list(info.items())[:2]:
            if isinstance(v, dict):
                print(f"  info[{k}] keys: {list(v.keys())[:10]}")
    print(f"  ball_pos: {ball_pos}")
    print(f"  player_pos: {player_pos}")
    print(f"  reward: {reward}")

    if done:
        obs = env.reset()

env.close()

if ball_pos is None:
    print("\n⚠ WARNING: ball_pos is None — reward shaping is NOT working!")
    print("Check info dict keys above and update _extract_ball_pos() accordingly.")
else:
    print("\n✓ Reward shaping is extracting ball position correctly.")
```

```bash
python verify_reward_shaping.py
```

### 3.2 可能的结果

**如果 ball_pos 不是 None** — reward shaping 正常工作，前任的训练结果是有效的。

**如果 ball_pos 是 None** — reward shaping 静默失效，前任 +1.91 的 reward 完全是稀疏奖励训练出来的。这意味着：
- reward shaping 还有巨大的未开发潜力
- 需要查看 info 的实际 key 结构，修复 `_extract_ball_pos()`
- 这是一个重大发现，需要写 ADR

### 3.3 记录结果

将验证结果附加到 Step 2 创建的 snapshot 中，或单独建一个 snapshot。

---

## Step 4: 下一步

完成以上验证后：
1. 将结果汇总到 snapshot
2. 回来讨论 ROADMAP — 根据胜率和 shaping 验证结果决定优化方向
3. 开始实验迭代

---

## 故障排查

| 问题 | 解决 |
|------|------|
| `ModuleNotFoundError: No module named 'soccer_twos'` | `pip install soccer-twos` 或检查 conda env 是否激活 |
| `mlagents_envs.exception.UnityWorkerInUseException` | 端口被占用，加 `--base_port 9200` 换端口 |
| Unity binary 找不到 | soccer_twos 会自动下载，检查网络连接 |
| `FileNotFoundError: checkpoint` | 确认路径正确，PACE 上注意 scratch vs home 路径 |
| `numpy` / `protobuf` 版本错误 | `bash scripts/setup.sh --verify` 检查，重装对应版本 |
| PACE 排队太久 | 避开 DDL 前几天，或用 `--partition` 指定空闲分区 |
