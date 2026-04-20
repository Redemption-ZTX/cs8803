# 工程规范

> 架构总览见 [overview.md](overview.md)。代码审计见 [code-audit.md](code-audit.md)（最新: [code-audit-000](code-audit-000.md)）。
> 文件保护等级与强制规则见根目录 `CLAUDE.md` 的 Rules 章节。

---

## 环境搭建

### 一键安装

```bash
bash scripts/setup/setup.sh              # 全量：conda + 依赖 + baseline + 验证
bash scripts/setup/setup.sh --verify     # 仅验证
bash scripts/setup/setup.sh --deps-only  # 跳过 conda 创建，只装依赖
bash scripts/setup/setup.sh --no-baseline  # 跳过 baseline 下载
```

### 手动安装

```bash
# 1. 创建 conda 环境
conda create --name soccertwos python=3.8 -y && conda activate soccertwos

# 2. 锁定构建工具
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge

# 3. 安装依赖
pip install -r requirements.txt
pip install protobuf==3.20.3 pydantic==1.10.13

# 4. Baseline agent — 已在 git 仓库中
#    如果缺失，从 Google Drive 下载:
#    https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view
#    解压到项目根目录的 ceia_baseline_agent/

# 5. 验证
bash scripts/setup/setup.sh --verify
```

### PACE 集群注意事项

- 通过 GT VPN → `ssh login-ice.pace.gatech.edu` 连接
- home 目录 15GB 限制，用 scratch 目录存大文件: `$SCRATCH` 或 `/storage/ice-shared/`
- conda 建议装到 scratch 下，避免 home 爆满
- 提交作业: `sbatch scripts/batch/starter/soccerstwos_job.batch`
- **不要在登录节点跑训练** — 必须通过 SLURM 提交
- 更多: [PACE-ICE 文档](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102)、[Slurm 指南](https://mshalimay.github.io/slurm_pace_guidelines/)

### 环境约束

| 项目 | 版本 | 说明 |
|------|------|------|
| Python | 3.8 (strict) | 上游兼容性，不可升级 |
| Ray / RLlib | 1.4.0 (strict) | 同上 |
| PyTorch | 随 Ray 1.4.0 安装 | 所有训练脚本 `framework: "torch"` |
| protobuf | 3.20.3 | 兼容性 pin |
| pydantic | 1.10.13 | 兼容性 pin |

---

## Git 协作

### 分支策略

```
main                  # 稳定分支，只接受 PR 合入
├── feat/xxx          # 新功能（如 feat/reward-shaping-v2）
├── fix/xxx           # 修复
├── exp/xxx           # 实验性分支（如 exp/curriculum-v2）
└── agent/xxx         # 特定 agent 开发（如 agent/imitation）
```

### Commit 流程

每次 commit（无论人还是 AI）必须按以下步骤执行：

```
1. Stage       git add <specific files>          # 禁止 git add -A 或 git add .
2. Check       git status                        # 确认只有目标文件被 staged
               git diff --cached                 # 逐行审查 staged 内容
3. Validate    - 没有 FROZEN 文件？
               - 没有 ray_results/, checkpoint*, *.ckpt, soccerstwos-*.out？
               - 没有 secrets (.env, credentials)？
               - 没有意外的大文件？
4. Message     git commit -m "<type>: <description>"
                 type: feat | fix | exp | docs | refactor | chore
                 description: 祈使语气，< 72 字符，英文
                 body (可选): 说 why，不说 what — diff 已经说了 what
5. Verify      git log --oneline -3              # 确认 commit 正确
               git diff HEAD~1 --stat            # 确认文件列表符合预期
```

**Commit Message 模板**:

```
# 单行（简单改动）
<type>: <description>

# 多行（需要解释 why）
<type>: <description>

<body: why this change, not what — the diff shows what>

# 带 scope（可选，标注影响范围）
<type>(<scope>): <description>
```

**示例**:

```bash
# 功能
feat: add potential-based reward shaping to RewardShapingWrapper
feat(selfplay): expand opponent pool from 3 to 5 snapshots

# 修复
fix: prevent silent reward shaping failure when info keys missing

# 实验
exp: test entropy_coeff=0.01 with 512x512 network

# 文档
docs: add snapshot-001 for entropy experiment

# 重构
refactor: extract checkpoint loading logic into checkpoint_utils.py

# 杂务
chore: update .gitignore to exclude SLURM logs
```

**规则**:
- **指定文件 stage** — `git add -A` 被禁止，太容易夹带私货
- **一个 commit 一件事** — 不要把功能、文档、格式混在一起
- **commit message 用英文** — 文档内容可以中文，但 git 历史保持英文
- **hook 失败不 amend** — 修问题，重新 stage，创建新 commit
- **`subtree / vendor / 第三方打包目录默认禁止推远端`** — 例如 `ceia_baseline_agent/` 以及未来任何 subtree / vendor / imported bundle，都视作只读依赖；除非用户明确授权，否则不做 remote push / subtree sync / upstream publish

### PR 流程

1. 从 `main` 拉分支开发
2. 完成后提 PR，另一人 review
3. 通过后合入 `main`，更新根目录 `CHANGELOG.md`
4. 纯实验分支可以不走 PR，但需在 [experiments/README.md](../experiments/README.md) 记录

---

## 代码风格

- Python 风格遵循 PEP 8
- 文件命名: `snake_case.py`
- 训练脚本前缀: `train_ray_*.py`
- 评估脚本前缀: `eval_*.py` / `evaluate_*.py`
- Agent 模块: 根目录独立文件夹，含 `__init__.py` + `README.md`（[提交格式要求](../references/Final%20Project%20Instructions%20Document.md#submission-integrity-10-points)）
- 配置使用 YAML 或环境变量，不要硬编码到训练脚本中

## 文件组织

不要提交的文件（确保在 `.gitignore` 中）:

```
ray_results/         # 训练结果（太大）
checkpoint*/         # checkpoint 文件
*.ckpt               # checkpoint 文件
soccerstwos-*.out    # SLURM 作业日志
__pycache__/
*.egg-info/
.env
```

---

## 迭代流程

### 代码迭代

```
拉分支 → 实现 → 自检 → commit → PR review → 合入 main → 更新文档
```

**自检清单**（commit 前必过）：
- [ ] 现有训练脚本能正常 `from cs8803drl.core.utils import ...`？
- [ ] agent 模块仍然满足提交格式（`__init__.py` + `AgentInterface`）？
- [ ] 新增的可调参数用环境变量，而非硬编码？
- [ ] 没有 FROZEN 文件在改动中？
- [ ] 没有意外提交 checkpoint / ray_results？

### 实验迭代

完整 12 步，必须按顺序执行：

1. **假设** — 你认为会发生什么？为什么？先写下来
2. **建 Snapshot** — 创建 `docs/experiments/snapshot-NNN-title.md`，实验开始前填好假设和配置
3. **确认 commit 干净** — `git status` 必须干净，脏状态 = 不可复现
4. **记录 commit hash** — 写到 snapshot 里
5. **配置** — 通过环境变量设置超参，不要改脚本默认值
6. **运行** — 在 PACE 或本地执行训练
7. **监控** — 查看 TensorBoard / progress.csv，关注早期发散信号
8. **记录结果** — 填入 snapshot：指标、训练曲线、观察
9. **分析** — 是否符合假设？为什么是/不是？
10. **同步 rank.md** — 任何 `official 500` / `1000ep` / `H2H` / `random eval` 出结论都**必须**立即同步到 [`docs/experiments/rank.md`](../experiments/rank.md)（单一真相源）。包括:
    - 新模型进 §1 Model Registry
    - baseline WR 入 §3
    - H2H 入 §5 矩阵（按 §0.2 验证方向后再填）
    - 更新 §6 排名（如有变动）
    - §8 Changelog 加一行
11. **决策** — 三选一：
    - **Iterate**: 调整参数 → 回到步骤 1，新 snapshot
    - **Adopt**: 合入代码到 main，更新 CHANGELOG
    - **Abandon**: 在 snapshot 中记录原因，继续下一个方向
12. **如果做了架构决策** → 写 ADR 到 `docs/architecture/adr/`

**关键原则**：
- **先写假设再跑实验** — snapshot 文件在实验开始前就要创建
- **commit hash 必须记录** — 没有 hash 的实验结果不可复现，等于没跑
- **只用环境变量调参** — 不要为了跑实验去改训练脚本里的默认值
- **失败也记录** — 知道什么不行和知道什么行一样重要
- **数字必须进 rank.md** — snapshot 里可以展开分析，但 baseline / H2H / random 的**原始数字**以 rank.md 为准。snapshot 和 rank.md 冲突时以 rank.md 为准
- **读 H2H log 前强制先按 [rank.md §0.2](../experiments/rank.md#02-正确读-h2h-的命令) 的 grep 命令验证方向** —— 2026-04-18 曾经因为假设"第一段是文件名前位 agent"把整个 snapshot-037 读反。不允许靠"第一段是谁"的猜测做 H2H 结论

### Checkpoint 选模规则

从 2026-04-12 的 shaping-v1 / v2 复核开始，checkpoint 选模统一采用以下规则：

1. **训练内评估只用于筛候选，不用于直接定 best**
   - 默认使用 `baseline 50` 局
   - 作用是缩小候选窗口，不做最终结论

2. **正式复核按 `top 5% + ties`**
   - 仅按训练内 `baseline` 胜率排序
   - 取前 `5%` 的 checkpoint
   - 如果第 `5%` cutoff 分数有并列，则同分 checkpoint 全部纳入

3. **最终选模先看 `baseline 500`**
   - 对 `top 5% + ties` 的全部候选跑官方 `500` 局 `baseline`
   - 先只比较 `baseline`，不混入 `random`

4. **`random` 只做最终补充确认**
   - 在 `baseline 500` 里筛出最终 `1-2` 个候选后
   - 再补 `random 500`
   - 不再用 `random` 作为主判据

5. **若大样本结果与训练内 `50` 局排序冲突，以 `500` 局为准**
   - 训练内 `50` 局可能只是在抓到噪声尖峰
   - 官方 `500` 局才是最终依据

6. **在文档中必须同时记录两层结果**
   - 训练内候选窗口（`50` 局）
   - 官方大样本复核结果（`500` 局）

7. **failure capture 的 checkpoint 选择跟随 `baseline 500` 结果**
   - 失败样本分析优先针对大样本复核后真正最强的 checkpoint
   - 不再默认分析训练内单个最高点

### Post-training eval SOP

训练完成后的 baseline / failure capture / H2H 三阶段 SOP 封装在 **`/post-train-eval` skill** 里 (`.claude/skills/post-train-eval/SKILL.md`)。

**用法**：在 Claude 会话里说 `/post-train-eval <lane>` 或 "对 <lane> 做 post-train eval"，Claude 会按 SOP 执行 3 stages，每阶段有 gating decision。

**Stage 概览**：
1. **Stage 1**: baseline 1000ep eval（top 5% + ties + ±1 整 window）
2. **Stage 2**: failure capture 500ep（1-2 个 baseline 最强 ckpt）
3. **Stage 3**: H2H 500ep（vs **snapshot 推荐对手**，不用硬编码列表）
4. **Stage 4 (optional)**: 基于 1-3 结果自行扩展（rerun, 加 H2H 对手, 第二个 ckpt capture）—— 但要 ask user

**关键纪律**：
- 数据 append-only 写回 snapshot + rank.md，必须包含原始 `---- Summary ----` / `---- H2H Recap ----` raw block 供复查
- H2H log 必读 `team0_overall_win_rate`（rank.md §0.2 防 037 错读）
- 100ep SE ±0.05 / 500ep SE ±0.022 / 1000ep SE ±0.016 — 不在 SE 内写"显著差异"

**辅助工具**：
- [`pick_top_ckpts.py`](../../scripts/eval/pick_top_ckpts.py) — top 5%+ties+±1 候选选取
- [`list_my_gpu_nodes.sh`](../../scripts/eval/list_my_gpu_nodes.sh) — 节点 scout（人工选 node 用）

skill 不会做的事：submit SLURM job、改训练脚本、auto-rerun 超过 Stage 4。

### Batch / 节点 / 端口 经验集（参考性，非强制）

实战中反复踩到的几个坑，记录下来供参考——不是必须遵守的规则，但当训练莫名挂起或 launch 失败时优先排查这些方向。

#### Port shelf 上限

Ray + Unity 训练每条 lane 大约占 `8 worker × 5 env = 40` 个连续端口；如果 batch 用 `PORT_OFFSET=$(( (PORT_SEED % 20) * 100 ))` 模式，PORT_OFFSET 0-1900。所以 `BASE_PORT_anchor + 1900 + 40 ≤ 65535` → **anchor 的安全上界 ≈ 63595**。

经验值：常用 anchor 落在 50000–63000 之间。具体每条 lane 用什么 shelf 可以从已有 batch 抄（搜 `BASE_PORT.*PORT_OFFSET`）；只要保证不和**当前 alive 的训练**重叠就行（不同节点之间端口不冲突）。

如果未来批量训练增多想要更宽 PORT_OFFSET 范围（比如 % 50 → 0-4900），那 anchor 上界要相应下调到 ~60500。

#### 失败 launch 留下的 socket 半绑定

如果一次 launch 因为代码 bug / port 冲突 fail 了，**Unity 进程可能没干净退出**，留下半绑定的 gRPC socket，TCP TIME_WAIT 几分钟才释放。如果立刻同 port shelf 重启，常见**第二次也 fail**（`UnityWorkerInUseException`）。

实战快速对策（按 ROI 排序）：
1. **换 port shelf 重启** — 最快，5 行 batch 改一下
2. 等 5 分钟 TCP TIME_WAIT 完成再用同 shelf
3. 在节点上 `pkill -u $USER -f "Soccer.x86_64"` 强杀所有 Unity binary（python pkill 可能漏掉它们）

#### 节点上的端口可能被其他用户占

PACE 节点大多 GPU 独占但 CPU/network 共享。`ss -tlnp` 看到的 LISTEN 端口可能是别人的进程，你**看不到 owner pid**（无 sudo）。所以如果 Unity 莫名 `worker number N is still in use` 但没看到自己的 zombie，**第一假设是别人占了**——立刻换 shelf。

#### `RESUME_CHECKPOINT` / `RESTORE_CHECKPOINT` 在 batch 里被 unset 的坑

每条 batch 头部常有一段 "scratch hygiene" unset 块（`unset RESTORE_CHECKPOINT` 等）防止上一次 leak 进来。这会破坏 wrapper 调用 batch 时通过 env var 传 **续跑 checkpoint** 的能力。

当前推荐口径是：

- **首选** `RESUME_CHECKPOINT`
- `RESTORE_CHECKPOINT` 仅作为向后兼容 alias

也就是说，默认语义现在是：

- `WARMSTART_CHECKPOINT` = 迁移初始化
- `RESUME_CHECKPOINT` = continuation / 续跑

如果你需要 batch 既能 scratch 也能续训，参考 [031B batch fix](../../scripts/batch/experiments/soccerstwos_h100_cpu32_team_level_031B_cross_attention_scratch_v2_512x512.batch)：

```bash
_RESUME_OVERRIDE="${RESUME_CHECKPOINT:-${RESTORE_CHECKPOINT:-}}"
unset RESUME_CHECKPOINT
unset RESTORE_CHECKPOINT
unset BC_WARMSTART_CHECKPOINT
unset WARMSTART_CHECKPOINT
... [其他 unsets]
[[ -n "$_RESUME_OVERRIDE" ]] && export RESUME_CHECKPOINT="$_RESUME_OVERRIDE"
```

不强制所有 batch 都立即改成这个 pattern，但当某条 lane 训练中断需要续跑时，加上这几行后就能直接用 `RESUME_CHECKPOINT=... bash <batch>` 续，不需要再 patch 训练脚本。

更重要的是，**默认续跑应该沿用同一个 `RUN_NAME` / 同一个 canonical run root 原地继续**。  
不要为了“怕脏写”而新建一个空目录 resume；那样会把 lineage 拆成多个顶层 run root，后面 summary / curve / eval 都更难读。只有当你是有意做分叉实验时，才应该新开目录。

#### 节点选取（人工判断，不要自动化）

`scripts/eval/list_my_gpu_nodes.sh` 列出所有 GPU job + 每节点的 tmux 数量、python 进程数、GPU 占用——**仅为信息源，不做决策**。

经验法则：
- **GPU 利用率 > 0%**：节点上有人在跑训练，eval 跑这里会和训练抢 CPU/memory
- **python 进程数 > 0**：可能是训练或 ray 残留；先 `pgrep -af python` 看具体
- **tmux 数量 > 0 但 python = 0**：sleeping wrapper，节点本质空闲，可以用
- **Remaining < 8h**：不适合启动 14h scratch 训练，但跑 eval（30-60 min）没问题

这些都是**经验启发式**，不是死规则——当观察和判断冲突时（比如 GPU 0% 但 ss 显示一堆 LISTEN port），相信观察。

---

## 环境变量速查

### `train_ray_team_vs_random_shaping.py`

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RESUME_CHECKPOINT` | 空 | 从 checkpoint 续跑（首选） |
| `RESTORE_CHECKPOINT` | 空 | `RESUME_CHECKPOINT` 的兼容 alias |
| `RESUME_TIMESTEPS_DELTA` | 0 | 续跑时在原 checkpoint 步数基础上追加的训练步数 |
| `RESTORE_TIMESTEPS_DELTA` | 0 | `RESUME_TIMESTEPS_DELTA` 的兼容 alias |
| `TIMESTEPS_TOTAL` | 15000000 | 总训练步数 |
| `TIME_TOTAL_S` | 7200 | 最大训练时间（秒） |
| `NUM_GPUS` | 1 | GPU 数量 |
| `FRAMEWORK` | torch | 训练框架 |
| `CHECKPOINT_FREQ` | 20 | checkpoint 保存频率 |
| `EVAL_INTERVAL` | 0 | 评估间隔（0=关闭） |
| `EVAL_EPISODES` | 10 | 每次评估局数 |
| `EVAL_BASE_PORT` | 7005 | 评估环境端口 |
| `EVAL_MAX_STEPS` | 1500 | 每局评估最大步数 |
| `RUN_NAME` | PPO_team_vs_random_shaping | 实验名 |

### `train_ray_selfplay.py`

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BASELINE_PROB` | 0.7 | 对手为 baseline 的概率 |
| `BASELINE_EVAL_INTERVAL` | 10 | 对战基线评估间隔 |
| `BASELINE_EVAL_EPISODES` | 10 | 评估局数 |
| `BASELINE_EVAL_MAX_STEPS` | 1500 | 每局评估最大步数 |
| `BASELINE_EVAL_BASE_PORT` | 19100 | 评估环境端口 |
| `SHAPING_TIME_PENALTY` | 0.001 | 每步时间惩罚 |
| `SHAPING_BALL_PROGRESS` | 0.01 | 球推进奖励系数 |
| `SHAPING_OPP_PROGRESS_PENALTY` | 0.0 | 对手推进惩罚（默认关闭） |
| `SHAPING_POSSESSION_DIST` | 1.25 | 控球判定距离 |
| `SHAPING_POSSESSION_BONUS` | 0.002 | 控球奖励 |

### `train_ray_curriculum.py`

无自定义环境变量。所有配置硬编码在脚本中（网络、停止条件等）。课程任务定义在 `curriculum.yaml`。

### `scripts/batch/starter/soccerstwos_job.batch` 覆盖值

PACE 作业脚本中预设了部分环境变量，会覆盖脚本默认值：

| 变量 | 脚本默认 | batch 覆盖 | 原因 |
|------|---------|-----------|------|
| `TIME_TOTAL_S` | 7200 | 14000 | 匹配 SLURM 4h 时间限制 |
| `NUM_GPUS` | 1 | 1 | 一致 |
| `CHECKPOINT_FREQ` | 20 | 20 | 一致 |

详见 [code-audit-000 § 2.1](code-audit-000.md#21-整体设计)。

---

## 文档更新速查

| 事件 | 更新什么 |
|------|---------|
| 代码合入 main | `CHANGELOG.md` |
| 架构变更 | `docs/architecture/overview.md` 或新 ADR |
| 开始实验 | 新建 `docs/experiments/snapshot-NNN.md` + 更新 `experiments/README.md` 索引 |
| 实验结束 | 补完 snapshot 的结果和结论 |
| 重大重构 | 新建 `docs/architecture/code-audit-NNN.md` + 更新 `code-audit.md` 索引 |
| 新增 ADR | 更新 `adr/README.md` 索引 + `adr/topic-map.md` |
| 新增/删除/重命名任何文件 | 同步更新 `docs/README.md` 文档地图 + 根 `README.md` 项目结构 + 相关子索引 |
| 每日工作 | `docs/management/WORKLOG.md` |
