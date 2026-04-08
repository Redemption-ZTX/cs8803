# 项目路线图

> 优先级排序来源: [code-audit-000 § 7](../architecture/code-audit-000.md#7-优先级排序)。
> 作业评分标准: [Final Project Instructions Document.md](../references/Final%20Project%20Instructions%20Document.md#rubric)。
> 前任实验分析: [snapshot-000](../experiments/snapshot-000-prior-team.md)。

---

## 已确认的关键发现

在接手阶段发现的问题，直接影响路线选择：

1. **Reward shaping 从未生效** — `_extract_ball_pos()` 对 single_player 模式返回 None，前任 +1.91 reward 完全是稀疏奖励训练的结果。修复后性能可能大幅提升。详见本地验证结果。
2. **没有实际胜率数据** — 前任只有 reward 曲线，没跑过 vs baseline 的胜率评估。需要先测一次确认当前水平。
3. **没有提交格式的 agent** — 已创建 `agent_performance/` 和 `agent_reward_mod/` 骨架，但还没有训练好的 checkpoint。
4. **代码重复已清理** — `checkpoint_utils.py` 已提取为 canonical source。
5. **selfplay 路线失败** — 前任的 selfplay 训练 reward 持续下降，需要诊断原因或放弃该路线。

---

## 评分目标对照

| 评分项 | 分值 | 当前状态 | 需要做什么 |
|--------|------|---------|-----------|
| [Submission Integrity](../references/Final%20Project%20Instructions%20Document.md#submission-integrity-10-points) | 10 | 骨架已建，缺 checkpoint | 训练 → 放入 agent 目录 → zip |
| [Reward/Obs Modification](../references/Final%20Project%20Instructions%20Document.md#reward-observation-or-architecture-modification-40-points) | 40 | reward shaping 代码存在但有 bug | 修复 bug → 改进设计 → 消融实验 |
| [Policy Performance](../references/Final%20Project%20Instructions%20Document.md#policy-performance-50-points) | 50 | 未测试，预估不达标 | 训练至 vs random 9/10 + vs baseline 9/10 |
| [Bonus: Novel Concept](../references/Final%20Project%20Instructions%20Document.md#reward-observation-or-architecture-modification-40-points) | +5 | 未开始 | curriculum / imitation learning |
| [Report](../references/Final%20Project%20Instructions%20Document.md#report-100-points) | 100 | 未开始 | 训练曲线 + 超参表 + 对比图 + 分析 |

---

## 路线抉择：待讨论

### 抉择 1: 修复 reward shaping 后重新训练 vs 直接用现有 checkpoint

- **选项 A**: 修复 `_extract_ball_pos` bug → 重新从零训练 → 预期效果更好（shaping 终于生效了）
- **选项 B**: 先用现有 checkpoint_000712 测胜率 → 如果够用就先提交，再训练改进版
- **建议**: 先 B 再 A（10 分钟测胜率，有数据再决定）

### 抉择 2: 训练路线选择

- **team_vs_policy + reward shaping** — 前任验证过能训到 +1.91（无 shaping 下），修复 shaping 后是最有希望的路线
- **selfplay** — 前任失败了，但可能是网络太小 / entropy=0 / shaping 没生效导致的
- **curriculum** — 上游未修改，和 shaping 还没组合过
- **建议**: 主攻 team_vs_policy + 修复后的 shaping，selfplay 作为备选

### 抉择 3: 工作环境分配

- **PACE 集群**: 主力训练环境（GPU + 长时间运行）
- **本地 (Windows + GPU + CUDA)**: 开发、调试、短时间训练、评估测试、可视化

---

## 立即执行（接手收尾）

- [ ] commit + push 当前所有变更
- [ ] PACE 上 clone 并 `bash scripts/setup.sh`
- [ ] 用 checkpoint_000712 跑 vs random 和 vs baseline 胜率测试
- [ ] 团队讨论上面的抉择 1-3，确定分工和优先级
- [ ] 确定后更新本文件，开始实验迭代

---

## 协作方式

两人平面协作，各自探索路线，定期同步讨论，取最优方案或融合精华。不做固定分工。

## 任务追踪

| 任务 | 状态 | 备注 |
|------|------|------|
| 接手收尾 commit + push | TODO | |
| 环境部署（PACE + 本地） | TODO | 本地已验证可用 |
| 现有 checkpoint 胜率测试 | TODO | |
| 修复 reward shaping bug | TODO | 已确认 _extract_ball_pos 失效 |
| 训练路线确定 | TODO | 待测试数据支撑 |
| Agent Performance 训练 | TODO | |
| Agent Reward Mod 训练 | TODO | |
| 消融实验 | TODO | |
| 报告 | TODO | |
| 打包提交 | TODO | |

> 状态: `TODO` → `IN PROGRESS` → `DONE`
