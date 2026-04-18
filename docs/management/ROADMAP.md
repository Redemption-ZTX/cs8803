# 项目路线图

> **作业要求 SSOT**: [Final Project Instructions Document.pdf](../references/Final%20Project%20Instructions%20Document.pdf)（PDF 为准，[Markdown 版](../references/Final%20Project%20Instructions%20Document.md)为辅）。
> 优先级排序来源: [code-audit-000 § 7](../architecture/code-audit-000.md#7-优先级排序)。
> 前任实验分析: [snapshot-000](../experiments/snapshot-000-prior-team.md)。

---

## 作业要求澄清

以下是从 SSOT 重新确认的要求，更正了之前文档中的误解：

1. **Modification 是 40 分一项**（Reward OR Observation OR Architecture），不是要分两个 agent 分别做。一个 agent 展示修改即可，另一个 agent 拼 Performance。
2. **提交多个 agent** — 但不是每个评分项一个 agent。只需要在报告和描述中说明哪个 agent 对应哪项评分。
3. **Report 100 分独立于代码 100 分** — 报告有具体的逐项评分（超参表 10 分、训练曲线 10 分、对比图 10 分、分析 30 分等），每次实验的 snapshot 记录要完整到可以直接导出报告。
4. **competitive agent2 bonus (+5)** — 对手尚未发布，暂不考虑。
5. **Action space 差异** — 作业文档描述 "continuous action space"，实际环境为 `MultiDiscrete([3,3,3])`，代码中用 `ActionFlattener` 展平为 `Discrete`。以代码为准。

---

## 已确认的关键发现

1. **Reward shaping 从未生效** — `_extract_ball_pos()` 对 single_player 模式返回 None（本地已验证）。前任 +1.91 reward 完全是稀疏奖励训练的结果。修复后性能可能大幅提升。
2. **没有实际胜率数据** — 前任只有 reward 曲线，需要先测一次 vs baseline 胜率。
3. **Agent 模块骨架已建** — `agents/_template/` 就绪，训练出 checkpoint 即可打包。
4. **代码重复已清理** — `checkpoint_utils.py` 为 canonical source。
5. **selfplay 路线失败** — reward 持续下降，原因待诊断。

---

## 评分目标对照

| 评分项 | 分值 | 当前状态 | 需要做什么 |
|--------|------|---------|-----------|
| [Submission Integrity](../references/Final%20Project%20Instructions%20Document.md#submission-integrity-10-points) | 10 | 模板就绪，缺 checkpoint | 训练 → 放入 agents/vNNN/ → 最终复制到根目录 → zip |
| [Modification](../references/Final%20Project%20Instructions%20Document.md#reward-observation-or-architecture-modification-40-points) | 40 | reward shaping 存在但有 bug | 修复 → 改进 → 消融实验证明有效 |
| [Performance](../references/Final%20Project%20Instructions%20Document.md#policy-performance-50-points) | 50 | 未测试 | 训练至 vs random 9/10 + vs baseline 9/10 |
| [Bonus](../references/Final%20Project%20Instructions%20Document.md#reward-observation-or-architecture-modification-40-points) | +5 | 未开始 | curriculum / imitation learning |
| [Report](../references/Final%20Project%20Instructions%20Document.md#report-100-points) | 100 | 未开始 | 需要：算法说明、超参表、修改描述、假设动机、训练曲线、对比图、标注清晰的图表、性能对比陈述、技术分析 |

---

## 路线抉择：待讨论

### 抉择 1: 修复 reward shaping 后重新训练 vs 先用现有 checkpoint

- **选项 A**: 修复 bug → 从零重训 → shaping 终于生效，预期大幅提升
- **选项 B**: 先用 checkpoint_000712 测胜率 → 有数据再决定
- **建议**: 先 B 再 A（10 分钟测胜率，有数据再决定）

### 抉择 2: 训练路线

- **team_vs_policy + reward shaping** — 前任无 shaping 下训到 +1.91，修复后最有希望
- **selfplay** — 前任失败，可能是网络太小 / entropy=0 / shaping 失效
- **curriculum** — 上游未修改，可和 shaping 组合
- **建议**: 主攻 team_vs_policy + 修复后的 shaping

### 抉择 3: 工作环境

- **PACE**: 主力 GPU 训练（必须用 `$SCRATCH`，`sbatch` 提交，不在登录节点跑）
- **本地 (Windows + GPU + CUDA)**: 开发、调试、短训练、评估、可视化（Unity binary 已内置，无需安装 Unity）

---

## 立即执行

- [x] commit + push 当前所有变更
- [ ] PACE 部署（按 [deploy-and-verify.md](deploy-and-verify.md) 操作）
- [ ] 用 checkpoint_000712 测 vs random 和 vs baseline 胜率
- [ ] 讨论抉择 1-2，确定路线
- [ ] 开始实验迭代

---

## 协作方式

两人平面协作，各自探索路线，定期同步讨论，取最优方案或融合精华。不做固定分工。

## 任务追踪

| 任务 | 状态 | 备注 |
|------|------|------|
| 接手收尾 | DONE | 已 commit + push |
| 环境部署 | IN PROGRESS | 本地已验证，PACE 待部署 |
| 现有 checkpoint 胜率测试 | TODO | |
| 修复 reward shaping bug | DONE | 已通过 `soccertwos` 真环境验证，见 `snapshot-001` |
| 训练路线确定 | TODO | 待测试数据 |
| Modification agent 训练 | TODO | |
| Performance agent 训练 | TODO | |
| 消融实验 | TODO | |
| 报告 | TODO | |
| 打包提交 | TODO | |

> 状态: `TODO` → `IN PROGRESS` → `DONE`
