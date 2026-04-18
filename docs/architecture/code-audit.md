# 代码审计

> 架构总览见 [overview.md](overview.md)。作业要求见 [Final Project Instructions Document.md](../references/Final%20Project%20Instructions%20Document.md)。

## 审计索引

| 编号 | 标题 | 日期 | 代码状态 |
|------|------|------|---------|
| [000](code-audit-000.md) | 接手时代码审计 | 2026-04-07 | 上游 + 前任修改，未做任何优化 |
| [001](code-audit-001.md) | 隐藏约束与默认假设审查（assumption-driven audit）| 2026-04-15 | 9-lane 数据齐备后，针对 `low_possession` 跨 lane 不变量做的 focused audit；3 🔴 + 5 🟡 + 7 ⚪ 新发现，含 actor-only-own-obs / policy 时间盲 / agent_id spawn 稳定性未验证 等|

## 命名规范

- `code-audit-NNN.md` — 每次重大代码变更后的审计 snapshot
- 编号递增，NNN 从 000 开始
- 本文件只做索引，不放具体审计内容
- 新审计应注明相比上一版的变更点
