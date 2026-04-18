# 计划文档

这里存放跨多轮实验的中期计划，不替代 `snapshot`。

## 用途

- `experiments/snapshot-*.md`
  - 记录单次实验、单条假设、单个 A/B 或单次复核
- `plan/*.md`
  - 记录跨多个 snapshot 的阶段性主线
  - 说明为什么转向、主线拆分、阶段目标、退出条件

## 当前计划

- [PLAN-001](plan-001-il-baseline-exploitation.md)
  - 从 shaping-only 路线转向 `imitation learning / baseline exploitation / ensemble`
- [PLAN-002](plan-002-il-mappo-dual-mainline.md)
  - 当前主线：`IL / BC + baseline weakness analysis + MAPPO 公平对照 + later ensemble`

## 维护原则

1. `plan` 只写阶段性路线，不记运行日志。
2. 每条 plan 必须链接到对应 snapshot。
3. 当主线发生实质性转向时，新增新的 `plan-00N-*.md`，不要覆写旧计划。
