# SNAPSHOT-007: Base Lane Reset 与目录重构

- **日期**: 2026-04-11
- **负责人**:
- **目标**:
  - 暂停继续叠加 PPO 变体与下游适配假设，回到 SSOT 与 starter example 重新定义基础模型路线
  - 将仓库从根目录平铺结构重构为带职能分层的包结构和脚本结构
  - 将后续实验矩阵重置为“先训练基础模型，再做 warm-start / adaptation”

## 1. 关键判断重置

重新对照 [SSOT](../references/Final%20Project%20Instructions%20Document.md) 和 starter example 后，明确两点：

1. 当前优先回答的问题，不是“再换一个 PPO 变体是否会突然过线”，而是“哪条 starter-aligned 基础模型主线最强”。
2. `warm-start` 只应该出现在下游适配阶段，不能和“基础模型训练”混在一个实验里解释。

因此，后续实验分成两条 lane：

- **Base Model Lane**：从零训练、对齐 starter / SSOT 场景，目标是得到更强 base checkpoint
- **Adaptation Lane**：在选定 base checkpoint 之上做 warm-start、fixed teammate、reward/observation 改造等

## 2. 目录重构

运行时代码从根目录平铺迁入结构化包 [cs8803drl/](../../cs8803drl)：

- [core/](../../cs8803drl/core)
- [training/](../../cs8803drl/training)
- [deployment/](../../cs8803drl/deployment)
- [evaluation/](../../cs8803drl/evaluation)
- [branches/](../../cs8803drl/branches)

脚本目录同步重构为：

- [scripts/setup/](../../scripts/setup)
- [scripts/eval/](../../scripts/eval)
- [scripts/tools/](../../scripts/tools)
- [scripts/batch/starter/](../../scripts/batch/starter)
- [scripts/batch/base/](../../scripts/batch/base)
- [scripts/batch/adaptation/](../../scripts/batch/adaptation)
- [scripts/batch/experiments/](../../scripts/batch/experiments)

目录治理规则同步更新于 [directory-governance.md](../management/directory-governance.md) 和 [scripts/README.md](../../scripts/README.md)。

## 3. 实验矩阵纠偏

明确记录：

- [soccerstwos_h100_cpu32_singleplayer_fixed_teammate_wide512x256.batch](../../scripts/batch/experiments/soccerstwos_h100_cpu32_singleplayer_fixed_teammate_wide512x256.batch)
  不是干净的结构对照
- 它同时改变了：
  - teammate 机制
  - 初始化方式
  - 模型宽度

因此后续不再拿它回答“更大模型是否更适合作为基础模型”。

## 4. 新的优先实验

当前优先级改为：

1. 做干净的基础模型 / 初始化对照
2. 再决定是否值得扩大模型容量
3. 最后才考虑 warm-start adaptation

已经新增的干净对照脚本：

- [soccerstwos_h100_cpu32_singleplayer_fixed_teammate.batch](../../scripts/batch/adaptation/soccerstwos_h100_cpu32_singleplayer_fixed_teammate.batch)
  - `256,128`
  - fixed teammate
  - warm-start
- [soccerstwos_h100_cpu32_singleplayer_fixed_teammate_scratch.batch](../../scripts/batch/adaptation/soccerstwos_h100_cpu32_singleplayer_fixed_teammate_scratch.batch)
  - `256,128`
  - fixed teammate
  - scratch

## 5. 结论

- 继续沿用“平铺目录 + 混杂实验”的方式已经不可持续
- 后续所有实验需要先回答“当前这条 lane 到底在比较哪个变量”
- 目录、脚本、运行时入口都已切到结构化路径，后续不再新增根目录平铺文件

## 相关

- [snapshot-006-fixed-teammate-and-dual-expert-rethink.md](snapshot-006-fixed-teammate-and-dual-expert-rethink.md)
- [directory-governance.md](../management/directory-governance.md)
- [scripts/README.md](../../scripts/README.md)
