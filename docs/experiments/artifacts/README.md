# 实验产物归档

本目录用于存放不适合继续堆在仓库根目录、但又需要保留以便复核实验过程的产物。

当前约定：

- `official-evals/`：官方 `soccer_twos.evaluate` 原始日志
- `official-scans/`：checkpoint 粗扫 / 精扫 CSV
- `slurm-logs/`：SLURM 或 batch 输出日志

说明：

- 这里存放的是“实验证据”，不是正式结论。
- 正式结论写入对应 `snapshot-NNN.md`。
- 最终选模以官方 evaluator 大样本结果为准。
