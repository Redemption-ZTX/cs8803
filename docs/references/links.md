# 外部链接与工具集

> 项目技术栈与版本约束见 [engineering-standards.md](../architecture/engineering-standards.md)。
> 作业要求见 [Final Project Instructions Document.md](Final%20Project%20Instructions%20Document.md)。

## 项目相关

| 链接 | 说明 | 相关文档 |
|------|------|----------|
| [mdas64/soccer-twos-starter](https://github.com/mdas64/soccer-twos-starter) | 上游 starter kit（课程指定） | [overview.md](../architecture/overview.md)、[原版 README](upstream-README.md) |
| [bryanoliveira/soccer-twos-env](https://github.com/bryanoliveira/soccer-twos-env) | Soccer-Twos 环境源码 | [code-audit-000 § 1.2](../architecture/code-audit-000.md#12-rewardshapingwrapperl56-225) |
| [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) | 底层环境框架 | — |

## 框架文档

| 链接 | 说明 |
|------|------|
| [Ray RLlib 1.4.0 文档](https://docs.ray.io/en/releases-1.4.0/rllib.html) | 训练框架，见 [ADR-001](../architecture/adr/001-training-framework.md) |
| [Ray Tune 1.4.0 文档](https://docs.ray.io/en/releases-1.4.0/tune/index.html) | 超参搜索与实验管理 |

## PACE 集群

| 链接 | 说明 |
|------|------|
| [PACE-ICE 文档](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102) | 集群使用指南 |
| [Slurm 指南](https://mshalimay.github.io/slurm_pace_guidelines/) | 作业脚本编写，对应 `scripts/soccerstwos_job.batch` |

<!-- 格式：一句话说明 + 链接 + 相关文档引用 -->
