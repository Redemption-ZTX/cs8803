# ADR-001: 训练框架选择 Ray RLlib

- **状态**: 继承
- **日期**: 2026-04-07
- **决策人**: 上游项目

## 背景

Soccer-Twos 环境需要一个支持多智能体训练的 RL 框架。上游 [starter kit](https://github.com/mdas64/soccer-twos-starter) 已选定 Ray RLlib。

## 决策

使用 Ray RLlib 1.4.0 作为训练框架，锁定 Python 3.8。

## 理由

- 上游 starter kit 基于 Ray RLlib，切换框架需重写所有训练脚本
- 原生支持多智能体（multi-agent）训练、self-play、curriculum learning
- 内置 PPO、DQN 等常用算法
- Ray Tune 提供超参搜索和实验管理

## 替代方案

| 方案 | 放弃原因 |
|------|----------|
| Stable Baselines3 | 多智能体支持较弱 |
| CleanRL | 单文件实现，多智能体需大量自定义 |
| RLlib 2.x | 与 starter kit API 不兼容 |

## 影响

- 锁定 Python 3.8 + Ray 1.4.0（较老版本），见 [工程规范](../engineering-standards.md)
- 需 protobuf/pydantic 版本 pin（兼容性问题）
- checkpoint 格式为 Ray 1.x pickle，恢复逻辑复杂，见 [code-audit-000 § 2.2](../code-audit-000.md#22-_sanitize_checkpoint_for_restorel56-103)
