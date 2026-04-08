# Agents

实验性 agent 版本目录。每个子目录是一个独立的 agent 模块，可以用 `python -m soccer_twos.watch -m agents.vNNN_xxx` 测试。

## 命名规范

```
agents/
├── v001_简短描述/
│   ├── __init__.py        # from .agent import RayAgent
│   ├── agent.py           # AgentInterface 实现
│   ├── README.md          # agent 信息（名字、作者、描述）
│   ├── checkpoint         # 训练好的权重
│   └── params.pkl         # 训练配置
├── v002_xxx/
└── ...
```

- 编号递增：`v001`, `v002`, ...
- 每个版本独立自包含，可以直接 zip 提交
- 对应的实验记录在 `docs/experiments/snapshot-NNN.md`

## 提交时

把最终选定的版本复制到根目录，改名为作业要求的 agent 名：

```bash
cp -r agents/vXXX_best_performance agent_performance
cp -r agents/vYYY_best_reward agent_reward_mod
```

然后分别 zip 提交。

## 公共代码

agent.py 中的 checkpoint 解析函数来自 `checkpoint_utils.py`（canonical source）。agent 模块必须自包含，所以是复制而非 import。修改 `checkpoint_utils.py` 后需要同步到各 agent 的 agent.py。

详见 [code-audit-000 § 6.1](../docs/architecture/code-audit-000.md#61-代码重复严重)。
