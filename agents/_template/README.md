# Agent: _版本名_

**Agent name:** _填写_

**Author(s):** _填写姓名_ (_填写邮箱_)

## Description

_训练方法、reward 设计、关键超参等_

## Experiment

对应实验记录: [snapshot-NNN](../../docs/experiments/snapshot-NNN.md)

## Usage

```bash
python -m soccer_twos.watch -m agents.vNNN_xxx
python evaluate_matches.py -m1 agents.vNNN_xxx -m2 ceia_baseline_agent
```
