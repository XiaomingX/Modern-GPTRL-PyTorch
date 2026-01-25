# Q-Learning 算法 (Table 实现)

本项目实现了经典的 **Q-Learning** 算法，这是一种基于表格 (Tabular) 的无模型 (Model-Free) 强化学习算法，适用于状态空间较小的离散环境。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `q_learning_agent.py` | **智能体定义**：维护一个 Q 表（哈希表），实现贝尔曼方程更新。 |
| `train_ql.py` | **训练脚本**：在 `FrozenLake-v1` 环境中训练智能体寻找路径。 |

## 核心算法原理

### 1. Q 表
Q-Learning 维护一个表格 $Q(s, a)$，记录在状态 $s$ 下采取动作 $a$ 能获得的期望累积回报。

### 2. 贝尔曼更新 (Off-Policy)
$$ Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] $$
其中：
*   $\alpha$: 学习率
*   $\gamma$: 折扣因子
*   $\max_{a'} Q(s', a')$: 下一个状态的最大 Q 值（贪心估计）

Q-Learning 是 Off-Policy 的，因为它在更新时使用的是最佳动作及其 Q 值（贪心），而实际执行策略可以是 $\epsilon$-Greedy（探索）。

## 使用方法

### 运行训练 (FrozenLake-v1)
```bash
python train_ql.py
```
FrozenLake 是一个网格世界，Agent 需要从起点走到终点，地面很滑（动作有概率执行失败导致偏移），掉进冰窟窿失败。Q-Learning 能学会最优路径。