# MA-DDPG (Multi-Agent DDPG) 算法

本项目实现了 **MA-DDPG** 算法，这是一种用于多智能体协作/竞争环境的扩展版 DDPG。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `simple_mpe.py` | **多智能体环境**：一个简化的 "Simple Spread" 环境。多个 Agent 需要协作覆盖地标，避免碰撞。 |
| `maddpg_agent.py` | **智能体定义**：包含 Local Actor 和 Global Critic。 |
| `train_maddpg.py` | **训练脚本**：集中式训练循环。 |

## 核心算法原理

### 1. 集中式训练，分布式执行 (CTDE)
在多智能体环境中，如果每个 Agent 只看自己的局部观测进行训练 (Independent DDPG)，环境会变得非稳态 (Non-stationary)，因为其他 Agent 的策略也在变。

MA-DDPG 引入了 **Global Critic**：
*   **训练时**：Critic 能够看到**所有智能体**的观测和动作 $Q_i(x, a_1, \dots, a_N)$。这样环境就变回稳态了。
*   **执行时**：Actor 只使用局部观测 $\pi_i(o_i)$，不需要通信。

### 2. 环境设置 (Simple MPE)
*   **目标**：N 个 Agent 覆盖 N 个 Landmark。
*   **奖励**：合作型奖励（覆盖距离 + 碰撞惩罚）。所有 Agent 共享同一个全局奖励（或者各自有独立奖励，公式里兼容）。

## 使用方法

### 运行训练
```bash
python train_maddpg.py
```
训练开始后，Agent 会逐渐学会分散开来去寻找最近的地标，而不是挤在一起。