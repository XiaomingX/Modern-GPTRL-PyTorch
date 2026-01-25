# DDPG 算法 (PyTorch + Gymnasium 实现)

本项目实现了 **DDPG (Deep Deterministic Policy Gradient)** 算法，这是一个用于**连续动作空间**的经典 Actor-Critic 算法。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `ddpg_agent.py` | **智能体定义**：包含 Deterministic Actor（输出确定性动作）和 Critic（Q 价值）。 |
| `train_ddpg.py` | **训练脚本**：训练 Pendulum-v1，包含 Replay Buffer 和 OU 噪声。 |

## 核心算法原理

### 1. 为什么用 DDPG？
DQN 无法直接应用于连续动作空间，因为在计算 $\max_a Q(s,a)$ 时需要穷举所有动作。DDPG 通过使用一个 Actor 网络 $\mu(s)$ 直接输出能最大化 Q 值的动作，解决了这个问题：
$$ \max_a Q(s,a) \approx Q(s, \mu(s)) $$

### 2. 核心组件
*   **确定性策略梯度**: Actor 的更新方向是沿着 Q 值增加的方向。
*   **目标网络 (Target Networks)**: 使用软更新 ($\tau \ll 1$) 来缓慢更新目标网络，极大地稳定了训练。
    $$ \theta_{target} \leftarrow \tau \theta + (1-\tau) \theta_{target} $$
*   **探索噪声**: 由于策略是确定性的，需要在动作上添加噪声（通常是 Ornstein-Uhlenbeck 噪声）来保持探索。
*   **经验回放 (Replay Buffer)**: 打破数据的时间相关性。

## 使用方法

### 运行训练 (Pendulum-v1)
```bash
python train_ddpg.py
```
Pendulum-v1 的目标是保持倒立，训练约 20,000 步后可以看到显著的奖励提升。