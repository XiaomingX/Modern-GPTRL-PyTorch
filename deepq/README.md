# DeepQ (DQN) 算法 (PyTorch + Gymnasium 实现)

本项目实现了 **DQN (Deep Q-Network)** 算法，这是现代深度强化学习的开山之作，适用于**离散动作空间**。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `dqn_agent.py` | **智能体定义**：包含 Q 网络定义和 $\epsilon$-Greedy 策略。 |
| `train_dqn.py` | **训练脚本**：训练 CartPole-v1，包含 Replay Buffer 和目标网络更新逻辑。 |

## 核心算法原理

### 1. 为什么叫 "Deep" Q-Network？
传统的 Q-Learning 使用表格 (Q-Table) 记录状态-动作价值。当状态空间巨大（如图像输入）时，表格无法容纳。DQN 使用**深度神经网络**来近似 Q 函数：
$$ Q(s, a; \theta) \approx Q^*(s, a) $$

### 2. 三大核心技术
为了解决神经网络训练的不稳定性，DQN 引入了以下技术：
*   **经验回放 (Experience Replay)**: 将交互数据 $(s, a, r, s', done)$ 存入缓冲区，随机采样进行训练。打破了数据的时序相关性，提高数据利用率。
*   **目标网络 (Target Network)**: 计算 TD 目标时使用一个参数固定的网络 $Q(s', a'; \theta^-)$，定期将主网络参数复制过去。这防止了目标值的剧烈震荡。
    $$ Target = r + \gamma \max_{a'} Q(s', a'; \theta^-) $$
*   **奖励截断 (Reward Clipping)**: (本项目针对 CartPole 未使用) 将奖励限制在 [-1, 1] 范围内，统一不同游戏的梯度尺度。

## 使用方法

### 运行训练 (CartPole-v1)
```bash
python train_dqn.py
```
CartPole-v1 的目标是保持平衡。训练约 5,000 步左右即可看到 Agent 学会控制策略。
