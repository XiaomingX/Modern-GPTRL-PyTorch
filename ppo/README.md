# PPO (PyTorch + Gymnasium 实现)

本项目提供了一个基于 **PyTorch** 和 **Gymnasium** 的 **PPO (Proximal Policy Optimization)** 算法标准实现。代码设计高度模块化，清晰展示了从环境交互到策略更新的完整闭环。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `ppo_agent.py` | **智能体定义**：包含 Actor（策略网络）和 Critic（价值网络）。 |
| `train_ppo.py` | **训练脚本**：包含环境初始化、Rollout 数据收集、GAE 计算和 PPO 训练循环。 |

## 核心算法原理

### 1. 算法架构 (Algorithm Architecture)
PPO 是一种 **Actor-Critic** 架构的策略梯度算法：
*   **Actor (策略网络)**: 输入状态 $S$，输出动作概率分布 $\pi(A|S)$。
*   **Critic (价值网络)**: 输入状态 $S$，输出状态价值估计 $V(S)$。

### 2. 关键技术点 (Key Components)

#### (1) GAE (Generalized Advantage Estimation)
优势函数 $A(s,a)$ 衡量了“在状态 $s$ 采取动作 $a$”比“平均情况”好多少。GAE 通过引入 $\lambda$ 参数，在偏差（Bias）和方差（Variance）之间取得平衡，提供更稳定的优势估计。
*代码对应*: `train_ppo.py` 中的 `Calculating Advantages` 部分。

#### (2) PPO Clipping
PPO 的核心思想是**限制策略更新的步长**，防止一次更新过大导致策略崩溃。它通过以下目标函数实现：
$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]
$$
其中 $r_t(\theta)$ 是新旧策略的概率比。Clipping 机制确保了 $r_t(\theta)$ 不会偏离 1 太远。
*代码对应*: `train_ppo.py` 中的 `PPO Clip Loss` 计算。

## 使用方法

### 1. 训练模型 (CartPole-v1)
```bash
python train_ppo.py
```
程序将自动启动 4 个并行环境进行训练，并实时打印 Reward。

### 2. 参数调整
你可以在 `train_ppo.py` 顶部的超参数区域调整配置：
```python
TOTAL_TIMESTEPS = 50000  # 总训练步数
LEARNING_RATE = 2.5e-4   # 学习率
NUM_ENVS = 4             # 并行环境数量
```
