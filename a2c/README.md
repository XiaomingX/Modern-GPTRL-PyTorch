# A2C 算法 (PyTorch + Gymnasium 实现)

本项目提供了一个标准且简洁的 **A2C (Advantage Actor-Critic)** 算法实现，适配经典的 CartPole 游戏。代码基于 **PyTorch** 和 **Gymnasium**，旨在帮助开发者理解 Actor-Critic 架构与 N-Step 回报计算。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `a2c_agent.py` | **智能体定义**：包含共享特征提取层、Actor 头（策略）和 Critic 头（价值）。 |
| `train_a2c.py` | **训练脚本**：包含同步环境 (SyncVectorEnv)、N-Step 数据收集和 A2C 更新逻辑。 |

## 核心算法原理

### 1. 算法架构
A2C 是 Actor-Critic 算法的**同步并行**版本。与异步的 A3C 不同，A2C 使用同步环境包装器收集多个环境的经验，组成一个 batch 进行更新，能更高效地利用 GPU。

### 2. 关键公式

#### (1) N-Step 折扣回报 (Bootstrapped Returns)
为了平衡偏差与方差，A2C 使用 N 步回报作为目标值：
$$ R_t = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n}) $$
其中 $V(s_{t+n})$ 是 Critic 对第 N 步状态的价值估计（Bootstrap）。
*代码对应*: `train_a2c.py` 中的 `Calculate Returns` 部分。

#### (2) 优势函数 (Advantage)
优势值表示当前动作比平均水平好多少：
$$ A(s_t, a_t) = R_t - V(s_t) $$

#### (3) 损失函数 (Loss Function)
总损失由三部分组成：
*   **策略损失**: $- \text{mean}(A \cdot \log \pi(a|s))$
*   **价值损失**: $\frac{1}{2} \text{mean}((R - V(s))^2)$
*   **熵惩罚**: $- \text{entropy}$ (鼓励探索)

## 使用方法

### 运行训练
```bash
python train_a2c.py
```
程序将启动 4 个并行环境进行训练，训练约 20,000 步即可在 CartPole-v1 上达到收敛。

### 超参数调整
可以在 `train_a2c.py` 中修改 `NUM_STEPS` (收集步数) 或 `LEARNING_RATE` (学习率) 观察效果。