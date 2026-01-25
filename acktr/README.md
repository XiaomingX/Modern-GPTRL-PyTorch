# ACKTR 算法 (PyTorch + K-FAC 实现)

本项目实现了 **ACKTR (Actor Critic using Kronecker-Factored Trust Region)** 算法的 PyTorch 版本。ACKTR 是一种利用**二阶优化信息（自然梯度）**的强化学习算法。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `kfac.py` | **核心优化器**：实现了 K-FAC (Kronecker-factored Approximate Curvature) 算法，作为 PyTorch 的自定义优化器。它对梯度进行**预条件处理 (Preconditioning)**，模拟自然梯度更新。 |
| `acktr_agent.py` | **智能体定义**：标准的 Actor-Critic 网络结构。 |
| `train_acktr.py` | **训练脚本**：训练循环，展示如何在 RL 训练中调用 K-FAC 优化器。 |

## 核心算法原理

### 为什么需要二阶优化？
传统的一阶优化算法（如 SGD, Adam）只考虑梯度的方向。而二阶优化算法（如自然梯度法）考虑了参数空间的**曲率结构**（Fisher 信息矩阵）。这使得每次更新的方向更符合参数分布的真实变化，从而能更高效地训练，尤其是在策略网络中。

### K-FAC 如何工作？
计算完整的 Fisher 矩阵逆极其昂贵。K-FAC 假设神经网络层的 Fisher 矩阵可以近似为两个小矩阵的 Kronecker 积：
$$ F \approx A \otimes G $$
其中 $A$ 是输入激活的协方差矩阵，$G$ 是输出梯度的协方差矩阵。
这样，$F^{-1}$ 可以高效计算为 $A^{-1} \otimes G^{-1}$。

## 代码实现细节
1.  **Hook 机制**: `kfac.py` 使用 PyTorch 的 `register_forward_pre_hook` 和 `register_full_backward_hook` 自动收集每一层的 $A$ 和 $G$。
2.  **预条件 (Preconditioning)**: 在 `optimizer.step()` 中，我们修改标准梯度 $\nabla \theta$，将其变换为 $F^{-1} \nabla \theta$。

## 使用方法

### 运行训练
```bash
python train_acktr.py
```
注意：由于 K-FAC 涉及矩阵求逆，计算量较大，但在 cartpole 这种简单环境上可以看到其收敛特性。
