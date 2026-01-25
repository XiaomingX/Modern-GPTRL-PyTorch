# Learning-to-Learn (元学习优化器)

本项目实现了 DeepMind 的 **Learning to Learn by Gradient Descent by Gradient Descent** 算法。该算法通过训练一个神经网络（通常是 LSTM）来自动学习如何优化另一个神经网络。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `l2l_optimizer.py` | **核心逻辑**：包含梯度预处理模块、LSTM 元优化器和元优化管理类。 |
| `train_l2l.py` | **训练脚本**：在简单二次函数问题上元训练优化器。 |

## 核心算法原理

元优化的基本思想是：既然优化器本身也是一个函数，那么它就可以被参数化，并通过梯度下降来学习。

### 1. 逐坐标 LSTM (Coordinate-wise LSTM)
为了让一个小型 LSTM 能够处理拥有数百万参数的大模型，L2L 假设所有参数的优化规则是相同的，因此采用逐坐标处理。LSTM 接收每个参数的梯度输入，并输出对应的更新增量 $\Delta \theta$。

### 2. 梯度预处理
为了处理高度变化的梯度量级，输入被映射到一个包含梯度 Log 量级和符号的 2 维向量中。

### 3. 未展开轨迹 (Unrolling)
元优化器的训练通过“展开”优化步数来实现。元损失是所有 $T$ 个步长的损失之和：
$$ \mathcal{L}(\phi) = E_f \left[ \sum_{t=1}^T w_t f(\theta_t) \right] $$
其中 $\phi$ 是元优化器的参数。

## 使用方法

### 运行训练
```bash
python train_l2l.py
```
元学习通过让模型“学会学习”，在处理具有相似曲率特性的任务族时，能够显著优于 Adam 或 SGD 等手工设计的优化器。
