# MuZero 算法 (PyTorch 实现)

本项目实现了 DeepMind 的 **MuZero** 算法。MuZero 的核心创新在于它能够在不给定环境模型的情况下，通过学习一个“隐变量模型”来实现基于搜索的强化学习。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `muzero_model.py` | **三合一模型**：包含表示网络 (Representation)、动力学网络 (Dynamics) 和预测网络 (Prediction)。 |
| `mcts.py` | **蒙特卡洛树搜索**：专门为 MuZero 设计的 MCTS，利用学习到的 Dynamics 网络在隐空间进行搜索。 |
| `train_muzero.py` | **训练脚本**：在 CartPole-v1 上验证算法。 |

## 核心算法原理

MuZero 与 AlphaZero 的最大区别在于，MuZero 学习三个子网络：

1.  **表示网络 (h)**: $s_0 = h(o_1 \dots o_t)$，将观察映射到隐状态。
2.  **动力学网络 (g)**: $s_k, r_k = g(s_{k-1}, a_k)$，在隐空间预测下一步状态和奖励。
3.  **预测网络 (f)**: $p_k, v_k = f(s_k)$，预测当前隐状态下的策略分布和价值。

在 MCTS 过程中，除了根节点使用真实输入以外，树内部的所有后续节点都完全依赖动力学网络 $g$ 在隐空间进行推演。这使得 MuZero 可以应用于任何复杂的视觉环境或逻辑环境。

## 使用方法

### 运行训练
```bash
python train_muzero.py
```
虽然 MuZero 极其复杂且计算量大（由于每一步都要跑几十次 MCTS），但它在处理具有延迟奖励和需要长程规划的任务时表现极其卓越。
