# TRPO (Trust Region Policy Optimization) 算法 (PyTorch 实现)

本项目实现了 **TRPO** 算法。TRPO 是一种保证策略单调提升的优化方法，通过在参数更新时添加 KL 散度约束来实现。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `trpo_agent.py` | **智能体定义**：实现了高难度的共轭梯度 (Conjugate Gradient) 和 Fisher 向量积 (Hessian-vector product)，以及线搜索 (Line Search)。 |
| `train_trpo.py` | **训练脚本**：在 CartPole-v1 上验证算法。 |

## 核心算法原理

普通梯度下降（如 REINFORCE）选择的最陡下降方向依赖于参数的某种度量（通常是欧氏距离）。但参数空间的欧氏距离并不能反映策略分布的真实变化。

TRPO 使用 **KL 散度** 来衡量策略变化的“距离”，并在一个受限的“信任域”内进行更新：
$$ \max_\theta E [ \frac{\pi_\theta}{\pi_{old}} A ] \quad \text{s.t.} \quad D_{KL}(\pi_{old} || \pi_\theta) \le \delta $$

由于直接计算 Hessian 矩阵逆太慢，TRPO 使用 **共轭梯度法 (Conjugate Gradient)** 来近似计算自然梯度方向 $F^{-1} g$。

## 使用方法

### 运行训练
```bash
python train_trpo.py
```
你将看到 CartPole 分数稳步上升至满分 (500)。
