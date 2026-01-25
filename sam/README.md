# SAM (Sharpness-Aware Minimization & Segment Anything)

该文件夹包含两个名为 "SAM" 的前沿技术实现：

## 1. 锐度感知最小化 (Sharpness-Aware Minimization)
`sam_optimizer.py` 实现了用于提升深度学习模型泛化能力的优化器。

### 核心原理
传统的优化器仅关注降低损失（Loss），但在复杂的地形中，模型可能会陷入“尖锐”的局部极小值点，这些点在测试数据上表现较差。
SAM 通过寻找一个邻域，使得该邻域内**最坏情况**的损失最小化，从而引导模型走向“平坦”的区域。平坦的极小值点通常具有更好的泛化性能。

### 使用方法
```python
from sam_optimizer import SAM
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)

# 第一步：寻找扰动位置
loss = criterion(model(inputs), targets)
loss.backward()
optimizer.first_step(zero_grad=True)

# 第二步：真实更新
criterion(model(inputs), targets).backward()
optimizer.second_step(zero_grad=True)
```

## 2. Segment Anything Model (SAM)
`segment_anything.py` 提供了 Meta AI 开源的视觉基础模型 SAM 的模块化实现。

### 特点
- **零样本泛化**：能够处理从未见过的图像和物体。
- **提示驱动**：支持点、框、掩码等多种形式的提示输入。
- **解耦结构**：强大的 ViT 图像编码器与轻量级的提示/掩码解码器分离。

---
虽然两个技术都叫 SAM，但一个侧重于**模型训练的泛化性**，另一个侧重于**视觉场景的普适分割能力**。
