"""
业务问题：实现《Learning to Learn by Gradient Descent by Gradient Descent》中的元优化器。
实现逻辑：
1. 元优化器是一个 LSTM 网络，它接收参数当前的梯度作为输入。
2. 梯度经过预处理（log + sign）后送入 LSTM。
3. LSTM 的输出作为参数的更新量（Update）。
4. 元损失是整个优化路径上的损失之和。
"""

import torch
import torch.nn as nn
import numpy as np

class L2LPreprocess(nn.Module):
    """
    梯度预处理：实现论文中的 log + sign 变换。
    将梯度的量级压缩到 log 空间，并保留方向（sign）。
    """
    def __init__(self, p=10.0):
        super().__init__()
        self.p = p

    def forward(self, gradients):
        # 增加一个维度用于 concat
        # log(abs(g) + eps) / p, sign(g)
        eps = 1e-8
        log_grad = torch.log(torch.abs(gradients) + eps) / self.p
        sign_grad = torch.clamp(gradients * np.exp(self.p), -1, 1) # 简化版 sign
        
        # 裁剪 log 到 [-1, 1]
        log_grad = torch.clamp(log_grad, -1, 1)
        
        return torch.stack([log_grad, sign_grad], dim=-1)

class LSTMOptimizer(nn.Module):
    """
    元优化器核心：逐坐标 (Coordinate-wise) LSTM。
    """
    def __init__(self, input_dim=2, hidden_dim=20, num_layers=2):
        super().__init__()
        self.preprocess = L2LPreprocess()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.scale = 0.01

    def forward(self, grad, state=None):
        # grad: [num_params]
        num_params = grad.numel()
        # 预处理 -> [num_params, 2]
        preprocessed = self.preprocess(grad.view(-1))
        # LSTM 输入 -> [1, num_params, 2]
        out, state = self.lstm(preprocessed.unsqueeze(0), state)
        # 输出更新量 -> [num_params, 1]
        update = self.linear(out.squeeze(0)) * self.scale
        return update.view(grad.shape), state

class MetaOptimizer:
    """
    管理元学习流程：展开轨迹、计算元损失并更新元优化器。
    """
    def __init__(self, optimizee_factory, lr=0.001):
        self.model = LSTMOptimizer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizee_factory = optimizee_factory

    def meta_train(self, num_epochs=100, unroll_len=20):
        for epoch in range(num_epochs):
            # 1. 实例化待优化问题（如简单二次函数）
            optimizee = self.optimizee_factory()
            state = None
            total_meta_loss = 0
            
            # 2. 展开优化轨迹 (Unrolling)
            current_params = optimizee.params
            for step in range(unroll_len):
                # 计算待优化问题的梯度
                # 注意：为了让元损失对元优化器参数可导，我们需要手动处理梯度
                loss = torch.sum(current_params ** 2)
                total_meta_loss += loss
                
                # 计算梯度 (需允许计算图保留)
                grads = torch.autograd.grad(loss, current_params, create_graph=True)[0]
                
                # 获取元优化器输出的更新
                update, state = self.model(grads, state)
                
                # 更新待优化问题的参数 (非原位更新)
                current_params = current_params + update
            
            # 3. 更新元优化器
            self.optimizer.zero_grad()
            total_meta_loss.backward()
            self.optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch: {epoch} | Meta Loss: {total_meta_loss.item():.4f}")

class QuadraticProblem:
    """待优化问题示例：f(x) = x^2"""
    def __init__(self):
        self.params = torch.tensor([10.0], requires_grad=True)
    
    def loss(self):
        return torch.sum(self.params ** 2)
