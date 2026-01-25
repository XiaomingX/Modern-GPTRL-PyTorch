"""
业务问题：实现 K-FAC (Kronecker-factored Approximate Curvature) 优化器。
实现逻辑：
1. 使用 Hook 机制捕获网络层的输入 (Activations) 和输出梯度 (Sensitivities)。
2. 计算 A 和 G 的协方差矩阵 (Covariance Matrices)。
3. 使用 Kronecker 积近似 Fisher 信息矩阵的逆，对梯度进行预条件处理 (Preconditioning)。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class KFACOptimizer(optim.Optimizer):
    def __init__(self, model, lr=0.25, momentum=0.9, stat_decay=0.99, kl_clip=0.001, damping=1e-2, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, stat_decay=stat_decay, kl_clip=kl_clip, damping=damping, weight_decay=weight_decay)
        super().__init__(model.parameters(), defaults)
        
        self.model = model
        self.modules = []
        self.stats = {} # 存储 A 和 G 的统计量
        self.inv_stats = {} # 存储 Fisher 逆矩阵
        
        # 1. 注册 Hooks 收集统计量
        self._register_hooks()

    def _register_hooks(self):
        """为所有 Linear 和 Conv2d 层注册 Forward/Backward Hooks"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear): # 目前简化只支持 Linear，CartPole 足够
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    def _save_input(self, module, input):
        """保存输入激活值 (A)"""
        # input 是 tuple，取第一个
        if isinstance(module, nn.Linear):
            self.stats[module] = {'a': input[0].data}

    def _save_grad_output(self, module, grad_input, grad_output):
        """保存输出梯度 (G)"""
        # grad_output 是 tuple，取第一个
        if isinstance(module, nn.Linear):
            self.stats[module]['g'] = grad_output[0].data

    def step(self, closure=None):
        """执行一步更新 (计算 Fisher -> 预条件梯度 -> 更新参数)"""
        # ACKTR 需要闭包来重新计算 Loss (如果做了 Trust Region)
        # 这里简化实现，直接利用已有的梯度
        
        loss = None
        if closure is not None:
            loss = closure()
            
        # 1. 更新 Fisher 统计量 (A 和 G 的协方差)
        self._update_fisher_stats()
        
        # 2. 计算 Fisher 逆 (每隔 T 步更新一次，这里简化为每步更新或按需)
        # 实际训练中 inverse 比较耗时，通常每 100 步一次，这里简化为每次都算
        self._update_inverse_fisher()
        
        # 3. 预条件梯度 (Precondition Gradients)
        fisher_norm = 0.0
        for group in self.param_groups:
            for module in self.modules:
                # 获取该层的权重和偏置
                weight = module.weight
                bias = module.bias
                
                # 获取梯度
                g_weight = weight.grad
                g_bias = bias.grad if bias is not None else None
                
                if g_weight is None:
                    continue
                    
                # 组合权重和偏置的梯度 (与 K-FAC 矩阵形式对应)
                # Linear: G_grad = g^T * a
                # 这里我们直接拿到 dL/dW, 需转换视角
                # 实际上 K-FAC 的 preconditioning 是对 flatten 的梯度做的：
                # F^{-1} vec(W_grad) = vec(A^{-1} * G_grad * G^{-1})
                
                # A_inv: [I, I], G_inv: [O, O]
                # W_grad: [O, I]
                # Preconditioned: G_inv * W_grad * A_inv
                
                if module in self.inv_stats:
                    a_inv = self.inv_stats[module]['a'] # [I+1, I+1] (如果含bias)
                    g_inv = self.inv_stats[module]['g'] # [O, O]
                    
                    # 拼接 weight 和 bias 的梯度
                    if bias is not None:
                        # [O, I] + [O, 1] -> [O, I+1]
                        grad = torch.cat([g_weight, g_bias.view(-1, 1)], dim=1)
                    else:
                        grad = g_weight
                        
                    # 预条件变换: G_inv @ grad @ A_inv
                    v_grad = g_inv @ grad @ a_inv
                    
                    # 计算 Fisher Norm (用于 KL Clip)
                    # v^T * F * v = v^T * grad
                    fisher_norm += (grad * v_grad).sum().item()
                    
                    # 拆分回 weight 和 bias
                    if bias is not None:
                        weight.grad.data = v_grad[:, :-1]
                        bias.grad.data = v_grad[:, -1].view_as(bias)
                    else:
                        weight.grad.data = v_grad
        
        # 4. KL Clipping 与 参数更新
        scale = 1.0
        if group['kl_clip'] > 0:
            if fisher_norm > group['kl_clip']:
                scale = math.sqrt(group['kl_clip'] / (fisher_norm + 1e-8))
        
        for p in self.model.parameters():
            if p.grad is not None:
                # 应用 Scale 和 LR
                p.data.add_(p.grad.data, alpha=-group['lr'] * scale)
                
        return loss

    def _update_fisher_stats(self):
        """计算并滑动平均更新 A 和 G"""
        for module in self.modules:
            stat = self.stats[module]
            if 'a' not in stat or 'g' not in stat:
                continue
                
            a = stat['a'] # [Batch, Input]
            g = stat['g'] # [Batch, Output]
            
            # 拼接 Bias (1) 到 A
            if module.bias is not None:
                ones = torch.ones(a.shape[0], 1, device=a.device)
                a = torch.cat([a, ones], dim=1)
                
            # 计算 Batch 平均的协方差: A^T * A / B
            batch_size = a.shape[0]
            aa_t = (a.t() @ a) / batch_size
            gg_t = (g.t() @ g) / batch_size
            
            # 滑动平均更新 (Simple Moving Average)
            decay = self.param_groups[0]['stat_decay']
            if module not in self.inv_stats: # 首次初始化
                self.inv_stats[module] = {
                    'aa_t': aa_t,
                    'gg_t': gg_t
                }
            else:
                self.inv_stats[module]['aa_t'].mul_(decay).add_(aa_t, alpha=1-decay)
                self.inv_stats[module]['gg_t'].mul_(decay).add_(gg_t, alpha=1-decay)
    
    def _update_inverse_fisher(self):
        """计算 Fisher 因子矩阵的逆 (添加阻尼)"""
        damping = self.param_groups[0]['damping']
        
        for module in self.modules:
            if module not in self.inv_stats:
                continue
                
            aa_t = self.inv_stats[module]['aa_t']
            gg_t = self.inv_stats[module]['gg_t']
            
            # 添加 Tikhonov Damping (对角线加 epsilon)
            # pi-damping (Martens & Grosse, 2015)：平衡 A 和 G 的数值尺度
            # damping_a = sqrt(tr(A)/tr(G)) * damping
            # 简化版：直接对每个矩阵加 sqrt(damping)
            
            d = math.sqrt(damping)
            aa_t_d = aa_t + torch.eye(aa_t.shape[0], device=aa_t.device) * d
            gg_t_d = gg_t + torch.eye(gg_t.shape[0], device=gg_t.device) * d
            
            # Cholesky 分解求逆 或 直接 inverse (PyTorch inverse 此时比较稳定)
            self.inv_stats[module]['a'] = torch.inverse(aa_t_d)
            self.inv_stats[module]['g'] = torch.inverse(gg_t_d)
    
