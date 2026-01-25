"""
业务问题：构建 TRPO (Trust Region Policy Optimization) 智能体。
实现逻辑：
1. 实现共轭梯度法 (Conjugate Gradient) 求解 Fisher 信息矩阵逆与梯度的乘积。
2. 实现 Fisher 向量积 (Fisher-Vector Product) 利用反向传播计算 Hessian * vector。
3. 实现线搜索 (Line Search) 确保更新满足 KL 约束且目标函数提升。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

def flat_grad(grads, params):
    """将梯度列表展平为向量"""
    grad_flatten = []
    for g, p in zip(grads, params):
        if g is None:
            grad_flatten.append(torch.zeros_like(p).view(-1))
        else:
            grad_flatten.append(g.view(-1))
    return torch.cat(grad_flatten)

def flat_params(params):
    """将参数列表展平为向量"""
    return torch.cat([p.data.view(-1) for p in params])

def update_model(model, new_params):
    """将展平的参数赋值给模型"""
    index = 0
    for p in model.parameters():
        param_len = p.numel()
        p.data.copy_(new_params[index:index+param_len].view(p.size()))
        index += param_len

class TRPOAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, device):
        super().__init__()
        self.device = device
        
        # 策略网络 (Policy Network)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim), 
            nn.Softmax(dim=-1) # 离散动作空间
        ).to(device)
        
        # 价值网络 (Value Network)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(device)
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def get_action(self, x):
        x = torch.Tensor(x).to(self.device)
        probs = self.actor(x)
        d = dist.Categorical(probs)
        action = d.sample()
        return action.item(), d.log_prob(action)

    def get_fisher_vector_product(self, v, states):
        """
        计算 F*v = \nabla^2 D_{KL} * v
        利用 autograd 计算 Hessian-vector product，无需显式构建 Hessian 矩阵
        """
        states = states.detach()
        probs = self.actor(states)
        d = dist.Categorical(probs)
        
        # 对自身的 KL 散度 (KL(old||new) 当 old=new 时 KL=0, 梯度=0, Hessian 正定)
        # 这里用一个小 trick: D_KL(theta || theta_prime) 对 theta_prime 求导
        # 为了方便，我们固定 old_probs
        # KL(p||q) = \sum p * (log p - log q)
        # \nabla^2 KL = \nabla^2 (- \sum p * log q)
        
        # 为了计算 Hessian，先计算一阶导
        kl = torch.sum(probs.detach() * torch.log(probs.detach() / probs), dim=1).mean()
        
        grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        flat_kl_grad = flat_grad(grads, self.actor.parameters())
        
        #计算 gradient * v
        kl_v = (flat_kl_grad * v).sum()
        
        # 再对 (grad * v) 求导，得到 Hessian * v
        grads_2nd = torch.autograd.grad(kl_v, self.actor.parameters())
        flat_kl_grad_2nd = flat_grad(grads_2nd, self.actor.parameters())
        
        return flat_kl_grad_2nd + v * 0.1 # 添加 damping，保证数值稳定性 (F + lambda*I)

    def conjugate_gradient(self, b, states, cg_iters=10, residual_tol=1e-10):
        """
        求解 Ax = b，其中 A 是 Fisher Information Matrix
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        
        for _ in range(cg_iters):
            # A * p
            Ap = self.get_fisher_vector_product(p, states)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def update(self, states, actions, returns, advantages, max_kl=0.01, damping=0.1):
        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        returns = torch.Tensor(returns).to(self.device)
        advantages = torch.Tensor(advantages).to(self.device)
        
        # 1. 计算 Value Loss 并更新 Critic
        # 多次更新 Critic 以逼近真实价值
        for _ in range(10): 
            v_pred = self.critic(states).squeeze()
            v_loss = (v_pred - returns).pow(2).mean()
            self.critic_optimizer.zero_grad()
            v_loss.backward()
            self.critic_optimizer.step()
        
        # 2. 计算 Policy Loss (Surrogate Objective)
        # L = E [ ratio * advantage ]
        probs = self.actor(states)
        d = dist.Categorical(probs)
        log_probs = d.log_prob(actions)
        old_log_probs = log_probs.detach()
        
        def get_loss():
            probs_new = self.actor(states)
            d_new = dist.Categorical(probs_new)
            log_probs_new = d_new.log_prob(actions)
            ratio = torch.exp(log_probs_new - old_log_probs)
            return -(ratio * advantages).mean() # Minimize negative reward

        loss = get_loss()
        
        # 3. 计算 Gradient g
        grads = torch.autograd.grad(loss, self.actor.parameters())
        g = flat_grad(grads, self.actor.parameters())
        
        # 4. 计算搜索方向 s = F^{-1} g using Conjugate Gradient
        # TRPO maximizing objective -> minimizing cost -> move opposite to gradient of cost
        # But here 'loss' is negative objective, so minimizing loss is maximizing objective.
        # Direction should be -F^{-1} * grad_loss(loss)
        # Wait, standard TRPO uses g as gradient of objective (positive), so direction is F^{-1}g
        # Here we use g as gradient of LOSS (negative), so direction is -F^{-1}g.
        # But let's stick to textbook: x = F^{-1} (-g) is descent direction for minimizing.
        
        step_dir = self.conjugate_gradient(-g, states)
        
        # 5. 计算步长 beta
        # beta = sqrt(2 * max_kl / (s^T * F * s))
        shs = 0.5 * (step_dir * self.get_fisher_vector_product(step_dir, states)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / max_kl) # Lagrangian multiplier roughly
        full_step = step_dir / lm[0] # scaled step
        
        neg_step_dir = -step_dir # Check sign? 
        # Actually: if we minimize loss, we want to go against gradient.
        # grad is pointing to increase loss. So -grad is good.
        # Natural gradient: - F^-1 g. 
        # Above I used -g inside CG, so step_dir is F^-1(-g). Correct.
        
        expected_improve = (-g * full_step).sum() # should be positive (loss decreases)
        
        # 6. Line Search (Backtracking)
        old_params = flat_params(self.actor.parameters())
        
        flag = False
        fraction = 1.0
        for i in range(10):
            new_params = old_params + fraction * full_step
            update_model(self.actor, new_params)
            
            new_loss = get_loss()
            
            # Recompute KL
            probs_new = self.actor(states)
            d_new = dist.Categorical(probs_new)
            # KL(old || new)
            # kl = torch.sum(probs.detach() * torch.log(probs.detach() / probs_new), dim=1).mean()
            # 简化计算：
            kl = dist.kl_divergence(dist.Categorical(probs.detach()), d_new).mean()
            
            if kl <= max_kl * 1.5 and new_loss < loss: # Simple sufficient ascent condition
                flag = True
                break
            
            fraction *= 0.5
            
        if not flag:
            # 如果找不到满足条件的步长，回退
            update_model(self.actor, old_params)
