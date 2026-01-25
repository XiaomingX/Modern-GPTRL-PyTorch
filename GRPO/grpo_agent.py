"""
业务问题：构建 GRPO (Group Relative Policy Optimization) 智能体。
实现逻辑：
1. GRPO 不需要 Critic 网络，而是通过一组 (Group) 样本的相对奖励来估计优势。
2. 优势计算公式：A_i = (R_i - mean(R_group)) / std(R_group)。
3. 损失函数包含：策略梯度损失 + KL 散度约束（针对旧策略）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRPOAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, device):
        super().__init__()
        self.device = device
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.beta = 0.01 # KL 权重

    def get_action_probs(self, obs):
        return self.actor(obs)

    def update(self, batch_obs, batch_actions, batch_rewards, old_probs):
        """
        batch_obs: [Group_Size, Obs_Dim]
        batch_actions: [Group_Size]
        batch_rewards: [Group_Size]
        old_probs: [Group_Size, Action_Dim]
        """
        # 1. 计算相对优势 (Standardize within the group)
        mean_reward = batch_rewards.mean()
        std_reward = batch_rewards.std() + 1e-8
        advantages = (batch_rewards - mean_reward) / std_reward
        
        # 2. 计算当前策略的概率
        current_probs = self.actor(batch_obs)
        dist = torch.distributions.Categorical(current_probs)
        log_probs = dist.log_prob(batch_actions)
        
        # 3. 计算旧策略的 log_probs (用于比例)
        old_dist = torch.distributions.Categorical(old_probs)
        old_log_probs = old_dist.log_prob(batch_actions)
        
        # 4. 策略損失 (Surrogate Objective)
        ratio = torch.exp(log_probs - old_log_probs)
        surrogate_loss = ratio * advantages
        
        # 5. KL 散度损失 (KL(old || current))
        # 限制当前策略与参考/旧策略的偏差
        kl_loss = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
        
        # 总损失 = -Surrogate + beta * KL
        loss = -surrogate_loss.mean() + self.beta * kl_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), kl_loss.item()
