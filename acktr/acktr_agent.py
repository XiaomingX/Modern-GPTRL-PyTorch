"""
业务问题：构建使用 ACKTR 算法的智能体。
实现逻辑：
1. 网络结构同 A2C (共享 Backbone + 双头)。
2. 使用 KFACOptimizer 替代标准 SGD/RMSProp。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ACKTRAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        n_actions = envs.single_action_space.n
        
        # 1. 网络定义 (保持简单以便 K-FAC 处理)
        # 注意：K-FAC 通常对 Linear/Conv 层效果最好，中间不要夹杂太复杂的结构
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        self.actor = layer_init(nn.Linear(64, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def compute_loss(self, obs, action, returns, advantages):
        """辅助函数：计算当前 Loss 以便 K-FAC 采样梯度"""
        _, newlogprob, entropy, newvalue = self.get_action_and_value(obs, action)
        
        # Policy Loss
        pg_loss = -(advantages * newlogprob).mean()
        
        # Value Loss
        v_loss = 0.5 * ((newvalue.view(-1) - returns) ** 2).mean()
        
        # Entropy Loss
        entropy_loss = entropy.mean()
        
        # ACKTR 总 Loss
        return pg_loss - 0.01 * entropy_loss + 0.5 * v_loss
