"""
业务问题：构建 A2C (Advantage Actor-Critic) 强化学习智能体。
实现逻辑：
1. 共享特征提取层 -> 提取状态特征。
2. Actor 头 -> 输出动作概率分布。
3. Critic 头 -> 输出状态价值。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化，提升训练稳定性"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class A2CAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # 获取观测空间维度 (假设是展平的向量)
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        # 获取动作空间维度 (假设是离散动作)
        n_actions = envs.single_action_space.n
        
        # 1. 共享特征提取网络 (Backbone)
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        # 2. Actor 头 (策略网络)
        self.actor = layer_init(nn.Linear(64, n_actions), std=0.01)
        
        # 3. Critic 头 (价值网络)
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)

    def get_value(self, x):
        """仅计算状态价值"""
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        """
        前向传播
        :param x: 观测状态
        :param action: 可选，用于计算给定动作的 log_prob
        :return: (action, log_prob, entropy, value)
        """
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
