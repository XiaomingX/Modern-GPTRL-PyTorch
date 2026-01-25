"""
业务问题：构建 PPO (Proximal Policy Optimization) 强化学习智能体，用于在连续或离散动作空间中做出决策。
实现逻辑：
1. Actor 网络 -> 输出动作概率分布 (Normal/Categorical)。
2. Critic 网络 -> 估计由当前状态出发的预期回报 (Value Function)。
3. 学习过程 -> 使用 PPO-Clip 损失函数限制策略更新幅度，保证训练稳定性。
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化权重，有助于 RL 训练收敛"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # 假设环境观测是展平的向量
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        # 假设是离散动作空间 (如 CartPole)
        # 如果是连续空间 (如 MuJoCo)，需修改为输出均值和标准差
        n_actions = envs.single_action_space.n 
        
        # Critic 网络: 状态 -> 价值
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Actor 网络: 状态 -> 动作 Logits
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_value(self, x):
        """获取状态价值"""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        前向传播
        :param x: 观测状态
        :param action: 如果提供，则计算该动作的 log_prob (用于训练)；否则采样新动作 (用于收集数据)
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
