"""
业务问题：构建 DDPG (Deep Deterministic Policy Gradient) 强化学习智能体。
实现逻辑：
1. Actor 网络 -> 输出确定性动作 (Deterministic Action)。
2. Critic 网络 -> 输出状态-动作价值 (Q-Value)。
3. 目标网络 (Target Networks) -> 软更新以稳定训练。
"""

import numpy as np
import torch
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        action_dim = np.array(env.action_space.shape).prod()
        # 获取动作边界 (假设是对称的，如 [-2, 2])
        self.action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32) 
        self.action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)

        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
            nn.Tanh(),
        )

    def forward(self, x):
        """输出归一化到 [-1, 1] 的动作，然后缩放到实际范围"""
        return self.net(x) * self.action_scale.to(x.device) + self.action_bias.to(x.device)

class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        action_dim = np.array(env.action_space.shape).prod()

        # Critic 输入是 状态 + 动作
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1))

class DDPGAgent(nn.Module):
    """
    DDPG 智能体容器，包含 Actor 和 Critic
    """
    def __init__(self, env):
        super().__init__()
        self.actor = Actor(env)
        self.critic = Critic(env)
