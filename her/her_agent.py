"""
业务问题：构建支持 HER 的 DDPG 智能体。
实现逻辑：
1. Actor/Critic 输入需包含 Goal (状态+目标)。
2. 网络结构同标准 DDPG。
"""

import numpy as np
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=1)
        return self.max_action * self.net(x)

class Critic(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, goal, action):
        x = torch.cat([obs, goal, action], dim=1)
        return self.net(x)

class HERAgent:
    def __init__(self, obs_dim, goal_dim, action_dim, max_action, device):
        self.actor = Actor(obs_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(obs_dim, goal_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, goal_dim, action_dim).to(device)
        self.critic_target = Critic(obs_dim, goal_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.max_action = max_action
        self.device = device
        
    def select_action(self, obs, goal, noise=0.0):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)
            action = self.actor(obs, goal).cpu().data.numpy().flatten()
            if noise != 0:
                action = (action + np.random.normal(0, noise, size=action.shape)).clip(-self.max_action, self.max_action)
            return action
