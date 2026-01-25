"""
业务问题：构建 DQN (Deep Q-Network) 智能体，用于离散动作空间控制。
实现逻辑：
1. Q 网络 (MLP) -> 输出每个动作的 Q 值。
2. 目标网络 (Target Network) -> 定期同步，计算 TD 目标。
3. Epsilon-Greedy 策略 -> 平衡探索与利用。
"""

import random
import numpy as np
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        action_dim = env.action_space.n
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_dim),
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """
    DQN 智能体容器，包含模型和选动作逻辑
    """
    def __init__(self, env, learning_rate=2.5e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q 网络 & 目标网络
        self.q_network = QNetwork(env).to(self.device)
        self.target_network = QNetwork(env).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
    def get_action(self, obs, epsilon=0.0):
        """Epsilon-Greedy 策略"""
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
            q_values = self.q_network(obs_t)
            return torch.argmax(q_values, dim=1).item()
            
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
        
    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
