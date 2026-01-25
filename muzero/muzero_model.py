import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationNetwork(nn.Module):
    """表示网络：将原始观测映射到隐藏状态（隐空间）"""
    def __init__(self, obs_dim, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
            nn.Tanh()  # 限制隐藏状态范围
        )

    def forward(self, obs):
        return self.net(obs)

class DynamicsNetwork(nn.Module):
    """动力学网络：预测 (当前隐藏状态, 动作) -> (下一隐藏状态, 奖励)"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim + 1) # state + reward
        )
        self.state_dim = state_dim

    def forward(self, state, action):
        # action expectation: one-hot or categorical
        x = torch.cat([state, action], dim=-1)
        out = self.net(x)
        next_state = torch.tanh(out[:, :self.state_dim])
        reward = out[:, self.state_dim:]
        return next_state, reward

class PredictionNetwork(nn.Module):
    """预测网络：预测当前隐藏状态下的 (Q值分布/价值, 策略/先验概率)"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 价值头
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # 策略头
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        value = self.value_net(state)
        policy_logits = self.policy_net(state)
        return value, policy_logits

class MuZeroNetwork(nn.Module):
    """汇总 MuZero 的三个子网络"""
    def __init__(self, obs_dim, action_dim, state_dim):
        super().__init__()
        self.representation = RepresentationNetwork(obs_dim, state_dim)
        self.dynamics = DynamicsNetwork(state_dim, action_dim)
        self.prediction = PredictionNetwork(state_dim, action_dim)
        self.action_dim = action_dim

    def initial_inference(self, obs):
        state = self.representation(obs)
        value, policy_logits = self.prediction(state)
        return state, value, policy_logits

    def recurrent_inference(self, state, action_idx):
        # 将动作索引转为 one-hot
        action_onehot = F.one_hot(action_idx, num_classes=self.action_dim).float()
        next_state, reward = self.dynamics(state, action_onehot)
        value, policy_logits = self.prediction(next_state)
        return next_state, reward, value, policy_logits
