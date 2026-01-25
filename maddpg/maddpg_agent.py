"""
业务问题：构建 MA-DDPG 智能体 (Centralized Training, Decentralized Execution)。
实现逻辑：
1. Actor (Local): 输入 obs[i]，输出 action[i]。
2. Critic (Global): 输入 all_obs + all_actions，输出 Q 值。
3. 训练时：Critic 能够看到所有人的信息；推理时：Actor 只看局部信息。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh()
        )
    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, sum_obs_dim, sum_act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sum_obs_dim + sum_act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, all_obs, all_act):
        x = torch.cat([all_obs, all_act], dim=1)
        return self.net(x)

class MADDPGAgent:
    def __init__(self, obs_dim, act_dim, agent_idx, args):
        self.agent_idx = agent_idx
        self.args = args
        self.device = args.device
        
        # Local Actor
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.target_actor = Actor(obs_dim, act_dim).to(self.device)
        hard_update(self.target_actor, self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.01)
        
        # Global Critic (for this agent's objective)
        self.critic = Critic(args.total_obs_dim, args.total_act_dim).to(self.device)
        self.target_critic = Critic(args.total_obs_dim, args.total_act_dim).to(self.device)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.01)

    def select_action(self, obs, noise=0.0):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        action = self.actor(obs).detach().cpu().numpy()[0]
        if noise != 0:
            action += np.random.randn(*action.shape) * noise
            action = np.clip(action, -1, 1)
        return action
    
    def update(self, agents, sample, logger):
        # sample: {'obs': [N, obs_n, dim], 'act': ..., 'next_obs': ...}
        # 转换数据格式
        # obs_n: list of tensors [batch, dim]
        obs_n = [sample['obs'][:, i, :] for i in range(len(agents))]
        act_n = [sample['act'][:, i, :] for i in range(len(agents))]
        next_obs_n = [sample['next_obs'][:, i, :] for i in range(len(agents))]
        rew = sample['rew'][:, self.agent_idx].unsqueeze(1) # [batch, 1]
        done = sample['done'][:, self.agent_idx].unsqueeze(1)
        
        # 1. 训练 Critic
        # 计算 Target Q
        with torch.no_grad():
            # 所有 Agents 的 Target Actor 输出下一动作
            next_act_n = [ag.target_actor(next_obs_n[i]) for i, ag in enumerate(agents)]
            # 拼接所有信息
            cat_next_obs = torch.cat(next_obs_n, dim=1)
            cat_next_act = torch.cat(next_act_n, dim=1)
            
            target_q = self.target_critic(cat_next_obs, cat_next_act)
            y = rew + self.args.gamma * (1 - done) * target_q
            
        # Current Q
        cat_obs = torch.cat(obs_n, dim=1)
        cat_act = torch.cat(act_n, dim=1)
        current_q = self.critic(cat_obs, cat_act)
        
        critic_loss = nn.MSELoss()(current_q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 2. 训练 Actor
        # 最大化 Q(s, a_1, ..., a_i_new, ..., a_n)
        # 要拿到当前Actor的最新动作，其他的用 buffer 里的旧动作? 
        # MA-DDPG 论文建议：During training of actor i, other agents' actions are sampled from their current policies? 
        # 或者是回放池里的动作? 通常是用 current policies。
        
        # 使用 Gumbel-Softmax 或 Deterministic gradients 获取当前策略动作
        curr_act_n = [ag.actor(obs_n[i]) if i == self.agent_idx else ag.actor(obs_n[i]).detach() 
                      for i, ag in enumerate(agents)]
        cat_curr_act = torch.cat(curr_act_n, dim=1)
        
        actor_loss = -self.critic(cat_obs, cat_curr_act).mean()
        # 加正则项减小输出
        actor_loss += (curr_act_n[self.agent_idx]**2).mean() * 1e-3
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 3. Soft Update
        soft_update(self.target_actor, self.actor, 0.01)
        soft_update(self.target_critic, self.critic, 0.01)
