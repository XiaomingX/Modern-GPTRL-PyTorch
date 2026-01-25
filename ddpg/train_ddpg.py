"""
业务问题：使用 DDPG 算法训练 Pendulum-v1 (经典控制任务：倒立摆)。
实现逻辑：
1. 本地 ReplayBuffer 存储经验。
2. Ornstein-Uhlenbeck 噪声用于探索。
3. 软更新目标网络 (Polyak Averaging)。
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from ddpg_agent import DDPGAgent

# 超参数
SEED = 42
TOTAL_TIMESTEPS = 20000
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
BUFFER_SIZE = int(1e5)
GAMMA = 0.99
TAU = 0.005 # 软更新系数
BATCH_SIZE = 64
NOISE_SIGMA = 0.2
WARMUP_STEPS = 1000 # 随机探索步数

def make_env(gym_id, seed):
    env = gym.make(gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

# OU 噪声 (Ornstein-Uhlenbeck Process)
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.state = float(mu) * np.ones(size)
        self.mu = float(mu)
        self.theta = theta
        self.sigma = sigma
        self.size = size

    def reset(self):
        self.state = float(self.mu) * np.ones(self.size)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

if __name__ == "__main__":
    env_id = "Pendulum-v1"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    env = make_env(env_id, SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化网络 (主网络 + 目标网络)
    agent = DDPGAgent(env).to(device)
    target_agent = DDPGAgent(env).to(device)
    target_agent.load_state_dict(agent.state_dict()) # 同步初始权重
    
    optimizer_actor = optim.Adam(agent.actor.parameters(), lr=LEARNING_RATE_ACTOR)
    optimizer_critic = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE_CRITIC)
    
    # Replay Buffer (使用简单的 list 或 numpy array)
    obs_buf = np.zeros((BUFFER_SIZE, env.observation_space.shape[0]), dtype=np.float32)
    next_obs_buf = np.zeros((BUFFER_SIZE, env.observation_space.shape[0]), dtype=np.float32)
    actions_buf = np.zeros((BUFFER_SIZE, env.action_space.shape[0]), dtype=np.float32)
    rewards_buf = np.zeros((BUFFER_SIZE), dtype=np.float32)
    dones_buf = np.zeros((BUFFER_SIZE), dtype=np.float32)
    ptr, size = 0, 0
    
    noise = OUNoise(env.action_space.shape[0], sigma=NOISE_SIGMA)
    
    obs, _ = env.reset(seed=SEED)
    noise.reset()
    
    print(f"Start Training DDPG on {env_id}...")
    
    for global_step in range(TOTAL_TIMESTEPS):
        # 1. 选择动作 (Warmup 期随机，之后用策略+噪声)
        if global_step < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = agent.actor(torch.Tensor(obs).to(device).view(1, -1))
                action = action.cpu().numpy().flatten()
                # 添加噪声并截断
                action += noise.sample()
                action = np.clip(action, env.action_space.low, env.action_space.high)
        
        # 2. 环境交互
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 3. 存入 Buffer
        obs_buf[ptr] = obs
        next_obs_buf[ptr] = next_obs
        actions_buf[ptr] = action
        rewards_buf[ptr] = reward
        dones_buf[ptr] = float(terminated)
        ptr = (ptr + 1) % BUFFER_SIZE
        size = min(size + 1, BUFFER_SIZE)
        
        obs = next_obs
        
        if "episode" in info:
            print(f"Global Step: {global_step} | Episode Reward: {info['episode']['r']}")
            obs, _ = env.reset(seed=SEED)
            noise.reset()
            
        # 4. 训练 (如果在 Warmup 之后)
        if global_step >= WARMUP_STEPS:
            # 随机采样 Batch
            idxs = np.random.randint(0, size, size=BATCH_SIZE)
            b_obs = torch.Tensor(obs_buf[idxs]).to(device)
            b_next_obs = torch.Tensor(next_obs_buf[idxs]).to(device)
            b_actions = torch.Tensor(actions_buf[idxs]).to(device)
            b_rewards = torch.Tensor(rewards_buf[idxs]).to(device)
            b_dones = torch.Tensor(dones_buf[idxs]).to(device)
            
            # --- 更新 Critic ---
            with torch.no_grad():
                # 目标 Q 值: r + gamma * Q_target(next_s, target_actor(next_s))
                next_state_actions = target_agent.actor(b_next_obs)
                q_next_target = target_agent.critic(b_next_obs, next_state_actions).flatten()
                target_q = b_rewards + (1 - b_dones) * GAMMA * q_next_target
            
            q_values = agent.critic(b_obs, b_actions).flatten()
            critic_loss = nn.functional.mse_loss(q_values, target_q)
            
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
            
            # --- 更新 Actor ---
            # 最大化 Q(s, actor(s))
            actor_loss = -agent.critic(b_obs, agent.actor(b_obs)).mean()
            
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()
            
            # --- 软更新目标网络 ---
            # param_target = tau * param + (1 - tau) * param_target
            for param, target_param in zip(agent.actor.parameters(), target_agent.actor.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(agent.critic.parameters(), target_agent.critic.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                
    env.close()
    print("Training Completed!")
