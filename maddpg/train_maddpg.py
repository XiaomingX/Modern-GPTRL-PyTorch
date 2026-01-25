"""
业务问题：训练 MA-DDPG 在 SimpleMPE 环境中协作覆盖地标。
"""

import numpy as np
import torch
import random
from simple_mpe import SimpleMPE
from maddpg_agent import MADDPGAgent

# Replay Buffer
class MultiAgentReplayBuffer:
    def __init__(self, capacity, n_agents, obs_dims, act_dims):
        self.capacity = capacity
        self.n = n_agents
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate numpy arrays
        # Shape: [capacity, n_agents, dim]
        self.obs = np.zeros((capacity, n_agents, max(obs_dims))) 
        # 注意：这里假设 obs_dim 不一样大的话取 max，实际我们的 SimpleMPE 一样大。
        # 更严谨的做法是 list of arrays，但 tensor 处理麻烦。这里定长。
        self.act = np.zeros((capacity, n_agents, max(act_dims)))
        self.rew = np.zeros((capacity, n_agents))
        self.next_obs = np.zeros((capacity, n_agents, max(obs_dims)))
        self.done = np.zeros((capacity, n_agents))
        
    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = np.stack(obs)
        self.act[self.ptr] = np.stack(act)
        self.rew[self.ptr] = np.array(rew)
        self.next_obs[self.ptr] = np.stack(next_obs)
        self.done[self.ptr] = np.array(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size, device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': torch.FloatTensor(self.obs[idxs]).to(device),
            'act': torch.FloatTensor(self.act[idxs]).to(device),
            'rew': torch.FloatTensor(self.rew[idxs]).to(device),
            'next_obs': torch.FloatTensor(self.next_obs[idxs]).to(device),
            'done': torch.FloatTensor(self.done[idxs]).to(device),
        }

class Args:
    pass

def train_maddpg():
    args = Args()
    args.device = torch.device("cpu")
    args.gamma = 0.95
    
    n_agents = 3
    env = SimpleMPE(n_agents=n_agents)
    
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    
    # 全局 Critic 需要的总维度
    args.total_obs_dim = obs_dim * n_agents
    args.total_act_dim = act_dim * n_agents
    
    agents = [MADDPGAgent(obs_dim, act_dim, i, args) for i in range(n_agents)]
    buffer = MultiAgentReplayBuffer(100000, n_agents, [obs_dim]*n_agents, [act_dim]*n_agents)
    
    print("Start MA-DDPG Training on SimpleMPE...")
    
    total_steps = 0
    max_episodes = 5000
    max_steps_per_ep = 25
    
    for episode in range(max_episodes):
        obs = env.reset() # [N, obs_dim]
        ep_rew = 0
        
        for t in range(max_steps_per_ep):
            # 1. Select Actions
            actions = []
            for i, ag in enumerate(agents):
                act = ag.select_action(obs[i], noise=0.1) # Add exploration noise
                actions.append(act)
            
            # 2. Step
            next_obs, rewards, dones, _ = env.step(actions)
            
            # 3. Store
            buffer.add(obs, actions, rewards, next_obs, dones)
            
            obs = next_obs
            ep_rew += sum(rewards) # Global Reward
            
            total_steps += 1
            
            # 4. Train
            if total_steps % 100 == 0 and buffer.size > 1024:
                sample = buffer.sample(1024, args.device)
                for ag in agents:
                    ag.update(agents, sample, None)
                    
        # Log
        if episode % 100 == 0:
            print(f"Episode: {episode} | Ep Reward: {ep_rew:.2f}")

    print("Training Completed!")

if __name__ == "__main__":
    train_maddpg()
