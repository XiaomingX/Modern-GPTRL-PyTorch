"""
业务问题：训练 GRPO 智能体解决一个简单的“数字对齐”任务。
任务描述：Agent 看到一个目标数字，需要通过多次动作之和逼近该数字。
GRPO 核心：对同一状态产生多个轨迹，计算它们的相对好坏。
"""

import torch
import numpy as np
import random
from grpo_agent import GRPOAgent

class SymbolicMathEnv:
    """极简环境：目标是输出动作索引，使索引之和等于目标值"""
    def __init__(self, target_sum=10):
        self.target_sum = target_sum
        self.obs_dim = 1
        self.action_dim = 11 # 0 to 10
        
    def get_obs(self):
        return np.array([self.target_sum], dtype=np.float32)
        
    def compute_reward(self, actions):
        # actions is a list of actions taken for one question
        total = sum(actions)
        # 距离越近奖励越高
        return -abs(total - self.target_sum)

def train_grpo():
    device = torch.device("cpu")
    env = SymbolicMathEnv(target_sum=10)
    agent = GRPOAgent(env.obs_dim, env.action_dim, device)
    
    group_size = 16 # 每个“问题”生成 16 个候选轨迹
    print(f"Start GRPO Training on SymbolicMath (Group Size: {group_size})...")
    
    for epoch in range(500):
        obs = env.get_obs()
        obs_tensor = torch.FloatTensor(obs).repeat(group_size, 1).to(device)
        
        # 1. 采样组内样本 (Group Rollout)
        with torch.no_grad():
            probs = agent.get_action_probs(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample() # [group_size]
            
        # 2. 计算奖励
        rewards = []
        for i in range(group_size):
            r = env.compute_reward([actions[i].item()])
            rewards.append(r)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        
        # 3. 更新 Agent
        # 这里 old_probs 就用本次采样的 probs
        loss, kl = agent.update(obs_tensor, actions, rewards_tensor, probs)
        
        if epoch % 50 == 0:
            avg_rew = rewards_tensor.mean().item()
            print(f"Epoch: {epoch} | Avg Reward: {avg_rew:.2f} | KL: {kl:.4f}")

    print("Training Completed!")

if __name__ == "__main__":
    train_grpo()