"""
业务问题：训练 HER 智能体。为了不依赖 Mujoco，我们实现经典的 BitFlipping 环境。
BitFlipping: 状态是 N 位 0/1，动作是翻转某一位，目标是达到特定 0/1 序列。
常规 RL 很难解决，因为奖励极其稀疏（只有完全匹配才赢）。HER 通过把“失败的尝试”标记为“成功的其他目标”来解决。
"""

import numpy as np
import torch
import torch.nn.functional as F
from her_agent import HERAgent
from her_replay_buffer import HERReplayBuffer

# 自定义环境
class BitFlippingEnv:
    def __init__(self, n_bits=10):
        self.n_bits = n_bits
        self.state = np.zeros(n_bits)
        self.target = np.zeros(n_bits)
        
    def reset(self):
        self.state = np.random.randint(2, size=self.n_bits).astype(np.float32)
        self.target = np.random.randint(2, size=self.n_bits).astype(np.float32)
        # 确保目标和初始状态不同
        while np.array_equal(self.state, self.target):
            self.target = np.random.randint(2, size=self.n_bits).astype(np.float32)
        return self.state.copy(), self.target.copy()
        
    def step(self, action):
        # 动作是连续值，从 [-1, 1] 映射到 [0, n_bits-1]
        bit_idx = int(np.clip(action[0] + 1, 0, 2) / 2 * self.n_bits) 
        bit_idx = np.clip(bit_idx, 0, self.n_bits - 1)
        
        self.state[bit_idx] = 1 - self.state[bit_idx] # 翻转
        
        done = np.array_equal(self.state, self.target)
        reward = 0.0 if done else -1.0
        
        return self.state.copy(), reward, done, {}

    def compute_reward(self, achieved_goal, desired_goal):
        # 距离为 0 则奖励 0，否则 -1
        # 这里的 goal 就是 state 本身
        dist = np.abs(achieved_goal - desired_goal).sum()
        return 0.0 if dist == 0 else -1.0

# 训练配置
N_BITS = 10
TOTAL_EPISODES = 5000
MAX_STEPS = N_BITS # 最优步数就是 n_bits
BATCH_SIZE = 128

def train_her():
    env = BitFlippingEnv(N_BITS)
    device = torch.device("cpu") # 简单环境 CPU 足够
    
    agent = HERAgent(
        obs_dim=N_BITS, 
        goal_dim=N_BITS, 
        action_dim=1, # 动作：选哪一位翻转
        max_action=1.0, 
        device=device
    )
    
    buffer = HERReplayBuffer(
        capacity=100000, 
        obs_dim=N_BITS, 
        goal_dim=N_BITS, 
        action_dim=1,
        reward_func=env.compute_reward
    )
    
    print(f"Start HER Training on BitFlipping({N_BITS} bits)...")
    
    success_rate = []
    
    for episode in range(TOTAL_EPISODES):
        obs, goal = env.reset()
        episode_transitions = []
        
        for t in range(MAX_STEPS):
            action = agent.select_action(obs, goal, noise=0.2)
            next_obs, reward, done, _ = env.step(action)
            
            # 先暂存轨迹，HER 需要完整的 episode 才能更好发挥 (这里用简化版)
            # 我们的 Buffer add 需要 achieved_goal, 这里 ag 就是 next_obs
            buffer.add(obs, action, next_obs, goal, next_obs)
            
            obs = next_obs
            if done:
                break
        
        success_rate.append(1.0 if done else 0.0)
        
        # 训练
        if buffer.size > BATCH_SIZE:
            for _ in range(5):
                batch = buffer.sample(BATCH_SIZE, her_ratio=0.8)
                
                b_obs = torch.FloatTensor(batch["obs"]).to(device)
                b_goals = torch.FloatTensor(batch["goals"]).to(device)
                b_actions = torch.FloatTensor(batch["actions"]).to(device)
                b_rewards = torch.FloatTensor(batch["rewards"]).to(device)
                b_next_obs = torch.FloatTensor(batch["next_obs"]).to(device)
                b_dones = torch.FloatTensor(batch["dones"]).to(device)
                
                # Critic Update
                # Q_target = r + gamma * Q_target(s', g, u')
                with torch.no_grad():
                    target_actions = agent.actor_target(b_next_obs, b_goals)
                    target_q = agent.critic_target(b_next_obs, b_goals, target_actions)
                    target_q = b_rewards + 0.98 * (1 - b_dones) * target_q
                    target_q = torch.clamp(target_q, -1.0/(1-0.98), 0.0) # Clip Q value range [-1/(1-gamma), 0]
                
                start_q = agent.critic(b_obs, b_goals, b_actions)
                critic_loss = F.mse_loss(start_q, target_q)
                
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                agent.critic_optimizer.step()
                
                # Actor Update
                actor_loss = -agent.critic(b_obs, b_goals, agent.actor(b_obs, b_goals)).mean()
                
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                agent.actor_optimizer.step()
                
                # Soft Update
                for param, target_param in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                    target_param.data.copy_(0.05 * param.data + 0.95 * target_param.data)
                for param, target_param in zip(agent.actor.parameters(), agent.actor_target.parameters()):
                    target_param.data.copy_(0.05 * param.data + 0.95 * target_param.data)

        if episode % 200 == 0:
            avg_success = np.mean(success_rate[-200:])
            print(f"Episode: {episode} | Success Rate: {avg_success:.2f}")

    print("Training Completed!")

if __name__ == "__main__":
    train_her()
