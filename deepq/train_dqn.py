"""
业务问题：使用 DQN 算法训练 CartPole-v1。
实现逻辑：
1. 经验回放池 (Replay Buffer) 存储交互数据。
2. 批量采样 -> 计算 MSE Loss -> 反向传播更新 Q 网络。
3. 定期更新目标网络。
"""

import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from dqn_agent import DQNAgent

# 超参数
SEED = 42
TOTAL_TIMESTEPS = 50000 # 50k 步通常足够 CartPole 收敛
LEARNING_RATE = 2.5e-4
BUFFER_SIZE = 10000
GAMMA = 0.99
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 1000 # 步数
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995 # 可选衰减方式，这里用简单的线性或指数
START_LEARNING_STEPS = 1000 # 存够多少数据开始训练

def make_env(gym_id, seed):
    env = gym.make(gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

# 简单的 Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_shape):
        self.capacity = capacity
        # 使用 numpy 预分配内存，比 list append 更高效
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.int64) 
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        
    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.obs[idxs],
            next_obs=self.next_obs[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            dones=self.dones[idxs]
        )

if __name__ == "__main__":
    env_id = "CartPole-v1"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    env = make_env(env_id, SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = DQNAgent(env, LEARNING_RATE, GAMMA)
    
    # 初始化 Buffer (CartPole action 是离散的标量)
    buffer = ReplayBuffer(BUFFER_SIZE, env.observation_space.shape, ())
    
    obs, _ = env.reset(seed=SEED)
    epsilon = EPSILON_START
    
    print(f"Start Training DQN on {env_id}...")
    
    for global_step in range(TOTAL_TIMESTEPS):
        # 1. 线性衰减 Epsilon
        # epsilon = max(EPSILON_END, EPSILON_START - global_step / TOTAL_TIMESTEPS) # 线性
        # 或者指数衰减
        if global_step > START_LEARNING_STEPS:
             epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # 2. 选动作
        action = agent.get_action(obs, epsilon)
        
        # 3. 环境交互
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 4. 存经验
        buffer.add(obs, action, reward, next_obs, float(terminated)) # 注意: truncated 不算环境终止状态
        
        obs = next_obs
        
        if done:
            if "episode" in info:
                print(f"Global Step: {global_step} | Episode Reward: {info['episode']['r']} | Epsilon: {epsilon:.3f}")
            obs, _ = env.reset(seed=SEED)
            
        # 5. 训练
        if global_step > START_LEARNING_STEPS:
            data = buffer.sample(BATCH_SIZE)
            
            b_obs = torch.Tensor(data['obs']).to(device)
            b_next_obs = torch.Tensor(data['next_obs']).to(device)
            b_actions = torch.LongTensor(data['actions']).to(device)
            b_rewards = torch.Tensor(data['rewards']).to(device)
            b_dones = torch.Tensor(data['dones']).to(device)
            
            with torch.no_grad():
                # 计算目标值: y = r + gamma * max Q_target(next_s, a)
                target_max, _ = agent.target_network(b_next_obs).max(dim=1)
                td_target = b_rewards + GAMMA * target_max * (1 - b_dones)
                
            # 计算当前值: Q(s, a)
            q_values = agent.q_network(b_obs)
            old_val = q_values.gather(1, b_actions.unsqueeze(1)).squeeze()
            
            loss = F.mse_loss(old_val, td_target)
            
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            
            # 6. 定期更新目标网络
            if global_step % TARGET_UPDATE_FREQ == 0:
                agent.target_network.load_state_dict(agent.q_network.state_dict())
                
    env.close()
    print("Training Completed!")
