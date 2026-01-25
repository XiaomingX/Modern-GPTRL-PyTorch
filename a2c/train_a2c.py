"""
业务问题：使用 A2C 算法训练智能体玩 CartPole 游戏。
实现逻辑：
1. 同步收集 N 步数据 (Rollout)。
2. 计算 N 步折扣回报 (Bootstrapped Returns)。
3. 计算优势 (Advantage) 和 损失 (Loss)。
4. 反向传播更新网络。
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from a2c_agent import A2CAgent

# 超参数配置
SEED = 42
TOTAL_TIMESTEPS = 20000
LEARNING_RATE = 7e-4
NUM_STEPS = 5        # A2C 特有的 N-Step Rollout (通常较短)
NUM_ENVS = 4         # 并行环境数
GAMMA = 0.99         # 折扣因子
ENT_COEF = 0.01      # 熵系数
VF_COEF = 0.5        # 价值损失系数
MAX_GRAD_NORM = 0.5  # 梯度裁剪

def make_env(gym_id, seed, idx, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    # 1. 环境与种子初始化
    run_name = f"A2C_CartPole_{SEED}_{int(time.time())}"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", SEED + i, i, run_name) for i in range(NUM_ENVS)]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = A2CAgent(envs).to(device)
    optimizer = optim.RMSprop(agent.parameters(), lr=LEARNING_RATE, eps=1e-5, alpha=0.99)
    
    # buffers
    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    
    # 初始状态
    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)
    
    print("Start Training A2C...")
    
    for update in range(1, num_updates + 1):
        # -------------------- 阶段 1: 收集 N 步数据 --------------------
        for step in range(0, NUM_STEPS):
            global_step += 1 * NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"Global Step: {global_step} | Episode Reward: {info['episode']['r']}")

        # -------------------- 阶段 2: 计算回报与损失 --------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            returns = torch.zeros_like(rewards).to(device)
            # 计算 Bootstrap Returns (从后往前)
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = returns[t + 1]
                returns[t] = rewards[t] + GAMMA * nextnonterminal * nextvalues
                
        # 展平 Batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_advantages = b_returns - b_values # A = R - V
        
        # -------------------- 阶段 3: 更新网络 --------------------
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions.long())
        
        # Policy Loss
        pg_loss = -(b_advantages * newlogprob).mean()
        
        # Value Loss
        v_loss = 0.5 * ((newvalue.view(-1) - b_returns) ** 2).mean()
        
        # Entropy Loss
        entropy_loss = entropy.mean()
        
        loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
    envs.close()
    print("Training Completed!")
