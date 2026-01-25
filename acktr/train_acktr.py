"""
业务问题：使用 ACKTR 算法训练 CartPole。
实现逻辑：
1. 收集数据 (同 A2C)。
2. 使用 K-FAC 优化器：
   - 先执行 backward() 收集梯度和 A/G 统计量。
   - optimizer.step() 计算预条件并更新。
"""

import time
import random
import numpy as np
import torch
import gymnasium as gym
from acktr_agent import ACKTRAgent
from kfac import KFACOptimizer

# 超参数
SEED = 42
TOTAL_TIMESTEPS = 20000
LEARNING_RATE = 0.25 # K-FAC 学习率通常较大
NUM_STEPS = 20       # 更长的 Rollout 有助于 K-FAC 估计
NUM_ENVS = 4
GAMMA = 0.99

def make_env(gym_id, seed, idx, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    run_name = f"ACKTR_CartPole_{SEED}_{int(time.time())}"
    
    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", SEED + i, i, run_name) for i in range(NUM_ENVS)]
    )
    
    device = torch.device("cpu") # K-FAC 涉及大量求逆，小模型在 CPU 上如果不针对 GPU 优化通常也很快
    agent = ACKTRAgent(envs).to(device)
    
    # 使用自定义 K-FAC 优化器
    optimizer = KFACOptimizer(agent, lr=LEARNING_RATE)
    
    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    
    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)
    
    print("Start ACKTR Training (PyTorch + K-FAC)...")
    
    for update in range(1, num_updates + 1):
        # 1. Rollout
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

        # 2. 计算 Returns (Bootstrap)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = returns[t + 1]
                returns[t] = rewards[t] + GAMMA * nextnonterminal * nextvalues
                
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_advantages = b_returns - values.reshape(-1)
        
        # 3. K-FAC Update
        optimizer.zero_grad()
        loss = agent.compute_loss(b_obs, b_actions.long(), b_returns, b_advantages)
        loss.backward() # 计算标准梯度，同时 K-FAC Hook 会捕获 Forward 和 Backward 的统计量
        
        # optimizer.step() 会自动执行预条件处理 (Preconditioning)
        optimizer.step()
        
    envs.close()
    print("Training Completed!")
