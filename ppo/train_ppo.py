"""
业务问题：使用 PPO 算法训练智能体玩 CartPole 游戏。
实现逻辑：
1. 收集数据 (Rollout): 智能体在环境中交互 T 步，存储 (State, Action, Reward, Value)。
2. 计算优势 (GAE): 估算动作相比于平均水平好多少。
3. 策略更新 (Optimize): 使用 Clipping 技术限制更新步长，最大化目标函数。
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from ppo_agent import Agent

# 超参数配置
SEED = 42
TOTAL_TIMESTEPS = 50000
LEARNING_RATE = 2.5e-4
NUM_STEPS = 128      # 每次更新收集的步数
NUM_ENVS = 4         # 并行环境数
GAMMA = 0.99         # 折扣因子
GAE_LAMBDA = 0.95    # GAE 平滑系数
CLIP_COEF = 0.2      # PPO 裁剪系数
ENT_COEF = 0.01      # 熵系数 (鼓励探索)
VF_COEF = 0.5        # 价值损失系数
UPDATE_EPOCHS = 4    # 每次更新迭代次数
BATCH_SIZE = NUM_ENVS * NUM_STEPS # 512

def make_env(gym_id, seed, idx, run_name):
    """创建环境的辅助函数"""
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env) # 记录回合奖励
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    # 1. 初始化环境与种子
    run_name = f"CartPole_v1_{SEED}_{int(time.time())}"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # 向量化环境 (同时跑4个环境)
    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", SEED + i, i, run_name) for i in range(NUM_ENVS)]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # 2. 训练循环
    # 存储数据的 Buffer
    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    
    # 初始观测
    next_obs, _ = envs.reset(seed=SEED) # VectorEnv reset 返回 (obs, info)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE
    
    print("Start training...")
    for update in range(1, num_updates + 1):
        # -------------------- 阶段 1: 收集数据 (Rollout) --------------------
        for step in range(0, NUM_STEPS):
            global_step += 1 * NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            # 环境交互
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            # 打印日志 (如果有环境结束)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"Global Step: {global_step} | Episode Reward: {info['episode']['r']}")
                        break

        # -------------------- 阶段 2: 计算优势 (GAE) --------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        # -------------------- 阶段 3: 策略更新 (Optimize) --------------------
        # 展平 batch 用于训练
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 多个 Epoch 更新
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, 64): # Mini-batch size 64
                end = start + 64
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                # 优势归一化
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # PPO Clip Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                # Entropy Loss
                entropy_loss = entropy.mean()
                
                # 总 Loss
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

    envs.close()
    print("Training compeleted!")
