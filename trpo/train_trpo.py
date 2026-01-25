"""
业务问题：使用 TRPO 算法训练 CartPole-v1。
实现逻辑：
1. 收集整条轨迹 (Trajectory) 进行蒙特卡洛估计或 GAE。
2. 计算优势函数。
3. 执行 TRPO 更新。
"""

import time
import numpy as np
import torch
import gymnasium as gym
from trpo_agent import TRPOAgent

# 超参数
MAX_KL = 0.01
DAMPING = 0.1
GAMMA = 0.99
LAMBDA = 0.95 # GAE lambda
TOTAL_TIMESTEPS = 30000
BATCH_SIZE = 4000 # TRPO 需要较大的 Batch 来准确估计 Fisher 矩阵

def train_trpo():
    env = gym.make("CartPole-v1")
    device = torch.device("cpu") # TRPO 计算密集，但 CartPole 模型小，CPU 够用
    
    agent = TRPOAgent(env.observation_space.shape[0], env.action_space.n, device)
    
    print("Start TRPO Training on CartPole-v1...")
    
    global_steps = 0
    
    while global_steps < TOTAL_TIMESTEPS:
        # 1. 收集数据 (Rollout)
        batch_obs, batch_acts, batch_rews = [], [], []
        batch_rets, batch_lens = [], []
        
        batch_steps = 0
        while batch_steps < BATCH_SIZE:
            obs, _ = env.reset()
            ep_rews = []
            while True:
                # 采样动作
                action, log_prob = agent.get_action([obs])
                
                batch_obs.append(obs)
                batch_acts.append(action)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_rews.append(reward)
                done = terminated or truncated
                
                if done:
                    batch_rets.append(sum(ep_rews))
                    batch_lens.append(len(ep_rews))
                    batch_steps += len(ep_rews)
                    batch_rews.append(ep_rews)
                    break
        
        global_steps += batch_steps
        
        # 2. 处理数据 (计算 Advantage 和 Return)
        obs_tensor = np.array(batch_obs)
        act_tensor = np.array(batch_acts)
        
        # Process rewards into returns & advantages
        all_returns = []
        all_advantages = []
        
        # 使用 GAE (Generalized Advantage Estimation)
        # 为此我们需要 Critic 对每个状态的 Value 估计
        v_preds = agent.critic(torch.Tensor(obs_tensor).to(device)).detach().cpu().numpy().flatten()
        
        # 将 v_preds 切分回轨迹
        v_preds_list = []
        cursor = 0
        for ep_r in batch_rews:
            v_preds_list.append(v_preds[cursor : cursor + len(ep_r)])
            cursor += len(ep_r)
            
        for i, ep_r in enumerate(batch_rews):
            v_p = v_preds_list[i]
            # 补一个 0 表示 terminal value
            v_p_next = np.append(v_p[1:], 0.0) 
            
            advs = []
            gae = 0.0
            for t in reversed(range(len(ep_r))):
                delta = ep_r[t] + GAMMA * v_p_next[t] - v_p[t]
                gae = delta + GAMMA * LAMBDA * gae
                advs.insert(0, gae)
                
            # Returns = Advantage + Value
            rets = np.array(advs) + v_p
            
            all_advantages.extend(advs)
            all_returns.extend(rets)
            
        # 3. TRPO 更新
        adv_tensor = np.array(all_advantages)
        # 归一化 Advantage，这对于优化稳定至关重要
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        
        ret_tensor = np.array(all_returns)
        
        agent.update(obs_tensor, act_tensor, ret_tensor, adv_tensor, max_kl=MAX_KL, damping=DAMPING)
        
        # 4. 打印日志
        avg_ret = np.mean(batch_rets)
        print(f"Global Step: {global_steps} | Avg Reward: {avg_ret:.2f} | Max Reward: {np.max(batch_rets)}")
        
    env.close()
    print("Training Completed!")

if __name__ == "__main__":
    train_trpo()
