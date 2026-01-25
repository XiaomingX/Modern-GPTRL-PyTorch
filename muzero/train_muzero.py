"""
业务问题：使用 MuZero 算法训练 CartPole-v1。
实现逻辑：
1. MuZero 的核心是不需要知道环境模型，而是通过学习一个隐空间的动力学模型。
2. 训练目标：预测的 Value、Policy、Reward 要与 MCTS 搜索结果及真实观察一致。
"""

import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
from muzero_model import MuZeroNetwork
from mcts import mcts_search

def train_muzero():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    state_dim = 16 # 隐藏状态维度
    
    network = MuZeroNetwork(obs_dim, action_dim, state_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    
    print("Start MuZero Training on CartPole-v1...")
    
    replay_buffer = []
    MAX_BUFFER = 5000
    
    for episode in range(1000):
        obs, _ = env.reset()
        game_history = []
        total_reward = 0
        
        # 1. 数据收集 (Self-play)
        while True:
            # 运行 MCTS 获取搜索改进后的策略
            root = mcts_search(network, obs, num_simulations=50)
            
            # 构造目标分布 (访问次数归一化)
            visit_counts = [child.visit_count for child in root.children.values()]
            policy_target = np.array(visit_counts) / sum(visit_counts)
            
            # 选择动作
            action = np.random.choice(action_dim, p=policy_target)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            game_history.append((obs, action, reward, policy_target, root.value))
            
            obs = next_obs
            total_reward += reward
            if done:
                break
        
        replay_buffer.append(game_history)
        if len(replay_buffer) > MAX_BUFFER // 10: # 简化版 buffer
            replay_buffer.pop(0)

        # 2. 训练更新 (训练隐空间模型)
        if len(replay_buffer) > 5:
            # 随机采样一个 episode 和一段长度
            batch_data = random.choice(replay_buffer)
            # 简化版：只更新一步预测
            # 输入当前的 obs，预测 value, policy, reward (0), 及 next_state
            indices = list(range(len(batch_data)))
            random.shuffle(indices)
            
            for i in indices[:10]: # 每次更新 10 步
                curr_obs, curr_act, curr_rew, target_pol, target_val = batch_data[i]
                
                # 初始推理
                state, pred_val, pred_logits = network.initial_inference(torch.FloatTensor(curr_obs).unsqueeze(0))
                
                # 损失 1: 价值损失
                value_loss = F.mse_loss(pred_val.squeeze(), torch.tensor(target_val).float())
                # 损失 2: 策略损失
                policy_loss = F.cross_entropy(pred_logits, torch.tensor([np.argmax(target_pol)]))
                
                total_loss = value_loss + policy_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        if episode % 20 == 0:
            print(f"Episode: {episode} | Reward: {total_reward}")

    print("Training Completed!")

if __name__ == "__main__":
    train_muzero()
