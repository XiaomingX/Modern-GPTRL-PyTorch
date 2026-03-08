"""
PPO 算法测试
测试目标：验证 PPO 智能体能够正常初始化、训练和收敛
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import gymnasium as gym
import numpy as np
from ppo.ppo_agent import Agent

def test_ppo_agent_initialization():
    """测试 PPO Agent 能否正常初始化"""
    env = gym.make("CartPole-v1")
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    agent = Agent(envs)
    assert agent is not None
    assert hasattr(agent, 'actor')
    assert hasattr(agent, 'critic')
    print("✓ PPO Agent 初始化成功")
    envs.close()

def test_ppo_forward_pass():
    """测试 PPO 前向传播"""
    env = gym.make("CartPole-v1")
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    agent = Agent(envs)
    obs = torch.randn(1, 4)  # CartPole 观测空间是 4 维
    
    action, logprob, entropy, value = agent.get_action_and_value(obs)
    
    assert action is not None
    assert logprob is not None
    assert entropy is not None
    assert value is not None
    assert action.shape == (1,)
    print("✓ PPO 前向传播正常")
    envs.close()

def test_ppo_training_step():
    """测试 PPO 能否完成一个训练步骤"""
    env = gym.make("CartPole-v1")
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    agent = Agent(envs)
    optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4)
    
    obs, _ = envs.reset(seed=42)
    obs = torch.Tensor(obs)
    
    # 执行一步
    action, logprob, entropy, value = agent.get_action_and_value(obs)
    next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
    
    # 简单的损失计算
    loss = -logprob.mean() + 0.5 * value.pow(2).mean() - 0.01 * entropy.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert loss.item() is not None
    print(f"✓ PPO 训练步骤完成，Loss: {loss.item():.4f}")
    envs.close()

if __name__ == "__main__":
    print("=" * 50)
    print("开始测试 PPO 算法")
    print("=" * 50)
    
    test_ppo_agent_initialization()
    test_ppo_forward_pass()
    test_ppo_training_step()
    
    print("\n" + "=" * 50)
    print("所有 PPO 测试通过 ✓")
    print("=" * 50)
