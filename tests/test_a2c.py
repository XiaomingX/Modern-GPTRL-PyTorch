"""
A2C 算法测试
测试目标：验证 A2C 智能体能够正常初始化和训练
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import gymnasium as gym
import numpy as np
from a2c.a2c_agent import A2CAgent

def test_a2c_agent_initialization():
    """测试 A2C Agent 能否正常初始化"""
    env = gym.make("CartPole-v1")
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    agent = A2CAgent(envs)
    assert agent is not None
    assert hasattr(agent, 'actor')
    assert hasattr(agent, 'critic')
    print("✓ A2C Agent 初始化成功")
    envs.close()

def test_a2c_forward_pass():
    """测试 A2C 前向传播"""
    env = gym.make("CartPole-v1")
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    agent = A2CAgent(envs)
    obs = torch.randn(1, 4)
    
    action, logprob, entropy, value = agent.get_action_and_value(obs)
    
    assert action is not None
    assert logprob is not None
    assert entropy is not None
    assert value is not None
    print("✓ A2C 前向传播正常")
    envs.close()

def test_a2c_value_estimation():
    """测试 A2C 价值估计"""
    env = gym.make("CartPole-v1")
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    agent = A2CAgent(envs)
    obs = torch.randn(4, 4)
    
    values = agent.get_value(obs)
    assert values.shape == (4, 1)
    print(f"✓ A2C 价值估计正常，值范围: [{values.min().item():.2f}, {values.max().item():.2f}]")
    envs.close()

if __name__ == "__main__":
    print("=" * 50)
    print("开始测试 A2C 算法")
    print("=" * 50)
    
    test_a2c_agent_initialization()
    test_a2c_forward_pass()
    test_a2c_value_estimation()
    
    print("\n" + "=" * 50)
    print("所有 A2C 测试通过 ✓")
    print("=" * 50)
