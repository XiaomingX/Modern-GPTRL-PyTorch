"""
DQN 算法测试
测试目标：验证 DQN 智能体能够正常初始化、选择动作和训练
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import gymnasium as gym
import numpy as np
from deepq.dqn_agent import DQNAgent

def test_dqn_agent_initialization():
    """测试 DQN Agent 能否正常初始化"""
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env, learning_rate=2.5e-4, gamma=0.99)
    
    assert agent is not None
    assert hasattr(agent, 'q_network')
    assert hasattr(agent, 'target_network')
    assert hasattr(agent, 'optimizer')
    print("✓ DQN Agent 初始化成功")
    env.close()

def test_dqn_action_selection():
    """测试 DQN 动作选择"""
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env, learning_rate=2.5e-4, gamma=0.99)
    
    obs, _ = env.reset(seed=42)
    
    # 测试贪婪动作
    action_greedy = agent.get_action(obs, epsilon=0.0)
    assert action_greedy in [0, 1]
    
    # 测试探索动作
    action_explore = agent.get_action(obs, epsilon=1.0)
    assert action_explore in [0, 1]
    
    print("✓ DQN 动作选择正常")
    env.close()

def test_dqn_q_value_computation():
    """测试 DQN Q 值计算"""
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env, learning_rate=2.5e-4, gamma=0.99)
    
    obs = torch.randn(4, 4)  # Batch of 4 observations
    q_values = agent.q_network(obs)
    
    assert q_values.shape == (4, 2)  # CartPole 有 2 个动作
    print(f"✓ DQN Q 值计算正常，Q 值范围: [{q_values.min().item():.2f}, {q_values.max().item():.2f}]")
    env.close()

def test_dqn_target_network_update():
    """测试 DQN 目标网络更新"""
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env, learning_rate=2.5e-4, gamma=0.99)
    
    # 获取初始参数
    initial_params = [p.clone() for p in agent.target_network.parameters()]
    
    # 更新 Q 网络
    obs = torch.randn(4, 4)
    q_values = agent.q_network(obs)
    loss = q_values.mean()
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    
    # 目标网络应该没变
    for p1, p2 in zip(initial_params, agent.target_network.parameters()):
        assert torch.allclose(p1, p2)
    
    # 手动更新目标网络
    agent.target_network.load_state_dict(agent.q_network.state_dict())
    
    # 现在应该不同了
    params_changed = False
    for p1, p2 in zip(initial_params, agent.target_network.parameters()):
        if not torch.allclose(p1, p2):
            params_changed = True
            break
    
    assert params_changed
    print("✓ DQN 目标网络更新机制正常")
    env.close()

if __name__ == "__main__":
    print("=" * 50)
    print("开始测试 DQN 算法")
    print("=" * 50)
    
    test_dqn_agent_initialization()
    test_dqn_action_selection()
    test_dqn_q_value_computation()
    test_dqn_target_network_update()
    
    print("\n" + "=" * 50)
    print("所有 DQN 测试通过 ✓")
    print("=" * 50)
