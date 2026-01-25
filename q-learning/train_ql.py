"""
业务问题：使用 Q-Learning 算法训练 FrozenLake-v1 (滑冰场寻路)。
实现逻辑：
1. 离散状态空间 (GridWorld)。
2. Q-Learning 更新。
3. 训练与测试分离。
"""

import time
import numpy as np
import gymnasium as gym
from q_learning_agent import QLearningAgent

# 超参数
TOTAL_EPISODES = 10000
LEARNING_RATE = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995

def train_ql(env_id="FrozenLake-v1", map_name="4x4", is_slippery=True):
    # 1. 创建环境 (is_slippery=True 增加了随机滑动的难度)
    env = gym.make(env_id, map_name=map_name, is_slippery=is_slippery)
    
    # 2. 初始化 Agent
    agent = QLearningAgent(
        action_space_n=env.action_space.n,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START
    )
    
    print(f"Start Training Q-Learning on {env_id} ({map_name})...")
    
    start_time = time.time()
    rewards_history = []
    
    for episode in range(1, TOTAL_EPISODES + 1):
        state, _ = env.reset()
        # FrozenLake 的 state 是一个整数，直接可用作 key
        
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state, train=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 训练 Agent
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
        # 衰减 Epsilon
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
        rewards_history.append(total_reward)
        
        if episode % 1000 == 0:
            avg_reward = np.mean(rewards_history[-1000:])
            print(f"Episode: {episode} | Avg Reward (last 1000): {avg_reward:.3f} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    print("Training Completed!")
    return agent

def test_ql(agent, env_id="FrozenLake-v1", map_name="4x4", is_slippery=True, episodes=5):
    # render_mode="human" 可以看到可视化界面
    env = gym.make(env_id, map_name=map_name, is_slippery=is_slippery, render_mode="human")
    
    print(f"\nStart Testing...")
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        print(f"Episode {ep+1} Start")
        
        while not done:
            action = agent.choose_action(state, train=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
        print(f"Episode {ep+1} Finished. Reward: {reward}")
        time.sleep(0.5)
        
    env.close()

if __name__ == "__main__":
    # 训练
    agent = train_ql()
    
    # 测试 (展示效果)
    # test_ql(agent)
    
    # 保存
    # agent.save("ql_model.pkl")
