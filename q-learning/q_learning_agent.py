"""
业务问题：构建 Q-Learning 智能体，用于离散状态和动作空间的表格型控制。
实现逻辑：
1. Q 表 (Q-Table) -> 存储状态-动作价值。
2. 贝尔曼方程更新 -> Q(s,a) = Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))。
3. Epsilon-Greedy 策略 -> 平衡探索与利用。
"""

import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, action_space_n, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.action_space_n = action_space_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {} # 使用字典存储稀疏状态，key: state, value: np.array([q_a1, q_a2...])

    def get_q(self, state):
        """获取状态的 Q 值，如果不存在则初始化为全 0"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_n)
        return self.q_table[state]

    def choose_action(self, state, train=True):
        """Epsilon-Greedy 策略选择动作"""
        if train and random.random() < self.epsilon:
            return random.randint(0, self.action_space_n - 1)
        else:
            q_values = self.get_q(state)
            # breaking ties randomly to encourage exploration in early stages
            max_q = np.max(q_values)
            return np.random.choice(np.flatnonzero(q_values == max_q))

    def update(self, state, action, reward, next_state):
        """Q-Learning 更新公式"""
        q_current = self.get_q(state)[action]
        q_next_max = np.max(self.get_q(next_state))
        
        # TD Target
        td_target = reward + self.gamma * q_next_max
        
        # 更新 Q 值
        self.q_table[state][action] += self.lr * (td_target - q_current)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)
