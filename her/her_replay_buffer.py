"""
业务问题：构建 HER 经验回放池。
实现逻辑：
1. 存储 transitions (s, a, r, s', g, result_g)。
2. Sample 时执行 Hindsight 策略：有一定概率将 achieved_goal 替换为 desired_goal 并重算奖励。
"""

import numpy as np
import random

class HERReplayBuffer:
    def __init__(self, capacity, obs_dim, goal_dim, action_dim, reward_func):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim))
        self.next_obs = np.zeros((capacity, obs_dim))
        self.desired_goals = np.zeros((capacity, goal_dim))
        self.achieved_goals = np.zeros((capacity, goal_dim)) # s' 达成的状态
        self.actions = np.zeros((capacity, action_dim))
        
        self.ptr = 0
        self.size = 0
        self.reward_func = reward_func # 奖励函数引用

    def add(self, obs, action, next_obs, desired_goal, achieved_goal):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.desired_goals[self.ptr] = desired_goal
        self.achieved_goals[self.ptr] = achieved_goal
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, her_ratio=0.8):
        """HER 采样：以 her_ratio 的概率将 goal 替换为 achieved_goal"""
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        obs = self.obs[idxs]
        next_obs = self.next_obs[idxs]
        actions = self.actions[idxs]
        achieved_goals = self.achieved_goals[idxs]
        
        # 决定哪些样本使用原始目标，哪些使用 HER 目标（Future Strategy 的简化版：直接用 achieved_goal）
        # 标准 HER 的 "future" 策略是从同一轨迹的后续时间步选一个 goal，这里为了代码简洁
        # 以及适配非 trajectory-aware 的 buffer，我们简化为 "final" 策略（假设 input 的 achieved_goal 就是轨迹终点或未来某点）
        # 或者更简单的：直接用当前的 achieved_goal 作为目标 (相当于 success)
        
        her_indices = np.where(np.random.rand(batch_size) < her_ratio)[0]
        
        goals = self.desired_goals[idxs].copy()
        # 对 HER 样本，将目标替换为实际达成的目标
        goals[her_indices] = achieved_goals[her_indices]
        
        # 重新计算奖励
        # VecEnv 风格的计算
        rewards = np.array([self.reward_func(ag, g) for ag, g in zip(achieved_goals, goals)])
        dones = (rewards == 0.0).astype(np.float32) # 稀疏奖励：0 是成功，-1 是失败
        
        return {
            "obs": obs,
            "actions": actions,
            "next_obs": next_obs,
            "goals": goals,
            "rewards": rewards.reshape(-1, 1),
            "dones": dones.reshape(-1, 1)
        }
