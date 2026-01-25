"""
业务问题：构建一个简化的多智能体环境 (Simple Spread)，用于测试 MA-DDPG。
环境描述：
- N 个 Agents，N 个 Landmarks (地标)。
- 目标：Agents 协作覆盖所有 Landmarks，且避免碰撞。
- 观测：自身位置/速度 + 其他 Agent 相对位置 + Landmark 相对位置。
- 奖励：全局奖励 (覆盖距离 + 碰撞惩罚)。
"""

import numpy as np

class SimpleMPE:
    def __init__(self, n_agents=3):
        self.n_agents = n_agents
        self.n_landmarks = n_agents
        self.world_size = 2.0 # -1 to 1 generally
        # 地标位置
        self.landmarks = np.zeros((self.n_landmarks, 2))
        # Agent 状态: p_pos (2), p_vel (2)
        self.agent_pos = np.zeros((self.n_agents, 2))
        self.agent_vel = np.zeros((self.n_agents, 2))
        
        self.dt = 0.1
        self.damping = 0.25 # 阻力
        
        # 观测空间维度: 自身(4) + landmarks(2*N) + 其他agents(2*(N-1))
        # 这里简化：相对位置
        self.obs_dim = 4 + 2 * self.n_landmarks + 2 * (self.n_agents - 1)
        # 动作空间维度: 连续 (vel_x, vel_y)
        self.act_dim = 2

    def reset(self):
        # 随机初始化位置
        self.landmarks = np.random.uniform(-1, 1, size=(self.n_landmarks, 2))
        self.agent_pos = np.random.uniform(-1, 1, size=(self.n_agents, 2))
        self.agent_vel = np.zeros((self.n_agents, 2))
        return self._get_obs()

    def _get_obs(self):
        obs_n = []
        for i in range(self.n_agents):
            obs = []
            # 1. 自身状态
            obs.append(self.agent_pos[i])
            obs.append(self.agent_vel[i])
            
            # 2. 地标相对位置
            for j in range(self.n_landmarks):
                obs.append(self.landmarks[j] - self.agent_pos[i])
                
            # 3. 其他 Agent 相对位置
            for j in range(self.n_agents):
                if i != j:
                    obs.append(self.agent_pos[j] - self.agent_pos[i])
            
            obs_n.append(np.concatenate(obs))
        return np.array(obs_n)

    def step(self, actions_n): # actions_n: list of arrays
        actions_n = np.array(actions_n)
        # 施加阻力
        self.agent_vel *= (1 - self.damping)
        # 施加动作 (加速度)
        self.agent_vel += actions_n * self.dt
        
        # 更新位置
        self.agent_pos += self.agent_vel * self.dt
        
        # 边界约束 (简单的反弹或限制)
        self.agent_pos = np.clip(self.agent_pos, -self.world_size, self.world_size)
        
        # 计算全局奖励
        # 1. 覆盖距离 (最小化 landmark 到最近 agent 的距离)
        dists = []
        for l_pos in self.landmarks:
             # 每个 landmark 到所有 agents 的距离
            d = np.linalg.norm(self.agent_pos - l_pos, axis=1)
            dists.append(min(d))
        dist_reward = -np.mean(dists)
        
        # 2. 碰撞惩罚
        collision_reward = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.agent_pos[i] - self.agent_pos[j]) < 0.1: # 半径 0.05
                    collision_reward -= 1.0
                    
        reward = dist_reward + collision_reward
        rewards_n = [reward] * self.n_agents # 合作型任务，奖励共享
        
        done = False # 无终止，依靠 max_steps 截断
        
        return self._get_obs(), rewards_n, [done]*self.n_agents, {}
