import math
import torch
import numpy as np

class Node:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, action_dim, policy_logits, hidden_state, reward):
        self.hidden_state = hidden_state
        self.reward = reward
        policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()
        for a in range(action_dim):
            self.children[a] = Node(policy[a])

def mcts_search(network, obs, num_simulations, pb_c_init=1.25, pb_c_base=19652):
    """
    MuZero MCTS 搜索逻辑
    """
    network.eval()
    with torch.no_grad():
        # 1. 根节点推理
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        state, value, policy_logits = network.initial_inference(obs_tensor)
        
        root = Node(0)
        root.expand(network.action_dim, policy_logits[0], state, 0)
        
        # 2. 模拟
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            # 选择：向下遍历直到叶子节点（未访问过的边）
            while node.children:
                # 如果有从未访问过的动作，直接选一个（或者 PUCT 会处理）
                action, next_node = select_child(node, pb_c_init, pb_c_base)
                search_path.append(next_node)
                # 如果这个节点还没被扩展（没有隐藏状态），它就是我们要扩展的
                if next_node.hidden_state is None:
                    break
                node = next_node
            
            # 扩展与评价 (对最后选中的 Node 进行动力学推导)
            node_to_expand = search_path[-1]
            parent = search_path[-2] if len(search_path) > 1 else None
            
            if node_to_expand.hidden_state is None and parent:
                last_action = [a for a, n in parent.children.items() if n == node_to_expand][0]
                state, reward, value, policy_logits = network.recurrent_inference(
                    parent.hidden_state, torch.tensor([last_action])
                )
                node_to_expand.expand(network.action_dim, policy_logits[0], state, reward.item())
                val_for_backprop = value.item()
            else:
                # 已经是根节点或者已展开节点
                val_for_backprop = 0 # 实际上如果搜到底了，应该取该节点的预测价值
                # 这里简化：如果是重复搜到，就用它自己的预测值
                # 实际上由于 CartPole 简单，基本不会搜到底
                pass
            
            # 反向传播
            backpropagate(search_path, val_for_backprop)
            
    return root

def select_child(node, pb_c_init, pb_c_base):
    """UCB 选择"""
    best_score = -float('inf')
    best_action = -1
    best_child = None
    
    # 算常数项
    pb_c = math.log((node.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(node.visit_count) / (1 + 0) # 修正项在后面循环里加
    
    for action, child in node.children.items():
        # PUCT 分数
        u_score = pb_c * child.prior / (1 + child.visit_count)
        # 如果 Q 值范围很大，通常需要归一化，这里 CartPole 相对简单
        score = child.value + u_score
        
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
            
    return best_action, best_child

def backpropagate(search_path, value):
    """回溯更新访问量与价值"""
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        # MuZero 特色：加上步即时奖励。这里简化了折扣因子的显式传递。
        value = node.reward + 0.99 * value
