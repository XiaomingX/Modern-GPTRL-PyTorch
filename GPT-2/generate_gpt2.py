"""
业务问题：利用预训练的 GPT-2 模型进行自回归文本生成。
实现逻辑：
1. 输入 Prompt -> Tokenize (可选，这里假设输入 ID) -> 模型 Forward。
2. 取最后一个 Token 的 Logits -> Top-K 采样 -> 得到下一个 Token。
3. 将新 Token 拼接到输入序列，重复步骤 1-2，直到达到最大长度。
"""

import torch
import torch.nn.functional as F
from gpt2_model import GPT2, GPT2Config

def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    GPT-2 文本生成函数
    :param model: GPT2 模型实例
    :param idx: 输入 Token IDs, 形状 (Batch, Seq_Len)
    :param max_new_tokens: 要生成的新 Token 数量
    :param temperature: 温度系数，控制生成随机性 (值越大越随机)
    :param top_k: Top-K 采样，保留概率最高的 K 个 Token
    """
    model.eval() # 切换到评估模式
    for _ in range(max_new_tokens):
        # 截断输入，防止超过最大上下文长度
        idx_cond = idx[:, -model.config.n_ctx:]
        
        # 1. 前向传播
        logits, _ = model(idx_cond)
        
        # 2. 取最后一个时间步的 Logits
        logits = logits[:, -1, :] / temperature
        
        # 3. Top-K 采样 (可选)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # 4. 计算概率并通过多项式分布采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 5. 拼接新生成的 Token
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

if __name__ == "__main__":
    # 简单测试代码
    config = GPT2Config()
    model = GPT2(config)
    
    # 模拟输入：BatchSize=1, 初始 Token=[66]
    start_ids = torch.tensor([[66]], dtype=torch.long)
    
    print("Generating...")
    generated_ids = generate(model, start_ids, max_new_tokens=20, top_k=10)
    print("Output IDs:", generated_ids.tolist())
