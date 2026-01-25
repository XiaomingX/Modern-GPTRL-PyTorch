"""
业务问题：提供 MiniMind 模型的训练脚本示例。
实现逻辑：
1. 初始化模型和优化器。
2. 运行简单的训练步骤（预测下一个 token）。
"""

import torch
from minimind_model import MiniMindModel

def train_minimind():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Start MiniMind Training Example on {device}...")
    
    # 1. 初始化小规模模型用于演示
    model = MiniMindModel(
        vocab_size=6400,
        dim=256,
        num_heads=8,
        num_layers=4
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 模拟数据：[batch, seq]
    input_tokens = torch.randint(0, 6400, (4, 32)).to(device)
    target_tokens = torch.randint(0, 6400, (4, 32)).to(device)
    
    # 2. 训练步
    model.train()
    logits = model(input_tokens)
    
    # 计算 Loss: [batch * seq, vocab] vs [batch * seq]
    loss = criterion(logits.view(-1, 6400), target_tokens.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Initial MiniMind Loss: {loss.item():.4f}")
    print("Training Step Completed!")

if __name__ == "__main__":
    train_minimind()
