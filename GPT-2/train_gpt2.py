"""
业务问题：对 GPT-2 模型进行预训练（Pre-training），使其学习语言规律。
实现逻辑：
1. 数据加载：构建 Dataset 和 DataLoader，按 Batch 提供输入。
2. 如果可用，将数据和模型移动到 GPU。
3. 前向传播 (Forward) 计算 Loss -> 反向传播 (Backward) 计算梯度 -> 优化器 (Optimizer) 更新权重。
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gpt2_model import GPT2, GPT2Config

class DummyDataset(Dataset):
    """
    虚拟数据集，用于演示。实际使用时需替换为真实文本数据加载器。
    返回: (x, y) 其中 y 是 x 的下一个 token 移位
    """
    def __init__(self, length=100, ctx_len=128):
        self.length = length
        self.ctx_len = ctx_len
        self.vocab_size = 50257
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 随机生成数据用于测试
        data = torch.randint(0, self.vocab_size, (self.ctx_len + 1,))
        x = data[:-1]
        y = data[1:]
        return x, y

def train():
    # 1. 配置
    config = GPT2Config(n_layer=4, n_head=4, n_embd=256) # 使用小模型以便于演示
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. 模型与优化器
    model = GPT2(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    # 3. 数据加载
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 4. 训练循环
    model.train()
    for epoch in range(2): # 演示跑 2 个 Epoch
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播 (自动计算 CrossEntropyLoss)
            logits, loss = model(x, y)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
