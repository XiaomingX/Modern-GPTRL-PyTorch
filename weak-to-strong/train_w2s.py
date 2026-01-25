"""
业务问题：演示 Weak-to-Strong Generalization 训练流程。
实现逻辑：
1. 弱模型：小型模型（如 GPT-2 Small）在子集数据上训练。
2. 强模型：大型模型（如 GPT-2 Medium/Large）在弱模型的标注下训练。
3. 展示强模型如何通过特定的 Loss 函数超越弱模型。
"""

import torch
from w2s_agent import TransformerWithHead, loss_w2s
from transformers import AutoTokenizer

def train_w2s():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Start Weak-to-Strong Training on {device}...")
    
    # 1. 初始化模型
    # 这里演示使用同一个基础模型，实际中弱模型和强模型规模不一
    weak_model = TransformerWithHead("gpt2", linear_probe=True).to(device)
    strong_model = TransformerWithHead("gpt2", linear_probe=False).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 模拟数据
    texts = ["This movie is great!", "I hated this film."]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    
    # 2. 弱模型生成“虚假”标签 (Teacher labels)
    with torch.no_grad():
        weak_logits = weak_model(inputs["input_ids"])
        weak_probs = torch.softmax(weak_logits, dim=-1)
        
    # 3. 强模型学习
    optimizer = torch.optim.Adam(strong_model.parameters(), lr=1e-5)
    
    strong_logits = strong_model(inputs["input_ids"])
    loss = loss_w2s(strong_logits, weak_probs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Initial W2S Loss: {loss.item():.4f}")
    print("Training Step Completed!")

if __name__ == "__main__":
    train_w2s()
