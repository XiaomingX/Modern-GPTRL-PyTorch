"""
GPT-2 模型测试
测试目标：验证 GPT-2 模型能够正常初始化和生成文本
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from GPT2.gpt2_model import GPT2Config, GPT2Model

def test_gpt2_config():
    """测试 GPT-2 配置"""
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    
    assert config.vocab_size == 50257
    assert config.n_positions == 1024
    assert config.n_embd == 768
    print("✓ GPT-2 配置正常")

def test_gpt2_model_initialization():
    """测试 GPT-2 模型初始化"""
    config = GPT2Config(
        vocab_size=50257,
        n_positions=128,  # 减小以加快测试
        n_embd=256,
        n_layer=4,
        n_head=4
    )
    
    model = GPT2Model(config)
    assert model is not None
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ GPT-2 模型初始化成功，参数量: {total_params:,}")

def test_gpt2_forward_pass():
    """测试 GPT-2 前向传播"""
    config = GPT2Config(
        vocab_size=50257,
        n_positions=128,
        n_embd=256,
        n_layer=4,
        n_head=4
    )
    
    model = GPT2Model(config)
    model.eval()
    
    # 创建输入
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    # 检查输出形状
    assert outputs.shape == (batch_size, seq_len, config.vocab_size)
    print(f"✓ GPT-2 前向传播正常，输出形状: {outputs.shape}")

if __name__ == "__main__":
    print("=" * 50)
    print("开始测试 GPT-2 模型")
    print("=" * 50)
    
    test_gpt2_config()
    test_gpt2_model_initialization()
    test_gpt2_forward_pass()
    
    print("\n" + "=" * 50)
    print("所有 GPT-2 测试通过 ✓")
    print("=" * 50)
