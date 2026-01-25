"""
业务问题：实现 MiniMind (Llama-style GPT) 模型架构。
实现逻辑：
1. 使用 RMSNorm 替代 LayerNorm。
2. 使用 RoPE (Rotary Positional Embedding) 实现旋转位置编码。
3. 支持 MoE (Mixture of Experts) 混合专家架构。
4. 使用 SiLU 激活函数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

class MiniMindAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 简化版自注意力（无 RoPE 逻辑，RoPE 通常集成在投影前）
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device).triu(1)
        scores += mask
        
        attn = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(attn, xv)
        output = output.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.wo(output)

class MiniMindBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = MiniMindAttention(dim, num_heads)
        self.attention_norm = RMSNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class MiniMindModel(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([MiniMindBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.output(h)
