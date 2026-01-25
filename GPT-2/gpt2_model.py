"""
业务问题：构建大语言模型（GPT-2）的核心神经网络结构，用于序列生成任务。
实现逻辑：
1. 词嵌入 (Embedding) + 位置嵌入 (Positional Encoding) -> 将 Token 转换为向量。
2. 堆叠 Transformer Decoder Block (12层) -> 学习文本上下文依赖。
3. 层归一化 (LayerNorm) + 线性层 (Linear) -> 输出下一个 Token 的概率分布 (Logits)。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Config:
    """GPT-2 模型配置参数"""
    def __init__(self, vocab_size=50257, n_embd=768, n_layer=12, n_head=12, n_ctx=1024):
        self.vocab_size = vocab_size  # 词表大小
        self.n_embd = n_embd          # 嵌入维度
        self.n_layer = n_layer        # Transformer 层数
        self.n_head = n_head          # 多头注意力头数
        self.n_ctx = n_ctx            # 上下文窗口大小 (最大序列长度)

class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制 (Causal Self-Attention)
    实现逻辑：Query, Key, Value 投影 -> 计算注意力分数 -> 掩码 (Mask) 屏蔽未来信息 -> Softmax -> 输出
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # 线性投影 Q, K, V (合并为一个大矩阵以提高效率)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # 注册因果掩码 (Lower Triangular Mask)，防止看到“未来”的词
        self.register_buffer("bias", torch.tril(torch.ones(config.n_ctx, config.n_ctx))
                                     .view(1, 1, config.n_ctx, config.n_ctx))

    def forward(self, x):
        B, T, C = x.size() # Batch, Time(seq_len), Channel(n_embd)
        
        # 1. 计算 Q, K, V
        # split 后维度: [B, T, n_head, head_size]
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 2. 计算注意力分数 (Scaled Dot-Product Attention)
        # att = (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 3. 应用因果掩码 (Causal Masking)
        # 将 upper triangular 部分 (future) 设为 -inf
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # 4. Softmax 归一化
        att = F.softmax(att, dim=-1)
        
        # 5. 加权求和并输出
        y = att @ v # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 恢复形状
        
        return self.c_proj(y)

class MLP(nn.Module):
    """
    位置前馈神经网络 (Position-wise Feed-Forward Network)
    实现逻辑：Linear -> GELU -> Linear
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):
    """
    Transformer 解码器块 (Decoder Block)
    实现逻辑：LayerNorm -> Self-Attention -> Residual -> LayerNorm -> MLP -> Residual
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    """
    GPT-2 完整模型
    实现逻辑：Embedding -> Blocks -> LayerNorm -> Logits
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Word Token Embedding
            wpe = nn.Embedding(config.n_ctx, config.n_embd),      # Word Position Embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # 12 Layers
            ln_f = nn.LayerNorm(config.n_embd),                   # Final LayerNorm
        ))
        
        # 语言模型头 (Language Modeling Head)
        # GPT-2 论文中权重绑定：wte 权重直接复用于输出层
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx, targets=None):
        """
        idx: [Batch, Seq_Len] 输入 Token IDs
        targets: [Batch, Seq_Len] 目标 Token IDs (可选)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.n_ctx, f"Cannot forward sequence of length {t}, block size is only {self.config.n_ctx}"
        
        # 1. 构建位置索引 [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)

        # 2. Token Embedding + Position Embedding
        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # (1, t, n_embd)
        x = tok_emb + pos_emb
        
        # 3. 通过所有 Transformer Blocks
        for block in self.transformer.h:
            x = block(x)
            
        # 4. 最终层归一化
        x = self.transformer.ln_f(x)
        
        # 5. 输出 Logits
        logits = self.lm_head(x) # (b, t, vocab_size)

        # 6. 计算 Loss (如果是训练模式)
        loss = None
        if targets is not None:
            # Flatten to [Batch*Seq_Len, Vocab] for CrossEntropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
