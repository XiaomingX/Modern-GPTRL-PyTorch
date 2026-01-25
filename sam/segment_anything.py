"""
业务问题：模块化 Segment Anything Model (SAM)。
实现逻辑：
1. 本文件整合了原 allinone.py 中的 Vision 相关部分。
2. 包含图像编码器 (ViT)、提示编码器和掩码解码器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np

# --- 基础组件 ---

class MLPBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embed_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

# --- 核心网络 ---

class ImageEncoderViT(nn.Module):
    """图像编码器：使用 ViT 提取特征"""
    def __init__(
        self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3,
        embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
        mlp_ratio: float = 4.0, out_chans: int = 256
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, img_size//patch_size, img_size//patch_size))

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                nn.LayerNorm(embed_dim),
                MLPBlock(embed_dim, int(embed_dim * mlp_ratio))
            ) for _ in range(depth)
        ])

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            LayerNorm2d(out_chans)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        for block in self.blocks:
            norm_x = block[0](x)
            attn_out, _ = block[1](norm_x, norm_x, norm_x)
            x = x + attn_out
            x = x + block[3](block[2](x))
        return self.neck(x.reshape(B, H, W, C).permute(0, 3, 1, 2))

class Sam(nn.Module):
    """Segment Anything Model 顶层封装"""
    def __init__(self, image_encoder, prompt_encoder, mask_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def forward(self, image):
        # 简化版前向传播：仅提取图像特征
        return self.image_encoder(image)

def build_sam_vit_b():
    """工厂函数：构建 ViT-B 版本的 SAM"""
    encoder = ImageEncoderViT(embed_dim=768, depth=12, num_heads=12)
    # 其它组件简化省略
    return Sam(encoder, None, None)
