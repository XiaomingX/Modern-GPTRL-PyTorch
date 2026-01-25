"""
业务问题：实现 Weak-to-Strong Generalization (弱到强泛化) 智能体。
实现逻辑：
1. 弱模型（Weak Model）在带有噪声或简单的标签上训练。
2. 强模型（Strong Model）利用弱模型的预测作为“有偏差”的监督信号进行训练。
3. 核心挑战：强模型如何超越弱模型的限制（即泛化到弱模型不懂的区域）。
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel

class TransformerWithHead(PreTrainedModel):
    """带分类头的 Transformer 模型，用于二分类任务（如情感分析）"""
    def __init__(self, model_name: str, num_labels: int = 2, linear_probe: bool = False):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self.num_labels = num_labels
        self.linear_probe = linear_probe
        
        # 基础 Transformer
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.transformer = self.lm.transformer
        
        hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.02)
        
        # 冻结权重（如果是线性探针模式）
        if linear_probe:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor):
        # 获取最后一个 token 的隐藏状态
        outputs = self.transformer(input_ids)
        hidden_states = outputs[0]
        
        # 简单取均值或最后一个有效 token (这里取最后一个)
        # 实际实现中应处理 padding mask
        last_hidden = hidden_states[:, -1, :]
        
        return self.classifier(last_hidden)

def loss_w2s(logits, weak_labels, beta=1.0):
    """
    弱到强训练损失。
    weak_labels 是弱模型的预测概率。
    """
    # 交叉熵损失：强模型的预测 vs 弱模型的软标签
    return nn.functional.cross_entropy(logits, weak_labels)
