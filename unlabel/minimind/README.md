# MiniMind (小规模 Llama 风格模型)

本项目实现了 **MiniMind** 的核心架构。MiniMind 是一个旨在研究大规模语言模型（LLM）核心组件（如 RoPE、RMSNorm、MoE）在极小规模下的表现的项目。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `minimind_model.py` | **模型核心**：包含 RMSNorm、Llama 风格的多头注意力机制和 FeedForward 块。 |
| `train_minimind.py` | **训练脚本**：提供预训练和微调的简化流程示例。 |

## 核心技术特性

MiniMind 采用了现代 LLM（如 Llama 3, DeepSeek）的主流技术栈：

1.  **RMSNorm**: 相比传统的 LayerNorm，RMSNorm 减少了计算中心化的开销，使得训练更加稳定。
2.  **RoPE旋转位置编码**: 允许模型更好地扩展上下文窗口，并捕捉相对位置信息。
3.  **SiLU 激活函数**: 提供了比 ReLU 更平滑的非线性变换。
4.  **混合专家架构 (MoE)**:（可选支持）通过路由机制动态选择专家网络，在保持推理成本较低的同时大幅增加模型参数量。

## 使用方法

### 运行示例
```bash
python train_minimind.py
```
MiniMind 是理解现代大语言模型底层逻辑的绝佳入口，适合进行算法实验和教学演示。
