# GPT-2 (PyTorch 实现)

本项目提供了一个极简但功能完备的 **GPT-2** PyTorch 实现。不仅包含模型定义，还重构了训练和推理流程，使其更易于理解和修改。

## 目录结构

| 文件 | 说明 |
| :--- | :--- |
| `gpt2_model.py` | **核心模型**：包含 Embeddings, Transformer Block, Attention, MLP 的完整定义。 |
| `train_gpt2.py` | **训练脚本**：包含 Dataset 构建、DataLoader 和主训练循环。 |
| `generate_gpt2.py` | **推理脚本**：基于自回归（Autoregressive）和 Top-K 采样的文本生成。 |

## 核心算法原理

### 1. 模型架构 (Model Architecture)
GPT-2 是一种**自回归语言模型 (Autoregressive Language Model)**，其核心由堆叠的 Transformer **解码器块 (Decoder Blocks)** 组成。

#### 关键组件：
*   **因果自注意力 (Causal/Masked Self-Attention)**:
    在计算注意力时，我们引入了一个**下三角掩码 (Lower Triangular Mask)**。这确保了位置 $t$ 的 token 只能“看到”位置 $0$ 到 $t$ 的信息，而无法看到未来的信息（$t+1$ 及以后）。这是 GPT 生成文本的基础。
    *代码对应*: `gpt2_model.py` 中的 `CausalSelfAttention` 类。

*   **位置编码 (Positional Embeddings)**:
    Transformer 本身不具备处理序列顺序的能力。GPT-2 使用可学习的位置嵌入 (`wpe`)，将其与词嵌入 (`wte`) 相加，赋予模型位置感知能力。

### 2. 生成策略 (Generation Strategy)
在推理阶段，GPT-2 逐个生成 Token。
*   **Top-K 采样**: 为了避免生成的文本过于单调（总是选概率最大的）或过于离谱（选了极低概率的词），我们只从概率最高的 $K$ 个词中按概率采样。这在保持连贯性的同时增加了多样性。
    *代码对应*: `generate_gpt2.py` 中的 `generate` 函数。

## 使用方法

### 1. 快速生成文本
```bash
python generate_gpt2.py
```
*输出示例*:
```text
Generating...
Output IDs: [66, 12, 856, ...]
```

### 2. 训练模型
```bash
python train_gpt2.py
```
*注意*: 默认使用虚拟数据进行演示。若需训练真实模型，请修改 `DummyDataset` 为真实的文本加载器。