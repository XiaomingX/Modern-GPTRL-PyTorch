# GPT-2 & RL Architecture Deep Dive (Chinese Edition)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 项目使命：弥合算法理论与工程实践的鸿沟

本项目是一个专为中文开发者设计的**深度学习与强化学习算法全栈实验室**。我们通过对 **GPT-2**、**RLHF**、**MuZero** 以及 **Alignment (GRPO, Weak-to-Strong)** 等前沿算法的现代化 PyTorch 重构，旨在提供一个“所见即所得”的学习与研究基准。

### 核心差异化价值
- **全栈重构**: 彻底告别不再维护的 TensorFlow 1.x / JAX 遗留代码，全面拥抱 **PyTorch 2.x** 生态。
- **理论实战闭环**: 每一行核心逻辑都配有详尽的**中文注释**，直接对应论文中的数学公式。
- **对齐技术前瞻**: 率先集成了 **GRPO (DeepSeek)**、**Weak-to-Strong (OpenAI)** 等 LLM 对齐关键算法。
- **现代化基础设施**: 采用 `uv` 进行闪电级的包管理，基于 `Gymnasium` 的标准化 RL 环境接口。

---

## 🗺️ 算法全景图

### 1. 语言模型基石 (LLM Foundations)
| 模块 | 核心技术 | 路径 |
| :--- | :--- | :--- |
| **GPT-2** | BPE 分词、Transformer Decoder、中文优化 | [`./GPT-2`](./GPT-2) |
| **MiniMind** | **MoE (混合专家)**、RoPE、RMSNorm、Llama 风格架构 | [`./unlabel/minimind`](./unlabel/minimind) |

### 2. 强化学习演进 (RL Evolution)
- **策略梯度系**: [PPO (Clip版)](./ppo) · [TRPO (信任区域)](./trpo) · [A2C (优势演员)](./a2c)
- **值函数系**: [DQN (Target网络)](./deepq) · [Q-Learning](./q-learning)
- **连续控制系**: [DDPG](./ddpg) · [TD3 (双延迟)](./ddpg)
- **高阶RL研究**:
    - **[MuZero](./muzero)**: 结合隐空间动力学模型与 MCTS 的模型驱动 RL。
    - **[ACKTR](./acktr)**: 基于 **K-FAC** 的自然梯度二阶优化。
    - **[MA-DDPG](./maddpg)**: 多智能体协作与对抗博弈。
    - **[HER](./her)**: 针对稀疏奖励的后验经验回放。

### 3. 模型对齐与元学习 (Alignment & Meta)
- **[GRPO](./grpo)**: DeepSeek 提出的组相对策略优化，彻底移除 Critic 依赖。
- **[Weak-to-Strong](./weak-to-strong)**: OpenAI 弱监督强模型的超级对齐（Superalignment）实验。
- **[Meta-Learning](./learning-to-learn)**: LSTM 元优化器，实现“用梯度下降学习梯度下降”。
- **[SAM](./sam)**: 锐度感知最小化（提高泛化性） & Segment Anything 视觉大模型。

---

## ⚡ 快速开始

### 现代化依赖管理 (推荐)
本项目强力推荐使用 [uv](https://github.com/astral-sh/uv) 进行高效的虚拟环境管理。

```bash
# 1. 安装依赖
uv pip install -r requirements.txt

# 2. 运行算法驱动示例 (以 PPO 为例)
uv run ppo/train_ppo.py
```

### Docker 容器化方案
我们也提供了预配置的 Docker 镜像，支持 CUDA 12.1 加速。
```bash
docker build -t gpt2-rl -f Dockerfile.gpu .
docker run --gpus all -it gpt2-rl
```

---

## 🛠️ 设计哲学
1. **模块化与独立性**: 每个算法目录都是一个独立的闭环，减少跨目录依赖，方便拆解学习。
2. **拒绝“黑盒”**: 核心公式计算处禁止使用过度抽象的第三方库，确保逻辑透明。
3. **中文语境优化**: 针对中文 GPT 语境优化了 Tokenizer 和样本数据集。

## 🤝 参与与支持
- **开发者文档**: 查看 [DEVELOPERS.md](./DEVELOPERS.md) 获取贡献指南。
- **寻求帮助**: 如果对算法推导有疑问，欢迎提交 Issue。
