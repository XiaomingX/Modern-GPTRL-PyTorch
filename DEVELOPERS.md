# Developer Guide

Welcome to the **GPT-2-rl-fundamental-Chinese** developer guide. This repository has been modernized to provide a clean, modular foundation for GPT and RL research.

## Environment Setup

We use `uv` for lightning-fast dependency management.

### Installation
```bash
# Install uv if you haven't (https://github.com/astral-sh/uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv pip install -r requirements.txt
```

## Project Structure

The repository is organized by algorithm families:

- `GPT-2/`: Core Transformer architecture and BPE logic.
- `ppo/`, `a2c/`, `acktr/`, `trpo/`: Policy Gradient family.
- `ddpg/`, `deepq/`, `q-learning/`: Value-based and Continuous control.
- `muzero/`: Model-based RL (Dynamics, MCTS).
- `grpo/`, `weak-to-strong/`: Alignment methods.
- `unlabel/minimind/`: Mini GPT for experimentation.

## Coding Standards

1.  **PyTorch First**: All new implementations must use PyTorch.
2.  **Gymnasium Integration**: Environments should follow the `gymnasium.Env` interface.
3.  **Self-Explaining Code**: Use Chinese comments to explain mathematical logic within the code.
4.  **No `allinone.py`**: Monolithic scripts are deprecated. Please split code into `model.py`, `agent.py`, and `train.py`.

## Running Experiments

Each directory contains its own `train_*.py` script.
```bash
# Example: Training TRPO
uv run trpo/train_trpo.py
```

## Testing

Before submitting a PR, ensure your script runs without crashing for at least a few episodes:
```bash
uv run <path_to_train_script>
```

---
For algorithm principles, please refer to the `README.md` within each sub-directory.
