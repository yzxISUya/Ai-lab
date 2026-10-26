一个边做边学的 AI 练习大杂烩垃圾桶。每个子目录是一个**独立、自包含**的弱智项目：
每个垃圾有自己的工作目录、自己的 `pyproject.toml`、自己的 `.venv`。

## 子项目

| 目录                     | 内容                                                             | 技术栈                          |
| ------------------------ | ---------------------------------------------------------------- | ------------------------------- |
| [`nano-gpt/`](nano-gpt/) | 在 TinyStories 上从零训练一个小 GPT，带采样、Gradio 界面和模型卡 | PyTorch · transformers · Gradio |
| [`snake_rl/`](snake_rl/) | 用贪吃蛇走一遍 RL 全流程：MDP 建模 → DQN 基线 → 消融 → PPO       | PyTorch · numpy · pygame        |

## 环境约定

- **系统**：WSL2 / Ubuntu 24.04
- **包管理**：[uv](https://docs.astral.sh/uv/)（两个子项目都用自己的 `.venv`，互不干扰）
- **Python**：由各项目的 `.python-version` 指定（nano-gpt 与 snake_rl 均为 3.12）

```bash
cd <子项目>
uv sync          # 创建 .venv 并安装依赖（含 dev 组）
uv run ...       # 在该项目环境里执行命令
```