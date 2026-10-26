# snake_rl

用贪吃蛇做的强化学习练习场：先把环境和测量工具做对，再逐步引入、定位并修正 DQN。

## 环境

- WSL2 / Ubuntu 24.04
- Python 3.12（见 `.python-version`）
- 包管理：uv

```bash
cd snake_rl
uv sync        # 创建 .venv 并安装依赖；torch 为 CPU 版
```

## 常用命令

```bash
# 测试
uv run pytest

# 训练（默认 500 局，输出到 runs/stage1_seed0）
uv run python scripts/train_dqn.py

# 冒烟测试：几秒钟跑完
uv run python scripts/train_dqn.py --episodes 5 --eval-every 0 \
    --checkpoint-every 0 --final-eval-episodes 2 --output-dir /tmp/snake_smoke

# 用 greedy 策略评估 checkpoint
uv run python scripts/evaluate.py --run-dir runs/stage1_seed0 --episodes 100

# 画训练曲线
uv run python scripts/plot_training.py --run-dir runs/stage1_seed0

# 随机基线（验证环境）
uv run python scripts/random_agent.py --episodes 100

# 人类试玩（方向键 / WASD）
uv run python scripts/play_human.py
```

`--device` 可选 `cpu` / `cuda` / `auto`。默认 `cpu`：网络只有
`11 → 256 → 3`，每步做一次 batch=1 更新，实测 GPU 反而慢约 5 倍。
详见 [`pyproject.toml`](pyproject.toml) 里的注释。

### WSL 下的线程数（重要）

WSL 里 PyTorch 默认吃满所有 CPU 线程，对这么小的网络是负优化。同一台机器、
同样 1587 步，实测：

| 设置                | 步/秒   |
| ------------------- | ------- |
| 默认（12 线程）     | 71      |
| `OMP_NUM_THREADS=4` | **739** |

跑训练前先限制线程数：

```bash
OMP_NUM_THREADS=4 uv run python scripts/train_dqn.py
```

## 目录结构

```
snake_rl/
├── snake_rl/          # 包
│   ├── config.py      # EnvConfig / DQNConfig / TrainConfig
│   ├── env/           # 纯逻辑环境、11 维观察、pygame 渲染
│   ├── models/        # QNet
│   ├── agents/        # DQNAgent + ReplayBuffer
│   └── training/      # 训练循环、评估、CSV 日志
├── scripts/           # 命令行入口
├── tests/             # pytest
├── teach/             # 本地课程与实验记录（未入库）
└── runs/              # 训练产物（未入库）
```

## 阶段进度

| 阶段    | 主题                                           | 状态   |
| ------- | ---------------------------------------------- | ------ |
| Stage 0 | 环境、渲染、随机基线                           | 完成   |
| Stage 1 | 复刻 DQN 基线                                  | 完成   |
| Stage 2 | 目标网络 / Double DQN / Huber / 截断 bootstrap | 下一步 |
| Stage 3 | 观察与奖励工程                                 | 待做   |
| Stage 4 | PPO 对比                                       | 待做   |

Stage 1 最终结果（100 局 greedy eval，训练/评估种子分离）：
**mean 27.14 / median 26 / max 52 / std 10.13**，95/100 死于 self collision。

## 依赖与产物

- 依赖的唯一真源是 [`pyproject.toml`](pyproject.toml) + `uv.lock`；
  需要 `requirements.txt` 时用 `uv export --no-dev > requirements.txt` 生成。
- torch 从 `pytorch-cpu` 索引安装（约 200MB）。要换 CUDA 版：删掉
  `pyproject.toml` 里的 `[[tool.uv.index]]` 和 `[tool.uv.sources]`，
  然后 `uv lock && uv sync`。
- 训练产物在 `runs/<run_name>/`：`config.json`、`train.csv`、`eval.csv`、
  `checkpoint.pt`、`final_eval.json`、`summary.json`、`curve.png`。
