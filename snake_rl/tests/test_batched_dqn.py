"""BatchedDQNAgent 单元测试。"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from snake_rl.agents.batched_dqn import BatchedDQNAgent
from snake_rl.agents.tensor_replay_buffer import BatchedTransition, TensorReplayBuffer
from snake_rl.config import DQNConfig, EnvConfig
from snake_rl.env.vector_env import VectorSnakeEnv


def test_batched_agent_select_actions() -> None:
    cfg = DQNConfig(
        obs_type="board",
        board_channels=5,
        board_height=10,
        board_width=10,
        adaptive_pool_size=(3, 3),
        seed=0,
    )
    agent = BatchedDQNAgent(cfg, device="cpu")

    # 模拟 8 个并行环境的状态
    obs = np.zeros((8, 5, 10, 10), dtype=np.uint8)
    actions = agent.select_actions(obs, epsilon=0.0)
    assert actions.shape == (8,)
    assert np.all((actions >= 0) & (actions < 3))

    # 带探索
    actions_eps = agent.select_actions(obs, epsilon=1.0)
    assert actions_eps.shape == (8,)


def test_batched_agent_train_step() -> None:
    cfg = DQNConfig(
        obs_type="board",
        board_channels=5,
        board_height=10,
        board_width=10,
        use_target_network=True,
        double_dqn=True,
        seed=0,
    )
    agent = BatchedDQNAgent(cfg, device="cpu")

    batch = BatchedTransition(
        obs=torch.zeros((16, 5, 10, 10), dtype=torch.float32),
        actions=torch.zeros((16,), dtype=torch.int64),
        rewards=torch.ones((16,), dtype=torch.float32),
        next_obs=torch.zeros((16, 5, 10, 10), dtype=torch.float32),
        terminated=torch.zeros((16,), dtype=torch.bool),
        truncated=torch.zeros((16,), dtype=torch.bool),
    )

    loss = agent.train_step(batch)
    assert math.isfinite(loss)
    assert agent.updates == 1


def test_batched_agent_save_and_load(tmp_path) -> None:
    cfg = DQNConfig(
        obs_type="board",
        board_channels=5,
        board_height=10,
        board_width=10,
        seed=42,
    )
    agent = BatchedDQNAgent(cfg, device="cpu")
    ckpt_path = tmp_path / "test_ckpt.pt"
    agent.save(ckpt_path)

    loaded = BatchedDQNAgent.load(ckpt_path, device="cpu")
    obs = torch.zeros((4, 5, 10, 10))
    # 验证权重一致
    q1 = agent.model(obs)
    q2 = loaded.model(obs)
    assert torch.allclose(q1, q2)


def test_batched_agent_adaptive_cross_dim() -> None:
    cfg = DQNConfig(
        obs_type="board",
        board_channels=5,
        adaptive_pool_size=(3, 3),
        seed=0,
    )
    agent = BatchedDQNAgent(cfg, device="cpu")

    # 可以同时处理 10x10 和 20x20
    a10 = agent.select_actions(np.zeros((4, 5, 10, 10), dtype=np.uint8))
    a20 = agent.select_actions(np.zeros((4, 5, 20, 20), dtype=np.uint8))

    assert a10.shape == (4,)
    assert a20.shape == (4,)
