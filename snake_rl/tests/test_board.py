"""全棋盘观察 + CNN Q 网络的测试。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from snake_rl.agents import Batch, DQNAgent, Transition
from snake_rl.config import DQNConfig, EnvConfig, TrainConfig
from snake_rl.env import SnakeEnv, observe_board
from snake_rl.models import BoardQNet


def make_env() -> SnakeEnv:
    return SnakeEnv(EnvConfig(width=10, height=8, initial_length=3), seed=0)


def test_board_observation_shape_and_channels() -> None:
    env = make_env()
    obs = observe_board(env)

    assert obs.shape == (5, 8, 10)
    assert obs.dtype == np.float32
    assert obs[0].sum() == len(env.snake)
    assert obs[1].sum() == 1
    assert obs[2].sum() == 1
    assert env.food is not None
    assert obs[3].sum() == 1
    assert 0.0 <= obs[4].min() <= obs[4].max() <= 1.0

    assert obs[1, env.head.y, env.head.x] == 1.0
    tail = env.snake[-1]
    assert obs[2, tail.y, tail.x] == 1.0
    assert obs[3, env.food.y, env.food.x] == 1.0


def test_board_observation_four_channels_and_invalid() -> None:
    env = make_env()
    obs = observe_board(env, channels=4)

    assert obs.shape == (4, 8, 10)
    with pytest.raises(ValueError):
        observe_board(env, channels=3)


def test_board_qnet_output_shape() -> None:
    net = BoardQNet(in_channels=5, height=8, width=10, hidden_size=16, output_size=3)
    out = net(torch.zeros(4, 5, 8, 10))
    assert out.shape == (4, 3)


def test_agent_board_observation_forward_and_update() -> None:
    config = DQNConfig(
        obs_type="board",
        board_channels=5,
        board_height=8,
        board_width=10,
        hidden_size=16,
        seed=0,
    )
    agent = DQNAgent(config)
    env = make_env()

    obs = agent.observation(env)
    assert obs.shape == (5, 8, 10)
    assert agent.q_values(obs).shape == (3,)

    batch = Batch.from_transitions(
        [
            Transition(
                obs=obs, action=0, reward=0.0, next_obs=obs,
                terminated=False, truncated=False,
            ),
            Transition(
                obs=obs, action=1, reward=1.0, next_obs=obs,
                terminated=True, truncated=False,
            ),
        ]
    )
    loss = agent.train_step(batch)
    assert math.isfinite(loss)


def test_agent_board_four_channels() -> None:
    config = DQNConfig(
        obs_type="board",
        board_channels=4,
        board_height=8,
        board_width=10,
        hidden_size=16,
        seed=0,
    )
    agent = DQNAgent(config)
    obs = agent.observation(make_env())

    assert obs.shape == (4, 8, 10)
    assert agent.q_values(obs).shape == (3,)


def test_agent_board_checkpoint_roundtrip(tmp_path) -> None:
    config = DQNConfig(
        obs_type="board",
        board_channels=5,
        board_height=8,
        board_width=10,
        hidden_size=16,
        seed=0,
    )
    agent = DQNAgent(config)
    obs = agent.observation(make_env())
    q_before = agent.q_values(obs)

    path = tmp_path / "board.pt"
    agent.save(path)
    loaded = DQNAgent.load(path, device="cpu")

    assert loaded.config.obs_type == "board"
    assert loaded.config.board_channels == 5
    np.testing.assert_allclose(loaded.q_values(obs), q_before, rtol=1e-6, atol=1e-6)


def test_invalid_observation_config() -> None:
    with pytest.raises(ValueError):
        DQNConfig(obs_type="pixels")
    with pytest.raises(ValueError):
        DQNConfig(obs_type="board", board_channels=3)


def test_train_config_replay_validation() -> None:
    with pytest.raises(ValueError):
        TrainConfig(replay_update_every=-1)
    with pytest.raises(ValueError):
        TrainConfig(min_replay_size=-1)
