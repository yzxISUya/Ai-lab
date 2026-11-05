"""VectorSnakeEnv 的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from snake_rl.config import EnvConfig
from snake_rl.env.vector_env import VectorSnakeEnv


def test_vector_env_board_shapes() -> None:
    cfg = EnvConfig(width=10, height=8, initial_length=3)
    num_envs = 4
    venv = VectorSnakeEnv(cfg, num_envs=num_envs, obs_type="board", as_uint8=True)

    obs = venv.reset()
    assert obs.shape == (num_envs, 5, 8, 10)
    assert obs.dtype == np.uint8

    actions = np.zeros(num_envs, dtype=np.int64)
    next_obs, rewards, terms, truncs, infos = venv.step(actions)

    assert next_obs.shape == (num_envs, 5, 8, 10)
    assert next_obs.dtype == np.uint8
    assert rewards.shape == (num_envs,)
    assert terms.shape == (num_envs,)
    assert truncs.shape == (num_envs,)
    assert len(infos) == num_envs


def test_vector_env_features11_shapes() -> None:
    cfg = EnvConfig(width=10, height=10, initial_length=3)
    num_envs = 8
    venv = VectorSnakeEnv(cfg, num_envs=num_envs, obs_type="features11")

    obs = venv.reset()
    assert obs.shape == (num_envs, 11)
    assert obs.dtype == np.float32

    actions = [0] * num_envs
    next_obs, rewards, terms, truncs, infos = venv.step(actions)
    assert next_obs.shape == (num_envs, 11)
    assert rewards.shape == (num_envs,)


def test_vector_env_auto_reset() -> None:
    # 设定极短超时，强制触发 auto-reset
    cfg = EnvConfig(width=6, height=6, initial_length=3, max_steps_without_food_factor=2)
    venv = VectorSnakeEnv(cfg, num_envs=2, obs_type="board")
    venv.reset()

    has_done = False
    for _ in range(50):
        # 持续直行，必然撞墙或超时
        _, _, terms, truncs, infos = venv.step([0, 0])
        for i in range(2):
            if terms[i] or truncs[i]:
                has_done = True
                assert "episode_score" in infos[i]
                assert "episode_steps" in infos[i]
        if has_done:
            break

    assert has_done, "50 步内应该至少触发一次环境重置"
