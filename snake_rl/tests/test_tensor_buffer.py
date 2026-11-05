"""TensorReplayBuffer 的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snake_rl.agents.tensor_replay_buffer import TensorReplayBuffer


def test_tensor_buffer_add_and_sample() -> None:
    capacity = 100
    obs_shape = (5, 8, 10)
    buffer = TensorReplayBuffer(capacity=capacity, obs_shape=obs_shape, as_uint8=True)

    batch_size = 16
    obs = np.ones((batch_size, *obs_shape), dtype=np.uint8) * 255
    next_obs = np.zeros((batch_size, *obs_shape), dtype=np.uint8)
    actions = np.zeros(batch_size, dtype=np.int64)
    rewards = np.ones(batch_size, dtype=np.float32)
    terms = np.zeros(batch_size, dtype=bool)
    truncs = np.zeros(batch_size, dtype=bool)

    buffer.add_batch(obs, actions, rewards, next_obs, terms, truncs)
    assert len(buffer) == 16
    assert buffer.ptr == 16

    batch = buffer.sample(8)
    assert len(batch) == 8
    assert batch.obs.shape == (8, *obs_shape)
    assert batch.obs.dtype == torch.float32
    # uint8 255 应该归一化为 1.0
    assert torch.allclose(batch.obs, torch.tensor(1.0))
    assert torch.allclose(batch.next_obs, torch.tensor(0.0))
    assert batch.rewards.shape == (8,)
    assert batch.actions.shape == (8,)


def test_tensor_buffer_circular_wrap() -> None:
    capacity = 20
    obs_shape = (2,)
    buffer = TensorReplayBuffer(capacity=capacity, obs_shape=obs_shape, as_uint8=False)

    # 第一次写入 15 条
    buffer.add_batch(
        obs=np.ones((15, 2)),
        actions=np.zeros(15),
        rewards=np.zeros(15),
        next_obs=np.ones((15, 2)),
        terminated=np.zeros(15),
        truncated=np.zeros(15),
    )
    assert len(buffer) == 15
    assert buffer.ptr == 15

    # 第二次写入 10 条（应该跨越边界并覆盖头部）
    buffer.add_batch(
        obs=np.full((10, 2), 2.0),
        actions=np.ones(10),
        rewards=np.ones(10),
        next_obs=np.full((10, 2), 2.0),
        terminated=np.ones(10),
        truncated=np.zeros(10),
    )
    assert len(buffer) == 20
    assert buffer.ptr == 5  # (15 + 10) % 20 = 5

    # 头部 5 条应该是第二次写入的数据 (value = 2.0)
    assert torch.allclose(buffer.obs[:5], torch.tensor(2.0))
    # 索引 5 到 14 依然是第一次写入的数据 (value = 1.0)
    assert torch.allclose(buffer.obs[5:15], torch.tensor(1.0))
    # 索引 15 到 19 也是第二次写入的数据 (value = 2.0)
    assert torch.allclose(buffer.obs[15:], torch.tensor(2.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_tensor_buffer_cuda() -> None:
    buffer = TensorReplayBuffer(capacity=50, obs_shape=(5, 10, 10), device="cuda")
    assert buffer.obs.device.type == "cuda"

    buffer.add_batch(
        obs=np.zeros((10, 5, 10, 10), dtype=np.uint8),
        actions=np.zeros(10),
        rewards=np.zeros(10),
        next_obs=np.zeros((10, 5, 10, 10), dtype=np.uint8),
        terminated=np.zeros(10),
        truncated=np.zeros(10),
    )
    batch = buffer.sample(4)
    assert batch.obs.device.type == "cuda"
