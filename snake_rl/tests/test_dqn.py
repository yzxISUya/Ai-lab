"""Stage 1 DQN 组件测试：观察之外的张量/目标/回放/checkpoint。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from snake_rl.agents import Batch, DQNAgent, ReplayBuffer, Transition
from snake_rl.config import DQNConfig
from snake_rl.models import QNet


def transition(
    *,
    action: int = 0,
    reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
) -> Transition:
    return Transition(
        obs=np.zeros(11, dtype=np.float32),
        action=action,
        reward=reward,
        next_obs=np.ones(11, dtype=np.float32),
        terminated=terminated,
        truncated=truncated,
    )


class FixedQNet(nn.Module):
    """忽略输入，永远输出固定 Q 值；用于测试 target 计算。"""

    def __init__(self, values: list[list[float]]):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values.expand(x.shape[0], -1).clone()


# ----------------------------------------------------------------------
# 网络与 batch
# ----------------------------------------------------------------------
def test_qnet_output_shape() -> None:
    net = QNet(input_size=11, hidden_size=16, output_size=3)
    out = net(torch.zeros(4, 11))
    assert out.shape == (4, 3)


def test_batch_from_transitions_shapes() -> None:
    batch = Batch.from_transitions(
        [
            transition(action=0, reward=1.0, terminated=True),
            transition(action=2, reward=-1.0, truncated=True),
        ]
    )

    assert len(batch) == 2
    assert batch.obs.shape == (2, 11)
    assert batch.actions.shape == (2,)
    assert batch.rewards.shape == (2,)
    assert batch.next_obs.shape == (2, 11)
    assert batch.terminated.dtype == np.bool_
    assert batch.truncated.dtype == np.bool_


def test_batch_from_empty_raises() -> None:
    with pytest.raises(ValueError):
        Batch.from_transitions([])


# ----------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------
def test_replay_buffer_capacity_and_sample_shapes() -> None:
    buffer = ReplayBuffer(capacity=2, seed=0)
    for i in range(3):
        buffer.add(transition(action=i % 3, reward=float(i)))

    assert len(buffer) == 2

    batch = buffer.sample(batch_size=10)
    assert len(batch) == 2  # 不足 batch_size 时返回全部
    assert batch.obs.shape == (2, 11)
    assert batch.actions.dtype == np.int64


def test_replay_buffer_sample_is_reproducible() -> None:
    buffer_a = ReplayBuffer(capacity=10, seed=42)
    buffer_b = ReplayBuffer(capacity=10, seed=42)
    for i in range(10):
        buffer_a.add(transition(action=i % 3, reward=float(i)))
        buffer_b.add(transition(action=i % 3, reward=float(i)))

    batch_a = buffer_a.sample(5)
    batch_b = buffer_b.sample(5)

    np.testing.assert_array_equal(batch_a.obs, batch_b.obs)
    np.testing.assert_array_equal(batch_a.actions, batch_b.actions)
    np.testing.assert_array_equal(batch_a.rewards, batch_b.rewards)


def test_replay_buffer_empty_sample_raises() -> None:
    buffer = ReplayBuffer(capacity=10, seed=0)
    with pytest.raises(ValueError):
        buffer.sample(4)


# ----------------------------------------------------------------------
# epsilon 与动作
# ----------------------------------------------------------------------
def test_epsilon_schedule() -> None:
    agent = DQNAgent(
        DQNConfig(epsilon_start=0.4, epsilon_end=0.0, epsilon_decay_games=80, seed=0)
    )

    assert agent.epsilon == pytest.approx(0.4)
    agent.n_games = 40
    assert agent.epsilon == pytest.approx(0.2)
    agent.n_games = 80
    assert agent.epsilon == pytest.approx(0.0)
    agent.n_games = 100
    assert agent.epsilon == pytest.approx(0.0)


def test_get_action_is_in_range_and_greedy_is_deterministic() -> None:
    agent = DQNAgent(DQNConfig(seed=0, epsilon_start=0.0, hidden_size=16))
    obs = np.zeros(11, dtype=np.float32)

    actions = [agent.get_action(obs, greedy=True) for _ in range(10)]
    assert all(0 <= a < 3 for a in actions)
    assert len(set(actions)) == 1


# ----------------------------------------------------------------------
# TD target
# ----------------------------------------------------------------------
def _target_batch() -> Batch:
    return Batch(
        obs=np.zeros((3, 11), dtype=np.float32),
        actions=np.array([0, 1, 2], dtype=np.int64),
        rewards=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        next_obs=np.zeros((3, 11), dtype=np.float32),
        terminated=np.array([True, False, False]),
        truncated=np.array([False, False, True]),
    )


def test_compute_targets_legacy_treats_truncation_as_terminal() -> None:
    agent = DQNAgent(DQNConfig(gamma=0.9, bootstrap_truncated=False, seed=0))
    agent.model = FixedQNet([[1.0, 2.0, 3.0]])
    batch = _target_batch()

    targets = agent.compute_targets(batch)

    # max Q(s') = 3；第 0 条 terminated，第 2 条 truncated（legacy 当终止）。
    expected = torch.tensor([1.0, 2.0 + 0.9 * 3.0, 3.0])
    assert torch.allclose(targets, expected)


def test_compute_targets_bootstraps_truncation_when_enabled() -> None:
    agent = DQNAgent(DQNConfig(gamma=0.9, bootstrap_truncated=True, seed=0))
    agent.model = FixedQNet([[1.0, 2.0, 3.0]])
    batch = _target_batch()

    targets = agent.compute_targets(batch)

    # 只有第 0 条 terminated；第 2 条 truncated 仍然 bootstrap。
    expected = torch.tensor([1.0, 2.0 + 0.9 * 3.0, 3.0 + 0.9 * 3.0])
    assert torch.allclose(targets, expected)


def test_compute_targets_uses_target_network_when_enabled() -> None:
    agent = DQNAgent(
        DQNConfig(gamma=0.9, bootstrap_truncated=False, use_target_network=True, seed=0)
    )
    agent.model = FixedQNet([[9.0, 9.0, 9.0]])
    agent.target_model = FixedQNet([[1.0, 2.0, 3.0]])

    targets = agent.compute_targets(_target_batch())

    # 如果用在线网络，max 会是 9；这里必须是目标网络的 max=3。
    expected = torch.tensor([1.0, 2.0 + 0.9 * 3.0, 3.0])
    assert torch.allclose(targets, expected)


def test_double_dqn_requires_target_network() -> None:
    with pytest.raises(ValueError):
        DQNConfig(double_dqn=True, use_target_network=False)


def test_huber_selects_smooth_l1_loss() -> None:
    assert isinstance(DQNAgent(DQNConfig(seed=0, huber=False)).criterion, nn.MSELoss)
    assert isinstance(DQNAgent(DQNConfig(seed=0, huber=True)).criterion, nn.SmoothL1Loss)


def test_compute_targets_double_dqn_separates_selection_and_evaluation() -> None:
    agent = DQNAgent(
        DQNConfig(
            gamma=0.9,
            bootstrap_truncated=False,
            use_target_network=True,
            double_dqn=True,
            seed=0,
        )
    )
    # 在线网络选动作 0；目标网络的 max 是动作 1，但动作 0 的值是 0。
    agent.model = FixedQNet([[10.0, 0.0, 0.0]])
    agent.target_model = FixedQNet([[0.0, 5.0, 0.0]])

    targets = agent.compute_targets(_target_batch())

    # Double：用在线网络选 0，再用目标网络取 Q(0)=0。
    expected = torch.tensor([1.0, 2.0 + 0.9 * 0.0, 3.0])
    assert torch.allclose(targets, expected)

    # 对照：普通 target max 会取目标网络的 5，target 变成 2 + 0.9*5。
    agent.config = DQNConfig(
        gamma=0.9,
        bootstrap_truncated=False,
        use_target_network=True,
        double_dqn=False,
        seed=0,
    )
    standard_targets = agent.compute_targets(_target_batch())
    expected_standard = torch.tensor([1.0, 2.0 + 0.9 * 5.0, 3.0])
    assert torch.allclose(standard_targets, expected_standard)


# ----------------------------------------------------------------------
# 目标网络
# ----------------------------------------------------------------------
def _update_batch() -> Batch:
    return Batch.from_transitions(
        [
            transition(action=0, reward=1.0),
            transition(action=1, reward=-1.0, terminated=True),
            transition(action=2, reward=0.0),
        ]
    )


def test_target_network_starts_equal_and_stays_frozen_between_syncs() -> None:
    agent = DQNAgent(
        DQNConfig(seed=0, hidden_size=16, use_target_network=True, target_sync_every=10)
    )

    for online, target in zip(agent.model.parameters(), agent.target_model.parameters()):
        assert torch.equal(online, target)

    target_before = [p.detach().clone() for p in agent.target_model.parameters()]
    agent.train_step(_update_batch())

    assert all(
        torch.equal(before, after)
        for before, after in zip(target_before, agent.target_model.parameters())
    )
    assert agent.updates == 1
    assert agent.target_syncs == 0


def test_sync_target_copies_online_parameters() -> None:
    agent = DQNAgent(
        DQNConfig(seed=0, hidden_size=16, use_target_network=True, target_sync_every=100)
    )
    agent.train_step(_update_batch())

    assert any(
        not torch.equal(online, target)
        for online, target in zip(agent.model.parameters(), agent.target_model.parameters())
    )

    agent.sync_target()

    for online, target in zip(agent.model.parameters(), agent.target_model.parameters()):
        assert torch.equal(online, target)
    assert agent.target_syncs == 1


def test_target_network_syncs_every_n_updates() -> None:
    agent = DQNAgent(
        DQNConfig(seed=0, hidden_size=16, use_target_network=True, target_sync_every=2)
    )
    batch = _update_batch()

    agent.train_step(batch)
    assert (agent.updates, agent.target_syncs) == (1, 0)

    agent.train_step(batch)
    assert (agent.updates, agent.target_syncs) == (2, 1)

    for online, target in zip(agent.model.parameters(), agent.target_model.parameters()):
        assert torch.equal(online, target)


# ----------------------------------------------------------------------
# 更新与 checkpoint
# ----------------------------------------------------------------------
def test_train_step_returns_finite_loss_and_updates_parameters() -> None:
    agent = DQNAgent(DQNConfig(hidden_size=16, lr=0.01, seed=0))
    batch = Batch.from_transitions(
        [
            transition(action=0, reward=10.0),
            transition(action=1, reward=-10.0, terminated=True),
            transition(action=2, reward=0.0),
            transition(action=0, reward=5.0, truncated=True),
        ]
    )
    before = [p.detach().clone() for p in agent.model.parameters()]

    loss = agent.train_step(batch)

    assert math.isfinite(loss)
    after = list(agent.model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_memory_helpers() -> None:
    agent = DQNAgent(DQNConfig(hidden_size=16, batch_size=2, seed=0))

    assert agent.train_long_memory() is None

    short_loss = agent.train_short_memory(transition(action=1, reward=1.0))
    assert math.isfinite(short_loss)

    agent.remember(transition(action=1, reward=1.0))
    agent.remember(transition(action=2, reward=-1.0, terminated=True))
    long_loss = agent.train_long_memory()
    assert long_loss is not None and math.isfinite(long_loss)


def test_checkpoint_roundtrip(tmp_path) -> None:
    agent = DQNAgent(DQNConfig(seed=123, hidden_size=16))
    agent.n_games = 7
    obs = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    q_before = agent.q_values(obs)

    path = tmp_path / "checkpoint.pt"
    agent.save(path)

    loaded = DQNAgent.load(path, device="cpu")

    assert loaded.n_games == 7
    np.testing.assert_allclose(loaded.q_values(obs), q_before, rtol=1e-6, atol=1e-6)


def test_checkpoint_roundtrip_preserves_target_network(tmp_path) -> None:
    agent = DQNAgent(
        DQNConfig(
            seed=123,
            hidden_size=16,
            use_target_network=True,
            target_sync_every=10,
            double_dqn=True,
            huber=True,
        )
    )
    agent.train_step(_update_batch())
    agent.sync_target()

    path = tmp_path / "checkpoint.pt"
    agent.save(path)
    loaded = DQNAgent.load(path, device="cpu")

    assert loaded.config.use_target_network
    assert loaded.config.double_dqn
    assert loaded.config.huber
    assert loaded.updates == agent.updates == 1
    assert loaded.target_syncs == agent.target_syncs == 1
    for expected, actual in zip(
        agent.target_model.parameters(), loaded.target_model.parameters()
    ):
        assert torch.equal(expected, actual)


def test_load_legacy_checkpoint_without_target_fields(tmp_path) -> None:
    agent = DQNAgent(DQNConfig(seed=123, hidden_size=16))
    path = tmp_path / "legacy.pt"
    agent.save(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("target_model", None)
    payload.pop("updates", None)
    payload.pop("target_syncs", None)
    torch.save(payload, path)

    loaded = DQNAgent.load(path, device="cpu")

    for online, target in zip(loaded.model.parameters(), loaded.target_model.parameters()):
        assert torch.equal(online, target)
    assert loaded.updates == 0
    assert loaded.target_syncs == 0
