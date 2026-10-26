"""Stage 1 的 DQN 智能体。

刻意保留参考项目的算法结构：

- 11 维观察 -> 2 层 MLP -> 3 个 Q 值；
- epsilon-greedy；
- 经验回放；
- 每步用单个 transition 做一次更新（short memory）；
- 每局结束用 replay 采样一个 batch 更新（long memory）；
- **没有目标网络**：target 用在线网络算；
- **MSE loss**；
- 默认把 truncation 当成 termination（``bootstrap_truncated=False``）。

Stage 2 会逐项修：目标网络、Double DQN、Huber、正确的 truncation bootstrap。
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_rl.config import DQNConfig
from snake_rl.models import QNet


class Transition(NamedTuple):
    """一条经验。"""

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool


class Batch(NamedTuple):
    """从 replay buffer 采样出的一批经验。"""

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @classmethod
    def from_transitions(cls, transitions: Sequence[Transition]) -> "Batch":
        if not transitions:
            raise ValueError("不能从空的 transition 列表构造 Batch")

        return cls(
            obs=np.stack([t.obs for t in transitions]).astype(np.float32),
            actions=np.asarray([t.action for t in transitions], dtype=np.int64),
            rewards=np.asarray([t.reward for t in transitions], dtype=np.float32),
            next_obs=np.stack([t.next_obs for t in transitions]).astype(np.float32),
            terminated=np.asarray([t.terminated for t in transitions], dtype=np.bool_),
            truncated=np.asarray([t.truncated for t in transitions], dtype=np.bool_),
        )

    def __len__(self) -> int:
        return len(self.actions)


class ReplayBuffer:
    """固定容量的经验回放池。

    用独立的 ``random.Random``，不污染全局随机状态；这样实验更容易复现。
    """

    def __init__(self, capacity: int, seed: int = 0):
        if capacity < 1:
            raise ValueError("capacity 至少为 1")
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Batch:
        if not self.buffer:
            raise ValueError("replay buffer 为空")
        k = min(batch_size, len(self.buffer))
        # random.sample 可以直接作用于 deque（Python 3.11+ 实测可用）。
        transitions = self.rng.sample(self.buffer, k)
        return Batch.from_transitions(transitions)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN 智能体：动作选择、经验存储、TD 更新、checkpoint。"""

    def __init__(self, config: DQNConfig | None = None, device: str = "cpu"):
        self.config = config or DQNConfig()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA，但当前 PyTorch 没有可用的 CUDA")
        self.device = torch.device(device)

        # 让网络初始化可复现。
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        self.model = QNet(
            input_size=self.config.obs_size,
            hidden_size=self.config.hidden_size,
            output_size=self.config.n_actions,
        ).to(self.device)
        # 目标网络：结构和在线网络相同，初始化时参数完全一致。
        # 即使 use_target_network=False 也保留它，checkpoint 格式更统一；
        # 此时 compute_targets 会忽略它。
        self.target_model = QNet(
            input_size=self.config.obs_size,
            hidden_size=self.config.hidden_size,
            output_size=self.config.n_actions,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.criterion = nn.SmoothL1Loss() if self.config.huber else nn.MSELoss()

        self.memory = ReplayBuffer(self.config.max_memory, seed=self.config.seed)
        self.rng = random.Random(self.config.seed + 1)

        self.n_games = 0
        self.updates = 0
        self.target_syncs = 0

    # ------------------------------------------------------------------
    # 行为
    # ------------------------------------------------------------------
    @property
    def epsilon(self) -> float:
        """当前探索概率（0 到 1 的线性衰减）。"""

        if self.config.epsilon_decay_games == 0:
            return self.config.epsilon_end
        progress = min(1.0, self.n_games / self.config.epsilon_decay_games)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def get_action(self, obs: np.ndarray, *, greedy: bool = False) -> int:
        """epsilon-greedy 选动作；``greedy=True`` 时只取 argmax。"""

        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.config.n_actions)

        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(x)
        return int(q.argmax(dim=1).item())

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """调试/分析用：返回三个动作的 Q 值。"""

        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(x)
        return q.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # 记忆
    # ------------------------------------------------------------------
    def remember(self, transition: Transition) -> None:
        self.memory.add(transition)

    def end_episode(self) -> None:
        self.n_games += 1

    # ------------------------------------------------------------------
    # 学习
    # ------------------------------------------------------------------
    def compute_targets(self, batch: Batch) -> torch.Tensor:
        """计算 TD target。

        Stage 1 默认 ``bootstrap_truncated=False``，等价于参考项目的
        ``done = terminated or truncated``：超时也把未来价值砍掉。
        Stage 2 把它设为 True，就会只对 terminated 砍 bootstrap。

        注意用 ``torch.no_grad()``：target 是“标签”，不应该反传到目标计算里。
        参考项目用 ``pred.clone()`` 的写法可能让 target 也带梯度，这里不复制这个 bug。
        """

        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.bool, device=self.device)
        truncated = torch.as_tensor(batch.truncated, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            # 有目标网络时用延迟参数算 target；否则退回 Stage 1 的在线网络。
            target_net = self.target_model if self.config.use_target_network else self.model
            target_values = target_net(next_obs)
            if self.config.double_dqn:
                # Double DQN：在线网络选动作，目标网络评估这个动作。
                next_actions = self.model(next_obs).argmax(dim=1, keepdim=True)
                next_q = target_values.gather(1, next_actions).squeeze(1)
            else:
                next_q = target_values.max(dim=1).values
            done = terminated if self.config.bootstrap_truncated else (terminated | truncated)
            return rewards + self.config.gamma * next_q * (~done).float()

    def train_step(self, batch: Batch) -> float:
        """对一批经验做一次梯度更新，返回 loss。"""

        if len(batch) == 0:
            raise ValueError("batch 不能为空")

        self.model.train()
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)

        pred = self.model(obs)  # (B, 3)
        pred_q = pred.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
        target = self.compute_targets(batch)  # (B,)

        loss = self.criterion(pred_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates += 1
        if (
            self.config.use_target_network
            and self.updates % self.config.target_sync_every == 0
        ):
            self.sync_target()
        return float(loss.item())

    def sync_target(self) -> None:
        """把在线网络参数复制到目标网络，并让目标网络保持 eval 模式。"""

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.target_syncs += 1

    def train_short_memory(self, transition: Transition) -> float:
        """参考项目的 per-step 单样本更新。"""

        return self.train_step(Batch.from_transitions([transition]))

    def train_long_memory(self) -> float | None:
        """每局结束从 replay 采样一个 batch 更新。"""

        if len(self.memory) == 0:
            return None
        return self.train_step(self.memory.sample(self.config.batch_size))

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "n_games": self.n_games,
            "updates": self.updates,
            "target_syncs": self.target_syncs,
            "dqn_config": asdict(self.config),
            "torch_rng_state": torch.get_rng_state(),
            "action_rng_state": self.rng.getstate(),
            "replay_rng_state": self.memory.rng.getstate(),
        }
        if self.device.type == "cuda":
            payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DQNAgent":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        payload = torch.load(path, map_location=device, weights_only=False)

        config = DQNConfig(**payload["dqn_config"])
        agent = cls(config, device=device)
        agent.model.load_state_dict(payload["model"])
        # 兼容 Stage 1 的旧 checkpoint：没有 target_model 时从在线网络复制。
        if "target_model" in payload:
            agent.target_model.load_state_dict(payload["target_model"])
        else:
            agent.target_model.load_state_dict(agent.model.state_dict())
        agent.target_model.eval()
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.n_games = int(payload.get("n_games", 0))
        agent.updates = int(payload.get("updates", 0))
        agent.target_syncs = int(payload.get("target_syncs", 0))

        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if "action_rng_state" in payload:
            agent.rng.setstate(payload["action_rng_state"])
        if "replay_rng_state" in payload:
            agent.memory.rng.setstate(payload["replay_rng_state"])
        if payload.get("cuda_rng_state") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        return agent
