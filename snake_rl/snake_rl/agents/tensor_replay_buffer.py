"""高性能显存/内存预分配张量经验回放池。

特点：
- 零 Python 对象开销：全部预分配为连续的 PyTorch 张量（支持 CUDA 或 CPU）；
- 紧凑显存存储：观测张量以 uint8 存储（10×10 棋盘 20 万条仅约 100MB 显存），采样时由 GPU 流水线并行归一化为 float32；
- 批量环形写入：单次写入 N 条经验（支持多环境并行收集）；
- GPU 纯张量采样：直接在 GPU 上生成随机索引并完成切片，无任何数据跨总线拷贝。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class BatchedTransition:
    """已切片好的 GPU 批处理张量包。"""

    obs: torch.Tensor          # (B, C, H, W) 或 (B, 11), float32
    actions: torch.Tensor      # (B,), int64
    rewards: torch.Tensor      # (B,), float32
    next_obs: torch.Tensor     # (B, C, H, W) 或 (B, 11), float32
    terminated: torch.Tensor   # (B,), bool
    truncated: torch.Tensor    # (B,), bool

    def __len__(self) -> int:
        return self.obs.shape[0]


class TensorReplayBuffer:
    """高性能预分配连续张量经验回放池。"""

    def __init__(
        self,
        capacity: int,
        obs_shape: Sequence[int],
        device: torch.device | str = "cpu",
        as_uint8: bool = True,
    ):
        if capacity < 1:
            raise ValueError("capacity 至少为 1")

        self.capacity = capacity
        self.obs_shape = tuple(obs_shape)
        self.device = torch.device(device)
        self.as_uint8 = as_uint8

        obs_dtype = torch.uint8 if as_uint8 else torch.float32

        # 连续内存/显存张量预分配
        self.obs = torch.empty((capacity, *self.obs_shape), dtype=obs_dtype, device=self.device)
        self.next_obs = torch.empty((capacity, *self.obs_shape), dtype=obs_dtype, device=self.device)
        self.actions = torch.empty((capacity,), dtype=torch.int64, device=self.device)
        self.rewards = torch.empty((capacity,), dtype=torch.float32, device=self.device)
        self.terminated = torch.empty((capacity,), dtype=torch.bool, device=self.device)
        self.truncated = torch.empty((capacity,), dtype=torch.bool, device=self.device)

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor,
        rewards: np.ndarray | torch.Tensor,
        next_obs: np.ndarray | torch.Tensor,
        terminated: np.ndarray | torch.Tensor,
        truncated: np.ndarray | torch.Tensor,
    ) -> None:
        """批量写入 N 条转移数据。"""
        batch_size = len(obs)
        if batch_size == 0:
            return

        # 准备张量并转移到指定设备
        def _to_tensor(x, dtype):
            if isinstance(x, torch.Tensor):
                return x.to(device=self.device, dtype=dtype)
            return torch.as_tensor(x, dtype=dtype, device=self.device)

        obs_dtype = torch.uint8 if self.as_uint8 else torch.float32
        t_obs = _to_tensor(obs, obs_dtype)
        t_next_obs = _to_tensor(next_obs, obs_dtype)
        t_actions = _to_tensor(actions, torch.int64)
        t_rewards = _to_tensor(rewards, torch.float32)
        t_terminated = _to_tensor(terminated, torch.bool)
        t_truncated = _to_tensor(truncated, torch.bool)

        # 环形写入逻辑
        if self.ptr + batch_size <= self.capacity:
            idx = slice(self.ptr, self.ptr + batch_size)
            self.obs[idx] = t_obs
            self.next_obs[idx] = t_next_obs
            self.actions[idx] = t_actions
            self.rewards[idx] = t_rewards
            self.terminated[idx] = t_terminated
            self.truncated[idx] = t_truncated
        else:
            # 跨越边界时分两段写入
            first_len = self.capacity - self.ptr
            second_len = batch_size - first_len

            # 第一段：写入尾部
            idx1 = slice(self.ptr, self.capacity)
            self.obs[idx1] = t_obs[:first_len]
            self.next_obs[idx1] = t_next_obs[:first_len]
            self.actions[idx1] = t_actions[:first_len]
            self.rewards[idx1] = t_rewards[:first_len]
            self.terminated[idx1] = t_terminated[:first_len]
            self.truncated[idx1] = t_truncated[:first_len]

            # 第二段：写入头部
            idx2 = slice(0, second_len)
            self.obs[idx2] = t_obs[first_len:]
            self.next_obs[idx2] = t_next_obs[first_len:]
            self.actions[idx2] = t_actions[first_len:]
            self.rewards[idx2] = t_rewards[first_len:]
            self.terminated[idx2] = t_terminated[first_len:]
            self.truncated[idx2] = t_truncated[first_len:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> BatchedTransition:
        """GPU 上纯张量随机切片采样。"""
        if self.size < batch_size:
            raise ValueError(f"样本不足：当前 {self.size} 条，请求采样 {batch_size} 条")

        # 在 GPU/CPU 上生成随机索引
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        sampled_obs = self.obs[indices]
        sampled_next_obs = self.next_obs[indices]

        # 若以 uint8 存储，则在 GPU 显存切片时高速归一化为 float32
        if self.as_uint8:
            sampled_obs = sampled_obs.to(torch.float32) / 255.0
            sampled_next_obs = sampled_next_obs.to(torch.float32) / 255.0

        return BatchedTransition(
            obs=sampled_obs,
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=sampled_next_obs,
            terminated=self.terminated[indices],
            truncated=self.truncated[indices],
        )
