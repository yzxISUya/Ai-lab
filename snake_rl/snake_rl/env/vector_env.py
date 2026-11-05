"""向量化并行 Snake 环境。

在单个进程中并行维护 N 个独立的 SnakeEnv 实例：
- 批处理 action 输入：``(num_envs,)``
- 批处理 observation 输出：``(num_envs, C, H, W)`` 或 ``(num_envs, 11)``
- 自动 Reset：任一子环境结束时原地 reset 并返回新局起点，外部训练循环无需停顿。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from snake_rl.config import EnvConfig
from snake_rl.env.board_observations import BOARD_CHANNELS, observe_board
from snake_rl.env.observations import observe
from snake_rl.env.snake_env import SnakeEnv


class VectorSnakeEnv:
    """同步向量化贪吃蛇环境。"""

    def __init__(
        self,
        config: EnvConfig,
        num_envs: int = 32,
        seed: int = 0,
        obs_type: str = "board",
        board_channels: int = BOARD_CHANNELS,
        as_uint8: bool = True,
    ):
        if num_envs < 1:
            raise ValueError("num_envs 至少为 1")
        if obs_type not in ("features11", "board"):
            raise ValueError("obs_type 必须是 'features11' 或 'board'")

        self.config = config
        self.num_envs = num_envs
        self.seed = seed
        self.obs_type = obs_type
        self.board_channels = board_channels
        self.as_uint8 = as_uint8

        # 创建 N 个独立环境，分配递增的种子
        self.envs = [SnakeEnv(config, seed=seed + i * 1000) for i in range(num_envs)]
        self._episode_counts = [0] * num_envs

        if self.obs_type == "board":
            self.obs_shape = (board_channels, config.height, config.width)
            dtype = np.uint8 if as_uint8 else np.float32
        else:
            self.obs_shape = (11,)
            dtype = np.float32

        self._obs_buf = np.zeros((num_envs, *self.obs_shape), dtype=dtype)

    def _observe_one(self, env: SnakeEnv) -> np.ndarray:
        if self.obs_type == "board":
            obs = observe_board(env, channels=self.board_channels)
            if self.as_uint8:
                return (obs * 255.0).round().astype(np.uint8)
            return obs
        return observe(env)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """重置所有环境，返回初始批处理观测。"""
        base_seed = self.seed if seed is None else seed
        for i, env in enumerate(self.envs):
            self._episode_counts[i] += 1
            env.reset(seed=base_seed + i * 1000 + self._episode_counts[i])
            self._obs_buf[i] = self._observe_one(env)
        return self._obs_buf.copy()

    def step(
        self, actions: np.ndarray | list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """对所有子环境执行一步。

        返回:
            (next_obs, rewards, terminateds, truncateds, infos)
            - next_obs: (num_envs, *obs_shape)
            - rewards: (num_envs,) float32
            - terminateds: (num_envs,) bool
            - truncateds: (num_envs,) bool
            - infos: list[dict] 长度为 num_envs，若本步结束则包含 "final_score" 等
        """
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            outcome = env.step(int(action))
            rewards[i] = outcome.reward
            term = outcome.terminated
            trunc = outcome.truncated
            terminateds[i] = term
            truncateds[i] = trunc

            info = dict(outcome.info)
            real_obs = self._observe_one(env)

            if term or trunc:
                info["episode_score"] = env.score
                info["episode_steps"] = env.steps
                info["final_observation"] = real_obs
                self._episode_counts[i] += 1
                # 自动原地重置
                env.reset(seed=self.seed + i * 1000 + self._episode_counts[i])
                self._obs_buf[i] = self._observe_one(env)
            else:
                self._obs_buf[i] = real_obs

            infos.append(info)

        return self._obs_buf.copy(), rewards, terminateds, truncateds, infos
