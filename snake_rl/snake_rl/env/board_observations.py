"""把完整棋盘编码成 CNN 可用的 ``(C, H, W)`` 张量。

通道顺序：

    0  body     整条蛇（含头尾）
    1  head     蛇头
    2  tail     蛇尾
    3  food     食物
    4  border   到最近棋盘的归一化距离（边缘 0，中心 1）

前 4 个通道是二值；第 5 个通道只在 ``channels=5`` 时存在。
这个编码不使用 BFS/最短路径/最优动作，只是把当前棋盘状态画出来；
真正的策略仍然由 CNN 从奖励里学习。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from snake_rl.env.snake_env import SnakeEnv

BOARD_CHANNEL_NAMES = ("body", "head", "tail", "food", "border")
BOARD_CHANNELS = len(BOARD_CHANNEL_NAMES)


@lru_cache(maxsize=None)
def _border_channel(height: int, width: int) -> np.ndarray:
    """边缘 0、中心 1 的归一化距离图；只和棋盘尺寸有关，缓存复用。"""

    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    distance = np.minimum(
        np.minimum(xs, width - 1 - xs),
        np.minimum(ys, height - 1 - ys),
    )
    # 用实际最大距离归一化：边缘 0，中心 1；小棋盘也不会除以 0。
    max_distance = float(distance.max())
    return distance / max_distance if max_distance > 0 else distance


def observe_board(env: SnakeEnv, *, channels: int = BOARD_CHANNELS) -> np.ndarray:
    """返回 ``(channels, height, width)`` 的 float32 棋盘观察。

    ``channels`` 只支持 4 或 5：4 是 body/head/tail/food，5 额外带边界距离。
    """

    if channels not in (4, 5):
        raise ValueError(f"channels 必须是 4 或 5，得到 {channels}")

    height = env.config.height
    width = env.config.width
    obs = np.zeros((channels, height, width), dtype=np.float32)

    body = np.asarray(env.snake, dtype=np.intp)  # (N, 2)，列顺序是 x, y
    obs[0, body[:, 1], body[:, 0]] = 1.0

    head = env.head
    obs[1, head.y, head.x] = 1.0

    tail = env.snake[-1]
    obs[2, tail.y, tail.x] = 1.0

    if env.food is not None:
        obs[3, env.food.y, env.food.x] = 1.0

    if channels == 5:
        obs[4] = _border_channel(height, width)

    return obs


def describe_board(obs: np.ndarray) -> dict[str, float]:
    """调试用：每个通道的均值/最大值。"""

    if obs.ndim != 3:
        raise ValueError(f"期望 (C, H, W)，得到 {obs.shape}")
    names = BOARD_CHANNEL_NAMES[: obs.shape[0]]
    return {
        name: float(obs[i].mean()) if obs[i].size else 0.0
        for i, name in enumerate(names)
    }
