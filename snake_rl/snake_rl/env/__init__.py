"""Snake 环境。

环境只负责游戏逻辑，不依赖 pygame；渲染在 :mod:`snake_rl.env.renderer` 中。
"""

from .snake_env import (
    Action,
    Direction,
    Point,
    SnakeEnv,
    SnakeState,
    StepOutcome,
    relative_action,
)

__all__ = [
    "Action",
    "Direction",
    "Point",
    "SnakeEnv",
    "SnakeState",
    "StepOutcome",
    "relative_action",
]
