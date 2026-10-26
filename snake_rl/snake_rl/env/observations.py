"""把环境的完整状态压成 11 维观察。

这个顺序刻意和参考项目 `agent.py::get_state` 对齐：

    0  正前方是否危险
    1  右方是否危险
    2  左方是否危险
    3  朝左
    4  朝右
    5  朝上
    6  朝下
    7  食物在左边
    8  食物在右边
    9  食物在上边
    10 食物在下边

注意：这是一个**部分观察**，不是完整状态。它没有距离、没有身体形状，
所以 Stage 1 的策略上限会被它限制。Stage 3 会做观察工程。
"""

from __future__ import annotations

import numpy as np

from snake_rl.env.snake_env import CLOCKWISE, Direction, SnakeEnv

OBSERVATION_NAMES = (
    "danger_straight",
    "danger_right",
    "danger_left",
    "dir_left",
    "dir_right",
    "dir_up",
    "dir_down",
    "food_left",
    "food_right",
    "food_up",
    "food_down",
)

OBSERVATION_SIZE = len(OBSERVATION_NAMES)


def observe(env: SnakeEnv, *, ignore_tail: bool = False) -> np.ndarray:
    """返回 11 维 float32 观察。

    ``ignore_tail=False`` 与参考项目一致：它把“走进当前尾巴”标成危险，
    即使我们的环境实际允许这一步。这个保守语义是 Stage 1 对齐基线的一部分；
    Stage 2/3 可以统一它。
    """

    state = env.state
    facing = state.direction
    idx = CLOCKWISE.index(facing)
    right_dir = CLOCKWISE[(idx + 1) % 4]
    left_dir = CLOCKWISE[(idx - 1) % 4]

    danger_straight = env.will_collide(facing, ignore_tail=ignore_tail)
    danger_right = env.will_collide(right_dir, ignore_tail=ignore_tail)
    danger_left = env.will_collide(left_dir, ignore_tail=ignore_tail)

    dir_left = facing == Direction.LEFT
    dir_right = facing == Direction.RIGHT
    dir_up = facing == Direction.UP
    dir_down = facing == Direction.DOWN

    if state.food is None:
        # 棋盘填满时没有食物；这是终止前的状态，四个食物位都为 0。
        food_left = food_right = food_up = food_down = False
    else:
        food_left = state.food.x < state.head.x
        food_right = state.food.x > state.head.x
        food_up = state.food.y < state.head.y
        food_down = state.food.y > state.head.y

    return np.array(
        [
            danger_straight,
            danger_right,
            danger_left,
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            food_left,
            food_right,
            food_up,
            food_down,
        ],
        dtype=np.float32,
    )


def describe_observation(obs: np.ndarray) -> dict[str, int]:
    """调试用：把 11 维向量变成可读的字典。"""

    if obs.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"期望形状 ({OBSERVATION_SIZE},)，得到 {obs.shape}")
    return {name: int(value) for name, value in zip(OBSERVATION_NAMES, obs)}
