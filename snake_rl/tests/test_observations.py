"""11 维观察的测试。

参考项目的位顺序：
danger_straight, danger_right, danger_left,
dir_left, dir_right, dir_up, dir_down,
food_left, food_right, food_up, food_down
"""

from __future__ import annotations

import numpy as np

from snake_rl.config import EnvConfig
from snake_rl.env import Direction, Point, SnakeEnv
from snake_rl.env.observations import (
    OBSERVATION_SIZE,
    describe_observation,
    observe,
)


def make_env() -> SnakeEnv:
    # 初始：头 (5,5)，身体 (4,5),(3,5)，朝右。
    return SnakeEnv(EnvConfig(width=10, height=10), seed=0)


def set_board(env: SnakeEnv, snake: list[Point], direction: Direction, food: Point | None) -> None:
    env.reset(seed=0)
    env.snake = list(snake)
    env.direction = direction
    env.food = food


def test_observation_shape_and_dtype() -> None:
    env = make_env()
    env.food = Point(8, 3)

    obs = observe(env)

    assert OBSERVATION_SIZE == 11
    assert obs.shape == (11,)
    assert obs.dtype == np.float32


def test_initial_observation_matches_expected_bits() -> None:
    env = make_env()
    env.food = Point(8, 3)  # 右 + 上

    obs = observe(env)

    # 前方 (6,5)、右方/下方 (5,6)、左方/上方 (5,4) 都安全；
    # 朝右；食物在右和上。
    expected = np.array(
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(obs, expected)


def test_danger_straight_at_wall() -> None:
    env = make_env()
    set_board(
        env,
        snake=[Point(9, 5), Point(8, 5), Point(7, 5)],
        direction=Direction.RIGHT,
        food=Point(0, 0),
    )

    obs = observe(env)

    assert obs[0] == 1
    assert obs[1] == 0
    assert obs[2] == 0


def test_danger_right_from_body() -> None:
    env = make_env()
    # 头 (5,5)，脖子在左，身体绕到下方 (5,6)（不是尾巴）：右转方向危险。
    set_board(
        env,
        snake=[Point(5, 5), Point(4, 5), Point(4, 6), Point(5, 6), Point(5, 7)],
        direction=Direction.RIGHT,
        food=Point(0, 0),
    )

    obs = observe(env)

    assert obs[0] == 0  # 正前方 (6,5) 安全
    assert obs[1] == 1  # 右转 -> 下 (5,6)，撞身体
    assert obs[2] == 0  # 左转 -> 上 (5,4) 安全


def test_danger_left_from_body() -> None:
    env = make_env()
    set_board(
        env,
        snake=[Point(5, 5), Point(4, 5), Point(4, 4), Point(5, 4), Point(5, 3)],
        direction=Direction.RIGHT,
        food=Point(0, 0),
    )

    obs = observe(env)

    assert obs[0] == 0
    assert obs[1] == 0
    assert obs[2] == 1  # 左转 -> 上 (5,4)，撞身体


def test_food_relative_bits() -> None:
    cases = {
        Point(2, 5): [1, 0, 0, 0],
        Point(8, 5): [0, 1, 0, 0],
        Point(5, 2): [0, 0, 1, 0],
        Point(5, 8): [0, 0, 0, 1],
    }
    for food, expected in cases.items():
        env = make_env()
        env.food = food
        obs = observe(env)
        assert obs[7:11].tolist() == expected, food


def test_food_none_sets_all_food_bits_to_zero() -> None:
    env = make_env()
    env.food = None

    obs = observe(env)

    assert obs[7:11].tolist() == [0, 0, 0, 0]


def test_tail_ahead_is_dangerous_by_default() -> None:
    env = make_env()
    # 尾巴就在正前方 (6,5)。
    set_board(
        env,
        snake=[Point(5, 5), Point(4, 5), Point(4, 6), Point(5, 6), Point(6, 6), Point(6, 5)],
        direction=Direction.RIGHT,
        food=Point(0, 0),
    )

    obs_conservative = observe(env)  # ignore_tail=False，与参考项目一致
    obs_ignore_tail = observe(env, ignore_tail=True)

    assert obs_conservative[0] == 1
    assert obs_ignore_tail[0] == 0


def test_describe_observation() -> None:
    env = make_env()
    env.food = Point(8, 3)
    obs = observe(env)

    desc = describe_observation(obs)

    assert desc["dir_right"] == 1
    assert desc["food_right"] == 1
    assert desc["food_up"] == 1
    assert sum(desc.values()) == 3
