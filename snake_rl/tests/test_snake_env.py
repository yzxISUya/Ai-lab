"""Stage 0 环境测试。

这些测试不涉及 RL，只验证 MDP 的转移、终止和随机性是否符合设计。
"""

from __future__ import annotations

import pytest

from snake_rl.config import EnvConfig
from snake_rl.env import (
    Action,
    Direction,
    Point,
    SnakeEnv,
    relative_action,
)


def make_env(**overrides) -> SnakeEnv:
    config = EnvConfig(width=10, height=10, **overrides)
    return SnakeEnv(config, seed=0)


def set_board(
    env: SnakeEnv,
    snake: list[Point],
    food: Point | None,
    direction: Direction,
) -> None:
    """测试辅助：直接摆一个局面。"""
    env.reset(seed=0)
    env.snake = list(snake)
    env.food = food
    env.direction = direction
    env.score = 0
    env.steps = 0
    env.steps_since_food = 0


# ----------------------------------------------------------------------
# reset / 初始状态
# ----------------------------------------------------------------------
def test_reset_initial_state() -> None:
    env = make_env()
    state = env.state

    assert len(state.snake) == 3
    assert state.head == state.snake[0] == Point(5, 5)
    assert state.direction is Direction.RIGHT
    assert state.score == 0
    assert state.food is not None
    assert state.food not in state.snake
    assert all(0 <= p.x < 10 and 0 <= p.y < 10 for p in state.snake)
    assert not env.done


def test_reset_seed_is_reproducible() -> None:
    env = make_env()
    env.reset(seed=123)
    food_a = env.food
    env.reset(seed=123)
    food_b = env.food
    assert food_a == food_b


def test_food_never_spawns_on_snake() -> None:
    env = make_env()
    for seed in range(20):
        state = env.reset(seed=seed)
        assert state.food is not None
        assert state.food not in state.snake


# ----------------------------------------------------------------------
# 相对动作
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("action", "expected_direction", "expected_head"),
    [
        (Action.STRAIGHT, Direction.RIGHT, Point(6, 5)),
        (Action.RIGHT, Direction.DOWN, Point(5, 6)),
        (Action.LEFT, Direction.UP, Point(5, 4)),
    ],
)
def test_relative_actions(
    action: Action,
    expected_direction: Direction,
    expected_head: Point,
) -> None:
    env = make_env()
    env.food = Point(0, 0)  # 放在远处，避免这一步吃到食物

    outcome = env.step(action)

    assert outcome.state.direction is expected_direction
    assert outcome.state.head == expected_head
    assert len(outcome.state.snake) == 3


def test_relative_action_helper() -> None:
    assert relative_action(Direction.RIGHT, Direction.RIGHT) is Action.STRAIGHT
    assert relative_action(Direction.RIGHT, Direction.DOWN) is Action.RIGHT
    assert relative_action(Direction.RIGHT, Direction.UP) is Action.LEFT
    assert relative_action(Direction.RIGHT, Direction.LEFT) is None


# ----------------------------------------------------------------------
# 碰撞与终止
# ----------------------------------------------------------------------
def test_wall_collision_terminates() -> None:
    env = make_env()
    set_board(
        env,
        snake=[Point(9, 5), Point(8, 5), Point(7, 5)],
        food=Point(0, 0),
        direction=Direction.RIGHT,
    )
    before = env.state

    outcome = env.step(Action.STRAIGHT)

    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.info["death_reason"] == "wall"
    assert outcome.reward == env.config.death_reward
    # 撞死后棋盘保持合法，不会出现越界的蛇头。
    assert outcome.state.snake == before.snake
    assert outcome.state.head == before.head


def test_self_collision_terminates() -> None:
    env = make_env()
    set_board(
        env,
        snake=[
            Point(5, 5),
            Point(6, 5),
            Point(6, 6),
            Point(5, 6),
            Point(4, 6),
        ],
        food=Point(0, 0),
        direction=Direction.RIGHT,
    )

    outcome = env.step(Action.STRAIGHT)

    assert outcome.terminated is True
    assert outcome.info["death_reason"] == "self"
    assert outcome.reward == env.config.death_reward


def test_moving_into_vacating_tail_is_safe() -> None:
    env = make_env()
    # 头 (5,5)，尾巴 (5,6)。朝下走会进入当前尾巴所在格，
    # 但尾巴这一步会腾出来，所以不算撞身体。
    set_board(
        env,
        snake=[Point(5, 5), Point(6, 5), Point(6, 6), Point(5, 6)],
        food=Point(0, 0),
        direction=Direction.DOWN,
    )

    outcome = env.step(Action.STRAIGHT)

    assert outcome.terminated is False
    assert outcome.state.head == Point(5, 6)
    assert len(outcome.state.snake) == 4


# ----------------------------------------------------------------------
# 吃食物与超时
# ----------------------------------------------------------------------
def test_eating_increases_score_and_length() -> None:
    env = make_env()
    env.food = Point(6, 5)

    outcome = env.step(Action.STRAIGHT)

    assert outcome.state.score == 1
    assert len(outcome.state.snake) == 4
    assert outcome.reward == env.config.food_reward
    assert outcome.state.food is not None
    assert outcome.state.food != Point(6, 5)
    assert outcome.state.food not in outcome.state.snake
    assert outcome.state.steps_since_food == 0


def test_timeout_is_truncation_not_termination() -> None:
    env = make_env(max_steps_without_food_factor=1)
    env.food = Point(0, 0)

    # 初始蛇长 3，上限 = 1 * 3 = 3。
    for _ in range(3):
        outcome = env.step(Action.STRAIGHT)
        assert outcome.truncated is False

    outcome = env.step(Action.STRAIGHT)

    assert outcome.truncated is True
    assert outcome.terminated is False
    assert outcome.info["timeout"] is True
    assert env.done is True


def test_step_after_done_raises() -> None:
    env = make_env()
    env.reset(seed=0)
    env._terminated = True

    with pytest.raises(RuntimeError):
        env.step(Action.STRAIGHT)


# ----------------------------------------------------------------------
# 确定性与配置校验
# ----------------------------------------------------------------------
def _rollout(seed: int) -> list:
    env = SnakeEnv(EnvConfig(width=10, height=10), seed=seed)
    actions = [
        Action.STRAIGHT,
        Action.RIGHT,
        Action.STRAIGHT,
        Action.LEFT,
        Action.STRAIGHT,
        Action.STRAIGHT,
    ]
    states = [env.state]
    for action in actions:
        if env.done:
            break
        states.append(env.step(action).state)
    return states


def test_same_seed_same_trajectory() -> None:
    assert _rollout(seed=7) == _rollout(seed=7)


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        EnvConfig(width=4)
    with pytest.raises(ValueError):
        EnvConfig(width=6, initial_length=5)
    with pytest.raises(ValueError):
        EnvConfig(max_steps_without_food_factor=0)
