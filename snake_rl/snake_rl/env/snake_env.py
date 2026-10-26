"""Snake 的纯逻辑环境（不依赖 pygame）。

设计目标
--------
1. **逻辑与渲染分离**：训练时完全不创建窗口，渲染只在试玩/演示时使用。
2. **格子坐标**：状态里是 (x, y) 整数格子，不是像素。
3. **相对动作**：直行 / 右转 / 左转，而不是绝对的上/下/左/右。
4. **termination 与 truncation 分开**：撞墙/撞自己是 terminated；超时是 truncated。

后面做 DQN 时，这两者的区别会直接影响 bootstrap：
- terminated：真正的终止状态，目标值 ``y = r``；
- truncated：时间限制导致的中断，目标值仍然是 ``y = r + γ max Q(s')``。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

from snake_rl.config import EnvConfig


class Direction(IntEnum):
    """绝对朝向。顺序是顺时针，方便做相对转向。"""

    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


# 顺时针顺序：右 -> 下 -> 左 -> 上
CLOCKWISE: tuple[Direction, ...] = (
    Direction.RIGHT,
    Direction.DOWN,
    Direction.LEFT,
    Direction.UP,
)


class Action(IntEnum):
    """相对动作，相对于蛇当前朝向。"""

    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2


class Point(NamedTuple):
    x: int
    y: int


@dataclass(frozen=True)
class SnakeState:
    """一步结束后的完整游戏状态。

    注意：这是环境的**底层状态**。Stage 1 的 DQN 不会直接看到它，
    而是通过 observation 函数把它压成 11 维特征。二者要区分开。
    """

    head: Point
    snake: tuple[Point, ...]  # 第 0 个是头，最后一个是尾
    food: Point | None  # 蛇填满棋盘后没有食物
    direction: Direction
    score: int
    steps: int  # 本局总步数
    steps_since_food: int  # 距离上次吃食物的步数


class StepOutcome(NamedTuple):
    """``env.step(action)`` 的返回值，字段顺序模仿 Gymnasium。

    可以这样解包::

        state, reward, terminated, truncated, info = env.step(action)
    """

    state: SnakeState
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class SnakeEnv:
    """极简 Snake 环境。

    典型用法::

        env = SnakeEnv(EnvConfig(width=10, height=10), seed=0)
        while not env.done:
            action = Action.STRAIGHT
            state, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------
    @property
    def head(self) -> Point:
        return self.snake[0]

    @property
    def done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def state(self) -> SnakeState:
        return SnakeState(
            head=self.snake[0],
            snake=tuple(self.snake),
            food=self.food,
            direction=self.direction,
            score=self.score,
            steps=self.steps,
            steps_since_food=self.steps_since_food,
        )

    def reset(self, seed: int | None = None) -> SnakeState:
        """重置环境，返回初始状态。

        传入 ``seed`` 会让食物位置可复现；不传则沿用当前随机数流。
        """

        if seed is not None:
            self.rng = random.Random(seed)

        cx = self.config.width // 2
        cy = self.config.height // 2

        self.direction = Direction.RIGHT
        self.snake: list[Point] = [Point(cx, cy)] + [
            Point(cx - offset, cy) for offset in range(1, self.config.initial_length)
        ]
        self.food: Point | None = None
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self._terminated = False
        self._truncated = False

        self._place_food()
        return self.state

    def step(self, action: Action | int) -> StepOutcome:
        """执行一个相对动作。

        ``action`` 用整数 0/1/2 表示 straight/right/left。
        """

        if self.done:
            raise RuntimeError("episode 已经结束，请先调用 reset()")

        action = Action(int(action))
        new_direction = self._turn(self.direction, action)
        new_head = self._next_head(self.head, new_direction)

        self.steps += 1
        self.steps_since_food += 1

        wall_collision = not self._in_bounds(new_head)

        # 不吃食物时，尾巴会在这一步腾出来，所以新头可以走到当前尾巴上。
        eating = (not wall_collision) and self.food is not None and new_head == self.food
        body_to_check = self.snake if eating else self.snake[:-1]
        self_collision = (not wall_collision) and new_head in body_to_check

        if wall_collision or self_collision:
            # 刻意不把蛇头移到越界/重叠的位置：保持状态合法，方便渲染和测试。
            self._terminated = True
            reason = "wall" if wall_collision else "self"
            info = self._info(
                death_reason=reason,
                attempted_head=new_head,
            )
            return StepOutcome(
                state=self.state,
                reward=float(self.config.step_reward + self.config.death_reward),
                terminated=True,
                truncated=False,
                info=info,
            )

        self.direction = new_direction
        self.snake.insert(0, new_head)

        reward = float(self.config.step_reward)
        terminated = False
        truncated = False
        info: dict = {}

        if eating:
            self.score += 1
            self.steps_since_food = 0
            reward += float(self.config.food_reward)

            if not self._place_food():
                # 棋盘被蛇填满：胜利，没有下一格可以放食物。
                terminated = True
                reward += float(self.config.win_reward)
                info["win"] = True
        else:
            self.snake.pop()

        # 超时用「距上次吃食物的步数」而不是总步数：
        # 否则一局活得越久，越容易在吃食物之后被无端截断。
        step_limit = self.config.max_steps_without_food_factor * len(self.snake)
        if not terminated and self.steps_since_food > step_limit:
            truncated = True
            reward += float(self.config.timeout_reward)
            info["timeout"] = True

        self._terminated = terminated
        self._truncated = truncated
        info.update(self._info())
        return StepOutcome(
            state=self.state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def will_collide(
        self,
        direction: Direction,
        *,
        ignore_tail: bool = True,
    ) -> bool:
        """假设蛇从当前头沿 ``direction`` 走一格，是否会撞墙或撞身体。

        这个函数主要给 Stage 1 的 observation 用：11 维特征里有 3 个
        “前方/右方/左方是否危险”的 bit。
        """

        next_head = self._next_head(self.head, direction)
        if not self._in_bounds(next_head):
            return True
        body = self.snake[:-1] if ignore_tail else self.snake
        return next_head in body

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _info(self, **extra: object) -> dict:
        info = {
            "score": self.score,
            "steps": self.steps,
            "steps_since_food": self.steps_since_food,
            "snake_length": len(self.snake),
        }
        info.update(extra)
        return info

    def _place_food(self) -> bool:
        """从空格子中均匀采样食物；没有空格子时返回 False（胜利）。"""

        occupied = set(self.snake)
        empty = [
            Point(x, y)
            for y in range(self.config.height)
            for x in range(self.config.width)
            if (x, y) not in occupied
        ]
        if not empty:
            self.food = None
            return False

        self.food = self.rng.choice(empty)
        return True

    def _in_bounds(self, point: Point) -> bool:
        return 0 <= point.x < self.config.width and 0 <= point.y < self.config.height

    @staticmethod
    def _next_head(head: Point, direction: Direction) -> Point:
        dx, dy = {
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.UP: (0, -1),
        }[direction]
        return Point(head.x + dx, head.y + dy)

    @staticmethod
    def _turn(current: Direction, action: Action) -> Direction:
        idx = CLOCKWISE.index(current)
        if action == Action.STRAIGHT:
            return current
        if action == Action.RIGHT:
            return CLOCKWISE[(idx + 1) % 4]
        if action == Action.LEFT:
            return CLOCKWISE[(idx - 1) % 4]
        raise ValueError(f"未知动作: {action}")


def relative_action(current: Direction, desired: Direction) -> Action | None:
    """把“我想朝 desired”翻译成相对动作。

    人类试玩时用绝对方向（方向键）更直观，但环境接收的是相对动作。
    如果 desired 与 current 相反（掉头），返回 None，表示这个动作不合法。
    """

    diff = (CLOCKWISE.index(desired) - CLOCKWISE.index(current)) % 4
    if diff == 0:
        return Action.STRAIGHT
    if diff == 1:
        return Action.RIGHT
    if diff == 3:
        return Action.LEFT
    return None
