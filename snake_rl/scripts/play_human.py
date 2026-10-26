r"""人类试玩。

运行::

    .\.venv\Scripts\python.exe scripts\play_human.py

操作：
- 方向键 / WASD：转向（绝对方向，脚本会自动翻译成相对动作）
- R：重开
- Esc：退出
"""

from __future__ import annotations

import argparse

import pygame

from snake_rl.config import EnvConfig
from snake_rl.env import Action, Direction, SnakeEnv, relative_action
from snake_rl.env.renderer import PygameRenderer

KEY_TO_DIRECTION = {
    pygame.K_UP: Direction.UP,
    pygame.K_w: Direction.UP,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_s: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_a: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_d: Direction.RIGHT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human-playable Snake")
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--cell-size", type=int, default=24)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EnvConfig(width=args.width, height=args.height)
    env = SnakeEnv(config, seed=args.seed)
    renderer = PygameRenderer(env, cell_size=args.cell_size)

    desired = env.direction
    status = "Arrows/WASD  R  Esc"
    renderer.draw(env.state, status)
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        env.reset(seed=args.seed)
                        desired = env.direction
                        status = "New game"
                    elif event.key in KEY_TO_DIRECTION:
                        desired = KEY_TO_DIRECTION[event.key]

            if not running:
                break

            if env.done:
                renderer.draw(env.state, status)
                renderer.tick(args.fps)
                continue

            action = relative_action(env.direction, desired)
            if action is None:
                # 不能 180 度掉头；忽略这次输入，继续直行。
                action = Action.STRAIGHT

            outcome = env.step(action)

            if outcome.terminated:
                reason = outcome.info.get("death_reason", "?")
                status = f"Dead ({reason})  R restart"
            elif outcome.truncated:
                status = "Timeout  R restart"
            else:
                status = ""

            renderer.draw(outcome.state, status)
            renderer.tick(args.fps)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
