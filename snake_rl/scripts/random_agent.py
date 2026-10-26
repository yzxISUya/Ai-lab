r"""随机策略基线。

它有两个作用：
1. 验证环境能跑通（不崩溃、episode 能结束）；
2. 给后续 DQN 一个“最差也应该超过”的下界。

运行::

    .\.venv\Scripts\python.exe scripts\random_agent.py --episodes 100
    .\.venv\Scripts\python.exe scripts\random_agent.py --episodes 20 --render --fps 12
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter

from snake_rl.config import EnvConfig
from snake_rl.env import Action, SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uniform-random Snake baseline")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-factor", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes 至少为 1")
    config = EnvConfig(
        width=args.width,
        height=args.height,
        max_steps_without_food_factor=args.timeout_factor,
    )
    env = SnakeEnv(config)
    action_rng = random.Random(args.seed)

    renderer = None
    if args.render:
        from snake_rl.env.renderer import PygameRenderer

        renderer = PygameRenderer(env)

    scores: list[int] = []
    steps: list[int] = []
    endings: Counter[str] = Counter()

    try:
        for episode in range(args.episodes):
            # 每局独立 seed，保证整次实验可复现。
            env.reset(seed=args.seed + episode)
            last_info: dict = {}

            while not env.done:
                action = Action(action_rng.randrange(3))
                outcome = env.step(action)
                last_info = outcome.info
                if renderer is not None:
                    renderer.draw(outcome.state)
                    renderer.tick(args.fps)

            scores.append(env.score)
            steps.append(env.steps)

            if last_info.get("win"):
                ending = "win"
            elif last_info.get("timeout"):
                ending = "timeout"
            elif "death_reason" in last_info:
                ending = last_info["death_reason"]
            else:
                ending = "unknown"
            endings[ending] += 1

            if args.verbose:
                print(
                    f"episode {episode:03d}  score={env.score:2d}  "
                    f"steps={env.steps:4d}  ending={ending}"
                )
    finally:
        if renderer is not None:
            renderer.close()

    print()
    print(f"episodes={args.episodes}  seed={args.seed}  grid={args.width}x{args.height}")
    print(f"scores: {scores}")
    print(
        "score  mean={:.2f}  median={:.1f}  max={}  min={}  std={:.2f}".format(
            statistics.mean(scores),
            statistics.median(scores),
            max(scores),
            min(scores),
            statistics.stdev(scores) if len(scores) > 1 else 0.0,
        )
    )
    print(
        "steps  mean={:.1f}  median={:.1f}  max={}".format(
            statistics.mean(steps),
            statistics.median(steps),
            max(steps),
        )
    )
    print(f"endings: {dict(endings)}")


if __name__ == "__main__":
    main()
