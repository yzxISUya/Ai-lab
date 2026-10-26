r"""用 greedy 策略评估一个 Stage 1 checkpoint。

运行::

    uv run python scripts/evaluate.py --run-dir runs/stage1_seed0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from snake_rl.agents import DQNAgent
from snake_rl.config import EnvConfig
from snake_rl.env import SnakeEnv
from snake_rl.training.evaluate import evaluate_agent, format_eval
from snake_rl.training.logging import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy evaluation of a trained DQN checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=str, default="runs/stage1_seed0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"找不到 {config_path}；请先训练，或用 --run-dir 指向正确的目录")

    cfg = load_json(config_path)
    env_config = EnvConfig(**cfg["env"])

    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoint.pt"
    if not checkpoint.exists():
        raise SystemExit(f"找不到 checkpoint: {checkpoint}")

    agent = DQNAgent.load(checkpoint, device=args.device)
    env = SnakeEnv(env_config)

    renderer = None
    if args.render:
        from snake_rl.env.renderer import PygameRenderer

        renderer = PygameRenderer(env, caption="Snake DQN evaluation")

    try:
        result = evaluate_agent(
            agent,
            env,
            episodes=args.episodes,
            seed=args.seed,
            renderer=renderer,
            fps=args.fps,
        )
    finally:
        if renderer is not None:
            renderer.close()

    print(format_eval(result))
    output = Path(args.output) if args.output else run_dir / "eval.json"
    save_json(output, result.to_dict())
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
