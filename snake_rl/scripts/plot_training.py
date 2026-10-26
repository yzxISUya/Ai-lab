r"""读取训练日志并画曲线。

运行::

    uv run python scripts/plot_training.py --run-dir runs/stage1_seed0

输出默认是 ``runs/stage1_seed0/curve.png``。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DQN training curves",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=str, default="runs/stage1_seed0")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    train_path = run_dir / "train.csv"
    if not train_path.exists():
        raise SystemExit(f"找不到 {train_path}")

    rows = read_rows(train_path)
    if not rows:
        raise SystemExit("train.csv 为空")

    episodes = [int(float(r["episode"])) for r in rows]
    scores = [as_float(r, "score") for r in rows]
    mean100 = [as_float(r, "mean100") for r in rows]
    epsilon = [as_float(r, "epsilon") for r in rows]
    steps = [as_float(r, "steps") for r in rows]
    short_loss = [as_float(r, "short_loss") for r in rows]
    long_loss = [as_float(r, "long_loss") for r in rows]

    eval_path = run_dir / "eval.csv"
    eval_episodes: list[int] = []
    eval_means: list[float] = []
    if eval_path.exists():
        eval_rows = read_rows(eval_path)
        eval_episodes = [int(float(r["episode"])) for r in eval_rows]
        eval_means = [as_float(r, "mean_score") for r in eval_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"DQN training — {run_dir}")

    ax = axes[0, 0]
    ax.plot(episodes, scores, color="#9ecae1", alpha=0.45, linewidth=1, label="train score")
    ax.plot(episodes, mean100, color="#1f77b4", linewidth=2, label="train mean100")
    if eval_episodes:
        ax.plot(eval_episodes, eval_means, "o-", color="#d62728", linewidth=1.5,
                markersize=3, label="greedy eval mean")
    ax.set_title("Score")
    ax.set_xlabel("episode")
    ax.set_ylabel("score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(episodes, short_loss, color="#ff7f0e", alpha=0.5, linewidth=1,
            label="short / per-step loss")
    ax.plot(episodes, long_loss, color="#9467bd", alpha=0.8, linewidth=1,
            label="replay / episode loss")
    ax.set_title("TD loss")
    ax.set_xlabel("episode")
    ax.set_ylabel("TD loss (MSE / Huber)")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(episodes, epsilon, color="#2ca02c", linewidth=1.5)
    ax.set_title("Epsilon")
    ax.set_xlabel("episode")
    ax.set_ylabel("epsilon")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(episodes, steps, color="#7f7f7f", alpha=0.6, linewidth=1)
    ax.set_title("Episode length")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    output = Path(args.output) if args.output else run_dir / "curve.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
