r"""多 Seed 并行训练调度器 (充分利用 GPU 闲置算力)。

在单卡 (如 NVIDIA L4 24GB) 上并发启动多个随机种子 (Seed) 独立训练：
- 4 个种子进程同时并发执行，互相隔离输出目录；
- 算力压榨：显存占用 ~2.8GB/24GB，GPU 利用率大幅提高；
- 科学评测：训练结束后自动聚合多种子统计指标 (均值、标准差、最佳值)；
- 自动绘制带有置信区间 (阴影带) 的学术级对比曲线。

运行示例::

    # 本地或 Colab 4 个 seed 并行跑 80 万步
    python scripts/train_multiseed.py --seeds 0 1 2 3 --total-steps 800000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多 Seed 并行向量化 DQN 训练调度器")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3], help="要并行运行的种子列表 (默认: 0 1 2 3)")
    p.add_argument("--num-envs", type=int, default=32, help="每个进程的并行环境数 (默认: 32)")
    p.add_argument("--total-steps", type=int, default=800_000, help="每个种子训练的总步数 (默认: 800,000)")
    p.add_argument("--width", type=int, default=10, help="棋盘宽度")
    p.add_argument("--height", type=int, default=10, help="棋盘高度")
    p.add_argument("--timeout-factor", type=float, default=50.0, help="步数限制因子 (默认: 50.0)")
    p.add_argument("--timeout-reward", type=float, default=-10.0, help="超时惩罚")
    p.add_argument("--obs-type", choices=["board", "features11"], default="board")
    p.add_argument("--board-channels", type=int, default=5)
    p.add_argument("--adaptive-pool-size", type=int, default=3)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--max-memory", type=int, default=200_000)
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--updates-per-step", type=int, default=2)
    p.add_argument("--min-replay-size", type=int, default=2000)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    p.add_argument("--target-sync-every", type=int, default=500)
    p.add_argument("--eval-every-steps", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--final-eval-episodes", type=int, default=100)
    p.add_argument("--output-base", type=str, default="runs/vector_board_10x10", help="输出目录前缀")
    p.add_argument("--init-checkpoint", type=str, default=None, help="预训练权重路径 (用于迁移/微调)")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def plot_multiseed_results(seed_dirs: list[Path], output_path: Path) -> None:
    """绘制带标准差阴影的多 Seed 训练与评测曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 绘制 Eval 曲线
    eval_data: dict[int, list[float]] = {}
    for sdir in seed_dirs:
        eval_csv = sdir / "eval.csv"
        if not eval_csv.exists():
            continue
        with eval_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row["step"])
                score = float(row["mean_score"])
                if step not in eval_data:
                    eval_data[step] = []
                eval_data[step].append(score)

    if eval_data:
        steps = sorted(eval_data.keys())
        means = [float(np.mean(eval_data[s])) for s in steps]
        stds = [float(np.std(eval_data[s])) for s in steps]
        mins = [float(np.min(eval_data[s])) for s in steps]
        maxs = [float(np.max(eval_data[s])) for s in steps]

        axes[0].plot(steps, means, color="#1f77b4", lw=2, label="Mean Eval Score")
        axes[0].fill_between(
            steps,
            np.maximum(0, np.array(means) - np.array(stds)),
            np.array(means) + np.array(stds),
            color="#1f77b4",
            alpha=0.25,
            label="±1 Std Dev",
        )
        axes[0].fill_between(
            steps, mins, maxs, color="#1f77b4", alpha=0.1, label="Min-Max Range"
        )
        axes[0].set_title("Greedy Evaluation Score across Seeds")
        axes[0].set_xlabel("Environment Steps")
        axes[0].set_ylabel("Score")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    # 2. 绘制各 Seed 训练 Loss
    for sdir in seed_dirs:
        train_csv = sdir / "train.csv"
        if not train_csv.exists():
            continue
        steps, losses = [], []
        with train_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
        if steps:
            seed_name = sdir.name.split("_")[-1]
            axes[1].plot(steps, losses, alpha=0.6, label=f"{seed_name}")

    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Environment Steps")
    axes[1].set_ylabel("Huber Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📊 多 Seed 综合曲线已保存至: {output_path}")


def main() -> None:
    args = parse_args()
    seeds = args.seeds
    num_runs = len(seeds)

    print("=" * 70)
    print(f"🚀 启动多 Seed 并行训练调度器 (并发数: {num_runs})")
    print(f"   Seeds          : {seeds}")
    print(f"   总环境步数/Seed: {args.total_steps:,} 步 (合计 {args.total_steps * num_runs:,} 步)")
    print(f"   设备           : {args.device}")
    print("=" * 70)

    processes: list[subprocess.Popen] = []
    log_files: list[Any] = []
    seed_dirs: list[Path] = []

    t0 = time.perf_counter()

    for seed in seeds:
        out_dir = Path(f"{args.output_base}_seed{seed}")
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_dirs.append(out_dir)

        log_path = out_dir / "train.log"
        log_file = log_path.open("w", encoding="utf-8")
        log_files.append(log_file)

        cmd = [
            sys.executable,
            "-u",
            "scripts/train_vector_dqn.py",
            "--seed", str(seed),
            "--num-envs", str(args.num_envs),
            "--total-steps", str(args.total_steps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--timeout-factor", str(args.timeout_factor),
            "--timeout-reward", str(args.timeout_reward),
            "--obs-type", args.obs_type,
            "--board-channels", str(args.board_channels),
            "--adaptive-pool-size", str(args.adaptive_pool_size),
            "--hidden-size", str(args.hidden_size),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--gamma", str(args.gamma),
            "--max-memory", str(args.max_memory),
            "--update-every", str(args.update_every),
            "--updates-per-step", str(args.updates_per_step),
            "--min-replay-size", str(args.min_replay_size),
            "--epsilon-start", str(args.epsilon_start),
            "--epsilon-end", str(args.epsilon_end),
            "--epsilon-decay-steps", str(args.epsilon_decay_steps),
            "--target-sync-every", str(args.target_sync_every),
            "--eval-every-steps", str(args.eval_every_steps),
            "--eval-episodes", str(args.eval_episodes),
            "--final-eval-episodes", str(args.final_eval_episodes),
            "--output-dir", str(out_dir),
            "--device", args.device,
        ]
        if args.init_checkpoint:
            cmd.extend(["--init-checkpoint", str(args.init_checkpoint)])

        # 启动后台独立进程
        p = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "2"},
        )
        processes.append(p)
        print(f"  [PID {p.pid:6d}] 启动 Seed {seed} -> {out_dir}")

    print("\n所有进程已启动，进入进度监控 (每 10 秒刷新一次)...\n")

    # 循环监控
    try:
        while True:
            all_done = True
            status_lines = []
            for i, (seed, p, sdir) in enumerate(zip(seeds, processes, seed_dirs)):
                poll = p.poll()
                if poll is None:
                    all_done = False
                    # 读取最新进度
                    train_csv = sdir / "train.csv"
                    latest_step = 0
                    latest_fps = 0.0
                    latest_mean = 0.0
                    if train_csv.exists():
                        try:
                            with train_csv.open("r", encoding="utf-8") as f:
                                rows = list(csv.DictReader(f))
                                if rows:
                                    last = rows[-1]
                                    latest_step = int(last.get("step", 0))
                                    latest_fps = float(last.get("fps", 0.0))
                                    latest_mean = float(last.get("mean100", 0.0))
                        except Exception:
                            pass
                    pct = (latest_step / max(1, args.total_steps)) * 100
                    status_lines.append(
                        f"Seed {seed}: {pct:5.1f}% ({latest_step:6d}/{args.total_steps}) "
                        f"FPS={latest_fps:5.0f} Mean100={latest_mean:4.1f}"
                    )
                else:
                    status_lines.append(f"Seed {seed}: 完成 (退出码 {poll})")

            elapsed = time.perf_counter() - t0
            print(f"[{elapsed:5.1f}s] " + " | ".join(status_lines), flush=True)

            if all_done:
                break
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n捕获中断信号，正在终止所有子进程...")
        for p in processes:
            p.terminate()
        sys.exit(1)
    finally:
        for f in log_files:
            f.close()

    total_time = time.perf_counter() - t0
    total_env_steps = args.total_steps * num_runs
    effective_fps = total_env_steps / total_time

    print("\n" + "=" * 70)
    print(f"🎉 所有 {num_runs} 个 Seed 训练完成！总耗时: {total_time:.1f}s ({total_time/60:.2f} 分钟)")
    print(f"⚡ 综合有效吞吐率: {effective_fps:,.1f} steps/s (总完成步数: {total_env_steps:,})")
    print("=" * 70)

    # 收集评测结果并打印汇总表
    eval_results: list[dict[str, Any]] = []
    for seed, sdir in zip(seeds, seed_dirs):
        final_json = sdir / "final_eval.json"
        if final_json.exists():
            with final_json.open("r", encoding="utf-8") as f:
                res = json.load(f)
                res["seed"] = seed
                eval_results.append(res)

    if eval_results:
        print("\n### 最终 100 局 Greedy 评测多 Seed 汇总表\n")
        print("| Seed | Eval Mean | Median | Max | Min | Std | Steps | 撞自身(%) | 撞墙(%) | 超时(%) |")
        print("| :--- | --------: | -----: | --: | --: | --: | ----: | --------: | ------: | ------: |")

        means = []
        maxs = []
        for r in eval_results:
            seed = r["seed"]
            mean_s = r["mean_score"]
            means.append(mean_s)
            max_s = r["max_score"]
            maxs.append(max_s)
            med = r.get("median_score", 0.0)
            min_s = r.get("min_score", 0)
            std_s = r.get("std_score", 0.0)
            steps = r.get("mean_steps", 0.0)
            endings = r.get("endings", {})
            self_pct = endings.get("self", 0)
            wall_pct = endings.get("wall", 0)
            timeout_pct = endings.get("timeout", 0)
            print(
                f"| seed {seed} | **{mean_s:.2f}** | {med:.1f} | {max_s} | {min_s} | "
                f"{std_s:.2f} | {steps:.1f} | {self_pct}% | {wall_pct}% | {timeout_pct}% |"
            )

        mean_of_means = float(np.mean(means))
        std_of_means = float(np.std(means))
        print(f"| **跨 Seed 均值 ± SD** | **{mean_of_means:.2f} ± {std_of_means:.2f}** | — | **{max(maxs)}** | — | — | — | — | — | — |")

    # 绘图
    plot_path = Path(args.output_base).parent / "multiseed_curve.png"
    plot_multiseed_results(seed_dirs, plot_path)


if __name__ == "__main__":
    main()
