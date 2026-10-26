"""Stage 1：训练一个可复现、可评估的 DQN 基线。

算法上刻意贴近参考项目：
- 11 维观察；
- 在线网络算 target（没有目标网络）；
- MSE；
- 每步单样本更新 + 每局 replay batch 更新；
- epsilon 从 0.4 线性降到 0；
- 默认把超时当终止（``bootstrap_truncated=False``）。

工程上做了这些事：
- headless 环境，训练不受 pygame 帧率限制；
- 固定环境/动作/replay/网络初始化种子；
- CSV 日志、greedy 评估、checkpoint；
- 训练和评估使用不同种子。
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

from snake_rl.agents import DQNAgent, Transition
from snake_rl.config import DQNConfig, EnvConfig, TrainConfig
from snake_rl.env import SnakeEnv
from snake_rl.env.observations import observe
from snake_rl.training.evaluate import evaluate_agent, format_eval
from snake_rl.training.logging import CSVLogger, save_json

TRAIN_FIELDS = [
    "episode",
    "score",
    "steps",
    "mean100",
    "epsilon",
    "short_loss",
    "long_loss",
    "target_syncs",
    "wall",
    "self",
    "timeout",
    "win",
    "elapsed_s",
]

EVAL_FIELDS = [
    "episode",
    "episodes",
    "mean_score",
    "median_score",
    "max_score",
    "min_score",
    "std_score",
    "mean_steps",
    "score_per_step",
    "wall",
    "self",
    "timeout",
    "win",
]


def run_training(
    env_config: EnvConfig,
    dqn_config: DQNConfig,
    train_config: TrainConfig,
) -> dict:
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        output_dir / "config.json",
        {
            "env": asdict(env_config),
            "dqn": asdict(dqn_config),
            "train": asdict(train_config),
        },
    )

    env = SnakeEnv(env_config, seed=train_config.seed)
    eval_env = SnakeEnv(env_config, seed=train_config.seed + 10_000)
    agent = DQNAgent(dqn_config, device=train_config.device)

    renderer = None
    if train_config.render:
        from snake_rl.env.renderer import PygameRenderer

        renderer = PygameRenderer(env, caption="Snake DQN training")

    scores_window: deque[int] = deque(maxlen=100)
    total_steps = 0
    start_time = time.perf_counter()
    final_eval = None

    print("Stage 1 DQN baseline")
    print(f"  device: {agent.device}")
    print(f"  env:    {env_config}")
    print(f"  dqn:    {dqn_config}")
    print(f"  output: {output_dir}")
    print()

    try:
        with CSVLogger(output_dir / "train.csv", TRAIN_FIELDS) as train_logger, CSVLogger(
            output_dir / "eval.csv", EVAL_FIELDS
        ) as eval_logger:
            try:
                for episode in range(1, train_config.episodes + 1):
                    env.reset(seed=train_config.seed + episode)
                    obs = observe(env)
                    short_losses: list[float] = []
                    last_info: dict = {}
                    steps = 0

                    while not env.done:
                        action = agent.get_action(obs)
                        outcome = env.step(action)
                        next_obs = observe(env)

                        transition = Transition(
                            obs=obs,
                            action=action,
                            reward=outcome.reward,
                            next_obs=next_obs,
                            terminated=outcome.terminated,
                            truncated=outcome.truncated,
                        )

                        if train_config.short_memory:
                            short_losses.append(agent.train_short_memory(transition))
                        agent.remember(transition)

                        obs = next_obs
                        last_info = outcome.info
                        steps += 1
                        total_steps += 1

                        if renderer is not None:
                            renderer.draw(outcome.state)
                            renderer.tick(train_config.render_fps)
                            if renderer.should_quit():
                                raise KeyboardInterrupt

                    score = env.score
                    epsilon = agent.epsilon
                    agent.end_episode()
                    long_loss = agent.train_long_memory()

                    scores_window.append(score)
                    mean100 = statistics.mean(scores_window)

                    if last_info.get("win"):
                        ending = "win"
                    elif last_info.get("timeout"):
                        ending = "timeout"
                    elif "death_reason" in last_info:
                        ending = last_info["death_reason"]
                    else:
                        ending = "unknown"

                    elapsed = time.perf_counter() - start_time
                    short_loss = statistics.mean(short_losses) if short_losses else float("nan")
                    train_logger.log(
                        {
                            "episode": episode,
                            "score": score,
                            "steps": steps,
                            "mean100": mean100,
                            "epsilon": epsilon,
                            "short_loss": short_loss,
                            "long_loss": "" if long_loss is None else long_loss,
                            "target_syncs": agent.target_syncs,
                            "wall": int(ending == "wall"),
                            "self": int(ending == "self"),
                            "timeout": int(ending == "timeout"),
                            "win": int(ending == "win"),
                            "elapsed_s": elapsed,
                        }
                    )

                    if episode == 1 or episode % train_config.log_every == 0:
                        long_loss_str = "n/a" if long_loss is None else f"{long_loss:.4f}"
                        print(
                            f"[{episode:4d}/{train_config.episodes:4d}] "
                            f"score={score:2d} steps={steps:3d} mean100={mean100:5.2f} "
                            f"eps={epsilon:.3f} short={short_loss:.4f} "
                            f"long={long_loss_str} end={ending} "
                            f"t={elapsed:6.1f}s",
                            flush=True,
                        )

                    if train_config.eval_every > 0 and episode % train_config.eval_every == 0:
                        result = evaluate_agent(
                            agent,
                            eval_env,
                            episodes=train_config.eval_episodes,
                            seed=10_000,
                        )
                        eval_logger.log(
                            {
                                "episode": episode,
                                "episodes": result.episodes,
                                "mean_score": result.mean_score,
                                "median_score": result.median_score,
                                "max_score": result.max_score,
                                "min_score": result.min_score,
                                "std_score": result.std_score,
                                "mean_steps": result.mean_steps,
                                "score_per_step": result.score_per_step,
                                "wall": result.endings.get("wall", 0),
                                "self": result.endings.get("self", 0),
                                "timeout": result.endings.get("timeout", 0),
                                "win": result.endings.get("win", 0),
                            }
                        )
                        print(f"  eval @ {episode}: {format_eval(result)}", flush=True)

                    if (
                        train_config.checkpoint_every > 0
                        and episode % train_config.checkpoint_every == 0
                    ):
                        agent.save(output_dir / "checkpoint.pt")

            except KeyboardInterrupt:
                print("Interrupted; saving checkpoint...")
                agent.save(output_dir / "checkpoint.pt")
                raise

            # 训练结束：固定种子的最终评估
            final_eval = evaluate_agent(
                agent,
                eval_env,
                episodes=train_config.final_eval_episodes,
                seed=20_000,
            )
            save_json(output_dir / "final_eval.json", final_eval.to_dict())
            agent.save(output_dir / "checkpoint.pt")
    finally:
        if renderer is not None:
            renderer.close()

    elapsed = time.perf_counter() - start_time
    summary = {
        "output_dir": str(output_dir),
        "train_seconds": elapsed,
        "train_steps": total_steps,
        "updates": agent.updates,
        "target_syncs": agent.target_syncs,
        "steps_per_second": (total_steps / elapsed) if elapsed else 0.0,
        "final_eval": None if final_eval is None else final_eval.to_dict(),
        "env_config": asdict(env_config),
        "dqn_config": asdict(dqn_config),
        "train_config": asdict(train_config),
    }
    save_json(output_dir / "summary.json", summary)

    if final_eval is not None:
        print()
        print(f"final greedy eval: {format_eval(final_eval)}")
        print(f"train: {total_steps} steps in {elapsed:.1f}s ({total_steps / elapsed:.1f} steps/s)")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Stage 1 DQN baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 训练规模
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-short-memory", action="store_false", dest="short_memory",
                        help="关闭参考项目的 per-step 单样本更新")
    parser.add_argument("--render", action="store_true", help="训练时渲染（很慢，只用于观察）")
    parser.add_argument("--render-fps", type=int, default=30)

    # 环境
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--timeout-factor", type=int, default=100)
    parser.add_argument("--timeout-reward", type=float, default=-10.0,
                        help="参考项目在超时/死亡时都给 -10；Stage 1 默认复刻")

    # DQN
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-memory", type=int, default=100_000)
    parser.add_argument("--epsilon-start", type=float, default=0.4)
    parser.add_argument("--epsilon-end", type=float, default=0.0)
    parser.add_argument("--epsilon-decay-games", type=int, default=80)
    parser.add_argument("--bootstrap-truncated", action="store_true",
                        help="Stage 2 的正确做法：截断时仍 bootstrap（Stage 1 默认关闭）")
    parser.add_argument("--use-target-network", action="store_true",
                        help="Stage 2：用延迟更新的目标网络计算 TD target")
    parser.add_argument("--target-sync-every", type=int, default=1000,
                        help="每多少次 train_step 把在线网络同步到目标网络")
    parser.add_argument("--double-dqn", action="store_true",
                        help="Stage 2：在线网络选动作、目标网络评估（需要目标网络）")

    # 其他
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    env_config = EnvConfig(
        width=args.width,
        height=args.height,
        max_steps_without_food_factor=args.timeout_factor,
        timeout_reward=args.timeout_reward,
    )
    dqn_config = DQNConfig(
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        max_memory=args.max_memory,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_games=args.epsilon_decay_games,
        bootstrap_truncated=args.bootstrap_truncated,
        use_target_network=args.use_target_network,
        target_sync_every=args.target_sync_every,
        double_dqn=args.double_dqn,
        seed=args.seed,
    )
    output_dir = args.output_dir or f"runs/stage1_seed{args.seed}"
    train_config = TrainConfig(
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        final_eval_episodes=args.final_eval_episodes,
        short_memory=args.short_memory,
        render=args.render,
        render_fps=args.render_fps,
        seed=args.seed,
        output_dir=output_dir,
        device=args.device,
    )

    run_training(env_config, dqn_config, train_config)


if __name__ == "__main__":
    main()
