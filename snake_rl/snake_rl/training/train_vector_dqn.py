"""高性能向量化 DQN 训练主循环。

专为充分释放 GPU (如 NVIDIA L4) 算力设计：
- 并行环境采样：利用 VectorSnakeEnv 同步推进 N 个环境（默认 N=32）；
- 显存张量回放池：TensorReplayBuffer 预分配连续空间，uint8 压缩存储；
- 批处理动作决策：BatchedDQNAgent 单次 GPU 前向推理决策 N 个动作；
- 原生 AMP 混合精度 (Bfloat16)；
- 统计真实的吞吐性能 (Steps/sec, FPS) 并记录曲线。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from snake_rl.agents.batched_dqn import BatchedDQNAgent
from snake_rl.agents.tensor_replay_buffer import TensorReplayBuffer
from snake_rl.config import DQNConfig, EnvConfig, TrainConfig
from snake_rl.env.snake_env import SnakeEnv
from snake_rl.env.vector_env import VectorSnakeEnv
from snake_rl.training.evaluate import evaluate_agent, format_eval
from snake_rl.training.logging import CSVLogger, save_json


def run_vector_training(
    env_config: EnvConfig,
    dqn_config: DQNConfig,
    num_envs: int = 32,
    total_steps: int = 200_000,
    update_every: int = 4,
    updates_per_step: int = 1,
    min_replay_size: int = 2000,
    eval_every_steps: int = 10_000,
    eval_episodes: int = 20,
    final_eval_episodes: int = 100,
    output_dir: str = "runs/vector_board_10x10",
    device: str = "auto",
    use_amp: bool = True,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # 1. 向量化并行环境
    venv = VectorSnakeEnv(
        env_config,
        num_envs=num_envs,
        seed=dqn_config.seed,
        obs_type=dqn_config.obs_type,
        board_channels=dqn_config.board_channels,
        as_uint8=True,
    )

    # 2. 独立单环境评测器
    eval_env = SnakeEnv(env_config, seed=dqn_config.seed + 10_000)

    # 3. 显存张量回放池
    buffer = TensorReplayBuffer(
        capacity=dqn_config.max_memory,
        obs_shape=venv.obs_shape,
        device=dev,
        as_uint8=True,
    )

    # 4. 批处理智能体
    agent = BatchedDQNAgent(
        dqn_config,
        device=dev,
        use_amp=use_amp,
    )

    save_json(
        out_dir / "config.json",
        {
            "env": asdict(env_config),
            "dqn": asdict(dqn_config),
            "vector": {
                "num_envs": num_envs,
                "total_steps": total_steps,
                "update_every": update_every,
                "updates_per_step": updates_per_step,
                "min_replay_size": min_replay_size,
                "device": str(dev),
                "use_amp": agent.use_amp,
            },
        },
    )

    train_fields = [
        "step",
        "episodes",
        "mean100",
        "epsilon",
        "loss",
        "fps",
        "target_syncs",
        "replay_updates",
        "elapsed_s",
    ]
    eval_fields = [
        "step",
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
    ]

    print("=== 高性能向量化 DQN 训练开始 ===")
    print(f"  设备           : {dev} (AMP Bfloat16: {agent.use_amp})")
    print(f"  并行环境数     : {num_envs}")
    print(f"  总环境步数     : {total_steps}")
    print(f"  批大小         : {dqn_config.batch_size}")
    print(f"  回放池容量     : {dqn_config.max_memory} (显存预分配 uint8)")
    print(f"  自适应池化     : {dqn_config.adaptive_pool_size}")
    print(f"  输出目录       : {out_dir}")
    print()

    obs = venv.reset()
    recent_scores = deque(maxlen=100)
    total_episodes = 0
    step_count = 0
    t0 = time.perf_counter()
    last_log_time = t0
    last_log_step = 0
    losses_window = deque(maxlen=50)

    # 探索率线性衰减计算
    decay_steps = dqn_config.epsilon_decay_games * 50  # 约合多少步

    with CSVLogger(out_dir / "train.csv", train_fields) as train_logger, CSVLogger(
        out_dir / "eval.csv", eval_fields
    ) as eval_logger:
        vector_steps = total_steps // num_envs
        for v_step in range(1, vector_steps + 1):
            # 1. 动态 Epsilon
            progress = min(1.0, step_count / max(1, decay_steps))
            epsilon = (
                dqn_config.epsilon_start
                - progress * (dqn_config.epsilon_start - dqn_config.epsilon_end)
            )

            # 2. 向量化并行动作决策
            actions = agent.select_actions(obs, epsilon=epsilon)

            # 3. 向量化并行步进
            next_obs, rewards, terms, truncs, infos = venv.step(actions)
            step_count += num_envs

            # 4. 批量直写张量回放池
            buffer.add_batch(obs, actions, rewards, next_obs, terms, truncs)
            obs = next_obs

            # 收集结束的回合统计
            for info in infos:
                if "episode_score" in info:
                    total_episodes += 1
                    recent_scores.append(info["episode_score"])

            # 5. 梯度反向传播
            effective_min_replay = max(min_replay_size, dqn_config.batch_size)
            if len(buffer) >= effective_min_replay and v_step % update_every == 0:
                for _ in range(updates_per_step):
                    batch = buffer.sample(dqn_config.batch_size)
                    loss_val = agent.train_step(batch)
                    losses_window.append(loss_val)

            # 6. 定期日志打印
            if v_step % 25 == 0 or v_step == vector_steps:
                now = time.perf_counter()
                elapsed = now - t0
                fps = (step_count - last_log_step) / (now - last_log_time)
                last_log_step = step_count
                last_log_time = now

                mean_loss = float(np.mean(losses_window)) if losses_window else 0.0
                mean100 = float(np.mean(recent_scores)) if recent_scores else 0.0

                train_logger.log(
                    {
                        "step": step_count,
                        "episodes": total_episodes,
                        "mean100": mean100,
                        "epsilon": epsilon,
                        "loss": mean_loss,
                        "fps": fps,
                        "target_syncs": agent.target_syncs,
                        "replay_updates": agent.updates,
                        "elapsed_s": elapsed,
                    }
                )

                print(
                    f"[{step_count:7d}/{total_steps:7d}] "
                    f"eps={total_episodes:4d} mean100={mean100:5.2f} "
                    f"loss={mean_loss:6.3f} eps_rate={epsilon:.3f} "
                    f"FPS={fps:6.1f} updates={agent.updates:5d} t={elapsed:5.1f}s",
                    flush=True,
                )

            # 7. 定期独立评测
            if (
                eval_every_steps > 0
                and step_count % eval_every_steps < num_envs
                and step_count > 0
            ):
                # 评测使用兼容的单体智能体包装
                from snake_rl.agents.dqn import DQNAgent

                temp_agent = DQNAgent(dqn_config, device=dev)
                temp_agent.model.load_state_dict(agent.model.state_dict())
                res = evaluate_agent(
                    temp_agent, eval_env, episodes=eval_episodes, seed=10_000
                )
                eval_logger.log(
                    {
                        "step": step_count,
                        "episodes": res.episodes,
                        "mean_score": res.mean_score,
                        "median_score": res.median_score,
                        "max_score": res.max_score,
                        "min_score": res.min_score,
                        "std_score": res.std_score,
                        "mean_steps": res.mean_steps,
                        "score_per_step": res.score_per_step,
                        "wall": res.endings.get("wall", 0),
                        "self": res.endings.get("self", 0),
                        "timeout": res.endings.get("timeout", 0),
                    }
                )
                print(f"  >>> Eval @ {step_count} 步: {format_eval(res)}", flush=True)

    # 最终评测
    print("\n运行最终贪心评测...")
    final_agent = DQNAgent(dqn_config, device=dev)
    final_agent.model.load_state_dict(agent.model.state_dict())
    final_res = evaluate_agent(
        final_agent, eval_env, episodes=final_eval_episodes, seed=20_000
    )
    print("最终评测结果 (20k seed):", format_eval(final_res))

    agent.save(out_dir / "checkpoint.pt")
    save_json(out_dir / "final_eval.json", final_res.to_dict())

    total_time = time.perf_counter() - t0
    summary = {
        "output_dir": str(out_dir),
        "total_steps": step_count,
        "total_episodes": total_episodes,
        "train_seconds": total_time,
        "steps_per_second": step_count / total_time,
        "updates": agent.updates,
        "target_syncs": agent.target_syncs,
        "final_eval": final_res.to_dict(),
    }
    save_json(out_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="高性能向量化并行 Snake DQN 训练")
    p.add_argument("--num-envs", type=int, default=32, help="并行环境数量 (默认: 32)")
    p.add_argument("--total-steps", type=int, default=200_000, help="总环境步数 (默认: 200,000)")
    p.add_argument("--width", type=int, default=10, help="棋盘宽度 (默认: 10)")
    p.add_argument("--height", type=int, default=10, help="棋盘高度 (默认: 10)")
    p.add_argument("--timeout-factor", type=float, default=2.0, help="步数限制因子 (默认: 2.0)")
    p.add_argument("--timeout-reward", type=float, default=-0.5, help="超时惩罚 (默认: -0.5)")
    p.add_argument("--obs-type", choices=["board", "features11"], default="board", help="观察空间类型")
    p.add_argument("--board-channels", type=int, default=5, choices=[3, 4, 5], help="棋盘通道数 (默认: 5)")
    p.add_argument("--adaptive-pool-size", type=int, default=3, help="自适应池化尺寸 (3 表示 3x3，<=0 表示禁用)")
    p.add_argument("--hidden-size", type=int, default=512, help="全连接隐藏层维度 (默认: 512)")
    p.add_argument("--batch-size", type=int, default=512, help="GPU 梯度批大小 (默认: 512)")
    p.add_argument("--lr", type=float, default=1e-4, help="学习率 (默认: 1e-4)")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子 (默认: 0.99)")
    p.add_argument("--max-memory", type=int, default=200_000, help="张量回放池容量 (默认: 200,000)")
    p.add_argument("--update-every", type=int, default=4, help="每多少个向量步执行梯度更新 (默认: 4)")
    p.add_argument("--updates-per-step", type=int, default=1, help="每次更新迭代次数 (默认: 1)")
    p.add_argument("--min-replay-size", type=int, default=2000, help="开始训练前的最小预热样本数 (默认: 2000)")
    p.add_argument("--epsilon-start", type=float, default=1.0, help="起始探索率 (默认: 1.0)")
    p.add_argument("--epsilon-end", type=float, default=0.05, help="最终探索率 (默认: 0.05)")
    p.add_argument("--epsilon-decay-games", type=int, default=200, help="探索衰减等效回合数 (默认: 200)")
    p.add_argument("--target-sync-every", type=int, default=1000, help="目标网络同步间隔步数 (默认: 1000)")
    p.add_argument("--eval-every-steps", type=int, default=10_000, help="评测周期步数 (默认: 10,000)")
    p.add_argument("--eval-episodes", type=int, default=20, help="中期评测回合数 (默认: 20)")
    p.add_argument("--final-eval-episodes", type=int, default=100, help="最终评测回合数 (默认: 100)")
    p.add_argument("--output-dir", type=str, default=None, help="输出目录 (留空自动生成)")
    p.add_argument("--device", type=str, default="auto", help="计算设备 (auto/cuda/cpu)")
    p.add_argument("--no-amp", action="store_true", help="禁用 Bfloat16 AMP 自动混合精度")
    p.add_argument("--seed", type=int, default=0, help="随机种子 (默认: 0)")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)

    pool_size = (args.adaptive_pool_size, args.adaptive_pool_size) if args.adaptive_pool_size > 0 else None

    env_config = EnvConfig(
        width=args.width,
        height=args.height,
        max_steps_without_food_factor=args.timeout_factor,
        timeout_reward=args.timeout_reward,
    )
    dqn_config = DQNConfig(
        obs_type=args.obs_type,
        board_channels=args.board_channels,
        board_height=args.height,
        board_width=args.width,
        adaptive_pool_size=pool_size,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        max_memory=args.max_memory,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_games=args.epsilon_decay_games,
        bootstrap_truncated=True,
        use_target_network=True,
        target_sync_every=args.target_sync_every,
        double_dqn=True,
        huber=True,
        seed=args.seed,
    )

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"runs/vector_{args.obs_type}_{args.width}x{args.height}_seed{args.seed}"

    return run_vector_training(
        env_config=env_config,
        dqn_config=dqn_config,
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        update_every=args.update_every,
        updates_per_step=args.updates_per_step,
        min_replay_size=args.min_replay_size,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        final_eval_episodes=args.final_eval_episodes,
        output_dir=output_dir,
        device=args.device,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
