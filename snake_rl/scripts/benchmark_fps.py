"""基准测试脚本：对比传统串行 DQN 与新向量化架构的吞吐性能 (FPS / Steps per Second)。

运行示例::

    .venv/bin/python scripts/benchmark_fps.py --steps 3000
"""

from __future__ import annotations

import argparse
import time

import torch

from snake_rl.agents.batched_dqn import BatchedDQNAgent
from snake_rl.agents.dqn import DQNAgent, Transition
from snake_rl.agents.tensor_replay_buffer import TensorReplayBuffer
from snake_rl.config import DQNConfig, EnvConfig
from snake_rl.env.snake_env import SnakeEnv
from snake_rl.env.vector_env import VectorSnakeEnv


def benchmark_serial(
    steps: int = 2000,
    width: int = 10,
    height: int = 10,
    device: str = "cpu",
) -> tuple[float, float]:
    """测试旧版单环境串行步进与训练的 FPS。"""
    env_cfg = EnvConfig(width=width, height=height)
    dqn_cfg = DQNConfig(
        obs_type="board",
        board_channels=5,
        board_width=width,
        board_height=height,
        batch_size=64,
        max_memory=10_000,
    )
    dev = torch.device(device)
    env = SnakeEnv(env_cfg, seed=42)
    agent = DQNAgent(dqn_cfg, device=dev)

    env.reset(seed=42)
    obs = agent.observation(env)
    t0 = time.perf_counter()
    for s in range(1, steps + 1):
        action = agent.get_action(obs)
        outcome = env.step(action)
        next_obs = agent.observation(env)
        agent.remember(
            Transition(
                obs=obs,
                action=action,
                reward=outcome.reward,
                next_obs=next_obs,
                terminated=outcome.terminated,
                truncated=outcome.truncated,
            )
        )
        if outcome.terminated or outcome.truncated:
            env.reset()
            obs = agent.observation(env)
        else:
            obs = next_obs

        if len(agent.memory) >= 200 and s % 4 == 0:
            agent.train_long_memory()

    elapsed = time.perf_counter() - t0
    fps = steps / elapsed
    return fps, elapsed


def benchmark_vector(
    num_envs: int = 32,
    steps: int = 2000,
    width: int = 10,
    height: int = 10,
    adaptive_pool: bool = True,
    device: str = "cpu",
) -> tuple[float, float]:
    """测试新版向量化并行架构的 FPS。"""
    env_cfg = EnvConfig(width=width, height=height)
    dqn_cfg = DQNConfig(
        obs_type="board",
        board_channels=5,
        board_width=width,
        board_height=height,
        adaptive_pool_size=(3, 3) if adaptive_pool else None,
        batch_size=512,
        max_memory=10_000,
    )
    dev = torch.device(device)
    venv = VectorSnakeEnv(env_cfg, num_envs=num_envs, seed=42, as_uint8=True)
    buffer = TensorReplayBuffer(10_000, obs_shape=venv.obs_shape, device=dev, as_uint8=True)
    agent = BatchedDQNAgent(dqn_cfg, device=dev, use_amp=True)

    obs = venv.reset()
    t0 = time.perf_counter()
    total_steps = 0
    v_steps = (steps + num_envs - 1) // num_envs

    for vs in range(1, v_steps + 1):
        actions = agent.select_actions(obs, epsilon=0.1)
        next_obs, rewards, terms, truncs, infos = venv.step(actions)
        buffer.add_batch(obs, actions, rewards, next_obs, terms, truncs)
        obs = next_obs
        total_steps += num_envs

        if len(buffer) >= 200 and vs % 4 == 0:
            batch = buffer.sample(min(len(buffer), dqn_cfg.batch_size))
            agent.train_step(batch)

    elapsed = time.perf_counter() - t0
    fps = total_steps / elapsed
    return fps, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Snake RL 架构性能基准测试")
    parser.add_argument("--steps", type=int, default=2000, help="基准步数 (默认: 2000)")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cuda/cpu)")
    args = parser.parse_args()

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    print("=" * 60)
    print(f"Snake RL 吞吐性能基准测试 (Device: {dev})")
    print("=" * 60)

    # 1. 串行基线
    print("\n[1/3] 正在测试 传统串行单环境架构 (N=1)...")
    fps_serial, time_serial = benchmark_serial(steps=args.steps, device=dev)
    print(f"  -> 串行架构: {fps_serial:7.1f} steps/s (耗时: {time_serial:.2f}s)")

    # 2. 向量化架构 (N=8)
    print("\n[2/3] 正在测试 新向量化架构 (N=8)...")
    fps_v8, time_v8 = benchmark_vector(num_envs=8, steps=args.steps, device=dev)
    print(f"  -> 向量化 N=8: {fps_v8:7.1f} steps/s (加速比: {fps_v8/fps_serial:.2f}x)")

    # 3. 向量化架构 (N=32)
    print("\n[3/3] 正在测试 新向量化架构 (N=32, 推荐配置)...")
    fps_v32, time_v32 = benchmark_vector(num_envs=32, steps=args.steps, device=dev)
    print(f"  -> 向量化 N=32: {fps_v32:7.1f} steps/s (加速比: {fps_v32/fps_serial:.2f}x)")

    print("\n" + "=" * 60)
    print("性能总结:")
    print(f"  - 串行基线 (N=1)  : {fps_serial:7.1f} steps/s (1.00x)")
    print(f"  - 向量并行 (N=8)  : {fps_v8:7.1f} steps/s ({fps_v8/fps_serial:.2f}x)")
    print(f"  - 向量并行 (N=32) : {fps_v32:7.1f} steps/s ({fps_v32/fps_serial:.2f}x)")
    print("=" * 60)


if __name__ == "__main__":
    main()
