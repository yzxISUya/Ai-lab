r"""高性能向量化 DQN 训练入口。

运行示例::

    # 本地或服务器快速测试 (10x10, 32 环境, 自适应池化 3x3)
    uv run python scripts/train_vector_dqn.py --total-steps 50000 --num-envs 32

    # L4 GPU 完整训练 (200,000 步, 批大小 512, 自适应池化)
    uv run python scripts/train_vector_dqn.py --total-steps 200000 --batch-size 512 --num-envs 32
"""

from snake_rl.training.train_vector_dqn import main

if __name__ == "__main__":
    main()
