r"""Stage 1 训练入口。

运行::

    .\.venv\Scripts\python.exe scripts\train_dqn.py --episodes 500
"""

from snake_rl.training.train_dqn import main

if __name__ == "__main__":
    main()
