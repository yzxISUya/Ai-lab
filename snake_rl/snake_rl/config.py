"""项目配置。

Stage 0 只有环境配置；Stage 1 会在这里加入 DQNConfig、训练配置等。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Snake 环境的全部可调参数。

    坐标系：左上角是 (0, 0)，x 向右，y 向下。
    棋盘有 ``width * height`` 个格子。
    """

    width: int = 20
    height: int = 20
    initial_length: int = 3

    # 连续多少步没吃到食物就截断 episode。
    # 实际上限 = factor * 当前蛇长；蛇越长，允许绕路的时间越多。
    max_steps_without_food_factor: int = 100

    food_reward: float = 10.0
    # 死亡这一步的总奖励 = step_reward + death_reward。
    death_reward: float = -10.0
    step_reward: float = 0.0
    timeout_reward: float = 0.0
    win_reward: float = 0.0

    def __post_init__(self) -> None:
        if self.width < 5 or self.height < 5:
            raise ValueError("width/height 至少为 5")
        if self.initial_length < 1:
            raise ValueError("initial_length 至少为 1")

        # 初始蛇水平向左延伸，所以必须放得下。
        max_length_that_fits = self.width // 2 + 1
        if self.initial_length > max_length_that_fits:
            raise ValueError(
                f"initial_length={self.initial_length} 放不进 width={self.width} "
                f"（最多 {max_length_that_fits}）"
            )
        if self.max_steps_without_food_factor < 1:
            raise ValueError("max_steps_without_food_factor 至少为 1")


@dataclass(frozen=True)
class DQNConfig:
    """Stage 1 的 DQN 超参数。

    这些默认值刻意接近参考项目：
    - 11 维观察、3 个动作；
    - 隐藏层 256；
    - lr=1e-3、gamma=0.9；
    - replay 容量 100k、batch 1000；
    - epsilon 从 0.4 线性降到 0，持续 80 局。

    ``bootstrap_truncated=False`` 表示 Stage 1 故意复刻参考项目的“把超时当终止”。
    Stage 2 把它改成 True，就能开始做正确的 bootstrap 消融。
    """

    obs_size: int = 11
    n_actions: int = 3
    hidden_size: int = 256
    lr: float = 1e-3
    gamma: float = 0.9
    batch_size: int = 1000
    max_memory: int = 100_000
    epsilon_start: float = 0.4
    epsilon_end: float = 0.0
    epsilon_decay_games: int = 80
    bootstrap_truncated: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.obs_size < 1:
            raise ValueError("obs_size 至少为 1")
        if self.n_actions < 1:
            raise ValueError("n_actions 至少为 1")
        if self.hidden_size < 1:
            raise ValueError("hidden_size 至少为 1")
        if self.lr <= 0:
            raise ValueError("lr 必须为正")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma 必须在 [0, 1]")
        if self.batch_size < 1:
            raise ValueError("batch_size 至少为 1")
        if self.max_memory < 1:
            raise ValueError("max_memory 至少为 1")
        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon_start 必须在 [0, 1]")
        if not 0.0 <= self.epsilon_end <= 1.0:
            raise ValueError("epsilon_end 必须在 [0, 1]")
        if self.epsilon_decay_games < 0:
            raise ValueError("epsilon_decay_games 不能为负")


@dataclass(frozen=True)
class TrainConfig:
    """训练循环的配置（与算法超参数分开）。"""

    episodes: int = 500
    eval_every: int = 50
    eval_episodes: int = 20
    checkpoint_every: int = 50
    log_every: int = 10
    final_eval_episodes: int = 100
    short_memory: bool = True
    render: bool = False
    render_fps: int = 30
    seed: int = 0
    output_dir: str = "runs/stage1_seed0"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes 至少为 1")
        if self.eval_every < 0 or self.checkpoint_every < 0:
            raise ValueError("eval_every/checkpoint_every 不能为负")
        if self.eval_episodes < 1 or self.final_eval_episodes < 1:
            raise ValueError("评估局数至少为 1")
        if self.log_every < 1:
            raise ValueError("log_every 至少为 1")
        if self.render_fps < 1:
            raise ValueError("render_fps 至少为 1")
