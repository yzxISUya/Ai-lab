"""Greedy 评估。

训练时的 epsilon-greedy 表现不是评估成绩：它带着探索，而且受 epsilon 调度影响。
这里固定用 greedy 策略、固定评估种子、默认不渲染，得到可比较的分数。
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import asdict, dataclass

from snake_rl.agents.dqn import DQNAgent
from snake_rl.env import SnakeEnv


@dataclass
class EvalResult:
    episodes: int
    mean_score: float
    median_score: float
    max_score: int
    min_score: int
    std_score: float
    mean_steps: float
    score_per_step: float
    endings: dict[str, int]
    scores: list[int]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_agent(
    agent: DQNAgent,
    env: SnakeEnv,
    *,
    episodes: int = 100,
    seed: int = 10_000,
    renderer=None,
    fps: int = 30,
) -> EvalResult:
    """跑若干局 greedy 评估。

    ``seed`` 是评估种子的起点：第 i 局用 ``seed + i``。训练和评估使用不同的种子，
    避免把训练时见过的局面当成泛化成绩。
    """

    if episodes < 1:
        raise ValueError("episodes 至少为 1")

    scores: list[int] = []
    steps: list[int] = []
    endings: Counter[str] = Counter()

    for i in range(episodes):
        env.reset(seed=seed + i)
        obs = agent.observation(env)
        last_info: dict = {}

        while not env.done:
            action = agent.get_action(obs, greedy=True)
            outcome = env.step(action)
            obs = agent.observation(env)
            last_info = outcome.info

            if renderer is not None:
                renderer.draw(outcome.state)
                renderer.tick(fps)
                if renderer.should_quit():
                    raise KeyboardInterrupt

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

    total_steps = sum(steps)
    return EvalResult(
        episodes=episodes,
        mean_score=statistics.mean(scores),
        median_score=statistics.median(scores),
        max_score=max(scores),
        min_score=min(scores),
        std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
        mean_steps=statistics.mean(steps),
        score_per_step=(sum(scores) / total_steps) if total_steps else 0.0,
        endings=dict(endings),
        scores=scores,
    )


def format_eval(result: EvalResult) -> str:
    """把评估结果格式化成一行/多行可读文本。"""

    endings = " ".join(f"{k}={v}" for k, v in sorted(result.endings.items()))
    return (
        f"episodes={result.episodes}  "
        f"mean={result.mean_score:.2f}  median={result.median_score:.1f}  "
        f"max={result.max_score}  min={result.min_score}  std={result.std_score:.2f}  "
        f"steps={result.mean_steps:.1f}  score/step={result.score_per_step:.4f}  "
        f"endings[{endings}]"
    )
