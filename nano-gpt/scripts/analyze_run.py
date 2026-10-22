"""从 Trainer 的 run 目录中打印一份紧凑的 loss 表。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="checkpoints/run-2k")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        # transformers 5 把完整的历史记录保存在每个 checkpoint 目录里。
        checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if not checkpoints:
            raise SystemExit(f"no trainer_state.json or checkpoint-* in {run_dir}")
        state_path = checkpoints[-1] / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state["log_history"]
    evals = {entry["step"]: entry["eval_loss"] for entry in history if "eval_loss" in entry}
    trains = [entry for entry in history if "loss" in entry]

    print(f"{'step':>6} {'train':>8} {'eval':>8} {'ppl':>7} {'lr':>10}")
    for entry in trains:
        step = entry["step"]
        eval_loss = evals.get(step)
        eval_col = f"{eval_loss:.4f}" if eval_loss is not None else ""
        ppl_col = f"{math.exp(eval_loss):.2f}" if eval_loss is not None else ""
        print(
            f"{step:>6} {entry['loss']:>8.4f} {eval_col:>8} {ppl_col:>7} "
            f"{entry.get('learning_rate', 0):>10.2e}"
        )

    print("\neval points:", {step: round(loss, 4) for step, loss in evals.items()})
    if evals:
        first, last = next(iter(evals.values())), list(evals.values())[-1]
        print(f"eval loss: {first:.4f} -> {last:.4f} | perplexity: {math.exp(last):.2f} chars")


if __name__ == "__main__":
    main()
