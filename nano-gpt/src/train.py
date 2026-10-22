"""在准备好的 TinyStories token 文件上训练一个小型 GPT-2 风格模型。

用法：
    uv run nano-gpt-train                         # 1000 步冒烟测试（约 10 分钟）
    uv run nano-gpt-train --max-steps 200         # 快速 sanity 检查
    uv run nano-gpt-train --max-steps 6000        # 更长的训练

训练器会保存 checkpoint，另外还会写出一个自包含的 ``final/`` 目录
（模型权重 + tokenizer.json），``nano-gpt-sample`` 可以直接加载它。

所有默认值见 ``src/config.py``，用命令行参数可以覆盖。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from src import config
from src.data import TokenWindowDataset
from src.model import build_model
from src.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.RUN_DIR)
    parser.add_argument("--block-size", type=int, default=config.BLOCK_SIZE)
    parser.add_argument("--n-layer", type=int, default=config.N_LAYER)
    parser.add_argument("--n-head", type=int, default=config.N_HEAD)
    parser.add_argument("--n-embd", type=int, default=config.N_EMBD)
    parser.add_argument("--dropout", type=float, default=config.DROPOUT)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=config.GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--warmup-steps", type=int, default=config.WARMUP_STEPS)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--eval-steps", type=int, default=config.EVAL_STEPS)
    parser.add_argument("--eval-windows", type=int, default=config.EVAL_WINDOWS)
    parser.add_argument("--save-steps", type=int, default=config.SAVE_STEPS)
    parser.add_argument("--logging-steps", type=int, default=config.LOGGING_STEPS)
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=config.TRAIN_SEED)
    parser.add_argument("--fp32", action="store_true", help="禁用 bf16 混合精度")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_bin = args.data_dir / "train.bin"
    val_bin = args.data_dir / "val.bin"
    tokenizer_path = args.data_dir / "tokenizer.json"
    for path in (train_bin, val_bin, tokenizer_path):
        if not path.exists():
            raise SystemExit(f"missing {path}; run `uv run nano-gpt-prepare` first")

    tokenizer = CharTokenizer.load(tokenizer_path)
    meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))

    train_ds = TokenWindowDataset(train_bin, args.block_size, meta["dtype"])
    val_ds = TokenWindowDataset(val_bin, args.block_size, meta["dtype"])
    # 用固定的切片做评估：开销小，而且不同 run 之间可比。
    val_ds = torch.utils.data.Subset(val_ds, range(min(len(val_ds), args.eval_windows)))
    print(f"train windows: {len(train_ds):,} | eval windows: {len(val_ds):,}")

    model = build_model(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lr_scheduler_type=config.LR_SCHEDULER,
        max_grad_norm=config.MAX_GRAD_NORM,
        bf16=not args.fp32,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_on_start=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=args.workers,
        dataloader_persistent_workers=args.workers > 0,
        remove_unused_columns=False,
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # 开始训练
    result = trainer.train()
    print(f"train loss: {result.training_loss:.4f}")

    final_dir = args.out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save(final_dir / "tokenizer.json")
    run_config = vars(args) | {"n_params": n_params}
    (final_dir / "run_config.json").write_text(
        json.dumps({k: str(v) for k, v in run_config.items()}, indent=2), encoding="utf-8"
    )
    # 把预处理数据的统计信息一并保存到权重旁边，方便复现。
    shutil.copy2(args.data_dir / "meta.json", final_dir / "data_meta.json")
    print(f"saved final model -> {final_dir}")


if __name__ == "__main__":
    main()
