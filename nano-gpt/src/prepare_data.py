"""下载 TinyStories，并转换成训练器可以用 memmap 读取的 token 文件。

输出（默认在 ``data/processed``）：
    tokenizer.json   字符词表
    train.bin        uint8 token id，每个字符一个
    val.bin          留出的验证集 token id（默认 2% 的故事）
    meta.json        数量统计与预处理设置

用法：
    uv run nano-gpt-prepare --max-chars 200000000

默认值见 ``src/config.py``，用命令行参数可以覆盖。
"""

from __future__ import annotations

import argparse # 命令行参数解析库
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from src import config
from src.tokenizer import EOT_ID, CharTokenizer


def download_parquet(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=config.DATASET_REPO_ID,
        filename=config.DATASET_FILENAME,
        repo_type="dataset",
        local_dir=str(raw_dir),
    )
    return Path(path)


def read_stories(parquet_path: Path, max_chars: int) -> list[str]:
    """读取故事，直到收集到大约 ``max_chars`` 个字符为止。"""
    parquet = pq.ParquetFile(parquet_path)
    stories: list[str] = []
    total = 0
    for batch in parquet.iter_batches(batch_size=8192, columns=["text"]):
        for text in batch.column(0).to_pylist():
            text = text.strip()
            if not text:
                continue
            stories.append(text)
            total += len(text)
            if len(stories) % 50_000 == 0:
                print(f"  read {len(stories):,} stories / {total / 1e6:.1f}M chars")
            if total >= max_chars:
                return stories
    return stories


def encode_stories(tokenizer: CharTokenizer, stories: list[str]) -> np.ndarray:
    """编码每条故事，并在其后追加 <|endoftext|>。"""
    total = sum(len(s) for s in stories) + len(stories) # 因为每条故事后面都要插入一个 EOT_ID，所以加 len(stories)
    ids = np.empty(total, dtype=tokenizer.dtype)
    lut = tokenizer.ascii_lut() # lut 是 lookup table 的缩写，即查找表，让字符的 ASCII 数值直接对应 Token ID，例如 a -> lut[97] -> 1
    pos = 0
    for story in stories:
        if lut is not None:
            codes = lut[np.frombuffer(story.encode("ascii"), dtype=np.uint8)]
        else:
            codes = np.asarray(tokenizer.encode(story), dtype=tokenizer.dtype)
        ids[pos : pos + len(codes)] = codes
        pos += len(codes)
        ids[pos] = EOT_ID
        pos += 1
    assert pos == total
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--max-chars", type=int, default=config.MAX_CHARS)
    parser.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=config.DATA_SEED)
    args = parser.parse_args()

    started = time.time()
    raw_dir = args.data_dir / "raw"
    out_dir = args.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] downloading {config.DATASET_REPO_ID} ...")
    parquet_path = download_parquet(raw_dir)
    print(f"      {parquet_path} ({parquet_path.stat().st_size / 1e6:.0f} MB)")

    print(f"[2/4] reading ~{args.max_chars / 1e6:.0f}M characters ...")
    stories = read_stories(parquet_path, args.max_chars)
    n_chars = sum(len(s) for s in stories)
    print(f"      {len(stories):,} stories / {n_chars / 1e6:.1f}M chars")

    print("[3/4] building character tokenizer ...")
    tokenizer = CharTokenizer.build(stories)
    print(f"      vocab_size={tokenizer.vocab_size} dtype={tokenizer.dtype}")

    # 划分前先打乱，让训练集和验证集的分布保持一致。
    random.Random(args.seed).shuffle(stories)
    n_val = max(1, int(len(stories) * args.val_fraction))
    train_stories, val_stories = stories[:-n_val], stories[-n_val:]

    train_ids = encode_stories(tokenizer, train_stories)
    val_ids = encode_stories(tokenizer, val_stories)

    print("[4/4] writing files ...")
    tokenizer.save(out_dir / "tokenizer.json")
    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")
    meta = {
        "repo_id": config.DATASET_REPO_ID,
        "max_chars": args.max_chars,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "vocab_size": tokenizer.vocab_size,
        "dtype": tokenizer.dtype.name,
        "stories": {"train": len(train_stories), "val": len(val_stories)},
        "tokens": {"train": int(len(train_ids)), "val": int(len(val_ids))},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"done in {time.time() - started:.1f}s -> {out_dir} "
        f"(train {len(train_ids) / 1e6:.1f}M tokens, val {len(val_ids) / 1e6:.1f}M)"
    )


if __name__ == "__main__":
    main()
