"""用训练好的模型把提示词续写成故事。

用法：
    uv run nano-gpt-sample --prompt "Once upon a time"
    uv run nano-gpt-sample --prompt "The little robot" --num-samples 5 --temperature 1.0

采样参数：
    temperature         越低越保守/越重复，越高越发散
    top-k               只保留概率最高的 k 个候选字符
    top-p               保留概率累计和达到 p 的最小字符集合
    repetition-penalty  大于 1 会抑制重复最近出现过的 token

默认值见 ``src/config.py``，用命令行参数可以覆盖。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from src import config
from src.generate import GenerationConfig, generate_story
from src.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    parser.add_argument("--prompt", type=str, default=config.PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--num-samples", type=int, default=config.NUM_SAMPLES)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--top-p", type=float, default=config.TOP_P)
    parser.add_argument("--repetition-penalty", type=float, default=config.REPETITION_PENALTY)
    parser.add_argument("--seed", type=int, default=config.SAMPLE_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.model_dir / "config.json").exists():
        raise SystemExit(f"no model in {args.model_dir}; run `uv run nano-gpt-train` first")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CharTokenizer.load(args.model_dir / "tokenizer.json")
    model = GPT2LMHeadModel.from_pretrained(args.model_dir).to(device).eval()
    model.config.use_cache = True
    print(
        f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
        f"context: {model.config.n_positions} tokens, device: {device}"
    )

    gen = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    for i in range(args.num_samples):
        torch.manual_seed(args.seed + i)
        text = generate_story(model, tokenizer, args.prompt, gen, device)
        print(f"\n{'=' * 20} sample {i + 1} {'=' * 20}\n{text.strip()}")


if __name__ == "__main__":
    main()
