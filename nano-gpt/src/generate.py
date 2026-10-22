"""与界面无关的生成核心，命令行和网页界面共用。

``stream_story`` 每采样一个 token 就 yield 一次，方便网页做流式显示；
``generate_story`` 则是它的非流式包装。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from transformers import GPT2LMHeadModel

from src import config
from src.tokenizer import EOT_ID, CharTokenizer


@dataclass(frozen=True)
class GenerationConfig:
    """单次生成使用的采样参数。"""

    max_new_tokens: int = config.MAX_NEW_TOKENS
    temperature: float = config.TEMPERATURE
    top_k: int = config.TOP_K
    top_p: float = config.TOP_P
    repetition_penalty: float = config.REPETITION_PENALTY


def _apply_repetition_penalty(
    logits: torch.Tensor, ids: list[int], penalty: float
) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    # 和 Hugging Face 的 RepetitionPenaltyLogitsProcessor 一致：
    # 出现过的 token，正 logit 除以 penalty，负 logit 乘以 penalty。
    tokens = torch.tensor(sorted(set(ids)), dtype=torch.long, device=logits.device)
    values = logits[tokens]
    logits = logits.clone()
    logits[tokens] = torch.where(values < 0, values * penalty, values / penalty)
    return logits


def _sample_next(logits: torch.Tensor, ids: list[int], gen: GenerationConfig) -> int:
    logits = _apply_repetition_penalty(logits.float(), ids, gen.repetition_penalty)

    if gen.temperature <= 0:
        return int(torch.argmax(logits))

    logits = logits / gen.temperature

    if gen.top_k > 0:
        k = min(gen.top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if 0 < gen.top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove_sorted = cumulative_probs > gen.top_p
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False
        remove = torch.zeros_like(remove_sorted)
        remove.scatter_(0, sorted_indices, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))


@torch.no_grad()
def stream_story(
    model: GPT2LMHeadModel,
    tokenizer: CharTokenizer,
    prompt: str,
    gen: GenerationConfig,
    device: str,
) -> Iterator[str]:
    """逐 token 续写，yield 新产生的文本（不含 prompt）。"""
    context_size = model.config.n_positions
    prompt_ids = tokenizer.encode(prompt) or [EOT_ID]
    ids = prompt_ids[-context_size:]
    past = None
    produced = 0

    while produced < gen.max_new_tokens:
        # 上下文满了就丢掉前一半并重算 KV cache，和原来的滑窗策略一致。
        if past is not None and len(ids) >= context_size:
            ids = ids[-max(1, context_size // 2) :]
            past = None

        if past is None:
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        else:
            input_ids = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)

        output = model(input_ids=input_ids, past_key_values=past, use_cache=True)
        past = output.past_key_values

        next_id = _sample_next(output.logits[0, -1], ids, gen)
        ids.append(next_id)
        produced += 1

        if next_id == EOT_ID:
            yield "\n"
            return
        yield tokenizer.decode([next_id])


def generate_story(
    model: GPT2LMHeadModel,
    tokenizer: CharTokenizer,
    prompt: str,
    gen: GenerationConfig,
    device: str,
) -> str:
    """非流式版本：返回 prompt + 新生成的文本。"""
    return prompt + "".join(stream_story(model, tokenizer, prompt, gen, device))
