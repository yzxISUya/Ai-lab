"""从零构建一个小型 GPT-2 风格的因果语言模型（随机初始化，不加载预训练权重）。"""

from __future__ import annotations

from transformers import GPT2Config, GPT2LMHeadModel

from src import config
from src.tokenizer import EOT_ID


def build_model(
    vocab_size: int,
    block_size: int = config.BLOCK_SIZE,
    n_layer: int = config.N_LAYER,
    n_head: int = config.N_HEAD,
    n_embd: int = config.N_EMBD,
    dropout: float = config.DROPOUT,
) -> GPT2LMHeadModel:
    """默认参数下约 10.7M 参数（默认值见 ``src/config.py``）。

    n_layer   : Transformer 块的数量（模型深度）
    n_head    : 每个块的注意力头数；head_dim = n_embd / n_head
    n_embd    : 隐藏层宽度；必须能被 n_head 整除
    block_size: 最大上下文长度（可学习的绝对位置编码，与 GPT-2 一样）
    """
    if n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    model_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * n_embd,
        activation_function="gelu_new",
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        bos_token_id=EOT_ID,
        eos_token_id=EOT_ID,
        tie_word_embeddings=True,  # 输入嵌入与 lm_head 共享权重
        use_cache=False,  # 训练时用不到；关掉可以省显存
    )
    return GPT2LMHeadModel(model_config)
