"""发现 ``checkpoints/`` 下的模型，并按需加载、缓存。"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from src import config
from src.tokenizer import CharTokenizer

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


@dataclass(frozen=True)
class ModelInfo:
    label: str
    path: Path
    tokenizer_path: Path
    block_size: int
    n_params: int
    max_steps: int | None
    modified: float


def _find_tokenizer(model_dir: Path) -> Path | None:
    """优先用模型自带的 tokenizer，最后回退到预处理目录。"""
    candidates = (
        model_dir / "tokenizer.json",
        model_dir.parent / "final" / "tokenizer.json",
        config.PROCESSED_DIR / "tokenizer.json",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _count_params(model_config: dict) -> int:
    """按 GPT-2 的结构计算参数量，不加载权重。"""
    vocab = int(model_config["vocab_size"])
    embd = int(model_config["n_embd"])
    positions = int(model_config["n_positions"])
    layers = int(model_config["n_layer"])
    inner = model_config.get("n_inner") or 4 * embd

    total = vocab * embd + positions * embd
    total += layers * (4 * embd * embd + 9 * embd + 2 * embd * inner + inner)
    total += 2 * embd
    if not model_config.get("tie_word_embeddings", True):
        total += vocab * embd
    return total


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def discover_models(checkpoints_dir: Path | str = config.CHECKPOINT_DIR) -> list[ModelInfo]:
    """扫描 ``*/final`` 和 ``*/checkpoint-*``，返回按修改时间倒序的模型列表。"""
    root = Path(checkpoints_dir)
    if not root.exists():
        return []

    candidates = list(root.glob("*/final")) + list(root.glob("*/checkpoint-*"))
    models: list[ModelInfo] = []

    for path in candidates:
        if not path.is_dir():
            continue
        config_path = path / "config.json"
        if not config_path.exists():
            continue
        if not any((path / name).exists() for name in _WEIGHT_FILES):
            continue

        tokenizer_path = _find_tokenizer(path)
        if tokenizer_path is None:
            continue

        model_config = _read_json(config_path)
        run_config = _read_json(path / "run_config.json")

        block_size = int(model_config.get("n_positions", 0))
        n_params = int(run_config.get("n_params") or _count_params(model_config))
        max_steps = run_config.get("max_steps")
        if max_steps is not None:
            max_steps = int(max_steps)
        elif path.name.startswith("checkpoint-"):
            max_steps = int(path.name.split("-")[-1])

        step_text = f"{max_steps} steps" if max_steps is not None else "steps ?"
        label = (
            f"{path.parent.name}/{path.name} · ctx {block_size} · "
            f"{step_text} · {n_params / 1e6:.1f}M"
        )
        models.append(
            ModelInfo(
                label=label,
                path=path,
                tokenizer_path=tokenizer_path,
                block_size=block_size,
                n_params=n_params,
                max_steps=max_steps,
                modified=path.stat().st_mtime,
            )
        )

    models.sort(key=lambda info: info.modified, reverse=True)
    return models


class ModelCache:
    """简单的 LRU 模型缓存；模型很小，保留几个即可。"""

    def __init__(self, max_size: int = config.MODEL_CACHE_SIZE) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_size = max_size
        self._items: OrderedDict[str, tuple[GPT2LMHeadModel, CharTokenizer]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, info: ModelInfo) -> tuple[GPT2LMHeadModel, CharTokenizer]:
        key = str(info.path.resolve())
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]

            tokenizer = CharTokenizer.load(info.tokenizer_path)
            model = (
                GPT2LMHeadModel.from_pretrained(info.path, local_files_only=True)
                .to(self.device)
                .eval()
            )
            model.config.use_cache = True

            self._items[key] = (model, tokenizer)
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                _, (evicted, _) = self._items.popitem(last=False)
                del evicted
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            return model, tokenizer
