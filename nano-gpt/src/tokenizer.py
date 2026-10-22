"""一个极简的字符级 tokenizer。

词表布局：
    0                     -> <|endoftext|>  （分隔故事 / 可用于终止生成）
    1 .. len(chars)       -> 训练语料中出现过的字符

对于英文 TinyStories，语料全是 ASCII，所以编码只需对原始字节做一次查表。
其他情况则走通用（较慢的）路径。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

EOT_TOKEN = "<|endoftext|>"
EOT_ID = 0


class CharTokenizer:
    version = 1

    def __init__(self, chars: Iterable[str]) -> None:
        self.chars = sorted(set(chars))
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.chars)} # enumerate() 可以在遍历对象时，同时获得元素的索引和元素本身
        self.itos = {i + 1: ch for i, ch in enumerate(self.chars)}
        self.itos[EOT_ID] = EOT_TOKEN
        self._lut: np.ndarray | None = None

    # ------------------------------------------------------------------ 构建
    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharTokenizer":
        charset: set[str] = set()
        for text in texts:
            charset.update(text)
        return cls(charset)

    # ------------------------------------------------------------------ 统计
    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 1

    @property
    def dtype(self) -> np.dtype:
        # 不超过 256 个 id 时，每个 token 只占一个字节：2 亿字符的语料也只有 200MB。
        return np.dtype(np.uint8) if self.vocab_size <= 256 else np.dtype(np.uint16)

    def ascii_lut(self) -> np.ndarray | None:
        """返回 256 项的「字节 -> id」查找表；若不是纯 ASCII 则返回 None。"""
        if any(ord(ch) > 127 for ch in self.chars): # ord() 是 Python 的内置函数，用于获取单个字符对应的 Unicode 码点，也就是一个整数
            # 对于 ASCII 字符，ord(ch) 得到的数值也就是它的 ASCII 编码，与 ord() 对应的还有 chr()，即ord的逆函数
            return None
        lut = np.full(256, EOT_ID, dtype=np.uint8) # 创建一个长度为 256 的一维数组，并把每个元素初始化为 EOT_ID
        for ch, idx in self.stoi.items():
            lut[ord(ch)] = idx
        return lut

    # -------------------------------------------------------------- 编码/解码
    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        lut = self.ascii_lut()
        if lut is not None:
            ids = lut[np.frombuffer(text.encode("ascii"), dtype=np.uint8)].tolist()
        else:
            ids = [self.stoi.get(ch, EOT_ID) for ch in text]
        if add_eot:
            ids.append(EOT_ID)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        # 把故事分隔符换成换行，让生成的文本读起来更自然。
        return "".join(
            self.itos.get(int(i), "") if int(i) != EOT_ID else "\n" for i in ids
        )

    # ------------------------------------------------------------- 持久化
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": self.version, "eot_id": EOT_ID, "chars": self.chars}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != cls.version:
            raise ValueError(f"unsupported tokenizer version: {payload.get('version')}")
        return cls(payload["chars"])
