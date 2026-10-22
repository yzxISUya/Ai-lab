"""通过 memmap 直接从 .bin 文件读取的定长 token 窗口。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset):
    """把 token 流切分成互不重叠的 ``block_size`` 窗口。

    每个样本里的输入和标签是同一段窗口：因果语言模型的 loss 内部会把
    标签右移一位，所以 token t 实际上预测的是 token t+1。
    """

    def __init__(self, bin_path: str | Path, block_size: int, dtype: str = "uint8") -> None:
        self.block_size = block_size
        self.data = np.memmap(bin_path, dtype=np.dtype(dtype), mode="r")
        self.n_windows = len(self.data) // block_size

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.block_size
        window = np.asarray(self.data[start : start + self.block_size], dtype=np.int64)
        ids = torch.from_numpy(window)
        return {"input_ids": ids, "labels": ids.clone()}
