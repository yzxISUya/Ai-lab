"""Snake 的 Q 网络。"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    """11 -> hidden -> 3 的 MLP。

    输入是 11 维观察，输出是三个相对动作的 Q 值。
    输出层没有激活函数，因为 Q 值可正可负。
    """

    def __init__(self, input_size: int = 11, hidden_size: int = 256, output_size: int = 3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
