"""全棋盘 CNN Q 网络。

输入是 ``(B, C, H, W)`` 的棋盘张量，输出 ``(B, n_actions)`` 的 Q 值。
网络刻意保持小规模：3 层 conv + 1 层 FC，适合 20×20 的棋盘。
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class BoardQNet(nn.Module):
    """``C×H×W -> 3`` 的小 CNN。"""

    def __init__(
        self,
        in_channels: int = 5,
        height: int = 20,
        width: int = 20,
        hidden_size: int = 256,
        output_size: int = 3,
    ):
        super().__init__()
        if in_channels < 1 or height < 1 or width < 1:
            raise ValueError("in_channels/height/width 至少为 1")

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # stride=2 把 20×20 压到 10×10，减少后面的全连接层。
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        out_h = (height + 2 * 1 - 3) // 2 + 1
        out_w = (width + 2 * 1 - 3) // 2 + 1
        flat_size = 64 * out_h * out_w

        self.fc1 = nn.Linear(flat_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.flat_size = flat_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
