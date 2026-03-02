"""Reusable neural network building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Pre-activation residual block: Conv → BN → ReLU → Conv → BN + skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class PolicyHead(nn.Module):
    """Policy head: Conv1x1 → flatten → Linear(num_actions)."""

    def __init__(self, num_channels: int, h: int, w: int, num_actions: int) -> None:
        super().__init__()
        self.conv = ConvBlock(num_channels, 2, kernel_size=1, padding=0)
        self.fc = nn.Linear(2 * h * w, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class ValueHead(nn.Module):
    """Value head: Conv1x1 → flatten → FC(256) → ReLU → FC(1) → tanh."""

    def __init__(self, num_channels: int, h: int, w: int) -> None:
        super().__init__()
        self.conv = ConvBlock(num_channels, 1, kernel_size=1, padding=0)
        self.fc1 = nn.Linear(h * w, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        return torch.tanh(self.fc2(out))
