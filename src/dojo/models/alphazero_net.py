"""AlphaZero dual-head ResNet: shared trunk → policy head + value head."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from dojo.core.model import NetworkOutput, PolicyValueNetwork
from dojo.models.blocks import ConvBlock, PolicyHead, ResidualBlock, ValueHead


class AlphaZeroNet(nn.Module):
    """Game-agnostic dual-head ResNet.

    Takes observation_shape (C, H, W) and num_actions from the Game object.
    Works for any grid-based game without modification.
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        num_res_blocks: int = 5,
        num_channels: int = 64,
    ) -> None:
        super().__init__()
        in_channels = observation_shape[0]
        h, w = observation_shape[1], observation_shape[2]

        # Shared trunk
        self.input_block = ConvBlock(in_channels, num_channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        self.policy_head = PolicyHead(num_channels, h, w, num_actions)
        self.value_head = ValueHead(num_channels, h, w)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: batch of observations, shape (B, C, H, W)

        Returns:
            policy_logits: (B, num_actions) — raw logits (apply log_softmax externally)
            value: (B, 1) — in [-1, 1] via tanh
        """
        out = self.input_block(x)
        out = self.res_blocks(out)
        return self.policy_head(out), self.value_head(out)


class AlphaZeroNetwork(PolicyValueNetwork):
    """Wraps AlphaZeroNet to implement the PolicyValueNetwork interface."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        num_res_blocks: int = 5,
        num_channels: int = 64,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.net = AlphaZeroNet(
            observation_shape=observation_shape,
            num_actions=num_actions,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
        ).to(self.device)

    @torch.no_grad()
    def predict(self, observation: npt.NDArray[np.float32]) -> NetworkOutput:
        # Caller must set eval_mode() before inference to ensure correct
        # BatchNorm/Dropout behavior. We don't call it here to avoid
        # silently overriding training mode during self-play.
        x = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        policy_logits, value = self.net(x)

        # Convert logits to probabilities, then to numpy
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        return NetworkOutput(policy=policy, value=value.item())

    def to_device(self, device: torch.device) -> None:
        self.device = device
        self.net = self.net.to(device)

    def train_mode(self) -> None:
        self.net.train()

    def eval_mode(self) -> None:
        self.net.eval()

    def parameters(self):
        return self.net.parameters()
