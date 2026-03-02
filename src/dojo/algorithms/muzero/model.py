"""MuZero neural networks: Representation + Dynamics + Prediction.

MuZero uses three learned functions instead of a game simulator:

    h(observation)          → hidden_state                (Representation)
    g(hidden_state, action) → (next_hidden, reward)       (Dynamics)
    f(hidden_state)         → (policy, value)             (Prediction)

During MCTS, only the root uses a real observation (via h). All subsequent
nodes are expanded using the learned dynamics (g) and evaluated with f.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from dojo.models.blocks import ConvBlock, PolicyHead, ResidualBlock, ValueHead


@dataclass
class MuZeroOutput:
    """Output from MuZero initial or recurrent inference."""

    hidden_state: torch.Tensor  # (1, C, H, W)
    policy: npt.NDArray[np.float32]  # (num_actions,)
    value: float
    reward: float = 0.0  # only from recurrent inference


def _scale_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """Min-max normalize hidden state to [0, 1] per sample.

    Prevents hidden representations from growing unbounded during
    multi-step dynamics unrolling — a key stability trick from the
    MuZero paper (Appendix G).
    """
    b = hidden.size(0)
    flat = hidden.view(b, -1)
    min_val = flat.min(dim=1, keepdim=True).values
    max_val = flat.max(dim=1, keepdim=True).values
    scale = max_val - min_val
    scale = torch.where(scale < 1e-5, torch.ones_like(scale), scale)
    flat = (flat - min_val) / scale
    return flat.view_as(hidden)


class MuZeroNet(nn.Module):
    """MuZero three-component network.

    All three sub-networks share the same spatial dimensions (H, W) and
    channel count, reusing the same ResidualBlock building blocks as
    AlphaZeroNet.
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
        self.num_actions = num_actions
        self.h = h
        self.w = w

        # --- Representation h(o) → s ---
        self.repr_input = ConvBlock(in_channels, num_channels)
        self.repr_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # --- Dynamics g(s, a) → (s', r) ---
        # Action encoded as a single spatial plane → input channels = num_channels + 1
        self.dyn_input = ConvBlock(num_channels + 1, num_channels)
        self.dyn_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )
        # Reward head
        self.reward_conv = ConvBlock(num_channels, 1, kernel_size=1, padding=0)
        self.reward_fc = nn.Linear(h * w, 1)

        # --- Prediction f(s) → (p, v) ---
        self.policy_head = PolicyHead(num_channels, h, w, num_actions)
        self.value_head = ValueHead(num_channels, h, w)

    def represent(self, observation: torch.Tensor) -> torch.Tensor:
        """h(o) → s: Encode observation into hidden state.

        Args:
            observation: (B, C_obs, H, W)

        Returns:
            hidden_state: (B, num_channels, H, W), scaled to [0, 1]
        """
        s = self.repr_input(observation)
        s = self.repr_blocks(s)
        return _scale_hidden(s)

    def dynamics(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """g(s, a) → (s', r): Predict next hidden state and reward.

        The action is encoded as a uniform spatial plane with value
        ``action / num_actions``, concatenated channel-wise to the hidden
        state.

        Args:
            hidden_state: (B, num_channels, H, W)
            action: (B,) integer action indices

        Returns:
            next_hidden: (B, num_channels, H, W), scaled to [0, 1]
            reward: (B, 1) predicted reward in [-1, 1]
        """
        b = hidden_state.size(0)
        action_plane = (
            action.float().view(b, 1, 1, 1).expand(b, 1, self.h, self.w)
            / self.num_actions
        )
        x = torch.cat([hidden_state, action_plane], dim=1)
        s = self.dyn_input(x)
        s = self.dyn_blocks(s)

        # Reward head (from pre-scale dynamics output)
        r = self.reward_conv(s)
        r = r.view(b, -1)
        r = torch.tanh(self.reward_fc(r))

        # Scale hidden state for stability
        s = _scale_hidden(s)
        return s, r

    def predict(
        self, hidden_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """f(s) → (p, v): Predict policy and value from hidden state.

        Args:
            hidden_state: (B, num_channels, H, W)

        Returns:
            policy_logits: (B, num_actions)
            value: (B, 1) in [-1, 1] via tanh
        """
        return self.policy_head(hidden_state), self.value_head(hidden_state)

    def initial_inference(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Representation + Prediction (used at the MCTS root).

        Returns:
            hidden_state, policy_logits, value
        """
        s = self.represent(observation)
        p, v = self.predict(s)
        return s, p, v

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dynamics + Prediction (used at MCTS internal nodes).

        Returns:
            next_hidden, reward, policy_logits, value
        """
        s, r = self.dynamics(hidden_state, action)
        p, v = self.predict(s)
        return s, r, p, v


class MuZeroNetwork:
    """Wraps MuZeroNet with numpy ↔ torch conversion for inference.

    Provides two entry points matching the MuZero paper:
    - ``initial_inference``: observation → (hidden, policy, value)
    - ``recurrent_inference``: (hidden, action) → (hidden, reward, policy, value)
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        num_res_blocks: int = 5,
        num_channels: int = 64,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.num_actions = num_actions
        self.net = MuZeroNet(
            observation_shape=observation_shape,
            num_actions=num_actions,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
        ).to(self.device)

    @torch.no_grad()
    def initial_inference(
        self, observation: npt.NDArray[np.float32]
    ) -> MuZeroOutput:
        """Run representation + prediction on a raw observation."""
        x = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        hidden, policy_logits, value = self.net.initial_inference(x)
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        return MuZeroOutput(
            hidden_state=hidden,
            policy=policy,
            value=value.item(),
        )

    @torch.no_grad()
    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: int
    ) -> MuZeroOutput:
        """Run dynamics + prediction from a hidden state and action."""
        action_t = torch.tensor([action], dtype=torch.long, device=self.device)
        hidden, reward, policy_logits, value = self.net.recurrent_inference(
            hidden_state, action_t
        )
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        return MuZeroOutput(
            hidden_state=hidden,
            policy=policy,
            value=value.item(),
            reward=reward.item(),
        )

    def to_device(self, device: torch.device) -> None:
        self.device = device
        self.net = self.net.to(device)

    def train_mode(self) -> None:
        self.net.train()

    def eval_mode(self) -> None:
        self.net.eval()

    def parameters(self):
        return self.net.parameters()
