"""Abstract base classes for neural network models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import numpy.typing as npt
import torch


@dataclass
class NetworkOutput:
    """Output from a policy-value network.

    policy: probabilities over actions (before masking).
    value: scalar value estimate in [-1, 1].
    """

    policy: npt.NDArray[np.float32]  # shape: (num_actions,)
    value: float


class PolicyValueNetwork(ABC):
    """Interface for any network that produces (policy, value) from an observation."""

    @abstractmethod
    def predict(self, observation: npt.NDArray[np.float32]) -> NetworkOutput:
        """Run inference on a single observation (no batch dim).

        Returns NetworkOutput with probabilities and scalar value.
        """

    @abstractmethod
    def to_device(self, device: torch.device) -> None:
        """Move the underlying model to a device."""

    @abstractmethod
    def train_mode(self) -> None:
        """Set the model to training mode."""

    @abstractmethod
    def eval_mode(self) -> None:
        """Set the model to evaluation mode."""

    @abstractmethod
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Return an iterator over model parameters (for optimizer)."""
