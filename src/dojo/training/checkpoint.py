"""Save and load model checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    path: Path,
    model_config: dict | None = None,
) -> None:
    """Save model + optimizer state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if model_config is not None:
        data["model_config"] = model_config
    torch.save(data, path)


def load_checkpoint_data(
    path: Path,
    device: torch.device | None = None,
) -> dict:
    """Load a raw checkpoint dict from disk.

    Unlike ``load_checkpoint``, this does **not** require a pre-constructed
    model/optimizer.  Useful when you need to inspect ``model_config`` or
    ``iteration`` before building a network.
    """
    map_location = device or torch.device("cpu")
    return torch.load(path, map_location=map_location, weights_only=True)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: Path,
    device: torch.device | None = None,
) -> int:
    """Load model + optimizer state from disk.

    Args:
        model: model to load weights into
        optimizer: optimizer to load state into (None to skip)
        path: checkpoint file path
        device: device to map tensors to

    Returns:
        The iteration number saved in the checkpoint
    """
    map_location = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
