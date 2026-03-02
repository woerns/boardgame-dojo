"""Metrics logging via TensorBoard."""

from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """Thin wrapper around TensorBoard SummaryWriter."""

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self._writer.close()
