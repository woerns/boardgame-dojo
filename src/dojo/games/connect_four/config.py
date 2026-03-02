"""Connect Four game configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConnectFourConfig:
    """Parameters for Connect Four.

    Default 6x7 board with 4-in-a-row to win (standard rules).
    """

    rows: int = 6
    cols: int = 7
    win_length: int = 4
