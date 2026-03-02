"""Connect Four game implementation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from dojo.core.game import Game, GameState
from dojo.core.types import (
    Action,
    ActionMask,
    ObservationTensor,
    PlayerId,
    TERMINAL_PLAYER,
)
from dojo.games.connect_four.config import ConnectFourConfig
from dojo.games.registry import register_game

# Win-check directions: vertical, horizontal, diagonal /, diagonal \
_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


class ConnectFourState(GameState):
    """Mutable Connect Four game state.

    Board is stored as a 2D int8 array: 0 = empty, 1 = player 0, 2 = player 1.
    Column heights are tracked for O(1) drop placement.
    """

    def __init__(self, config: ConnectFourConfig) -> None:
        self._cfg = config
        # Board: rows x cols, 0 = empty, 1 = P0, 2 = P1
        self._board = np.zeros((config.rows, config.cols), dtype=np.int8)
        # Height of each column (next available row from bottom)
        self._heights = np.zeros(config.cols, dtype=np.int32)
        self._current_player: PlayerId = 0
        self._winner: int | None = None  # None = ongoing, 0/1 = winner, -1 = draw
        self._move_count = 0

    # ── GameState interface ──────────────────────────────────────────

    def current_player(self) -> PlayerId:
        if self._winner is not None:
            return TERMINAL_PLAYER
        return self._current_player

    def legal_actions(self) -> list[Action]:
        return [c for c in range(self._cfg.cols) if self._heights[c] < self._cfg.rows]

    def legal_actions_mask(self) -> ActionMask:
        return self._heights < self._cfg.rows

    def apply_action(self, action: Action) -> None:
        col = action
        row = self._heights[col]
        piece = self._current_player + 1  # 1 or 2

        self._board[row, col] = piece
        self._heights[col] += 1
        self._move_count += 1

        if self._check_win(row, col, piece):
            self._winner = self._current_player
        elif self._move_count == self._cfg.rows * self._cfg.cols:
            self._winner = -1  # draw
        else:
            self._current_player = 1 - self._current_player

    def returns(self) -> list[float]:
        if self._winner is None or self._winner == -1:
            return [0.0, 0.0]  # ongoing or draw
        # Winner gets +1, loser gets -1
        r = [0.0, 0.0]
        r[self._winner] = 1.0
        r[1 - self._winner] = -1.0
        return r

    def observation_tensor(self, player: PlayerId | None = None) -> ObservationTensor:
        """3-channel tensor: [current_player_pieces, opponent_pieces, turn_indicator].

        If *player* is given, the tensor is from that player's perspective.
        Otherwise, from the current player's perspective.
        """
        p = player if player is not None else self._current_player
        mine = p + 1
        theirs = (1 - p) + 1

        obs = np.zeros((3, self._cfg.rows, self._cfg.cols), dtype=np.float32)
        obs[0] = (self._board == mine).astype(np.float32)
        obs[1] = (self._board == theirs).astype(np.float32)
        obs[2] = float(self._current_player == p)  # 1.0 if it's my turn

        return obs

    def clone(self) -> ConnectFourState:
        s = ConnectFourState.__new__(ConnectFourState)
        s._cfg = self._cfg
        s._board = self._board.copy()
        s._heights = self._heights.copy()
        s._current_player = self._current_player
        s._winner = self._winner
        s._move_count = self._move_count
        return s

    def __str__(self) -> str:
        rows = self._cfg.rows
        cols = self._cfg.cols
        symbols = {0: ".", 1: "X", 2: "O"}
        lines = []
        # Print top-to-bottom (row 5 at top for standard 6-row board)
        for r in range(rows - 1, -1, -1):
            lines.append(" ".join(symbols[self._board[r, c]] for c in range(cols)))
        lines.append(" ".join(str(c) for c in range(cols)))
        return "\n".join(lines)

    # ── Internal helpers ─────────────────────────────────────────────

    def _check_win(self, row: int, col: int, piece: int) -> bool:
        """Check if placing *piece* at (row, col) wins the game."""
        for dr, dc in _DIRECTIONS:
            count = 1
            # Check both directions along the axis
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while (
                    0 <= r < self._cfg.rows
                    and 0 <= c < self._cfg.cols
                    and self._board[r, c] == piece
                ):
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self._cfg.win_length:
                return True
        return False

    # ── Extra accessors (useful for rendering / debugging) ───────────

    @property
    def board(self) -> np.ndarray:
        view = self._board.view()
        view.flags.writeable = False
        return view

    @property
    def winner(self) -> int | None:
        return self._winner

    @property
    def move_count(self) -> int:
        return self._move_count


@register_game("connect_four")
class ConnectFourGame(Game):
    """Connect Four game factory."""

    def __init__(self, **kwargs: int) -> None:
        self._cfg = ConnectFourConfig(**kwargs)

    def name(self) -> str:
        return "connect_four"

    def num_players(self) -> int:
        return 2

    def num_actions(self) -> int:
        return self._cfg.cols

    def observation_tensor_shape(self) -> Sequence[int]:
        return (3, self._cfg.rows, self._cfg.cols)

    def new_initial_state(self) -> ConnectFourState:
        return ConnectFourState(self._cfg)

    @property
    def config(self) -> ConnectFourConfig:
        return self._cfg
