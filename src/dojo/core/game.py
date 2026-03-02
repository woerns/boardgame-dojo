"""Abstract base classes for games and game states."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from dojo.core.types import (
    Action,
    ActionMask,
    ObservationTensor,
    PlayerId,
)


class GameState(ABC):
    """Mutable game state.

    MCTS creates millions of copies via clone(), so mutable state with
    selective cloning is more efficient than immutable snapshots.
    """

    @abstractmethod
    def current_player(self) -> PlayerId:
        """Return the player to act.

        Returns a non-negative PlayerId for decision nodes, or one of the
        sentinel values (CHANCE_PLAYER, SIMULTANEOUS_PLAYER, TERMINAL_PLAYER).
        """

    @abstractmethod
    def legal_actions(self) -> list[Action]:
        """Return the list of legal actions at this state."""

    @abstractmethod
    def legal_actions_mask(self) -> ActionMask:
        """Return a boolean mask over the full action space.

        mask[a] is True iff action a is legal. Used for neural-net policy
        masking — much faster than iterating a list.
        """

    @abstractmethod
    def apply_action(self, action: Action) -> None:
        """Apply an action in-place, advancing the state."""

    @abstractmethod
    def returns(self) -> list[float]:
        """Return the payoff for each player at a terminal state.

        Convention: +1 win, -1 loss, 0 draw. Only valid when
        current_player() == TERMINAL_PLAYER.
        """

    @abstractmethod
    def observation_tensor(self, player: PlayerId | None = None) -> ObservationTensor:
        """Return the board from *player*'s perspective.

        For perfect-information games the player argument can be ignored.
        For imperfect-info games, returns only what *player* can see.
        Shape must match Game.observation_tensor_shape().
        """

    @abstractmethod
    def clone(self) -> GameState:
        """Return a deep copy of this state."""

    @abstractmethod
    def __str__(self) -> str:
        """Human-readable representation (for debugging / CLI play)."""


class Game(ABC):
    """Factory and metadata for a particular game type."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'connect_four'."""

    @abstractmethod
    def num_players(self) -> int:
        """Number of players (e.g. 2 for Connect Four)."""

    @abstractmethod
    def num_actions(self) -> int:
        """Size of the flat action space."""

    @abstractmethod
    def observation_tensor_shape(self) -> Sequence[int]:
        """Shape of the observation tensor (C, H, W) or similar."""

    @abstractmethod
    def new_initial_state(self) -> GameState:
        """Return a fresh initial game state."""
