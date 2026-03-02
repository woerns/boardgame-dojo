"""Agent abstractions — the interface between players and games."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dojo.core.game import Game, GameState
from dojo.core.types import Action


class Agent(ABC):
    """Base class for all agents (human, random, RL, etc.)."""

    @abstractmethod
    def select_action(self, state: GameState, game: Game) -> Action:
        """Choose an action given the current game state."""

    def reset(self) -> None:
        """Called at the start of each new game (optional hook)."""


class RandomAgent(Agent):
    """Uniformly random legal-move agent — useful as a baseline."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, state: GameState, game: Game) -> Action:
        legal = state.legal_actions()
        return self._rng.choice(legal)


class HumanAgent(Agent):
    """CLI-based human player — prompts for input on stdin."""

    def select_action(self, state: GameState, game: Game) -> Action:
        legal = state.legal_actions()
        print(state)
        while True:
            try:
                action = int(input(f"Your move {legal}: "))
                if action in legal:
                    return action
                print(f"Illegal action {action}. Legal: {legal}")
            except (ValueError, EOFError):
                print(f"Enter an integer from {legal}")
