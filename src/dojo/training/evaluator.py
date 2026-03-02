"""Arena evaluation — pit two agents against each other."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from dojo.core.agent import Agent
from dojo.core.game import Game


@dataclass
class ArenaResult:
    """Result of an arena evaluation."""

    wins: int
    losses: int
    draws: int

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0


def evaluate_agents(
    game: Game,
    agent1: Agent,
    agent2: Agent,
    num_games: int = 40,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ArenaResult:
    """Pit agent1 vs agent2 over multiple games.

    Each pair of games swaps who goes first to eliminate first-mover advantage.
    Results are from agent1's perspective.

    Args:
        game: game factory
        agent1: the challenger
        agent2: the current best
        num_games: total games to play (should be even)

    Returns:
        ArenaResult from agent1's perspective
    """
    wins = losses = draws = 0

    for i in range(num_games):
        # Alternate who goes first
        if i % 2 == 0:
            agents = [agent1, agent2]
            agent1_player = 0
        else:
            agents = [agent2, agent1]
            agent1_player = 1

        agents[0].reset()
        agents[1].reset()

        state = game.new_initial_state()
        while state.current_player() >= 0:
            player = state.current_player()
            action = agents[player].select_action(state, game)
            state.apply_action(action)

        returns = state.returns()
        agent1_return = returns[agent1_player]

        if agent1_return > 0:
            wins += 1
        elif agent1_return < 0:
            losses += 1
        else:
            draws += 1

        if progress_callback is not None:
            progress_callback(i + 1, num_games)

    return ArenaResult(wins=wins, losses=losses, draws=draws)
