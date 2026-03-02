"""AlphaZero agent — uses MCTS + neural network for inference/play."""

from __future__ import annotations

import numpy as np

from dojo.algorithms.alphazero.self_play import AlphaZeroNodeEvaluator
from dojo.algorithms.mcts import MCTSConfig, get_action_probs, run_mcts
from dojo.core.agent import Agent
from dojo.core.game import Game, GameState
from dojo.core.model import PolicyValueNetwork
from dojo.core.types import Action


class AlphaZeroAgent(Agent):
    """Agent that uses AlphaZero MCTS for action selection."""

    def __init__(
        self,
        network: PolicyValueNetwork,
        mcts_config: MCTSConfig | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._network = network
        self._mcts_config = mcts_config or MCTSConfig(num_simulations=50)
        self._temperature = temperature
        self._rng = np.random.default_rng(seed)
        self._evaluator = AlphaZeroNodeEvaluator(network)

    def select_action(self, state: GameState, game: Game) -> Action:
        root = run_mcts(state, self._evaluator, self._mcts_config, self._rng)
        probs = get_action_probs(root, game.num_actions(), self._temperature)

        if self._temperature == 0.0:
            return int(np.argmax(probs))
        return int(self._rng.choice(game.num_actions(), p=probs))
