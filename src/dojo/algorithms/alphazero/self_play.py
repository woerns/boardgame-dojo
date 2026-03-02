"""AlphaZero self-play: generate training data via MCTS + neural network."""

from __future__ import annotations

import numpy as np

from dojo.algorithms.mcts import MCTSConfig, get_action_probs, run_mcts
from dojo.core.game import Game, GameState
from dojo.core.model import PolicyValueNetwork
from dojo.core.replay_buffer import GameRecord


class AlphaZeroNodeEvaluator:
    """NodeEvaluator for AlphaZero: runs the neural network on real game states.

    Caches the last observation tensor to avoid recomputation when the
    root state's observation is needed for training data recording.
    """

    def __init__(self, network: PolicyValueNetwork) -> None:
        self._network = network
        self.last_observation: np.ndarray | None = None

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        obs = state.observation_tensor()
        self.last_observation = obs
        output = self._network.predict(obs)
        return output.policy, output.value


def generate_self_play_game(
    game: Game,
    network: PolicyValueNetwork,
    mcts_config: MCTSConfig,
    temperature_threshold: int = 30,
    rng: np.random.Generator | None = None,
) -> GameRecord:
    """Play a complete game via MCTS self-play, recording training data.

    Args:
        game: game factory
        network: policy-value network for MCTS evaluation
        mcts_config: MCTS hyperparameters
        temperature_threshold: after this many moves, use greedy action selection
        rng: random generator for reproducibility

    Returns:
        GameRecord with observations, MCTS policies, and final returns
    """
    if rng is None:
        rng = np.random.default_rng()

    state = game.new_initial_state()
    evaluator = AlphaZeroNodeEvaluator(network)
    record = GameRecord()

    move_number = 0
    while state.current_player() >= 0:
        # Capture the root observation BEFORE MCTS (which overwrites the cache)
        root_obs = state.observation_tensor()

        # Run MCTS
        root = run_mcts(state, evaluator, mcts_config, rng)

        # Extract action probabilities
        temperature = 1.0 if move_number < temperature_threshold else 0.0
        action_probs = get_action_probs(root, game.num_actions(), temperature)

        # Record training data
        record.add_step(root_obs, action_probs, state.current_player())

        # Select and apply action
        action = rng.choice(game.num_actions(), p=action_probs)
        state.apply_action(action)
        move_number += 1

    # Record final returns
    record.set_returns(state.returns())
    return record
