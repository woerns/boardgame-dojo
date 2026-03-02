"""Tests for MCTS search."""

import numpy as np
import pytest

from dojo.algorithms.mcts import (
    MCTSConfig,
    MCTSNode,
    NodeEvaluator,
    get_action_probs,
    run_mcts,
)
from dojo.core.game import GameState
from dojo.core.types import TERMINAL_PLAYER
from dojo.games.connect_four.game import ConnectFourGame


class UniformEvaluator:
    """Dummy evaluator: uniform prior, value = 0."""

    def __init__(self, num_actions: int) -> None:
        self._num_actions = num_actions

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        prior = np.ones(self._num_actions, dtype=np.float32) / self._num_actions
        return prior, 0.0


class BiasedEvaluator:
    """Evaluator that gives high prior to a specific action."""

    def __init__(self, num_actions: int, preferred_action: int) -> None:
        self._num_actions = num_actions
        self._preferred = preferred_action

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        prior = np.ones(self._num_actions, dtype=np.float32) * 0.01
        prior[self._preferred] = 0.9
        prior /= prior.sum()
        return prior, 0.0


class TestMCTSBasics:
    """Basic MCTS functionality."""

    def test_root_is_expanded(self):
        game = ConnectFourGame()
        state = game.new_initial_state()
        evaluator = UniformEvaluator(game.num_actions())

        config = MCTSConfig(num_simulations=10)
        root = run_mcts(state, evaluator, config)

        assert root.is_expanded()
        assert len(root.children) == 7  # All 7 columns legal

    def test_visit_counts_sum(self):
        game = ConnectFourGame()
        state = game.new_initial_state()
        evaluator = UniformEvaluator(game.num_actions())

        config = MCTSConfig(num_simulations=50)
        root = run_mcts(state, evaluator, config)

        child_visits = sum(c.visit_count for c in root.children.values())
        # Root visit count = num_simulations + 1 (the initial expansion)
        # Actually root is expanded once then simulations add to children
        # Total child visits should be num_simulations
        assert child_visits == config.num_simulations

    def test_state_not_modified(self):
        game = ConnectFourGame()
        state = game.new_initial_state()
        original_board = state.board.copy()
        evaluator = UniformEvaluator(game.num_actions())

        config = MCTSConfig(num_simulations=50)
        run_mcts(state, evaluator, config)

        # Original state should be unchanged
        np.testing.assert_array_equal(state.board, original_board)
        assert state.current_player() == 0

    def test_illegal_actions_not_in_children(self):
        # Use small board so we can fill a column quickly
        game = ConnectFourGame(rows=3, cols=4, win_length=4)
        state = game.new_initial_state()
        # Fill column 0 (3 rows): P0→col0, P1→col0, P0→col0
        # After 3 moves, col 0 is full, P1 to play
        state.apply_action(0)  # P0 → row 0
        state.apply_action(0)  # P1 → row 1
        state.apply_action(0)  # P0 → row 2 (col 0 now full)

        evaluator = UniformEvaluator(game.num_actions())
        config = MCTSConfig(num_simulations=20)
        root = run_mcts(state, evaluator, config)

        assert 0 not in root.children  # col 0 is full
        assert len(root.children) == 3  # cols 1, 2, 3 remain


class TestMCTSActionSelection:
    """Action probability extraction."""

    def test_action_probs_sum_to_one(self):
        game = ConnectFourGame()
        state = game.new_initial_state()
        evaluator = UniformEvaluator(game.num_actions())

        config = MCTSConfig(num_simulations=50)
        root = run_mcts(state, evaluator, config)

        probs = get_action_probs(root, game.num_actions(), temperature=1.0)
        assert probs.shape == (7,)
        np.testing.assert_almost_equal(probs.sum(), 1.0)

    def test_greedy_temperature(self):
        game = ConnectFourGame()
        state = game.new_initial_state()
        evaluator = UniformEvaluator(game.num_actions())

        config = MCTSConfig(num_simulations=100)
        root = run_mcts(state, evaluator, config)

        probs = get_action_probs(root, game.num_actions(), temperature=0)
        # Exactly one action should have probability 1.0
        assert np.count_nonzero(probs) == 1
        assert probs.max() == 1.0

    def test_biased_prior_influences_search(self):
        game = ConnectFourGame()
        state = game.new_initial_state()
        # Evaluator that strongly prefers column 3
        evaluator = BiasedEvaluator(game.num_actions(), preferred_action=3)

        config = MCTSConfig(num_simulations=50, dirichlet_epsilon=0.0)
        root = run_mcts(state, evaluator, config)

        probs = get_action_probs(root, game.num_actions(), temperature=1.0)
        # Column 3 should get the most visits
        assert probs[3] == probs.max()


class TestMCTSTerminal:
    """Handling terminal states during search."""

    def test_near_terminal_state(self):
        """MCTS should handle states close to terminal correctly."""
        game = ConnectFourGame(rows=4, cols=4, win_length=3)
        state = game.new_initial_state()
        # P0: col 0, P1: col 1, P0: col 0, P1: col 1, P0: col 0 → P0 wins vertically
        # But let's test a state one move before win
        for action in [0, 1, 0, 1]:
            state.apply_action(action)
        # P0 has 2 in col 0, needs one more for win
        evaluator = UniformEvaluator(game.num_actions())
        config = MCTSConfig(num_simulations=30)
        root = run_mcts(state, evaluator, config)
        probs = get_action_probs(root, game.num_actions(), temperature=1.0)
        assert probs.shape == (4,)
        np.testing.assert_almost_equal(probs.sum(), 1.0)
