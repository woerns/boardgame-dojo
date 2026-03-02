"""Monte Carlo Tree Search shared by AlphaZero and MuZero.

The MCTS logic (selection, expansion, backup) is algorithm-agnostic.
Algorithm-specific behavior is injected via the NodeEvaluator protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dojo.core.game import GameState
from dojo.core.types import Action, ActionMask


@dataclass
class MCTSConfig:
    """MCTS hyperparameters."""

    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0


class NodeEvaluator(Protocol):
    """Protocol that MCTS calls to evaluate a leaf node.

    AlphaZero: clones state, applies action, runs neural net.
    MuZero: runs dynamics network to predict next hidden state + reward.
    """

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """Evaluate a leaf state.

        Returns:
            prior: probability distribution over actions (sums to ~1)
            value: state value from current player's perspective in [-1, 1]
        """
        ...


class MCTSNode:
    """A node in the MCTS search tree."""

    __slots__ = (
        "prior",
        "visit_count",
        "value_sum",
        "children",
        "action_mask",
    )

    def __init__(self, prior: float = 0.0) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[Action, MCTSNode] | None = None  # lazy — allocated on expansion
        self.action_mask: ActionMask | None = None

    @property
    def q_value(self) -> float:
        """Mean action-value."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return self.children is not None and len(self.children) > 0


def _select_child(node: MCTSNode, c_puct: float) -> tuple[Action, MCTSNode]:
    """Select the child with the highest PUCT score.

    PUCT formula: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    """
    best_score = -float("inf")
    best_action = -1
    best_child = None

    # Compute sqrt once for all children of this parent
    sqrt_parent = c_puct * math.sqrt(node.visit_count)

    for action, child in node.children.items():
        exploration = sqrt_parent * child.prior / (1 + child.visit_count)
        # child.q_value is from the child node's player-to-move perspective.
        # Selection happens from the parent player's perspective, so negate.
        score = -child.q_value + exploration
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    assert best_child is not None
    return best_action, best_child


def _expand_node(
    node: MCTSNode,
    prior: np.ndarray,
    action_mask: ActionMask,
) -> None:
    """Expand a leaf node: create children for all legal actions."""
    node.action_mask = action_mask
    node.children = {}

    # Mask and renormalize prior
    masked_prior = prior * action_mask
    prior_sum = masked_prior.sum()
    if prior_sum > 0:
        masked_prior /= prior_sum
    else:
        # Uniform over legal actions if prior is all zeros after masking
        masked_prior = action_mask.astype(np.float32)
        masked_prior /= masked_prior.sum()

    for action in range(len(action_mask)):
        if action_mask[action]:
            node.children[action] = MCTSNode(prior=masked_prior[action])


def _add_dirichlet_noise(
    node: MCTSNode, alpha: float, epsilon: float, rng: np.random.Generator
) -> None:
    """Add Dirichlet noise to the root node's priors for exploration."""
    actions = list(node.children.keys())
    noise = rng.dirichlet([alpha] * len(actions))
    for i, action in enumerate(actions):
        child = node.children[action]
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


def _backpropagate(
    search_path: list[MCTSNode], value: float, to_play: list[int], value_player: int
) -> None:
    """Backpropagate the value through the search path.

    Args:
        search_path: nodes visited during selection (root → leaf)
        value: value from *value_player*'s perspective
        to_play: player-to-act at each node in the search path
        value_player: the player whose perspective *value* is from
    """
    for i, node in enumerate(search_path):
        if to_play[i] == value_player:
            node.value_sum += value
        else:
            node.value_sum -= value
        node.visit_count += 1


def run_mcts(
    state: GameState,
    evaluator: NodeEvaluator,
    config: MCTSConfig,
    rng: np.random.Generator | None = None,
) -> MCTSNode:
    """Run MCTS from the given state and return the root node.

    Args:
        state: current game state (not modified)
        evaluator: provides (prior, value) for leaf nodes
        config: MCTS hyperparameters
        rng: random generator for Dirichlet noise

    Returns:
        root MCTSNode with visit counts reflecting the search
    """
    if rng is None:
        rng = np.random.default_rng()

    root = MCTSNode()

    # Expand root
    prior, value = evaluator.evaluate(state)
    action_mask = state.legal_actions_mask()
    _expand_node(root, prior, action_mask)
    _add_dirichlet_noise(root, config.dirichlet_alpha, config.dirichlet_epsilon, rng)

    for _ in range(config.num_simulations):
        node = root
        sim_state = state.clone()
        search_path = [node]
        to_play = [sim_state.current_player()]

        # Selection: walk down the tree
        while node.is_expanded() and sim_state.current_player() >= 0:
            action, node = _select_child(node, config.c_puct)
            sim_state.apply_action(action)
            search_path.append(node)
            to_play.append(sim_state.current_player())

        # Evaluate leaf
        if sim_state.current_player() >= 0:
            # Non-terminal: expand and evaluate
            # value is from current player's perspective (per NodeEvaluator contract)
            prior, value = evaluator.evaluate(sim_state)
            value_player = sim_state.current_player()
            action_mask = sim_state.legal_actions_mask()
            _expand_node(node, prior, action_mask)
        else:
            # Terminal: use actual game returns
            # Value from the perspective of the player who made the last move
            value_player = to_play[-2] if len(to_play) >= 2 else 0
            value = sim_state.returns()[value_player]

        # Backpropagate
        _backpropagate(search_path, value, to_play, value_player)

    return root


def get_action_probs(
    root: MCTSNode,
    num_actions: int,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert visit counts to a probability distribution over actions.

    Args:
        root: the MCTS root node after search
        num_actions: size of the full action space
        temperature: controls exploration (1.0 = proportional, →0 = greedy)

    Returns:
        action probability vector of shape (num_actions,)
    """
    visits = np.zeros(num_actions, dtype=np.float32)
    for action, child in root.children.items():
        visits[action] = child.visit_count

    if temperature == 0:
        # Greedy: put all mass on the most-visited action
        best = np.argmax(visits)
        probs = np.zeros(num_actions, dtype=np.float32)
        probs[best] = 1.0
        return probs

    # Temperature-scaled: visits^(1/temp), normalized
    visits_temp = visits ** (1.0 / temperature)
    total = visits_temp.sum()
    if total > 0:
        return visits_temp / total

    # Fallback (shouldn't happen if root was expanded)
    return np.ones(num_actions, dtype=np.float32) / num_actions
