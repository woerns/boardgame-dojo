"""MuZero self-play: MCTS with learned dynamics model.

Unlike AlphaZero, MuZero MCTS operates on hidden states (latent
representations) rather than real game states.  Only the root node
uses the actual observation; all deeper nodes are expanded using the
learned dynamics network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from dojo.algorithms.mcts import (
    MCTSConfig,
    MCTSNode,
    _add_dirichlet_noise,
    _backpropagate,
    _expand_node,
    _select_child,
    get_action_probs,
)
from dojo.algorithms.muzero.model import MuZeroNetwork
from dojo.core.game import Game


@dataclass
class MuZeroGameRecord:
    """A complete self-play game recorded for MuZero training.

    Stores the full trajectory: at each step we record the observation,
    the action taken, the MCTS policy, the reward received, and who
    was playing.  The final game returns are filled in after the game ends.
    """

    observations: list[npt.NDArray[np.float32]] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    policies: list[npt.NDArray[np.float32]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    players: list[int] = field(default_factory=list)
    returns: list[float] | None = None

    def add_step(
        self,
        observation: npt.NDArray[np.float32],
        action: int,
        policy: npt.NDArray[np.float32],
        reward: float,
        player: int,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.policies.append(policy)
        self.rewards.append(reward)
        self.players.append(player)

    def set_returns(self, returns: list[float]) -> None:
        self.returns = returns

    def __len__(self) -> int:
        return len(self.observations)


def run_muzero_mcts(
    observation: npt.NDArray[np.float32],
    legal_mask: npt.NDArray[np.bool_],
    network: MuZeroNetwork,
    config: MCTSConfig,
    num_actions: int,
    current_player: int,
    rng: np.random.Generator,
) -> MCTSNode:
    """Run MCTS using MuZero's learned model.

    The root is evaluated using the real observation via ``initial_inference``.
    All deeper nodes are expanded using ``recurrent_inference`` (the learned
    dynamics model), with no access to the true game state.

    Args:
        observation: real observation at the root
        legal_mask: legal-actions mask at the root (from the real game)
        network: MuZero network wrapper
        config: MCTS hyperparameters
        num_actions: size of the action space
        current_player: player to move at the root
        rng: random generator for Dirichlet noise

    Returns:
        root MCTSNode with visit counts reflecting the search
    """
    root = MCTSNode()

    # Root: use real observation → representation + prediction
    root_output = network.initial_inference(observation)
    _expand_node(root, root_output.policy, legal_mask)
    _add_dirichlet_noise(root, config.dirichlet_alpha, config.dirichlet_epsilon, rng)

    # Store hidden states keyed by node identity
    hidden_states: dict[int, object] = {id(root): root_output.hidden_state}

    # All internal actions are treated as legal (the policy learns illegality)
    full_mask = np.ones(num_actions, dtype=bool)

    for _ in range(config.num_simulations):
        node = root
        search_path = [node]
        to_play = [current_player]
        last_action = -1

        # Selection: walk down the tree via PUCT
        while node.is_expanded():
            last_action, child = _select_child(node, config.c_puct)
            node = child
            search_path.append(node)
            # Alternate players (2-player assumption)
            to_play.append(1 - to_play[-1])

        # Expansion: run dynamics from parent's hidden state
        parent = search_path[-2]
        parent_hidden = hidden_states[id(parent)]

        output = network.recurrent_inference(parent_hidden, last_action)
        hidden_states[id(node)] = output.hidden_state

        _expand_node(node, output.policy, full_mask)

        # Backup (value from current player's perspective)
        _backpropagate(search_path, output.value, to_play, to_play[-1])

    return root



def generate_muzero_self_play_game(
    game: Game,
    network: MuZeroNetwork,
    mcts_config: MCTSConfig,
    temperature_threshold: int = 30,
    rng: np.random.Generator | None = None,
) -> MuZeroGameRecord:
    """Play a complete game via MuZero MCTS self-play.

    The real game state is used to:
    - provide observations (via ``representation``)
    - determine legal actions at the root of each MCTS search
    - determine when the game ends and compute returns

    Inside each MCTS search, only the learned model is used.

    Args:
        game: game factory
        network: MuZero network for MCTS evaluation
        mcts_config: MCTS hyperparameters
        temperature_threshold: after this many moves, use greedy selection
        rng: random generator for reproducibility

    Returns:
        MuZeroGameRecord with full trajectory data
    """
    if rng is None:
        rng = np.random.default_rng()

    state = game.new_initial_state()
    record = MuZeroGameRecord()
    num_actions = game.num_actions()

    move_number = 0
    while state.current_player() >= 0:
        obs = state.observation_tensor()
        legal_mask = state.legal_actions_mask()
        current_player = state.current_player()

        # Run MuZero MCTS
        root = run_muzero_mcts(
            observation=obs,
            legal_mask=legal_mask,
            network=network,
            config=mcts_config,
            num_actions=num_actions,
            current_player=current_player,
            rng=rng,
        )

        # Extract action probabilities
        temperature = 1.0 if move_number < temperature_threshold else 0.0
        action_probs = get_action_probs(root, num_actions, temperature)

        # Select action
        action = int(rng.choice(num_actions, p=action_probs))

        # Apply action to real game state
        state.apply_action(action)

        # Reward: 0 for intermediate steps, actual return for terminal
        if state.current_player() < 0:
            reward = state.returns()[current_player]
        else:
            reward = 0.0

        record.add_step(obs, action, action_probs, reward, current_player)
        move_number += 1

    record.set_returns(state.returns())
    return record
