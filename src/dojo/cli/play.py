"""CLI entry point for playing against a trained agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import dojo.games  # noqa: F401
from dojo.algorithms.alphazero.agent import AlphaZeroAgent
from dojo.algorithms.mcts import MCTSConfig
from dojo.core.agent import HumanAgent
from dojo.core.types import TERMINAL_PLAYER
from dojo.games.registry import create_game
from dojo.models.alphazero_net import AlphaZeroNetwork
from dojo.training.checkpoint import load_checkpoint_data


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play against a trained agent")
    parser.add_argument("--game", type=str, default="connect_four")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--human-player", type=int, default=0, choices=[0, 1])
    parsed = parser.parse_args(args)

    game = create_game(parsed.game)

    # Build network from checkpoint architecture + weights
    data = load_checkpoint_data(Path(parsed.checkpoint))
    model_config = data.get("model_config", {})
    network = AlphaZeroNetwork(
        observation_shape=game.observation_tensor_shape(),
        num_actions=game.num_actions(),
        num_res_blocks=model_config.get("num_res_blocks", 5),
        num_channels=model_config.get("num_channels", 64),
    )
    network.net.load_state_dict(data["model_state_dict"])

    network.eval_mode()
    ai_agent = AlphaZeroAgent(
        network=network,
        mcts_config=MCTSConfig(
            num_simulations=parsed.simulations,
            dirichlet_epsilon=0.0,  # no exploration noise during play
        ),
    )
    human_agent = HumanAgent()

    if parsed.human_player == 0:
        agents = [human_agent, ai_agent]
    else:
        agents = [ai_agent, human_agent]

    state = game.new_initial_state()
    while state.current_player() != TERMINAL_PLAYER:
        player = state.current_player()
        action = agents[player].select_action(state, game)
        state.apply_action(action)

    print(state)
    returns = state.returns()
    if returns[parsed.human_player] > 0:
        print("You win!")
    elif returns[parsed.human_player] < 0:
        print("AI wins!")
    else:
        print("Draw!")


if __name__ == "__main__":
    main()
