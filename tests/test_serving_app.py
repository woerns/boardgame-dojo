"""Tests for serving app checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch

import dojo.games.connect_four  # noqa: F401
from dojo.games.registry import create_game
from dojo.models.alphazero_net import AlphaZeroNetwork
from dojo.serving.app import create_app
from dojo.training.checkpoint import save_checkpoint


def _write_checkpoint(
    path: Path, num_res_blocks: int, num_channels: int
) -> None:
    game = create_game("connect_four")
    network = AlphaZeroNetwork(
        observation_shape=game.observation_tensor_shape(),
        num_actions=game.num_actions(),
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    save_checkpoint(
        network.net,
        optimizer,
        iteration=1,
        path=path,
        model_config={
            "num_res_blocks": num_res_blocks,
            "num_channels": num_channels,
        },
    )


def test_create_app_loads_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    _write_checkpoint(checkpoint_path, num_res_blocks=2, num_channels=16)

    app = create_app(
        game_name="connect_four",
        checkpoint_path=str(checkpoint_path),
        mcts_simulations=7,
    )
    paths = {route.path for route in app.routes}
    assert "/api/game/new" in paths
    assert "/api/game/{session_id}" in paths
    assert "/ws/game/{session_id}" in paths
