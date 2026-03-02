"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import dojo.games  # noqa: F401
from dojo.games.registry import create_game
from dojo.models.alphazero_net import AlphaZeroNetwork
from dojo.serving.game_session import SessionManager
from dojo.serving.routes import router, set_session_manager
from dojo.training.checkpoint import load_checkpoint_data


def create_app(
    game_name: str = "connect_four",
    checkpoint_path: str = "",
    mcts_simulations: int = 50,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Boardgame Dojo", version="0.1.0")

    # Create game
    game = create_game(game_name)

    if not checkpoint_path:
        raise ValueError("checkpoint_path is required")

    # Build network from checkpoint
    obs_shape = game.observation_tensor_shape()
    num_actions = game.num_actions()
    data = load_checkpoint_data(Path(checkpoint_path))
    model_config = data.get("model_config", {})
    network = AlphaZeroNetwork(
        obs_shape,
        num_actions,
        num_res_blocks=model_config.get("num_res_blocks", 5),
        num_channels=model_config.get("num_channels", 64),
    )
    network.net.load_state_dict(data["model_state_dict"])
    network.eval_mode()

    # Session manager
    manager = SessionManager(
        game=game,
        network=network,
        mcts_simulations=mcts_simulations,
    )
    set_session_manager(manager)

    # Routes
    app.include_router(router)

    # Static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
