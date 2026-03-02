"""CLI entry point for the web server."""

from __future__ import annotations

import argparse


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start the web UI server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--game", type=str, default="connect_four")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=50,
        help="Number of MCTS simulations per AI move in serving",
    )
    parsed = parser.parse_args(args)

    import uvicorn

    from dojo.serving.app import create_app

    app = create_app(
        game_name=parsed.game,
        checkpoint_path=parsed.checkpoint,
        mcts_simulations=parsed.mcts_simulations,
    )
    uvicorn.run(app, host=parsed.host, port=parsed.port)


if __name__ == "__main__":
    main()
