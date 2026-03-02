"""Main CLI dispatcher for `dojo` command."""

from __future__ import annotations

import sys


def app() -> None:
    if len(sys.argv) < 2:
        print("Usage: dojo <command> [args]")
        print("Commands: train, play, serve")
        sys.exit(1)

    command = sys.argv[1]
    # Remove the subcommand from argv so argparse in each module works correctly
    sys.argv = [f"dojo {command}"] + sys.argv[2:]

    if command == "train":
        from dojo.cli.train import main
        main()
    elif command == "play":
        from dojo.cli.play import main
        main()
    elif command == "serve":
        from dojo.cli.serve import main
        main()
    else:
        print(f"Unknown command: {command}")
        print("Commands: train, play, serve")
        sys.exit(1)


if __name__ == "__main__":
    app()
