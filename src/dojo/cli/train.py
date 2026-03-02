"""CLI entry point for training."""

from __future__ import annotations

import argparse
from pathlib import Path

from dojo.algorithms.alphazero.trainer import AlphaZeroTrainer
from dojo.core.config import load_config


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train an AlphaZero agent")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume from checkpoint. No value: auto-detect latest. "
        "With path: resume from that checkpoint file.",
    )
    parsed = parser.parse_args(args)

    config = load_config(parsed.config)
    trainer = AlphaZeroTrainer(config)

    resume_from: Path | None = None
    if parsed.resume is True:
        resume_from = True  # type: ignore[assignment]
    elif parsed.resume is not None:
        resume_from = Path(parsed.resume)

    trainer.train(resume_from=resume_from)


if __name__ == "__main__":
    main()
