"""Pydantic configuration classes with YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class MCTSConfig(BaseModel):
    """MCTS hyperparameters (Pydantic config, convertible to runtime dataclass)."""

    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0

    def to_runtime(self):
        """Convert to the runtime MCTSConfig dataclass used by run_mcts."""
        from dojo.algorithms.mcts import MCTSConfig as RuntimeMCTSConfig

        return RuntimeMCTSConfig(**self.model_dump())


class ModelConfig(BaseModel):
    """Neural network architecture config."""

    num_res_blocks: int = 5
    num_channels: int = 64


class TrainingConfig(BaseModel):
    """Full training configuration."""

    # Game
    game_name: str = "connect_four"
    game_params: dict[str, Any] = {}

    # Model
    model: ModelConfig = ModelConfig()

    # MCTS
    mcts: MCTSConfig = MCTSConfig()

    # Self-play
    num_self_play_games: int = 100
    num_self_play_workers: int = 1
    temperature_threshold: int = 30

    # Training
    num_iterations: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_train_steps_per_iteration: int = 10

    # Replay buffer
    replay_buffer_capacity: int = 100_000

    # Evaluation
    num_eval_games: int = 40
    num_eval_workers: int = 1
    eval_mcts_num_simulations: int = 25
    best_model_win_rate_threshold: float = 0.55

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1

    # Device
    device: str = "cuda"  # "cuda" or "cpu"


def load_config(path: str | Path) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return TrainingConfig(**data)
