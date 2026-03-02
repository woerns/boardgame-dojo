# Boardgame Dojo

Boardgame Dojo is a playground to experiment with agents to play turn-based board games. It contains a reinforcement learning training framework and a web UI for interactive play against the agent.

## Features

- AlphaZero training loop (self-play, optimization, evaluation, checkpointing)
- CLI for training, terminal play, and web serving
- FastAPI + WebSocket web app for interactive play

## Requirements

- Python `>=3.10,<3.13`
- Poetry

## Install

```bash
poetry install --with dev
```

## CLI Reference

Main entrypoint:

```bash
poetry run dojo <command> [args]
```

Commands: `train`, `play`, `serve`

### `dojo train`

Train an AlphaZero agent from config.

```bash
poetry run dojo train --config configs/training/alphazero_connect_four.yaml
```

Resume from the latest checkpoint found under `checkpoint_dir`:

```bash
poetry run dojo train --config configs/training/alphazero_connect_four.yaml --resume
```

Resume from a specific checkpoint:

```bash
poetry run dojo train \
  --config configs/training/alphazero_connect_four.yaml \
  --resume checkpoints/connect_four/alphazero_connect_four_YYYYMMDD_HHMM/iter0008.pt
```

Helpful flag docs:

```bash
poetry run dojo train --help
```

### `dojo play`

Play in terminal against a trained AI (checkpoint required):

```bash
poetry run dojo play \
  --game connect_four \
  --checkpoint checkpoints/connect_four/alphazero_connect_four_YYYYMMDD_HHMM/best.pt \
  --simulations 150 \
  --human-player 0
```

Key flags:

- `--game`: game name (default `connect_four`)
- `--checkpoint`: path to `.pt` checkpoint (required)
- `--simulations`: MCTS simulations per move (default `100`)
- `--human-player`: `0` (first) or `1` (second)

### `dojo serve`

Start the FastAPI web server (checkpoint required).

```bash
poetry run dojo serve \
  --host 0.0.0.0 \
  --port 8000 \
  --game connect_four \
  --checkpoint checkpoints/connect_four/alphazero_connect_four_YYYYMMDD_HHMM/best.pt \
  --mcts-simulations 200
```

Key flags:

- `--host`: bind host (default `0.0.0.0`)
- `--port`: bind port (default `8000`)
- `--game`: game name (default `connect_four`)
- `--checkpoint`: path to `.pt` checkpoint (required)
- `--mcts-simulations`: simulations per AI move in serving (default `50`)

## Training Config

Example config: [`configs/training/alphazero_connect_four.yaml`](configs/training/alphazero_connect_four.yaml)

Important fields:

- `game_name`: game registry key
- `model`: network size (`num_res_blocks`, `num_channels`)
- `mcts`: self-play search settings
- `num_iterations`, `num_self_play_games`, `num_train_steps_per_iteration`
- `num_eval_games`, `best_model_win_rate_threshold`
- `checkpoint_dir`, `checkpoint_interval`
- `device`: `cpu` or `cuda`

## Outputs

By default, training writes to:

```text
checkpoints/<game>/alphazero_<game>_<timestamp>/
```

Typical files:

- `iter0001.pt`, `iter0002.pt`, ...
- `best.pt`

TensorBoard logs are written under `logs/`.

## Development

Run tests:

```bash
poetry run pytest tests/ -v
```

Available make targets:

```bash
make install
make test
make train
make serve CHECKPOINT=checkpoints/connect_four/alphazero_connect_four_YYYYMMDD_HHMM/best.pt
make lint
```
