"""AlphaZero training loop."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import dojo.games  # noqa: F401 — trigger game registration
from dojo.algorithms.alphazero.agent import AlphaZeroAgent
from dojo.algorithms.alphazero.self_play import generate_self_play_game
from dojo.algorithms.mcts import MCTSConfig as MCTSRuntimeConfig
from dojo.core.agent import RandomAgent
from dojo.core.config import TrainingConfig
from dojo.core.replay_buffer import ReplayBuffer
from dojo.games.registry import create_game
from dojo.models.alphazero_net import AlphaZeroNetwork
from dojo.training.checkpoint import load_checkpoint, load_checkpoint_data, save_checkpoint
from dojo.training.evaluator import ArenaResult, evaluate_agents
from dojo.training.logging import TrainingLogger


_WORKER_GAME = None
_WORKER_NETWORK = None
_WORKER_MCTS_CONFIG = None
_WORKER_TEMPERATURE_THRESHOLD = 30

_EVAL_WORKER_GAME = None
_EVAL_WORKER_NETWORK = None
_EVAL_WORKER_MCTS_SIMS = 25


def _create_worker_game_and_network(
    game_name: str,
    game_params: dict,
    model_config: dict,
    model_state_dict: dict,
) -> tuple:
    """Common setup for spawned worker processes: create game + network on CPU.

    Handles game registration, thread limiting, network reconstruction from
    state_dict, and eval-mode activation.  Called by both self-play and eval
    worker initializers.
    """
    import dojo.games  # noqa: F401 — trigger game registration in worker process

    # Prevent CPU oversubscription across many worker processes.
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    game = create_game(game_name, **game_params)
    network = AlphaZeroNetwork(
        observation_shape=game.observation_tensor_shape(),
        num_actions=game.num_actions(),
        num_res_blocks=model_config["num_res_blocks"],
        num_channels=model_config["num_channels"],
        device=torch.device("cpu"),
    )
    network.net.load_state_dict(model_state_dict)
    network.eval_mode()
    return game, network


def _init_self_play_worker(
    game_name: str,
    game_params: dict,
    model_config: dict,
    mcts_config: MCTSRuntimeConfig,
    temperature_threshold: int,
    model_state_dict: dict,
) -> None:
    """Initialize reusable worker state for self-play generation."""
    global _WORKER_GAME, _WORKER_NETWORK, _WORKER_MCTS_CONFIG, _WORKER_TEMPERATURE_THRESHOLD

    _WORKER_GAME, _WORKER_NETWORK = _create_worker_game_and_network(
        game_name, game_params, model_config, model_state_dict,
    )
    _WORKER_MCTS_CONFIG = mcts_config
    _WORKER_TEMPERATURE_THRESHOLD = temperature_threshold


def _generate_self_play_chunk(task: tuple[int, int]) -> list:
    """Generate a chunk of self-play games in a worker process."""
    num_games, seed = task
    if _WORKER_GAME is None or _WORKER_NETWORK is None or _WORKER_MCTS_CONFIG is None:
        raise RuntimeError("Self-play worker not initialized")

    rng = np.random.default_rng(seed)
    records = []
    for _ in range(num_games):
        record = generate_self_play_game(
            game=_WORKER_GAME,
            network=_WORKER_NETWORK,
            mcts_config=_WORKER_MCTS_CONFIG,
            temperature_threshold=_WORKER_TEMPERATURE_THRESHOLD,
            rng=rng,
        )
        records.append(record)
    return records


def _split_games(num_games: int, num_workers: int) -> list[int]:
    """Split game count into near-equal worker chunks."""
    base = num_games // num_workers
    remainder = num_games % num_workers
    return [base + (1 if i < remainder else 0) for i in range(num_workers)]


def _build_parallel_chunks(total_games: int, num_workers: int) -> list[int]:
    """Build smaller chunks for smoother progress updates and load balancing."""
    num_chunks = min(total_games, max(num_workers * 3, num_workers))
    return [c for c in _split_games(total_games, num_chunks) if c > 0]


def _init_eval_worker(
    game_name: str,
    game_params: dict,
    model_config: dict,
    model_state_dict: dict,
    eval_mcts_num_simulations: int,
) -> None:
    """Initialize reusable worker state for parallel evaluation."""
    global _EVAL_WORKER_GAME, _EVAL_WORKER_NETWORK, _EVAL_WORKER_MCTS_SIMS

    _EVAL_WORKER_GAME, _EVAL_WORKER_NETWORK = _create_worker_game_and_network(
        game_name, game_params, model_config, model_state_dict,
    )
    _EVAL_WORKER_MCTS_SIMS = eval_mcts_num_simulations


def _play_eval_chunk(task: tuple[int, int, int]) -> tuple[int, int, int]:
    """Play a chunk of evaluation games in a worker process.

    Args:
        task: (start_index, num_games, seed) — start_index preserves
              first-mover alternation across chunks.

    Returns:
        (wins, losses, draws) from the AlphaZero agent's perspective.
    """
    start_index, num_games, seed = task
    if _EVAL_WORKER_GAME is None or _EVAL_WORKER_NETWORK is None:
        raise RuntimeError("Eval worker not initialized")

    agent = AlphaZeroAgent(
        network=_EVAL_WORKER_NETWORK,
        mcts_config=MCTSRuntimeConfig(
            num_simulations=_EVAL_WORKER_MCTS_SIMS,
            dirichlet_epsilon=0.0,
        ),
        temperature=0.0,
    )
    random_agent = RandomAgent(seed=seed)

    wins = losses = draws = 0
    for i in range(num_games):
        game_index = start_index + i
        # Alternate first-mover, consistent with serial evaluate_agents
        if game_index % 2 == 0:
            agents = [agent, random_agent]
            agent_player = 0
        else:
            agents = [random_agent, agent]
            agent_player = 1

        agents[0].reset()
        agents[1].reset()

        state = _EVAL_WORKER_GAME.new_initial_state()
        while state.current_player() >= 0:
            player = state.current_player()
            action = agents[player].select_action(state, _EVAL_WORKER_GAME)
            state.apply_action(action)

        agent_return = state.returns()[agent_player]
        if agent_return > 0:
            wins += 1
        elif agent_return < 0:
            losses += 1
        else:
            draws += 1

    return wins, losses, draws


class AlphaZeroTrainer:
    """Orchestrates the AlphaZero training pipeline.

    Each iteration:
    1. Self-play: generate games → store in replay buffer
    2. Train: sample batches, update network
    3. Evaluate: pit new network vs best, promote if better
    4. Checkpoint: save best model
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu"
            else "cpu"
        )

        # Game
        self.game = create_game(config.game_name, **config.game_params)

        # Network
        self.network = AlphaZeroNetwork(
            observation_shape=self.game.observation_tensor_shape(),
            num_actions=self.game.num_actions(),
            num_res_blocks=config.model.num_res_blocks,
            num_channels=config.model.num_channels,
            device=self.device,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)

        # MCTS config (runtime version)
        self.mcts_config = config.mcts.to_runtime()

        # Logging
        self.logger = TrainingLogger()

        # Best-checkpoint tracking
        self._best_checkpoint_path: Path | None = None
        self._best_network: AlphaZeroNetwork | None = None

    def _find_latest_checkpoint(self) -> Path | None:
        """Find the most recent checkpoint across all runs for this game."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        game = self.game.name()
        candidates = sorted(checkpoint_dir.glob(f"{game}/alphazero_{game}_*/iter*.pt"))
        return candidates[-1] if candidates else None

    def train(self, resume_from: Path | bool | None = None) -> None:
        """Run the full training loop.

        Args:
            resume_from: If a Path, load that checkpoint. If True, auto-detect
                the latest checkpoint. If None, start from scratch.
        """
        start_iteration = 0

        if resume_from is True:
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path is None:
                print("No existing checkpoint found — starting from scratch.")
            else:
                resume_from = checkpoint_path

        if isinstance(resume_from, Path):
            start_iteration = load_checkpoint(
                self.network.net, self.optimizer, resume_from, self.device
            )
            print(f"Resuming from {resume_from} (iteration {start_iteration})")
            # Check for best.pt in the resumed checkpoint's directory
            best_path = resume_from.parent / "best.pt"
            if best_path.exists():
                self._best_network = self._load_best_network(best_path)
                self._best_checkpoint_path = best_path
                print(f"  Loaded best checkpoint from {best_path}")

        # Create a new run directory for this training session
        game = self.game.name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self._run_dir = Path(self.config.checkpoint_dir) / game / f"alphazero_{game}_{timestamp}"

        print(f"Training AlphaZero on {self.game.name()} | device={self.device}")
        print(f"  iterations={self.config.num_iterations}")
        print(f"  self_play_games={self.config.num_self_play_games}")
        print(f"  self_play_workers={self.config.num_self_play_workers}")
        print(f"  eval_workers={self.config.num_eval_workers}")
        print(f"  mcts_sims={self.config.mcts.num_simulations}")
        print()

        for iteration in range(start_iteration + 1, self.config.num_iterations + 1):
            print(f"=== Iteration {iteration}/{self.config.num_iterations} ===")
            iteration_start = time.perf_counter()

            # Phase 1: Self-play
            self_play_start = time.perf_counter()
            added_samples = self._self_play(iteration)
            self_play_time = time.perf_counter() - self_play_start
            print(
                f"  [self-play] completed in {self_play_time:.2f}s | "
                f"samples_added={added_samples} | buffer_size={len(self.replay_buffer)}",
                flush=True,
            )

            # Phase 2: Train network
            train_start = time.perf_counter()
            avg_loss = self._train_network(iteration)
            train_time = time.perf_counter() - train_start

            # Phase 3: Evaluate vs random agent
            eval_start = time.perf_counter()
            win_rate = self._evaluate(iteration)
            eval_time = time.perf_counter() - eval_start

            # Phase 4: Evaluate vs best checkpoint + promotion
            eval_vs_best_time = 0.0
            best_promoted = False
            if self._best_network is None:
                # First time: bootstrap by saving current as best
                eval_vs_best_start = time.perf_counter()
                best_path = self._save_best_checkpoint(iteration)
                eval_vs_best_time = time.perf_counter() - eval_vs_best_start
                best_promoted = True
                print(
                    f"  [best-model] no previous best — saved initial best "
                    f"checkpoint {best_path}",
                    flush=True,
                )
            else:
                eval_vs_best_start = time.perf_counter()
                win_rate_vs_best = self._evaluate_vs_best(iteration)
                if (
                    win_rate_vs_best is not None
                    and win_rate_vs_best >= self.config.best_model_win_rate_threshold
                ):
                    best_path = self._save_best_checkpoint(iteration)
                    best_promoted = True
                    print(
                        f"  [best-model] promoted! win_rate_vs_best="
                        f"{win_rate_vs_best:.1%} >= "
                        f"{self.config.best_model_win_rate_threshold:.1%} — "
                        f"saved {best_path}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [best-model] not promoted (win_rate_vs_best="
                        f"{win_rate_vs_best:.1%} < "
                        f"{self.config.best_model_win_rate_threshold:.1%})",
                        flush=True,
                    )
                eval_vs_best_time = time.perf_counter() - eval_vs_best_start

            self.logger.log_scalar(
                "eval/best_promoted", int(best_promoted), iteration
            )

            # Phase 5: Periodic checkpoint
            checkpoint_time = 0.0
            if iteration % self.config.checkpoint_interval == 0:
                checkpoint_start = time.perf_counter()
                checkpoint_path = self._save_checkpoint(iteration)
                checkpoint_time = time.perf_counter() - checkpoint_start
                print(
                    f"  [checkpoint] saved {checkpoint_path} in {checkpoint_time:.2f}s",
                    flush=True,
                )

            iteration_time = time.perf_counter() - iteration_start

            print(
                f"  loss={avg_loss:.4f} | "
                f"win_rate_vs_random={win_rate:.1%} | "
                f"buffer_size={len(self.replay_buffer)}"
            )
            print(
                f"  timing: self_play={self_play_time:.2f}s | "
                f"train={train_time:.2f}s | "
                f"eval={eval_time:.2f}s | "
                f"eval_vs_best={eval_vs_best_time:.2f}s | "
                f"checkpoint={checkpoint_time:.2f}s | "
                f"iteration={iteration_time:.2f}s",
                flush=True,
            )
            print()

        self.logger.close()
        print("Training complete.")

    def _self_play(self, iteration: int) -> int:
        """Generate self-play games and add to replay buffer.

        Returns:
            Number of samples added to the replay buffer.
        """
        self.network.eval_mode()
        samples_added = 0
        total_games = self.config.num_self_play_games
        worker_count = max(1, self.config.num_self_play_workers)
        if worker_count == 1 or total_games <= 1:
            print(
                f"  [self-play] running {total_games} games in serial mode...",
                flush=True,
            )
            rng = np.random.default_rng(iteration)
            progress_interval = max(1, total_games // 10)
            serial_start = time.perf_counter()
            for game_idx in range(total_games):
                record = generate_self_play_game(
                    game=self.game,
                    network=self.network,
                    mcts_config=self.mcts_config,
                    temperature_threshold=self.config.temperature_threshold,
                    rng=rng,
                )
                samples_added += self.replay_buffer.add_game(record)
                done = game_idx + 1
                if done % progress_interval == 0 or done == total_games:
                    elapsed = time.perf_counter() - serial_start
                    speed = done / elapsed if elapsed > 0 else 0.0
                    print(
                        f"    [self-play] progress {done}/{total_games} games "
                        f"({speed:.2f} games/s)",
                        flush=True,
                    )
        else:
            # Use CPU workers for self-play to avoid CUDA multi-process contention/issues.
            print(
                f"  [self-play] running {total_games} games with "
                f"{worker_count} worker processes...",
                flush=True,
            )
            model_state_dict = self._cpu_state_dict()
            chunks = _build_parallel_chunks(total_games, worker_count)
            tasks = [
                (chunk_size, iteration * 10_000 + task_idx)
                for task_idx, chunk_size in enumerate(chunks)
            ]

            completed_games = 0
            parallel_start = time.perf_counter()
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
                initializer=_init_self_play_worker,
                initargs=(
                    self.config.game_name,
                    self.config.game_params,
                    self.config.model.model_dump(),
                    self.mcts_config,
                    self.config.temperature_threshold,
                    model_state_dict,
                ),
            ) as executor:
                futures = [executor.submit(_generate_self_play_chunk, task) for task in tasks]
                for future in as_completed(futures):
                    records = future.result()
                    completed_games += len(records)
                    for record in records:
                        samples_added += self.replay_buffer.add_game(record)
                    elapsed = time.perf_counter() - parallel_start
                    speed = completed_games / elapsed if elapsed > 0 else 0.0
                    print(
                        f"    [self-play] progress {completed_games}/{total_games} games "
                        f"({speed:.2f} games/s)",
                        flush=True,
                    )

        self.logger.log_scalar(
            "self_play/buffer_size", len(self.replay_buffer), iteration
        )
        return samples_added

    def _train_network(self, iteration: int) -> float:
        """Train the network on replay buffer data. Returns average loss."""
        if len(self.replay_buffer) < self.config.batch_size:
            print(
                "  [train] skipped (replay buffer smaller than batch size).",
                flush=True,
            )
            return 0.0

        self.network.train_mode()
        print(
            f"  [train] running {self.config.num_train_steps_per_iteration} optimization steps...",
            flush=True,
        )
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        progress_interval = max(1, self.config.num_train_steps_per_iteration // 5)
        train_start = time.perf_counter()

        for batch_idx in range(self.config.num_train_steps_per_iteration):
            obs, policy_targets, value_targets = self.replay_buffer.sample_batch(
                self.config.batch_size
            )

            obs_t = torch.from_numpy(obs).to(self.device)
            policy_t = torch.from_numpy(policy_targets).to(self.device)
            value_t = torch.from_numpy(value_targets).to(self.device)

            # Forward pass
            policy_logits, value_pred = self.network.net(obs_t)

            # Loss: cross-entropy for policy + MSE for value
            policy_loss = -torch.sum(policy_t * F.log_softmax(policy_logits, dim=1)) / obs_t.size(0)
            value_loss = F.mse_loss(value_pred.squeeze(-1), value_t)
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

            done = batch_idx + 1
            if done % progress_interval == 0 or done == self.config.num_train_steps_per_iteration:
                elapsed = time.perf_counter() - train_start
                print(
                    f"    [train] progress {done}/{self.config.num_train_steps_per_iteration} "
                    f"steps | avg_loss={total_loss / max(num_batches, 1):.4f}",
                    flush=True,
                )

        avg_loss = total_loss / max(num_batches, 1)
        avg_policy_loss = total_policy_loss / max(num_batches, 1)
        avg_value_loss = total_value_loss / max(num_batches, 1)
        self.logger.log_scalars(
            {
                "train/loss": avg_loss,
                "train/policy_loss": avg_policy_loss,
                "train/value_loss": avg_value_loss,
            },
            iteration,
        )
        print(
            f"  [train] completed in {time.perf_counter() - train_start:.2f}s | "
            f"avg_loss={avg_loss:.4f} | policy_loss={avg_policy_loss:.4f} | "
            f"value_loss={avg_value_loss:.4f}",
            flush=True,
        )
        return avg_loss

    def _evaluate(self, iteration: int) -> float:
        """Evaluate current network vs random agent."""
        self.network.eval_mode()
        total_games = self.config.num_eval_games
        worker_count = max(1, self.config.num_eval_workers)
        print(
            f"  [eval] running {total_games} games vs random "
            f"({self.config.eval_mcts_num_simulations} MCTS sims/move)...",
            flush=True,
        )
        eval_start = time.perf_counter()

        if worker_count == 1 or total_games <= 1:
            progress_interval = max(1, total_games // 5)

            def _on_eval_progress(done: int, total: int) -> None:
                if done % progress_interval == 0 or done == total:
                    elapsed = time.perf_counter() - eval_start
                    speed = done / elapsed if elapsed > 0 else 0.0
                    print(
                        f"    [eval] progress {done}/{total} games "
                        f"({speed:.2f} games/s)",
                        flush=True,
                    )

            agent = AlphaZeroAgent(
                network=self.network,
                mcts_config=MCTSRuntimeConfig(
                    num_simulations=self.config.eval_mcts_num_simulations,
                    dirichlet_epsilon=0.0,  # no exploration noise during evaluation
                ),
                temperature=0.0,
            )
            random_agent = RandomAgent(seed=iteration)

            result = evaluate_agents(
                game=self.game,
                agent1=agent,
                agent2=random_agent,
                num_games=total_games,
                progress_callback=_on_eval_progress,
            )
        else:
            print(
                f"  [eval] using {worker_count} worker processes...",
                flush=True,
            )
            model_state_dict = self._cpu_state_dict()
            chunks = _build_parallel_chunks(total_games, worker_count)

            # Compute starting game indices so each chunk preserves first-mover alternation
            start_indices = []
            idx = 0
            for chunk_size in chunks:
                start_indices.append(idx)
                idx += chunk_size

            tasks = [
                (start_idx, chunk_size, iteration * 10_000 + task_idx)
                for task_idx, (start_idx, chunk_size) in enumerate(
                    zip(start_indices, chunks)
                )
            ]

            wins = losses = draws = 0
            completed_games = 0
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
                initializer=_init_eval_worker,
                initargs=(
                    self.config.game_name,
                    self.config.game_params,
                    self.config.model.model_dump(),
                    model_state_dict,
                    self.config.eval_mcts_num_simulations,
                ),
            ) as executor:
                futures = [executor.submit(_play_eval_chunk, task) for task in tasks]
                for future in as_completed(futures):
                    w, l, d = future.result()
                    wins += w
                    losses += l
                    draws += d
                    completed_games += w + l + d
                    elapsed = time.perf_counter() - eval_start
                    speed = completed_games / elapsed if elapsed > 0 else 0.0
                    print(
                        f"    [eval] progress {completed_games}/{total_games} games "
                        f"({speed:.2f} games/s)",
                        flush=True,
                    )

            result = ArenaResult(wins=wins, losses=losses, draws=draws)

        self.logger.log_scalars(
            {
                "eval/win_rate": result.win_rate,
                "eval/wins": result.wins,
                "eval/draws": result.draws,
            },
            iteration,
        )
        print(
            f"  [eval] completed in {time.perf_counter() - eval_start:.2f}s | "
            f"wins={result.wins} losses={result.losses} draws={result.draws} | "
            f"win_rate={result.win_rate:.1%}",
            flush=True,
        )
        return result.win_rate

    def _load_best_network(self, path: Path) -> AlphaZeroNetwork:
        """Load a checkpoint into a fresh AlphaZeroNetwork on CPU."""
        data = load_checkpoint_data(path, device=torch.device("cpu"))
        model_config = data.get("model_config", self.config.model.model_dump())
        network = AlphaZeroNetwork(
            observation_shape=self.game.observation_tensor_shape(),
            num_actions=self.game.num_actions(),
            num_res_blocks=model_config["num_res_blocks"],
            num_channels=model_config["num_channels"],
            device=torch.device("cpu"),
        )
        network.net.load_state_dict(data["model_state_dict"])
        network.eval_mode()
        return network

    def _save_best_checkpoint(self, iteration: int) -> Path:
        """Save current model as best.pt and reload _best_network."""
        path = self._run_dir / "best.pt"
        save_checkpoint(
            self.network.net,
            self.optimizer,
            iteration,
            path,
            model_config=self.config.model.model_dump(),
        )
        self._best_checkpoint_path = path
        self._best_network = self._load_best_network(path)
        return path

    def _evaluate_vs_best(self, iteration: int) -> float | None:
        """Evaluate current network against best checkpoint.

        Returns win rate or None if no best exists.
        """
        if self._best_network is None:
            return None

        self.network.eval_mode()
        total_games = self.config.num_eval_games
        print(
            f"  [eval-vs-best] running {total_games} games vs best checkpoint "
            f"({self.config.eval_mcts_num_simulations} MCTS sims/move)...",
            flush=True,
        )
        eval_start = time.perf_counter()

        progress_interval = max(1, total_games // 5)

        def _on_progress(done: int, total: int) -> None:
            if done % progress_interval == 0 or done == total:
                elapsed = time.perf_counter() - eval_start
                speed = done / elapsed if elapsed > 0 else 0.0
                print(
                    f"    [eval-vs-best] progress {done}/{total} games "
                    f"({speed:.2f} games/s)",
                    flush=True,
                )

        mcts_config = MCTSRuntimeConfig(
            num_simulations=self.config.eval_mcts_num_simulations,
            dirichlet_epsilon=0.0,
        )
        current_agent = AlphaZeroAgent(
            network=self.network,
            mcts_config=mcts_config,
            temperature=0.0,
        )
        best_agent = AlphaZeroAgent(
            network=self._best_network,
            mcts_config=mcts_config,
            temperature=0.0,
        )

        result = evaluate_agents(
            game=self.game,
            agent1=current_agent,
            agent2=best_agent,
            num_games=total_games,
            progress_callback=_on_progress,
        )

        self.logger.log_scalars(
            {
                "eval/win_rate_vs_best": result.win_rate,
                "eval/wins_vs_best": result.wins,
                "eval/draws_vs_best": result.draws,
            },
            iteration,
        )
        print(
            f"  [eval-vs-best] completed in {time.perf_counter() - eval_start:.2f}s | "
            f"wins={result.wins} losses={result.losses} draws={result.draws} | "
            f"win_rate={result.win_rate:.1%}",
            flush=True,
        )
        return result.win_rate

    def _cpu_state_dict(self) -> dict:
        """Copy the network state_dict to CPU for shipping to worker processes."""
        return {k: v.detach().cpu() for k, v in self.network.net.state_dict().items()}

    def _save_checkpoint(self, iteration: int) -> Path:
        """Save model checkpoint."""
        path = self._run_dir / f"iter{iteration:04d}.pt"
        save_checkpoint(
            self.network.net,
            self.optimizer,
            iteration,
            path,
            model_config=self.config.model.model_dump(),
        )
        return path
