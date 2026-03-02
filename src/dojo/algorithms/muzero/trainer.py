"""MuZero training loop with unrolled dynamics loss.

Unlike AlphaZero which trains on independent (obs, policy, value) samples,
MuZero trains on K-step unrolled trajectories:

1. Encode the real observation at time t with the representation network
2. For k = 0..K-1, unroll the dynamics network with the recorded actions
3. At each unrolled step, compute prediction loss (policy + value) and
   reward loss against the recorded targets

This trains all three networks (representation, dynamics, prediction) jointly
through gradient backpropagation across the unrolled steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

import dojo.games  # noqa: F401 — trigger game registration
from dojo.algorithms.alphazero.agent import AlphaZeroAgent
from dojo.algorithms.mcts import MCTSConfig as MCTSRuntimeConfig
from dojo.algorithms.muzero.model import MuZeroNetwork
from dojo.algorithms.muzero.self_play import MuZeroGameRecord, generate_muzero_self_play_game
from dojo.core.agent import RandomAgent
from dojo.core.config import TrainingConfig
from dojo.core.model import NetworkOutput, PolicyValueNetwork
from dojo.games.registry import create_game
from dojo.training.checkpoint import load_checkpoint_data, save_checkpoint
from dojo.training.evaluator import evaluate_agents
from dojo.training.logging import TrainingLogger


@dataclass
class MuZeroSample:
    """A single unrolled training sample for MuZero.

    Contains the initial observation plus K steps of targets for
    actions, policies, values, and rewards.
    """

    observation: npt.NDArray[np.float32]  # initial observation (C, H, W)
    actions: list[int]  # K actions to unroll
    policy_targets: list[npt.NDArray[np.float32]]  # K+1 policies (including initial)
    value_targets: list[float]  # K+1 values
    reward_targets: list[float]  # K rewards


class MuZeroReplayBuffer:
    """Replay buffer that stores full game trajectories.

    Samples unrolled K-step subsequences for MuZero training.
    When a position near the end of a game is sampled, the trajectory
    is padded with absorbing states (zero policy, zero reward, terminal value).
    """

    def __init__(self, capacity: int = 5000, unroll_steps: int = 5) -> None:
        self._capacity = capacity
        self._games: list[MuZeroGameRecord] = []
        self._write_idx = 0
        self._unroll_steps = unroll_steps
        self._rng = np.random.default_rng()
        self._total_positions = 0

    def add_game(self, record: MuZeroGameRecord) -> None:
        if len(self._games) < self._capacity:
            self._total_positions += len(record)
            self._games.append(record)
        else:
            self._total_positions += len(record) - len(self._games[self._write_idx])
            self._games[self._write_idx] = record
        self._write_idx = (self._write_idx + 1) % self._capacity

    def sample_batch(self, batch_size: int, num_actions: int) -> list[MuZeroSample]:
        """Sample a batch of unrolled trajectories.

        For each sample, picks a random game and a random starting position,
        then extracts K steps of (action, policy, value, reward) targets.
        """
        # Pre-allocate uniform policy used for absorbing-state padding
        uniform_policy = np.full(num_actions, 1.0 / num_actions, dtype=np.float32)

        samples = []
        for _ in range(batch_size):
            # Pick a random game
            game_idx = int(self._rng.integers(0, len(self._games)))
            game = self._games[game_idx]

            # Pick a random starting position
            t = int(self._rng.integers(0, len(game)))

            # Build targets for K unrolled steps
            obs = game.observations[t]
            actions: list[int] = []
            policy_targets: list[npt.NDArray[np.float32]] = []
            value_targets: list[float] = []
            reward_targets: list[float] = []

            # Initial policy and value targets (at position t)
            policy_targets.append(game.policies[t])
            value_targets.append(game.returns[game.players[t]])

            for k in range(self._unroll_steps):
                idx = t + k
                if idx < len(game):
                    actions.append(game.actions[idx])
                    reward_targets.append(game.rewards[idx])

                    # Policy and value targets at position t+k+1
                    next_idx = idx + 1
                    if next_idx < len(game):
                        policy_targets.append(game.policies[next_idx])
                        value_targets.append(game.returns[game.players[next_idx]])
                    else:
                        # Past end of game: absorbing state
                        policy_targets.append(uniform_policy)
                        # Use the final return for the last acting player
                        value_targets.append(game.returns[game.players[-1]])
                else:
                    # Past end of game: pad with absorbing state
                    actions.append(0)
                    reward_targets.append(0.0)
                    policy_targets.append(uniform_policy)
                    value_targets.append(game.returns[game.players[-1]])

            samples.append(MuZeroSample(
                observation=obs,
                actions=actions,
                policy_targets=policy_targets,
                value_targets=value_targets,
                reward_targets=reward_targets,
            ))

        return samples

    @property
    def total_positions(self) -> int:
        """Total number of game positions stored."""
        return self._total_positions

    def __len__(self) -> int:
        return len(self._games)


class MuZeroTrainer:
    """Orchestrates the MuZero training pipeline.

    Each iteration:
    1. Self-play: generate games using MuZero MCTS → store in replay buffer
    2. Train: sample unrolled trajectories, update all three networks jointly
    3. Evaluate: pit against a random baseline
    4. Checkpoint: save if win rate exceeds threshold
    """

    def __init__(self, config: TrainingConfig, unroll_steps: int = 5) -> None:
        self.config = config
        self.unroll_steps = unroll_steps
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu"
            else "cpu"
        )

        # Game
        self.game = create_game(config.game_name, **config.game_params)

        # Network
        self.network = MuZeroNetwork(
            observation_shape=self.game.observation_tensor_shape(),
            num_actions=self.game.num_actions(),
            num_res_blocks=config.model.num_res_blocks,
            num_channels=config.model.num_channels,
            device=self.device,
        )

        # Optimizer (all three networks share parameters in MuZeroNet)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Replay buffer (stores full games, not individual samples)
        self.replay_buffer = MuZeroReplayBuffer(
            capacity=config.replay_buffer_capacity,
            unroll_steps=unroll_steps,
        )

        # MCTS config
        self.mcts_config = config.mcts.to_runtime()

        # Logging
        self.logger = TrainingLogger()

        # Best-checkpoint tracking
        self._best_checkpoint_path: Path | None = None
        self._best_network: _MuZeroPolicyValueAdapter | None = None

    def train(self) -> None:
        """Run the full training loop."""
        # Create a new run directory for this training session
        game = self.game.name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self._run_dir = Path(self.config.checkpoint_dir) / game / f"muzero_{game}_{timestamp}"

        print(f"Training MuZero on {self.game.name()} | device={self.device}")
        print(f"  iterations={self.config.num_iterations}")
        print(f"  self_play_games={self.config.num_self_play_games}")
        print(f"  unroll_steps={self.unroll_steps}")
        print()

        for iteration in range(1, self.config.num_iterations + 1):
            print(f"=== Iteration {iteration}/{self.config.num_iterations} ===")

            self._self_play(iteration)
            avg_loss = self._train_network(iteration)
            win_rate = self._evaluate(iteration)

            # Evaluate vs best checkpoint + promotion
            best_promoted = False
            if self._best_network is None:
                best_path = self._save_best_checkpoint(iteration)
                best_promoted = True
                print(
                    f"  [best-model] no previous best — saved initial best "
                    f"checkpoint {best_path}",
                    flush=True,
                )
            else:
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

            self.logger.log_scalar(
                "eval/best_promoted", int(best_promoted), iteration
            )

            if iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint(iteration)

            print(
                f"  loss={avg_loss:.4f} | "
                f"win_rate_vs_random={win_rate:.1%} | "
                f"buffer_games={len(self.replay_buffer)} "
                f"({self.replay_buffer.total_positions} positions)"
            )
            print()

        self.logger.close()
        print("Training complete.")

    def _self_play(self, iteration: int) -> None:
        """Generate self-play games and add to replay buffer."""
        self.network.eval_mode()
        rng = np.random.default_rng(iteration)

        for _ in range(self.config.num_self_play_games):
            record = generate_muzero_self_play_game(
                game=self.game,
                network=self.network,
                mcts_config=self.mcts_config,
                temperature_threshold=self.config.temperature_threshold,
                rng=rng,
            )
            self.replay_buffer.add_game(record)

        self.logger.log_scalar(
            "self_play/buffer_games", len(self.replay_buffer), iteration
        )

    def _train_network(self, iteration: int) -> float:
        """Train with unrolled dynamics loss. Returns average loss."""
        if len(self.replay_buffer) < 1:
            return 0.0

        self.network.train_mode()
        num_actions = self.game.num_actions()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward_loss = 0.0
        num_batches = 0

        for _ in range(self.config.num_train_steps_per_iteration):
            batch = self.replay_buffer.sample_batch(
                self.config.batch_size, num_actions
            )
            loss, p_loss, v_loss, r_loss = self._compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += p_loss
            total_value_loss += v_loss
            total_reward_loss += r_loss
            num_batches += 1

        n = max(num_batches, 1)
        avg_loss = total_loss / n
        self.logger.log_scalars(
            {
                "train/loss": avg_loss,
                "train/policy_loss": total_policy_loss / n,
                "train/value_loss": total_value_loss / n,
                "train/reward_loss": total_reward_loss / n,
            },
            iteration,
        )
        return avg_loss

    def _compute_loss(
        self, batch: list[MuZeroSample]
    ) -> tuple[torch.Tensor, float, float, float]:
        """Compute the MuZero unrolled loss for a batch.

        For each sample:
        1. Encode observation → hidden state via representation
        2. Predict policy + value at step 0
        3. Unroll K steps: dynamics(hidden, action) → next_hidden, reward;
           predict(next_hidden) → policy, value
        4. Sum cross-entropy (policy) + MSE (value) + MSE (reward) at each step

        Returns:
            total_loss (differentiable), and scalar policy/value/reward losses
        """
        b = len(batch)
        k = self.unroll_steps
        device = self.device

        # Collate observations
        obs = torch.from_numpy(
            np.array([s.observation for s in batch], dtype=np.float32)
        ).to(device)

        # Step 0: representation + prediction
        hidden = self.network.net.represent(obs)
        policy_logits, value = self.network.net.predict(hidden)

        # Policy targets at step 0
        policy_target_0 = torch.from_numpy(
            np.array([s.policy_targets[0] for s in batch], dtype=np.float32)
        ).to(device)
        value_target_0 = torch.tensor(
            [s.value_targets[0] for s in batch], dtype=torch.float32, device=device
        )

        policy_loss = -torch.sum(
            policy_target_0 * F.log_softmax(policy_logits, dim=1)
        ) / b
        value_loss = F.mse_loss(value.squeeze(-1), value_target_0)
        reward_loss = torch.tensor(0.0, device=device)

        # Unroll K steps
        for step in range(k):
            actions = torch.tensor(
                [s.actions[step] for s in batch], dtype=torch.long, device=device
            )
            hidden, reward_pred, policy_logits, value = (
                self.network.net.recurrent_inference(hidden, actions)
            )

            # Targets at step+1
            policy_target = torch.from_numpy(
                np.array(
                    [s.policy_targets[step + 1] for s in batch], dtype=np.float32
                )
            ).to(device)
            value_target = torch.tensor(
                [s.value_targets[step + 1] for s in batch],
                dtype=torch.float32,
                device=device,
            )
            reward_target = torch.tensor(
                [s.reward_targets[step] for s in batch],
                dtype=torch.float32,
                device=device,
            )

            policy_loss += (
                -torch.sum(policy_target * F.log_softmax(policy_logits, dim=1)) / b
            )
            value_loss += F.mse_loss(value.squeeze(-1), value_target)
            reward_loss += F.mse_loss(reward_pred.squeeze(-1), reward_target)

        total_loss = policy_loss + value_loss + reward_loss
        return (
            total_loss,
            policy_loss.item(),
            value_loss.item(),
            reward_loss.item(),
        )

    def _evaluate(self, iteration: int) -> float:
        """Evaluate current network vs random agent using AlphaZero-style MCTS.

        For evaluation, we use the MuZero network's initial_inference
        as a one-step policy-value evaluator (no dynamics unrolling needed
        for a fair comparison with a random baseline).
        """
        # Wrap MuZero network as a PolicyValueNetwork for the AlphaZero agent
        muzero_pv = _MuZeroPolicyValueAdapter(self.network)

        agent = AlphaZeroAgent(
            network=muzero_pv,
            mcts_config=MCTSRuntimeConfig(
                num_simulations=self.config.eval_mcts_num_simulations
            ),
            temperature=0.0,
        )
        random_agent = RandomAgent(seed=iteration)

        result = evaluate_agents(
            game=self.game,
            agent1=agent,
            agent2=random_agent,
            num_games=self.config.num_eval_games,
        )

        self.logger.log_scalars(
            {
                "eval/win_rate": result.win_rate,
                "eval/wins": result.wins,
                "eval/draws": result.draws,
            },
            iteration,
        )
        return result.win_rate

    def _load_best_network(self, path: Path) -> _MuZeroPolicyValueAdapter:
        """Load a checkpoint into a fresh MuZero network on CPU, wrapped as PolicyValueNetwork."""
        data = load_checkpoint_data(path, device=torch.device("cpu"))
        model_config = data.get("model_config", self.config.model.model_dump())
        network = MuZeroNetwork(
            observation_shape=self.game.observation_tensor_shape(),
            num_actions=self.game.num_actions(),
            num_res_blocks=model_config["num_res_blocks"],
            num_channels=model_config["num_channels"],
            device=torch.device("cpu"),
        )
        network.net.load_state_dict(data["model_state_dict"])
        network.eval_mode()
        return _MuZeroPolicyValueAdapter(network)

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

        mcts_config = MCTSRuntimeConfig(
            num_simulations=self.config.eval_mcts_num_simulations,
            dirichlet_epsilon=0.0,
        )
        current_agent = AlphaZeroAgent(
            network=_MuZeroPolicyValueAdapter(self.network),
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

    def _save_checkpoint(self, iteration: int) -> None:
        path = self._run_dir / f"iter{iteration:04d}.pt"
        save_checkpoint(
            self.network.net,
            self.optimizer,
            iteration,
            path,
            model_config=self.config.model.model_dump(),
        )


class _MuZeroPolicyValueAdapter(PolicyValueNetwork):
    """Adapts a MuZeroNetwork to the PolicyValueNetwork interface.

    Uses representation + prediction (no dynamics) to produce a
    single-step (policy, value) estimate, allowing MuZero to be
    evaluated with the same AlphaZero MCTS agent used for baselines.
    """

    def __init__(self, muzero_network: MuZeroNetwork) -> None:
        self._net = muzero_network

    def predict(self, observation: npt.NDArray[np.float32]) -> NetworkOutput:
        output = self._net.initial_inference(observation)
        return NetworkOutput(policy=output.policy, value=output.value)

    def to_device(self, device: torch.device) -> None:
        self._net.to_device(device)

    def train_mode(self) -> None:
        self._net.train_mode()

    def eval_mode(self) -> None:
        self._net.eval_mode()

    def parameters(self):
        return self._net.parameters()
