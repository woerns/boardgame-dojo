"""Replay buffer for storing and sampling self-play data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class TrainingSample:
    """A single training example extracted from a game.

    observation: board state as seen by the current player
    policy_target: MCTS visit-count distribution (search policy)
    value_target: final game outcome from this player's perspective
    """

    observation: npt.NDArray[np.float32]
    policy_target: npt.NDArray[np.float32]
    value_target: float


@dataclass
class GameRecord:
    """A complete self-play game.

    Stores per-move observations, MCTS policies, and the players who acted.
    The final outcome is filled in after the game ends.
    """

    observations: list[npt.NDArray[np.float32]] = field(default_factory=list)
    policies: list[npt.NDArray[np.float32]] = field(default_factory=list)
    players: list[int] = field(default_factory=list)
    returns: list[float] | None = None  # filled after game ends

    def add_step(
        self,
        observation: npt.NDArray[np.float32],
        policy: npt.NDArray[np.float32],
        player: int,
    ) -> None:
        self.observations.append(observation)
        self.policies.append(policy)
        self.players.append(player)

    def set_returns(self, returns: list[float]) -> None:
        self.returns = returns

    def to_samples(self) -> list[TrainingSample]:
        """Convert to training samples using the final returns."""
        assert self.returns is not None, "Game not finished — call set_returns() first"
        samples = []
        for obs, policy, player in zip(self.observations, self.policies, self.players):
            value = self.returns[player]
            samples.append(TrainingSample(
                observation=obs,
                policy_target=policy,
                value_target=value,
            ))
        return samples


class ReplayBuffer:
    """Fixed-capacity buffer of training samples with uniform sampling.

    Uses a list-backed circular buffer for O(1) random access (deque
    indexing is O(n), which is too slow for random sampling at scale).
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = capacity
        self._samples: list[TrainingSample] = []
        self._write_idx = 0
        self._rng = np.random.default_rng()

    def add_game(self, record: GameRecord) -> int:
        """Add all samples from a completed game.

        Returns:
            Number of samples added.
        """
        samples = record.to_samples()
        for sample in samples:
            if len(self._samples) < self._capacity:
                self._samples.append(sample)
            else:
                self._samples[self._write_idx] = sample
            self._write_idx = (self._write_idx + 1) % self._capacity
        return len(samples)

    def sample_batch(
        self, batch_size: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Sample a random batch of training data.

        Returns:
            observations: (B, C, H, W)
            policy_targets: (B, num_actions)
            value_targets: (B,)
        """
        indices = self._rng.integers(0, len(self._samples), size=batch_size)
        obs = np.array([self._samples[i].observation for i in indices], dtype=np.float32)
        policies = np.array([self._samples[i].policy_target for i in indices], dtype=np.float32)
        values = np.array([self._samples[i].value_target for i in indices], dtype=np.float32)
        return obs, policies, values

    def __len__(self) -> int:
        return len(self._samples)
