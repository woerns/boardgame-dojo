"""Tests for MuZero: model, MCTS, self-play, and training."""

from __future__ import annotations

import numpy as np
import torch

import dojo.games  # noqa: F401
from dojo.algorithms.mcts import MCTSConfig, get_action_probs
from dojo.algorithms.muzero.model import MuZeroNet, MuZeroNetwork, _scale_hidden
from dojo.algorithms.muzero.self_play import (
    MuZeroGameRecord,
    generate_muzero_self_play_game,
    run_muzero_mcts,
)
from dojo.algorithms.muzero.trainer import MuZeroReplayBuffer, MuZeroSample
from dojo.games.registry import create_game


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _small_game():
    """A tiny 4x4 Connect Four for fast tests."""
    return create_game("connect_four", rows=4, cols=4, win_length=3)


def _small_network(game):
    """A tiny MuZero network for testing."""
    return MuZeroNetwork(
        observation_shape=game.observation_tensor_shape(),
        num_actions=game.num_actions(),
        num_res_blocks=1,
        num_channels=8,
    )


# ---------------------------------------------------------------------------
# MuZeroNet tests
# ---------------------------------------------------------------------------

class TestMuZeroNet:
    def test_representation_shape(self):
        net = MuZeroNet(
            observation_shape=(3, 4, 4),
            num_actions=4,
            num_res_blocks=1,
            num_channels=8,
        )
        obs = torch.randn(2, 3, 4, 4)
        hidden = net.represent(obs)
        assert hidden.shape == (2, 8, 4, 4)

    def test_dynamics_shape(self):
        net = MuZeroNet(
            observation_shape=(3, 4, 4),
            num_actions=4,
            num_res_blocks=1,
            num_channels=8,
        )
        hidden = torch.randn(2, 8, 4, 4)
        action = torch.tensor([1, 3])
        next_hidden, reward = net.dynamics(hidden, action)
        assert next_hidden.shape == (2, 8, 4, 4)
        assert reward.shape == (2, 1)

    def test_prediction_shape(self):
        net = MuZeroNet(
            observation_shape=(3, 4, 4),
            num_actions=4,
            num_res_blocks=1,
            num_channels=8,
        )
        hidden = torch.randn(2, 8, 4, 4)
        policy_logits, value = net.predict(hidden)
        assert policy_logits.shape == (2, 4)
        assert value.shape == (2, 1)

    def test_initial_inference(self):
        net = MuZeroNet(
            observation_shape=(3, 4, 4),
            num_actions=4,
            num_res_blocks=1,
            num_channels=8,
        )
        obs = torch.randn(1, 3, 4, 4)
        hidden, policy, value = net.initial_inference(obs)
        assert hidden.shape == (1, 8, 4, 4)
        assert policy.shape == (1, 4)
        assert value.shape == (1, 1)

    def test_recurrent_inference(self):
        net = MuZeroNet(
            observation_shape=(3, 4, 4),
            num_actions=4,
            num_res_blocks=1,
            num_channels=8,
        )
        hidden = torch.randn(1, 8, 4, 4)
        action = torch.tensor([2])
        next_hidden, reward, policy, value = net.recurrent_inference(hidden, action)
        assert next_hidden.shape == (1, 8, 4, 4)
        assert reward.shape == (1, 1)
        assert policy.shape == (1, 4)
        assert value.shape == (1, 1)

    def test_hidden_state_scaling(self):
        """Hidden states should be in [0, 1] after scaling."""
        hidden = torch.randn(3, 8, 4, 4) * 100  # large range
        scaled = _scale_hidden(hidden)
        assert scaled.min() >= -1e-6
        assert scaled.max() <= 1.0 + 1e-6

    def test_value_in_range(self):
        """Value output should be in [-1, 1] (tanh)."""
        net = MuZeroNet(
            observation_shape=(3, 4, 4),
            num_actions=4,
            num_res_blocks=1,
            num_channels=8,
        )
        obs = torch.randn(5, 3, 4, 4)
        _, _, value = net.initial_inference(obs)
        assert (value >= -1.0).all() and (value <= 1.0).all()


# ---------------------------------------------------------------------------
# MuZeroNetwork wrapper tests
# ---------------------------------------------------------------------------

class TestMuZeroNetwork:
    def test_initial_inference_numpy(self):
        game = _small_game()
        network = _small_network(game)
        network.eval_mode()

        state = game.new_initial_state()
        obs = state.observation_tensor()
        output = network.initial_inference(obs)

        assert output.policy.shape == (game.num_actions(),)
        assert abs(output.policy.sum() - 1.0) < 1e-5
        assert -1.0 <= output.value <= 1.0
        assert output.hidden_state.shape[1:] == (8, 4, 4)

    def test_recurrent_inference_numpy(self):
        game = _small_game()
        network = _small_network(game)
        network.eval_mode()

        state = game.new_initial_state()
        obs = state.observation_tensor()
        initial = network.initial_inference(obs)

        output = network.recurrent_inference(initial.hidden_state, action=0)
        assert output.policy.shape == (game.num_actions(),)
        assert -1.0 <= output.value <= 1.0
        assert -1.0 <= output.reward <= 1.0


# ---------------------------------------------------------------------------
# MuZero MCTS tests
# ---------------------------------------------------------------------------

class TestMuZeroMCTS:
    def test_mcts_produces_valid_probs(self):
        game = _small_game()
        network = _small_network(game)
        network.eval_mode()

        state = game.new_initial_state()
        config = MCTSConfig(num_simulations=10)
        rng = np.random.default_rng(42)

        root = run_muzero_mcts(
            observation=state.observation_tensor(),
            legal_mask=state.legal_actions_mask(),
            network=network,
            config=config,
            num_actions=game.num_actions(),
            current_player=state.current_player(),
            rng=rng,
        )

        probs = get_action_probs(root, game.num_actions(), temperature=1.0)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert all(p >= 0 for p in probs)

    def test_mcts_visit_counts(self):
        game = _small_game()
        network = _small_network(game)
        network.eval_mode()

        config = MCTSConfig(num_simulations=20)
        state = game.new_initial_state()
        rng = np.random.default_rng(0)

        root = run_muzero_mcts(
            observation=state.observation_tensor(),
            legal_mask=state.legal_actions_mask(),
            network=network,
            config=config,
            num_actions=game.num_actions(),
            current_player=0,
            rng=rng,
        )

        total_visits = sum(c.visit_count for c in root.children.values())
        assert total_visits == config.num_simulations

    def test_illegal_actions_masked_at_root(self):
        """Only legal actions should appear as root children."""
        game = _small_game()
        network = _small_network(game)
        network.eval_mode()

        state = game.new_initial_state()
        legal = set(state.legal_actions())
        config = MCTSConfig(num_simulations=5)
        rng = np.random.default_rng(0)

        root = run_muzero_mcts(
            observation=state.observation_tensor(),
            legal_mask=state.legal_actions_mask(),
            network=network,
            config=config,
            num_actions=game.num_actions(),
            current_player=0,
            rng=rng,
        )

        assert set(root.children.keys()) == legal


# ---------------------------------------------------------------------------
# Self-play tests
# ---------------------------------------------------------------------------

class TestMuZeroSelfPlay:
    def test_generate_game(self):
        game = _small_game()
        network = _small_network(game)
        network.eval_mode()

        config = MCTSConfig(num_simulations=5)
        rng = np.random.default_rng(42)

        record = generate_muzero_self_play_game(
            game=game,
            network=network,
            mcts_config=config,
            rng=rng,
        )

        assert record.returns is not None
        assert len(record.observations) == len(record.actions)
        assert len(record.observations) == len(record.policies)
        assert len(record.observations) == len(record.rewards)
        assert len(record.observations) > 0

        # Check all actions are valid integers
        for action in record.actions:
            assert 0 <= action < game.num_actions()

        # Check policies are valid distributions
        for policy in record.policies:
            assert abs(policy.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Replay buffer tests
# ---------------------------------------------------------------------------

class TestMuZeroReplayBuffer:
    def _make_record(self, num_steps: int = 10, num_actions: int = 4):
        """Create a dummy MuZeroGameRecord."""
        record = MuZeroGameRecord()
        for i in range(num_steps):
            obs = np.random.randn(3, 4, 4).astype(np.float32)
            action = i % num_actions
            policy = np.ones(num_actions, dtype=np.float32) / num_actions
            reward = 0.0 if i < num_steps - 1 else 1.0
            player = i % 2
            record.add_step(obs, action, policy, reward, player)
        record.set_returns([1.0, -1.0])
        return record

    def test_add_and_sample(self):
        buf = MuZeroReplayBuffer(capacity=100, unroll_steps=3)
        record = self._make_record()
        buf.add_game(record)

        samples = buf.sample_batch(batch_size=4, num_actions=4)
        assert len(samples) == 4

        for s in samples:
            assert len(s.actions) == 3  # unroll_steps
            assert len(s.policy_targets) == 4  # unroll_steps + 1
            assert len(s.value_targets) == 4
            assert len(s.reward_targets) == 3

    def test_capacity_eviction(self):
        buf = MuZeroReplayBuffer(capacity=3, unroll_steps=2)
        for _ in range(5):
            buf.add_game(self._make_record())
        assert len(buf) == 3  # oldest 2 evicted

    def test_padding_near_end(self):
        """Sampling near the end of a game should pad with absorbing state."""
        buf = MuZeroReplayBuffer(capacity=10, unroll_steps=5)
        record = self._make_record(num_steps=3)  # only 3 steps
        buf.add_game(record)

        samples = buf.sample_batch(batch_size=10, num_actions=4)
        for s in samples:
            assert len(s.actions) == 5
            assert len(s.policy_targets) == 6
            assert len(s.value_targets) == 6
            assert len(s.reward_targets) == 5


# ---------------------------------------------------------------------------
# End-to-end mini training test
# ---------------------------------------------------------------------------

class TestMuZeroEndToEnd:
    def test_mini_training_loop(self):
        """Smoke test: run a tiny training loop to verify nothing crashes."""
        from dojo.core.config import MCTSConfig as PydanticMCTSConfig, ModelConfig, TrainingConfig

        config = TrainingConfig(
            game_name="connect_four",
            game_params={"rows": 4, "cols": 4, "win_length": 3},
            model=ModelConfig(num_res_blocks=1, num_channels=8),
            mcts=PydanticMCTSConfig(num_simulations=5),
            num_iterations=1,
            num_self_play_games=2,
            batch_size=4,
            num_train_steps_per_iteration=2,
            num_eval_games=2,
            replay_buffer_capacity=100,
            checkpoint_interval=1,  # checkpoint every iteration
            device="cpu",
        )

        from dojo.algorithms.muzero.trainer import MuZeroTrainer

        trainer = MuZeroTrainer(config, unroll_steps=3)
        trainer.train()

        # Verify some self-play data was collected
        assert len(trainer.replay_buffer) > 0
