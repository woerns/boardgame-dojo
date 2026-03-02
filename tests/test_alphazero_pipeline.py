"""End-to-end tests for the AlphaZero training pipeline."""

import numpy as np
import pytest
import torch

import dojo.games.connect_four  # noqa: F401
from dojo.algorithms.alphazero.agent import AlphaZeroAgent
from dojo.algorithms.alphazero.self_play import (
    AlphaZeroNodeEvaluator,
    generate_self_play_game,
)
from dojo.algorithms.mcts import MCTSConfig
from dojo.core.agent import RandomAgent
from dojo.core.config import TrainingConfig, load_config
from dojo.core.replay_buffer import GameRecord, ReplayBuffer, TrainingSample
from dojo.games.registry import create_game
from dojo.models.alphazero_net import AlphaZeroNetwork
from dojo.training.evaluator import ArenaResult, evaluate_agents


class TestReplayBuffer:
    """Replay buffer tests."""

    def test_game_record_to_samples(self):
        record = GameRecord()
        obs = np.zeros((3, 6, 7), dtype=np.float32)
        policy = np.ones(7, dtype=np.float32) / 7
        record.add_step(obs, policy, player=0)
        record.add_step(obs, policy, player=1)
        record.set_returns([1.0, -1.0])

        samples = record.to_samples()
        assert len(samples) == 2
        assert samples[0].value_target == 1.0   # player 0 won
        assert samples[1].value_target == -1.0   # player 1 lost

    def test_buffer_add_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        record = GameRecord()
        obs = np.zeros((3, 6, 7), dtype=np.float32)
        policy = np.ones(7, dtype=np.float32) / 7
        for i in range(10):
            record.add_step(obs, policy, player=i % 2)
        record.set_returns([0.0, 0.0])
        buf.add_game(record)

        assert len(buf) == 10
        obs_b, pol_b, val_b = buf.sample_batch(4)
        assert obs_b.shape == (4, 3, 6, 7)
        assert pol_b.shape == (4, 7)
        assert val_b.shape == (4,)

    def test_buffer_capacity_eviction(self):
        buf = ReplayBuffer(capacity=5)
        for _ in range(3):
            record = GameRecord()
            obs = np.zeros((3, 6, 7), dtype=np.float32)
            policy = np.ones(7, dtype=np.float32) / 7
            for j in range(4):
                record.add_step(obs, policy, player=j % 2)
            record.set_returns([0.0, 0.0])
            buf.add_game(record)
        # 12 samples added, capacity=5 → only 5 remain
        assert len(buf) == 5


class TestSelfPlay:
    """Self-play generation tests."""

    def test_generate_game(self):
        game = create_game("connect_four")
        network = AlphaZeroNetwork(
            observation_shape=game.observation_tensor_shape(),
            num_actions=game.num_actions(),
            num_res_blocks=1,
            num_channels=8,
        )
        network.eval_mode()

        config = MCTSConfig(num_simulations=5)
        record = generate_self_play_game(
            game=game,
            network=network,
            mcts_config=config,
            rng=np.random.default_rng(42),
        )

        assert record.returns is not None
        assert len(record.observations) > 0
        assert len(record.observations) == len(record.policies)
        # Policies should sum to ~1
        for p in record.policies:
            np.testing.assert_almost_equal(p.sum(), 1.0, decimal=5)

    def test_evaluator_interface(self):
        game = create_game("connect_four")
        network = AlphaZeroNetwork(
            observation_shape=game.observation_tensor_shape(),
            num_actions=game.num_actions(),
            num_res_blocks=1,
            num_channels=8,
        )
        network.eval_mode()
        evaluator = AlphaZeroNodeEvaluator(network)

        state = game.new_initial_state()
        prior, value = evaluator.evaluate(state)
        assert prior.shape == (7,)
        assert -1.0 <= value <= 1.0
        assert prior.sum() > 0  # priors should be positive after exp


class TestAlphaZeroAgent:
    """Agent inference tests."""

    def test_agent_selects_legal_action(self):
        game = create_game("connect_four")
        network = AlphaZeroNetwork(
            observation_shape=game.observation_tensor_shape(),
            num_actions=game.num_actions(),
            num_res_blocks=1,
            num_channels=8,
        )
        network.eval_mode()

        agent = AlphaZeroAgent(
            network=network,
            mcts_config=MCTSConfig(num_simulations=5),
            seed=42,
        )

        state = game.new_initial_state()
        action = agent.select_action(state, game)
        assert action in state.legal_actions()


class TestArenaEvaluation:
    """Arena evaluation tests."""

    def test_random_vs_random(self):
        game = create_game("connect_four")
        result = evaluate_agents(
            game=game,
            agent1=RandomAgent(seed=0),
            agent2=RandomAgent(seed=1),
            num_games=10,
        )
        assert result.total == 10
        assert result.wins + result.losses + result.draws == 10
        assert 0.0 <= result.win_rate <= 1.0


class TestConfig:
    """Configuration tests."""

    def test_default_config(self):
        config = TrainingConfig()
        assert config.game_name == "connect_four"
        assert config.mcts.num_simulations == 100
        assert config.eval_mcts_num_simulations == 25
        assert config.best_model_win_rate_threshold == 0.55

    def test_load_yaml(self, tmp_path):
        yaml_content = """
game_name: connect_four
num_iterations: 5
mcts:
  num_simulations: 10
eval_mcts_num_simulations: 11
device: cpu
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml_content)
        config = load_config(config_path)
        assert config.num_iterations == 5
        assert config.mcts.num_simulations == 10
        assert config.eval_mcts_num_simulations == 11
        assert config.device == "cpu"


class TestEndToEnd:
    """Smoke test: run a tiny training loop."""

    def test_mini_training_loop(self, tmp_path):
        """Train for 2 iterations to verify pipeline + best checkpoint."""
        config = TrainingConfig(
            game_name="connect_four",
            model={"num_res_blocks": 1, "num_channels": 8},
            mcts={"num_simulations": 5},
            num_self_play_games=2,
            num_iterations=2,
            batch_size=4,
            num_train_steps_per_iteration=2,
            num_eval_games=2,
            device="cpu",
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        from dojo.algorithms.alphazero.trainer import AlphaZeroTrainer

        trainer = AlphaZeroTrainer(config)
        trainer.train()

        # Verify some data was generated
        assert len(trainer.replay_buffer) > 0
        # best.pt should exist after iteration 1 bootstraps it
        assert (trainer._run_dir / "best.pt").exists()


class TestEvalVsBest:
    """Best-checkpoint evaluation and promotion tests."""

    def test_eval_vs_best_promotes(self, tmp_path):
        """With threshold=0.0, promotion should always succeed."""
        config = TrainingConfig(
            game_name="connect_four",
            model={"num_res_blocks": 1, "num_channels": 8},
            mcts={"num_simulations": 5},
            num_self_play_games=2,
            num_iterations=2,
            batch_size=4,
            num_train_steps_per_iteration=2,
            num_eval_games=2,
            best_model_win_rate_threshold=0.0,
            device="cpu",
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        from dojo.algorithms.alphazero.trainer import AlphaZeroTrainer

        trainer = AlphaZeroTrainer(config)
        trainer.train()

        # best.pt should exist
        best_path = trainer._run_dir / "best.pt"
        assert best_path.exists()
        # With threshold=0.0, the best network should have been promoted
        # at least once (iteration 1 bootstraps, iteration 2 should promote)
        assert trainer._best_checkpoint_path == best_path
        assert trainer._best_network is not None


class TestParallelSelfPlay:
    """Self-play worker splitting tests."""

    def test_split_games_evenly(self):
        from dojo.algorithms.alphazero.trainer import _split_games

        assert _split_games(10, 3) == [4, 3, 3]
        assert _split_games(4, 8) == [1, 1, 1, 1, 0, 0, 0, 0]
