"""Tests for Connect Four game logic."""

import numpy as np
import pytest

from dojo.core.types import TERMINAL_PLAYER
from dojo.games.connect_four.game import ConnectFourGame, ConnectFourState
from dojo.games.registry import create_game, list_games


class TestConnectFourBasics:
    """Basic game mechanics."""

    def test_initial_state(self, c4_game: ConnectFourGame):
        state = c4_game.new_initial_state()
        assert state.current_player() == 0
        assert state.move_count == 0
        assert len(state.legal_actions()) == 7

    def test_legal_actions_mask(self, c4_game: ConnectFourGame):
        state = c4_game.new_initial_state()
        mask = state.legal_actions_mask()
        assert mask.shape == (7,)
        assert mask.all()  # All columns open initially

    def test_alternating_players(self, c4_game: ConnectFourGame):
        state = c4_game.new_initial_state()
        assert state.current_player() == 0
        state.apply_action(0)
        assert state.current_player() == 1
        state.apply_action(1)
        assert state.current_player() == 0

    def test_column_fills_up(self, c4_game: ConnectFourGame):
        state = c4_game.new_initial_state()
        # Fill column 0 with alternating pieces (6 rows)
        for i in range(6):
            assert 0 in state.legal_actions()
            state.apply_action(0)
        # Column 0 should now be full
        assert 0 not in state.legal_actions()
        assert not state.legal_actions_mask()[0]

    def test_clone_independence(self, c4_game: ConnectFourGame):
        state = c4_game.new_initial_state()
        state.apply_action(3)
        clone = state.clone()
        clone.apply_action(4)
        # Original should be unaffected
        assert state.move_count == 1
        assert clone.move_count == 2


class TestConnectFourWinDetection:
    """Win detection across all directions."""

    def _play_sequence(self, game: ConnectFourGame, actions: list[int]) -> ConnectFourState:
        state = game.new_initial_state()
        for a in actions:
            state.apply_action(a)
        return state

    def test_vertical_win(self, c4_game: ConnectFourGame):
        # P0 plays col 0, P1 plays col 1, alternating.
        # P0: col 0 four times → vertical win
        actions = [0, 1, 0, 1, 0, 1, 0]  # P0 stacks 4 in col 0
        state = self._play_sequence(c4_game, actions)
        assert state.current_player() == TERMINAL_PLAYER
        assert state.returns() == [1.0, -1.0]

    def test_horizontal_win(self, c4_game: ConnectFourGame):
        # P0: cols 0,1,2,3 — but P1 must play somewhere else in between
        # P0: 0, P1: 0, P0: 1, P1: 1, P0: 2, P1: 2, P0: 3 → P0 horizontal win
        actions = [0, 0, 1, 1, 2, 2, 3]
        state = self._play_sequence(c4_game, actions)
        assert state.current_player() == TERMINAL_PLAYER
        assert state.returns() == [1.0, -1.0]

    def test_diagonal_win_ascending(self, c4_game: ConnectFourGame):
        # P0 diagonal: (0,0), (1,1), (2,2), (3,3)
        # P1 wastes moves on col 6 to avoid accidental horizontal win
        # Move 0 (P0): col 0 → (0,0)=P0
        # Move 1 (P1): col 1 → (0,1)=P1
        # Move 2 (P0): col 1 → (1,1)=P0
        # Move 3 (P1): col 2 → (0,2)=P1
        # Move 4 (P0): col 2 → (1,2)=P0
        # Move 5 (P1): col 6 → (0,6)=P1  (waste — avoids row-0 horizontal)
        # Move 6 (P0): col 2 → (2,2)=P0
        # Move 7 (P1): col 3 → (0,3)=P1
        # Move 8 (P0): col 3 → (1,3)=P0
        # Move 9 (P1): col 3 → (2,3)=P1
        # Move10 (P0): col 3 → (3,3)=P0 ✓ ascending diagonal!
        actions = [0, 1, 1, 2, 2, 6, 2, 3, 3, 3, 3]
        state = self._play_sequence(c4_game, actions)
        assert state.current_player() == TERMINAL_PLAYER
        assert state.returns() == [1.0, -1.0]

    def test_diagonal_win_descending(self, c4_game: ConnectFourGame):
        # Mirror of ascending: P0 gets (3,0), (2,1), (1,2), (0,3)
        # Build columns so P0 pieces land on the right rows
        # Col 3: P0 at (0,3)
        # Col 2: P1 at (0,2), P0 at (1,2)
        # Col 1: fill to row 2 for P0 → P1, P0, P0 — wait, need (2,1) for P0
        #   Col 1: P1(0,1), P1(1,1), P0(2,1)
        # Col 0: fill to row 3 for P0 → need 3 below
        #   Col 0: P1(0,0), P1(1,0), P1(2,0), P0(3,0)
        # Sequence:
        # P0: col3 → (0,3)=1
        # P1: col2 → (0,2)=2
        # P0: col2 → (1,2)=1
        # P1: col1 → (0,1)=2
        # P0: col4 → (0,4)=1  (waste move)
        # P1: col1 → (1,1)=2
        # P0: col1 → (2,1)=1
        # P1: col0 → (0,0)=2
        # P0: col5 → (0,5)=1  (waste move)
        # P1: col0 → (1,0)=2
        # P0: col6 → (0,6)=1  (waste move)
        # P1: col0 → (2,0)=2
        # P0: col0 → (3,0)=1  ✓ diagonal!
        actions = [3, 2, 2, 1, 4, 1, 1, 0, 5, 0, 6, 0, 0]
        state = self._play_sequence(c4_game, actions)
        assert state.current_player() == TERMINAL_PLAYER
        assert state.returns() == [1.0, -1.0]

    def test_player1_wins(self, c4_game: ConnectFourGame):
        # P1 wins vertically in col 1
        # P0: col 0, P1: col 1 × 4, P0: col 0, etc.
        actions = [0, 1, 0, 1, 0, 1, 2, 1]  # P1 stacks 4 in col 1
        state = self._play_sequence(c4_game, actions)
        assert state.current_player() == TERMINAL_PLAYER
        assert state.returns() == [-1.0, 1.0]


class TestConnectFourDraw:
    """Draw detection."""

    def test_draw_small_board(self):
        # Use a tiny 2x2 board with win_length=3 (impossible to win)
        game = ConnectFourGame(rows=2, cols=2, win_length=3)
        state = game.new_initial_state()
        # Fill all 4 cells: P0(0,0), P1(0,1), P0(1,0), P1(1,1)
        for col in [0, 1, 0, 1]:
            state.apply_action(col)
        assert state.current_player() == TERMINAL_PLAYER
        assert state.returns() == [0.0, 0.0]


class TestConnectFourObservation:
    """Observation tensor correctness."""

    def test_observation_shape(self, c4_game: ConnectFourGame):
        state = c4_game.new_initial_state()
        obs = state.observation_tensor()
        assert obs.shape == (3, 6, 7)
        assert obs.dtype == np.float32

    def test_observation_empty_board(self, c4_game: ConnectFourGame):
        state = c4_game.new_initial_state()
        obs = state.observation_tensor()
        # No pieces → channels 0 and 1 are all zeros
        assert obs[0].sum() == 0
        assert obs[1].sum() == 0
        # Current player is 0 (it's player 0's turn and we're viewing as player 0)
        assert obs[2, 0, 0] == 1.0  # turn indicator = my turn

    def test_observation_perspective(self, c4_game: ConnectFourGame):
        state = c4_game.new_initial_state()
        state.apply_action(3)  # P0 plays col 3

        # From P0's perspective (channel 0 = my pieces, channel 1 = opponent)
        obs_p0 = state.observation_tensor(player=0)
        assert obs_p0[0, 0, 3] == 1.0  # P0's piece
        assert obs_p0[1, 0, 3] == 0.0

        # From P1's perspective (channel 0 = my pieces, channel 1 = opponent)
        obs_p1 = state.observation_tensor(player=1)
        assert obs_p1[0, 0, 3] == 0.0
        assert obs_p1[1, 0, 3] == 1.0  # P0's piece is opponent's from P1 view


class TestRegistry:
    """Game registry tests."""

    def test_connect_four_registered(self):
        assert "connect_four" in list_games()

    def test_create_game(self):
        game = create_game("connect_four")
        assert game.name() == "connect_four"
        assert game.num_actions() == 7

    def test_create_game_with_params(self):
        game = create_game("connect_four", rows=4, cols=5, win_length=3)
        assert game.num_actions() == 5

    def test_unknown_game(self):
        with pytest.raises(KeyError, match="Unknown game"):
            create_game("chess")


class TestConnectFourStr:
    """String representation."""

    def test_str_initial(self, c4_game: ConnectFourGame):
        state = c4_game.new_initial_state()
        s = str(state)
        assert "." in s  # Empty cells shown as dots
        assert "0" in s  # Column numbers
