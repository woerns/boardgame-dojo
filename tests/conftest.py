"""Shared test fixtures."""

import pytest

# Ensure all games are registered before tests run
import dojo.games.connect_four  # noqa: F401
from dojo.games.connect_four.game import ConnectFourGame


@pytest.fixture
def c4_game() -> ConnectFourGame:
    """Standard 6x7 Connect Four game."""
    return ConnectFourGame()


@pytest.fixture
def small_game() -> ConnectFourGame:
    """Small 4x5 Connect Four for faster tests."""
    return ConnectFourGame(rows=4, cols=5, win_length=3)
