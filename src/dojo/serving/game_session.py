"""Game session management for human vs AI play."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dojo.algorithms.alphazero.agent import AlphaZeroAgent
from dojo.algorithms.mcts import MCTSConfig
from dojo.core.game import Game, GameState
from dojo.core.model import PolicyValueNetwork
from dojo.core.types import TERMINAL_PLAYER, Action


@dataclass
class GameSession:
    """A single human-vs-AI game session."""

    session_id: str
    game: Game
    state: GameState
    ai_agent: AlphaZeroAgent | None
    human_player: int = 0  # which player the human controls

    def get_state_dict(self) -> dict[str, Any]:
        """Serialize current state for the frontend.

        Uses game-generic methods where possible. The ``board`` key is
        game-specific — each game's renderer expects its own format.
        """
        current = self.state.current_player()
        is_terminal = current == TERMINAL_PLAYER

        result: dict[str, Any] = {
            "session_id": self.session_id,
            "current_player": current,
            "is_terminal": is_terminal,
            "legal_actions": self.state.legal_actions() if current >= 0 else [],
            "returns": self.state.returns() if is_terminal else None,
            "human_player": self.human_player,
            "is_human_turn": current == self.human_player,
            "game_name": self.game.name(),
        }

        # Game-specific board serialization for the frontend renderer
        if hasattr(self.state, "board"):
            result["board"] = self.state.board.tolist()

        return result

    def apply_human_action(self, action: Action) -> dict[str, Any]:
        """Apply human move, then AI move if applicable."""
        if self.state.current_player() != self.human_player:
            return {"error": "Not your turn"}
        if action not in self.state.legal_actions():
            return {"error": f"Illegal action: {action}"}

        self.state.apply_action(action)
        ai_action = None

        # If game is not over and it's AI's turn, let AI play
        if (
            self.ai_agent is not None
            and self.state.current_player() >= 0
            and self.state.current_player() != self.human_player
        ):
            ai_action = self.ai_agent.select_action(self.state, self.game)
            self.state.apply_action(ai_action)

        result = self.get_state_dict()
        if ai_action is not None:
            result["ai_action"] = ai_action
        return result


class SessionManager:
    """Manages active game sessions."""

    def __init__(
        self,
        game: Game,
        network: PolicyValueNetwork | None = None,
        mcts_simulations: int = 50,
    ) -> None:
        self._game = game
        self._network = network
        self._mcts_sims = mcts_simulations
        self._sessions: dict[str, GameSession] = {}

    def create_session(self, human_player: int = 0) -> GameSession:
        """Create a new game session."""
        session_id = uuid.uuid4().hex[:8]

        ai_agent = AlphaZeroAgent(
            network=self._network,
            mcts_config=MCTSConfig(
                num_simulations=self._mcts_sims,
                dirichlet_epsilon=0.0,
            ),
            temperature=0.0,
        ) if self._network else None

        state = self._game.new_initial_state()
        session = GameSession(
            session_id=session_id,
            game=self._game,
            state=state,
            ai_agent=ai_agent,
            human_player=human_player,
        )

        # If AI goes first, let it play
        if ai_agent and human_player != 0:
            ai_action = ai_agent.select_action(state, self._game)
            state.apply_action(ai_action)

        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> GameSession | None:
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
