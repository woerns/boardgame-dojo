"""FastAPI REST and WebSocket routes."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from dojo.serving.game_session import SessionManager

router = APIRouter()

# SessionManager is set by create_app()
_manager: SessionManager | None = None


def set_session_manager(manager: SessionManager) -> None:
    global _manager
    _manager = manager


@router.post("/api/game/new")
async def new_game(human_player: int = 0):
    """Create a new game session."""
    if human_player not in (0, 1):
        raise HTTPException(status_code=400, detail="human_player must be 0 or 1")
    # Session creation may run AI first-move; offload to thread pool
    loop = asyncio.get_running_loop()
    session = await loop.run_in_executor(
        None, _manager.create_session, human_player
    )
    return session.get_state_dict()


@router.get("/api/game/{session_id}")
async def get_game(session_id: str):
    """Get current game state."""
    session = _manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.get_state_dict()


@router.websocket("/ws/game/{session_id}")
async def game_ws(websocket: WebSocket, session_id: str):
    """WebSocket for real-time gameplay.

    Client sends: {"action": <int>}
    Server sends: full game state dict after each move
    """
    await websocket.accept()

    session = _manager.get_session(session_id)
    if session is None:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return

    # Send initial state
    await websocket.send_json(session.get_state_dict())

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            if "action" in msg:
                action = int(msg["action"])
                # Run AI computation in thread pool to avoid blocking event loop
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, session.apply_human_action, action
                )
                await websocket.send_json(result)
    except WebSocketDisconnect:
        _manager.remove_session(session_id)
