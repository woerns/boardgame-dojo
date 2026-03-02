/**
 * Boardgame Dojo — frontend application.
 *
 * Communicates with the server via REST (new game) + WebSocket (gameplay).
 */

let ws = null;
let currentState = null;

// Registry of game renderers
const renderers = {
    connect_four: ConnectFourRenderer,
};

function getRenderer(gameName) {
    return renderers[gameName] || ConnectFourRenderer;
}

function updateUI(state) {
    currentState = state;

    if (state.error) {
        document.getElementById('status').textContent = state.error;
        return;
    }

    const renderer = getRenderer(state.game_name);
    const container = document.getElementById('board-container');
    const status = document.getElementById('status');

    renderer.render(container, state, sendAction);
    status.textContent = renderer.getStatusMessage(state);
}

function sendAction(action) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!currentState || !currentState.is_human_turn) return;

    // Disable interaction while waiting for response
    document.getElementById('status').textContent = 'AI is thinking...';

    ws.send(JSON.stringify({ action: action }));
}

async function startNewGame(humanPlayer) {
    // Close existing WebSocket
    if (ws) {
        ws.close();
        ws = null;
    }

    document.getElementById('status').textContent = 'Starting new game...';

    try {
        // Create game session via REST
        const response = await fetch(`/api/game/new?human_player=${humanPlayer}`, {
            method: 'POST',
        });
        const state = await response.json();

        if (state.error) {
            document.getElementById('status').textContent = state.error;
            return;
        }

        // Connect WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/game/${state.session_id}`;
        ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateUI(data);
        };

        ws.onerror = () => {
            document.getElementById('status').textContent = 'Connection error';
        };

        ws.onclose = () => {
            // Only show disconnected if game is still in progress
            if (currentState && !currentState.is_terminal) {
                document.getElementById('status').textContent = 'Disconnected';
            }
        };
    } catch (err) {
        document.getElementById('status').textContent = 'Failed to connect to server';
    }
}
