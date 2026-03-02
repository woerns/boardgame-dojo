/**
 * Connect Four SVG renderer.
 *
 * Renders the board as an SVG with clickable columns.
 * Board data format: 2D array [row][col], 0=empty, 1=P0, 2=P1.
 * Row 0 is the bottom row.
 */

const ConnectFourRenderer = {
    CELL_SIZE: 60,
    PADDING: 10,
    PIECE_RADIUS: 23,

    /**
     * Render the board into the given container element.
     * @param {HTMLElement} container
     * @param {object} state - game state from server
     * @param {function} onColumnClick - callback(colIndex)
     */
    render(container, state, onColumnClick) {
        const board = state.board;
        const rows = board.length;
        const cols = board[0].length;
        const cs = this.CELL_SIZE;
        const pad = this.PADDING;

        const svgWidth = cols * cs + 2 * pad;
        const svgHeight = rows * cs + 2 * pad;

        let svg = `<svg class="c4-board" width="${svgWidth}" height="${svgHeight}"
                        viewBox="0 0 ${svgWidth} ${svgHeight}"
                        xmlns="http://www.w3.org/2000/svg">`;

        // Background
        svg += `<rect x="0" y="0" width="${svgWidth}" height="${svgHeight}"
                      rx="12" fill="#0f3460"/>`;

        // Cells (render top-to-bottom visually, but board[0] is bottom)
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const visualRow = rows - 1 - r;  // flip for display
                const cx = pad + c * cs + cs / 2;
                const cy = pad + visualRow * cs + cs / 2;
                const piece = board[r][c];

                let pieceClass = 'c4-hole';
                if (piece === 1) pieceClass = 'c4-piece-1';
                else if (piece === 2) pieceClass = 'c4-piece-2';

                svg += `<circle cx="${cx}" cy="${cy}" r="${this.PIECE_RADIUS}"
                               class="${pieceClass}"/>`;
            }
        }

        // Clickable column zones (only if it's human's turn)
        if (state.is_human_turn && !state.is_terminal) {
            const legalActions = state.legal_actions || [];
            for (const col of legalActions) {
                const x = pad + col * cs;
                svg += `<rect x="${x}" y="${pad}" width="${cs}" height="${rows * cs}"
                              class="c4-drop-zone"
                              data-col="${col}"/>`;
            }
        }

        svg += `</svg>`;
        container.innerHTML = svg;

        // Attach click handlers
        container.querySelectorAll('.c4-drop-zone').forEach(zone => {
            zone.addEventListener('click', () => {
                const col = parseInt(zone.getAttribute('data-col'));
                onColumnClick(col);
            });
        });
    },

    /**
     * Get a status message for the current state.
     */
    getStatusMessage(state) {
        if (state.is_terminal) {
            const returns = state.returns;
            const hp = state.human_player;
            if (returns[hp] > 0) return 'You win!';
            if (returns[hp] < 0) return 'AI wins!';
            return 'Draw!';
        }
        if (state.is_human_turn) {
            return 'Your turn — click a column';
        }
        return 'AI is thinking...';
    }
};
