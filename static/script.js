document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    // --- DOM Elements ---
    const boardElement = document.getElementById('board');
    const qValuesContainer = document.getElementById('q-values-container');
    const statusElement = document.getElementById('status');
    const thinkingElement = document.getElementById('thinking-status');
    const newGameBtn = document.getElementById('new-game-btn');
    const rowsInput = document.getElementById('rows-input');
    const colsInput = document.getElementById('cols-input');
    const analysisModeCheck = document.getElementById('analysis-mode');
    const aiRedModeCheck = document.getElementById('ai-red-mode');
    const aiBlackModeCheck = document.getElementById('ai-black-mode');
    const progressBar = document.getElementById('ai-progress');

    // --- State ---
    let isThinking = false;
    let isAnalysis = false;

    // --- Drawing Functions ---
    const drawBoard = (game) => {
        const validMoves = game.validMoves;
        boardElement.innerHTML = '';
        qValuesContainer.innerHTML = '';
        boardElement.style.gridTemplateColumns = `repeat(${game.cols}, 1fr)`;
        qValuesContainer.style.gridTemplateColumns = `repeat(${game.cols}, 1fr)`;
        
        for (let c = 0; c < game.cols; c++) {
            qValuesContainer.appendChild(document.createElement('div')).classList.add('q-value-cell');
        }

        for (let r = 0; r < game.rows; r++) {
            for (let c = 0; c < game.cols; c++) {
                const cell = document.createElement('div');
                cell.classList.add('board-cell');
                cell.dataset.col = c;
                if (validMoves.includes(c) && !game.gameOver && (!isThinking || isAnalysis)) {
                    cell.classList.add('valid');
                }
                const pieceValue = game.board[r][c];
                if (pieceValue !== 0) {
                    const piece = document.createElement('div');
                    piece.classList.add('piece', `player${pieceValue}`);
                    cell.appendChild(piece);
                }
                boardElement.appendChild(cell);
            }
        }
    };

    const drawQValues = (qValues) => {
        const qCells = qValuesContainer.querySelectorAll('.q-value-cell');
        qCells.forEach((cell, index) => {
            const qValue = qValues[index];
            if (qValue !== undefined && qValue > -2.0) {
                const winRate = ((qValue + 1) / 2) * 100;
                cell.textContent = `${winRate.toFixed(1)}%`;
                if (winRate > 60) cell.style.color = '#4caf50';
                else if (winRate < 40) cell.style.color = '#f44336';
                else cell.style.color = '#ffeb3b';
            } else {
                cell.textContent = '';
            }
        });
    };
    
    // --- Socket Event Handlers ---
    socket.on('connect', () => console.log('Socket.IO: Connected to server!'));

    socket.on('game_update', (game) => {
        console.log('Socket.IO: Received game_update', game);
        drawBoard(game);
        statusElement.textContent = game.status;
    });

    socket.on('modes_update', (modes) => {
        console.log('Socket.IO: Received modes_update', modes);
        isAnalysis = modes.analysis;
        analysisModeCheck.checked = modes.analysis;
        aiRedModeCheck.checked = modes.ai_red;
        aiBlackModeCheck.checked = modes.ai_black;
    });

    socket.on('analysis_update', (data) => {
        if (data.q_values) drawQValues(data.q_values);
    });

    socket.on('thinking_update', (data) => {
        console.log('Socket.IO: Received thinking_update', data);
        isThinking = data.isThinking;
        thinkingElement.textContent = isThinking ? data.status : '';
        boardElement.style.cursor = isThinking && !isAnalysis ? 'wait' : 'default';
        if (!isThinking) {
            progressBar.value = 0;
            // Redraw board to show valid moves for human
            socket.emit('request_game_state'); // Simple way to ask for a redraw
        }
    });
    
    // Add a handler for our new request
    socket.on('request_game_state', () => socket.emit('get_game_state'));

    socket.on('progress_update', (data) => {
        progressBar.max = data.total_sims;
        progressBar.value = data.sims;
    });

    // --- UI Event Listeners ---
    newGameBtn.addEventListener('click', () => {
        console.log('UI: Clicked New Game');
        socket.emit('request_new_game', { rows: rowsInput.value, cols: colsInput.value });
    });

    boardElement.addEventListener('click', (event) => {
        const cell = event.target.closest('.board-cell');
        if (cell && cell.classList.contains('valid')) {
            console.log(`UI: Clicked valid column ${cell.dataset.col}`);
            socket.emit('make_move', { col: parseInt(cell.dataset.col, 10) });
        } else {
            console.log('UI: Clicked an invalid area or while thinking.');
        }
    });

    [analysisModeCheck, aiRedModeCheck, aiBlackModeCheck].forEach(checkbox => {
        checkbox.addEventListener('change', (event) => {
            const modeId = event.target.id;
            const modeKey = modeId.replace('-mode', '').replace('ai-','ai_');
            
            // 互斥逻辑：分析模式和人机模式不能同时开启
            if (modeKey === 'analysis' && event.target.checked) {
                // 开启分析模式时关闭所有AI模式
                aiRedModeCheck.checked = false;
                aiBlackModeCheck.checked = false;
                // 发送所有模式更新
                socket.emit('toggle_mode', { mode: 'ai_red', isActive: false });
                socket.emit('toggle_mode', { mode: 'ai_black', isActive: false });
            } 
            else if ((modeKey === 'ai_red' || modeKey === 'ai_black') && event.target.checked) {
                // 开启AI模式时关闭分析模式
                analysisModeCheck.checked = false;
                socket.emit('toggle_mode', { mode: 'analysis', isActive: false });
            }
            
            console.log(`UI: Toggled mode ${modeKey} to ${event.target.checked}`);
            socket.emit('toggle_mode', {
                mode: modeKey,
                isActive: event.target.checked
            });
        });
    });
});