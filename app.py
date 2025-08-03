# app.py
import os
import sys
import queue
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from game_logic import ConnectFourGame
# [FIX] Import the DEVICE constant along with the model and ai_move function
from ai_player import ai_move, AI_MODEL, DEVICE
from mcts import MCTS
from config import (
    DEFAULT_ROWS, DEFAULT_COLS, MCTS_SIMULATIONS_PLAY,
    MCTS_SIMULATIONS_LIVE_ANALYSIS
)

# --- Flask & SocketIO App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_change_me'
socketio = SocketIO(app, async_mode='threading')

sessions = {}

# --- Helper Functions ---

def get_session_id():
    return request.sid

def get_game_status(game, modes):
    if game.game_over:
        return "It's a Draw!" if game.winner == -1 else ("Red Wins!" if game.winner == 1 else "Black Wins!")
    else:
        player_name = "Red (P1)" if game.current_player == 1 else "Black (P2)"
        is_ai_turn = (modes['ai_red'] and game.current_player == 1) or \
                     (modes['ai_black'] and game.current_player == 2)
        return f"AI ({player_name}) is thinking..." if is_ai_turn else f"Your turn, {player_name}."

def emit_game_update(sid):
    client_data = sessions.get(sid)
    if not client_data: return
    game, modes = client_data['game'], client_data['modes']
    game_state = {
        'board': game.board.tolist(), 'rows': game.rows, 'cols': game.cols,
        'currentPlayer': game.current_player, 'gameOver': game.game_over,
        'winner': game.winner, 'validMoves': game.get_valid_moves(),
        'status': get_game_status(game, modes)
    }
    socketio.emit('game_update', game_state, room=sid)

def emit_modes_update(sid):
    if (client_data := sessions.get(sid)):
        socketio.emit('modes_update', client_data['modes'], room=sid)

def stop_analysis_thread(sid):
    if (client_data := sessions.get(sid)) and (thread := client_data.get('analysis_thread')):
        client_data['is_analysis_active'][0] = False
        thread.join(timeout=0.5)
        client_data['analysis_thread'] = None
        print(f"Session {sid}: Stopped analysis thread.")

def restart_analysis_if_active(sid):
    client_data = sessions.get(sid)
    if not client_data or not client_data['modes']['analysis'] or client_data['game'].game_over:
        return
    stop_analysis_thread(sid) # Ensure any old thread is gone
    client_data['is_analysis_active'][0] = True
    client_data['analysis_thread'] = socketio.start_background_task(target=_run_analysis_loop, sid=sid)
    print(f"Session {sid}: Restarted analysis thread.")


def stop_all_ai_tasks(sid):
    if not (client_data := sessions.get(sid)): return
    if client_data.get('is_thinking_flag'):
        client_data['is_thinking_flag'][0] = False
        print(f"Session {sid}: Stopped main AI thinking task.")
    client_data['is_thinking_flag'] = None
    stop_analysis_thread(sid)

def trigger_ai_move(sid):
    client_data = sessions.get(sid)
    if not client_data or (client_data.get('is_thinking_flag') and client_data['is_thinking_flag'][0]) or client_data['game'].game_over:
        return

    is_thinking_flag = [True]
    client_data['is_thinking_flag'] = is_thinking_flag
    update_queue, game = queue.Queue(), client_data['game']

    player_name = "Red (P1)" if game.current_player == 1 else "Black (P2)"
    socketio.emit('thinking_update', {'isThinking': True, 'status': f'AI ({player_name}) is thinking...'}, room=sid)
    emit_game_update(sid)

    def _handle_ai_updates():
        while is_thinking_flag[0]:
            try:
                data = update_queue.get(timeout=1)
                if 'move' in data:
                    if is_thinking_flag[0]: game.make_move(data['move'])
                    break
                elif 'sims' in data:
                    socketio.emit('progress_update', {'sims': data['sims'], 'total_sims': MCTS_SIMULATIONS_PLAY}, room=sid)
                    if 'q_values' in data:
                        socketio.emit('analysis_update', {'q_values': data['q_values'].tolist()}, room=sid)
            except queue.Empty: continue

        client_data['is_thinking_flag'] = None
        socketio.emit('thinking_update', {'isThinking': False, 'status': ''}, room=sid)
        emit_game_update(sid)
        restart_analysis_if_active(sid)
        check_and_trigger_ai(sid)

    socketio.start_background_task(_handle_ai_updates)
    socketio.start_background_task(ai_move, game=game, update_queue=update_queue, is_thinking_flag=is_thinking_flag, simulations=MCTS_SIMULATIONS_PLAY)

def _run_analysis_loop(sid):
    client_data = sessions.get(sid)
    if not client_data: return
    is_analysis_active, game, mcts = client_data['is_analysis_active'], client_data['game'], client_data['mcts_instance']

    while is_analysis_active[0]:
        if game.game_over: break
        mcts.set_game_state(game)
        mcts.ponder(num_sims=MCTS_SIMULATIONS_LIVE_ANALYSIS)
        socketio.emit('analysis_update', {'q_values': mcts.get_q_values(game.cols).tolist()}, room=sid)
        socketio.sleep(0.1)
    print(f"Session {sid}: Analysis loop terminated.")

def check_and_trigger_ai(sid):
    client_data = sessions.get(sid)
    if not client_data or client_data['game'].game_over: return
    game, modes = client_data['game'], client_data['modes']
    is_ai_turn = (modes['ai_red'] and game.current_player == 1) or \
                 (modes['ai_black'] and game.current_player == 2)
    if is_ai_turn and not (client_data.get('is_thinking_flag') and client_data['is_thinking_flag'][0]):
        socketio.sleep(0.5)
        trigger_ai_move(sid)

# --- Socket.IO Event Handlers ---

@socketio.on('connect')
def handle_connect():
    sid = get_session_id()
    print(f"Client connected: {sid}")
    sessions[sid] = {
        'game': ConnectFourGame(rows=DEFAULT_ROWS, cols=DEFAULT_COLS),
        'modes': {'analysis': False, 'ai_red': False, 'ai_black': True},
        # [FIXED] Use the imported DEVICE constant instead of hardcoding "cpu".
        # This ensures the MCTS instance and the model are on the same device.
        'mcts_instance': MCTS(model=AI_MODEL, device=DEVICE, simulations=0),
        'is_thinking_flag': None,
        'analysis_thread': None,
        'is_analysis_active': [False],
    }
    emit_game_update(sid)
    emit_modes_update(sid)

@socketio.on('disconnect')
def handle_disconnect():
    sid = get_session_id()
    print(f"Client disconnected: {sid}")
    stop_all_ai_tasks(sid)
    if sid in sessions: del sessions[sid]

@socketio.on('request_new_game')
def handle_new_game(data):
    sid = get_session_id()
    print(f"Session {sid}: Received request for a new game.")
    stop_all_ai_tasks(sid)
    if not (client_data := sessions.get(sid)): return
    try: rows, cols = int(data.get('rows', DEFAULT_ROWS)), int(data.get('cols', DEFAULT_COLS))
    except (ValueError, TypeError): rows, cols = DEFAULT_ROWS, DEFAULT_COLS
    client_data['game'] = ConnectFourGame(rows=rows, cols=cols)
    emit_game_update(sid)
    restart_analysis_if_active(sid)
    check_and_trigger_ai(sid)

@socketio.on('make_move')
def handle_make_move(data):
    sid = get_session_id()
    if not (client_data := sessions.get(sid)): return
    game, modes = client_data['game'], client_data['modes']
    is_ai_thinking = client_data.get('is_thinking_flag') and client_data['is_thinking_flag'][0]
    is_ai_turn = (modes['ai_red'] and game.current_player == 1) or (modes['ai_black'] and game.current_player == 2)

    if game.game_over or is_ai_thinking or is_ai_turn: return

    if (col := data.get('col')) in game.get_valid_moves():
        stop_analysis_thread(sid) # Stop analysis during the move
        game.make_move(col)
        emit_game_update(sid)
        restart_analysis_if_active(sid) # Restart for the new state
        check_and_trigger_ai(sid)

@socketio.on('toggle_mode')
def handle_toggle_mode(data):
    sid = get_session_id()
    if not (client_data := sessions.get(sid)): return
    mode, is_active = data.get('mode'), data.get('isActive')
    modes = client_data['modes']
    
    if mode in modes:
        # 互斥逻辑：分析模式和人机模式不能同时开启
        if mode == 'analysis' and is_active:
            # 开启分析模式时关闭所有AI模式
            modes['ai_red'] = False
            modes['ai_black'] = False
        elif mode in ('ai_red', 'ai_black') and is_active:
            # 开启AI模式时关闭分析模式
            modes['analysis'] = False
        
        # 更新当前模式状态
        modes[mode] = is_active
        print(f"Session {sid}: Toggled mode '{mode}' to {is_active}")

    emit_modes_update(sid)
    emit_game_update(sid)

    if mode == 'analysis':
        if is_active: 
            restart_analysis_if_active(sid)
        else: 
            stop_analysis_thread(sid)
    else:
        check_and_trigger_ai(sid)

@socketio.on('get_game_state')
def handle_get_game_state():
    emit_game_update(get_session_id())

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Connect Four server...")
    print("Open http://127.0.0.1:5000 in your browser.")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)