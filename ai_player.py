import torch
import numpy as np
import os
import re # Import regular expressions
import pickle
import queue

from mcts import MCTS
from heuristics import find_immediate_win_loss_search
from neural_network import UniversalConnectFourNet
# Import constants from config
from config import MODEL_DIR, MODEL_BASENAME, CHALLENGER_MODEL_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latest_champion_model_path():
    """
    Finds the model with the highest version number in the models directory.
    Example: Finds 'universal_model_v5.pth' over 'universal_model_v4.pth'.
    """
    if not os.path.exists(MODEL_DIR):
        return None

    model_pattern = re.compile(f"^{MODEL_BASENAME}_v(\\d+)\\.pth$")
    latest_version = -1
    latest_model_path = None

    for filename in os.listdir(MODEL_DIR):
        match = model_pattern.match(filename)
        if match:
            version = int(match.group(1))
            if version > latest_version:
                latest_version = version
                latest_model_path = os.path.join(MODEL_DIR, filename)

    if latest_model_path:
        print(f"AI Player: Found latest champion model: {os.path.basename(latest_model_path)}")
        return latest_model_path
    
    # Fallback if no versioned champion is found
    if os.path.exists(CHALLENGER_MODEL_PATH):
        print(f"AI Player: No champion model found. Falling back to default: {CHALLENGER_MODEL_PATH}")
        return CHALLENGER_MODEL_PATH
        
    return None

# --- Main Model Loading Logic ---
AI_MODEL = UniversalConnectFourNet().to(DEVICE)
BEST_MODEL_PATH = get_latest_champion_model_path()

if BEST_MODEL_PATH and os.path.exists(BEST_MODEL_PATH):
    try:
        AI_MODEL.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(f"AI Player: Successfully loaded model from '{BEST_MODEL_PATH}'")
    except Exception as e:
        print(f"AI Player: WARNING - Could not load model from '{BEST_MODEL_PATH}'. Error: {e}. AI will perform poorly.")
else:
    print("AI Player: WARNING - No trained model found anywhere. AI will perform poorly.")

AI_MODEL.eval()

# The rest of the file (ai_move, _get_nn_q_values, etc.) remains unchanged.
# ... (paste the rest of your original ai_player.py code here)
def _get_nn_q_values(game):
    q_values = np.full(game.cols, -2.0)
    valid_moves = game.get_valid_moves()
    for move in valid_moves:
        temp_game = pickle.loads(pickle.dumps(game))
        temp_game.make_move(move)
        state_tensor = torch.FloatTensor(temp_game.get_board_state()).unsqueeze(0).to(DEVICE)
        board = temp_game.board; rows, cols = board.shape
        pieces_in_col = np.sum(board != 0, axis=0)
        open_rows = (rows - pieces_in_col - 1).astype(np.int64)
        target_rows_tensor = torch.from_numpy(open_rows).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, value = AI_MODEL(state_tensor, target_rows_tensor)
        q_values[move] = -value.item()
    return q_values

def ai_move(game, update_queue, is_thinking_flag, simulations, analysis_only=False):
    """
    [FIXED] 修正了MCTS的实例化，不再传递多余的参数。
    """
    if analysis_only:
        # 在分析模式下，我们不使用这个函数，逻辑已移至GUI
        return

    # --- 以下是用于电脑下棋的、任务导向的逻辑 ---
    heuristic_result = find_immediate_win_loss_search(game)
    final_result = None

    # [FIX] 创建一个MCTS实例以备后用
    mcts = MCTS(model=AI_MODEL, device=DEVICE, simulations=simulations)

    if heuristic_result:
        move_type, move_col = heuristic_result
        if move_type == 'WIN':
            q_values = _get_nn_q_values(game)
            q_values[move_col] = 1.0
            final_result = {'move': move_col, 'q_values': q_values, 'status': '必胜!'}
        elif move_type == 'BLOCK':
            # 在必须防守时，我们依然需要运行MCTS来获取防守步的真实胜率
            mcts.get_move(game, update_queue, is_thinking_flag)
            if not is_thinking_flag[0]: return

            final_q_values = mcts.get_final_q_values(game)
            display_q_values = np.full(game.cols, -2.0)
            for move in game.get_valid_moves(): display_q_values[move] = -1.0
            display_q_values[move_col] = final_q_values[move_col]
            final_result = {'move': move_col, 'q_values': display_q_values, 'status': '必须防守!'}
    
    if not final_result:
        # 标准流程：运行MCTS计算最佳棋步
        best_move = mcts.get_move(game, update_queue, is_thinking_flag)
        if not is_thinking_flag[0]: return
        
        final_q_values = mcts.get_final_q_values(game)
        final_result = {'move': best_move, 'q_values': final_q_values, 'sims': simulations}

    if is_thinking_flag[0]:
        update_queue.put(final_result)