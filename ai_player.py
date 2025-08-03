# ai_player.py
import torch
import numpy as np
import os
import time
import pickle
import queue

from mcts import MCTS
from neural_network import UniversalConnectFourNet
from heuristics import find_immediate_win_loss_search
from config import MODEL_SAVE_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AI_MODEL = UniversalConnectFourNet().to(DEVICE)
if os.path.exists(MODEL_SAVE_PATH):
    try:
        AI_MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
        print(f"AI Player: Loaded universal model from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"AI Player: WARNING - Could not load model. Error: {e}. AI will perform poorly.")
else:
    print("AI Player: WARNING - No trained model found. AI will perform poorly.")
AI_MODEL.eval()

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