import torch
import numpy as np
import os
import re # 导入正则表达式
import pickle
import queue

from mcts import MCTS
# 移除了 'from heuristics import find_immediate_win_loss_search'
from neural_network import UniversalConnectFourNet
# 从 config 导入常量
from config import MODEL_DIR, MODEL_BASENAME, CHALLENGER_MODEL_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latest_champion_model_path():
    """
    在模型目录中找到版本号最高的模型。
    例如: 找到 'universal_model_v5.pth' 而不是 'universal_model_v4.pth'。
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
        print(f"AI Player: 找到最新的冠军模型: {os.path.basename(latest_model_path)}")
        return latest_model_path
    
    # 如果没有找到带版本的冠军模型，则回退
    if os.path.exists(CHALLENGER_MODEL_PATH):
        print(f"AI Player: 未找到冠军模型。回退到默认模型: {CHALLENGER_MODEL_PATH}")
        return CHALLENGER_MODEL_PATH
        
    return None

# --- 主要模型加载逻辑 ---
AI_MODEL = UniversalConnectFourNet().to(DEVICE)
BEST_MODEL_PATH = get_latest_champion_model_path()

if BEST_MODEL_PATH and os.path.exists(BEST_MODEL_PATH):
    try:
        AI_MODEL.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(f"AI Player: 成功从 '{BEST_MODEL_PATH}' 加载模型")
    except Exception as e:
        print(f"AI Player: 警告 - 无法从 '{BEST_MODEL_PATH}' 加载模型。错误: {e}。AI 表现会很差。")
else:
    print("AI Player: 警告 - 未找到任何训练好的模型。AI 表现会很差。")

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
    [已修改] 简化的函数，完全依赖 MCTS 进行决策。
    """
    if analysis_only:
        return

    # --- 标准流程：运行MCTS计算最佳棋步 ---
    mcts = MCTS(model=AI_MODEL, device=DEVICE, simulations=simulations)

    # 运行 MCTS 计算最佳棋步
    best_move = mcts.get_move(game, update_queue, is_thinking_flag)
    
    # 如果思考被中断，则提前返回
    if not is_thinking_flag[0]:
        return

    # 获取最终的 Q 值并准备结果
    final_q_values = mcts.get_final_q_values(game)
    final_result = {'move': best_move, 'q_values': final_q_values, 'sims': simulations, 'status': 'MCTS 搜索'}

    # 将最终结果发送回主线程
    if is_thinking_flag[0]:
        update_queue.put(final_result)