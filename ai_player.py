# ai_player.py
import torch
import numpy as np
import os

# 导入新的类
from game_logic import ConnectFourGame
from neural_network import UniversalConnectFourNet as ConnectFourNet
from mcts import MCTS
from config import MODEL_SAVE_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 全局加载模型以提高效率 ---
AI_MODEL = ConnectFourNet().to(DEVICE)
if os.path.exists(MODEL_SAVE_PATH):
    # 使用 weights_only=True 以消除安全警告
    AI_MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    print(f"AI Player: Loaded universal model from {MODEL_SAVE_PATH}")
else:
    print("AI Player: WARNING - No trained model found. AI will perform poorly.")
AI_MODEL.eval()

def ai_move(board, player):
    """
    接收棋盘状态和当前玩家，返回最佳走法和每一步的Q值。
    """
    rows, cols = board.shape
    game = ConnectFourGame(rows=rows, cols=cols)
    game.board = np.array(board)
    game.current_player = player
    
    mcts = MCTS(game, AI_MODEL, DEVICE)
    
    # 调用新的分析方法，获取策略和Q值
    analysis = mcts.get_move_analysis(game, game.cols)
    action_probs = analysis["policy"]
    q_values = analysis["q_values"]
    
    valid_moves = game.get_valid_moves()
    masked_probs = np.zeros_like(action_probs)
    
    # 确保valid_moves不是空的
    if valid_moves:
        masked_probs[valid_moves] = action_probs[valid_moves]

    if np.sum(masked_probs) == 0:
        # 如果所有概率都为0（不太可能但作为保险），随机选择
        best_move = np.random.choice(valid_moves) if valid_moves else 0
        # 如果没有分析数据，返回一个全零的数组
        q_values = np.zeros(cols) 
    else:
        # 选择访问次数最多的走法作为最终决策
        best_move = np.argmax(masked_probs)

    # 返回最佳走法和 Q值数组
    return int(best_move), q_values