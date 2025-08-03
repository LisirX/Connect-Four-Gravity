# ai_player.py
import torch
import numpy as np
import os

# CHANGED: 导入新的类
from game_logic import ConnectFourGame
from neural_network import UniversalConnectFourNet as ConnectFourNet
from mcts import MCTS
from config import MODEL_SAVE_PATH, MCTS_SIMULATIONS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 全局加载模型以提高效率 ---
AI_MODEL = ConnectFourNet().to(DEVICE)
if os.path.exists(MODEL_SAVE_PATH):
    AI_MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    print(f"AI Player: Loaded universal model from {MODEL_SAVE_PATH}")
else:
    print("AI Player: WARNING - No trained model found. AI will perform poorly.")
AI_MODEL.eval()

def ai_move(board, player):
    rows, cols = board.shape
    game = ConnectFourGame(rows=rows, cols=cols)
    game.board = np.array(board)
    game.current_player = player
    
    mcts = MCTS(game, AI_MODEL, DEVICE)
    # 传入列数
    action_probs = mcts.get_move_probs(game, game.cols)
    
    valid_moves = game.get_valid_moves()
    masked_probs = np.zeros_like(action_probs)
    masked_probs[valid_moves] = action_probs[valid_moves]

    if np.sum(masked_probs) == 0:
        return np.random.choice(valid_moves) if valid_moves else 0

    best_move = np.argmax(masked_probs)
    return int(best_move)