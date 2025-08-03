# ai_player.py
import torch
import numpy as np
import os

from game_logic import ConnectFourGame
from neural_network import UniversalConnectFourNet as ConnectFourNet
from mcts import MCTS
# CHANGED: 导入新的配置项
from config import MODEL_SAVE_PATH, MCTS_SIMULATIONS_PLAY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AI_MODEL = ConnectFourNet().to(DEVICE)
if os.path.exists(MODEL_SAVE_PATH):
    AI_MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    print(f"AI Player: Loaded universal model from {MODEL_SAVE_PATH}")
else:
    print("AI Player: WARNING - No trained model found. AI will perform poorly.")
AI_MODEL.eval()

def ai_move(board, player):
    rows, cols = board.shape
    game = ConnectFourGame(rows=rows, cols=cols)
    game.board = np.array(board)
    game.current_player = player
    
    # CHANGED: 初始化MCTS时传入对战专用的模拟次数
    mcts = MCTS(game, AI_MODEL, DEVICE, simulations=MCTS_SIMULATIONS_PLAY)
    
    analysis = mcts.get_move_analysis(game, game.cols)
    action_probs = analysis["policy"]
    q_values = analysis["q_values"]
    
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        return None, np.zeros(cols)
    
    masked_probs = np.zeros_like(action_probs)
    masked_probs[valid_moves] = action_probs[valid_moves]

    if np.sum(masked_probs) == 0:
        best_move = np.random.choice(valid_moves)
        q_values = np.zeros(cols) 
    else:
        best_move = np.argmax(masked_probs)

    return int(best_move), q_values