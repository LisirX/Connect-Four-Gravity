# self_play.py
import os
import torch
import numpy as np
import pickle
import random
from collections import deque
from tqdm import tqdm

# CHANGED: 导入新的游戏和网络类
from game_logic import ConnectFourGame
from neural_network import UniversalConnectFourNet as ConnectFourNet
from mcts import MCTS
from config import SELF_PLAY_GAMES, MODEL_SAVE_PATH, TRAINING_DATA_PATH, DATA_MAX_SIZE

def self_play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型现在是通用的
    model = ConnectFourNet().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
        print("Loaded existing universal model.")
    else:
        print("No existing model found, starting with a random one.")
    model.eval()

    if os.path.exists(TRAINING_DATA_PATH):
        with open(TRAINING_DATA_PATH, 'rb') as f:
            training_data = pickle.load(f)
        print(f"Loaded {len(training_data)} existing data points.")
    else:
        training_data = deque(maxlen=DATA_MAX_SIZE)
    
    for i in tqdm(range(SELF_PLAY_GAMES), desc="Self-Play Games"):
        # --- NEW: 混合尺寸训练 ---
        # 每一局游戏都随机选择一个棋盘尺寸
        rows = random.randint(5, 8)
        cols = random.randint(5, 8)
        # 确保四子连珠是可能的
        if min(rows, cols) < 4 and max(rows, cols) < 4:
            rows, cols = 6, 7 # 如果太小，则使用默认值

        game = ConnectFourGame(rows=rows, cols=cols)
        # MCTS现在需要知道列数
        mcts = MCTS(game, model, device)
        game_history = []

        while not game.game_over:
            state = game.get_board_state()
            # MCTS现在需要知道cols
            action_probs = mcts.get_move_probs(game, game.cols)
            game_history.append([state, action_probs, game.current_player])
            
            action = np.random.choice(len(action_probs), p=action_probs)
            game.make_move(action)

        winner = game.winner
        for state, probs, player in game_history:
            if winner == -1: value = 0
            else: value = 1 if player == winner else -1
            
            # 存储 state, probs, 和 value
            # 注意：这里的 state 和 probs 尺寸是可变的！
            training_data.append({'state': state, 'probs': probs, 'value': value})

    with open(TRAINING_DATA_PATH, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"Self-play finished. Total data points: {len(training_data)}")

if __name__ == "__main__":
    self_play()