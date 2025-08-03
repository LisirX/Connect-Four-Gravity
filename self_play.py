# self_play.py
import os
import torch
import numpy as np
import pickle
import random
from collections import deque
from tqdm import tqdm

from game_logic import ConnectFourGame
from neural_network import UniversalConnectFourNet as ConnectFourNet
from mcts import MCTS
# CHANGED: 导入训练专用的模拟次数
from config import SELF_PLAY_GAMES, MODEL_SAVE_PATH, TRAINING_DATA_PATH, DATA_MAX_SIZE, MCTS_SIMULATIONS_TRAIN

def self_play():
    # ... (文件开头的代码保持不变) ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    
    postfix_stats = {}
    pbar = tqdm(range(SELF_PLAY_GAMES), desc="Self-Play Games")
    
    for i in pbar:
        rows = random.randint(5, 8)
        cols = random.randint(5, 8)
        if min(rows, cols) < 4 and max(rows, cols) < 4:
            rows, cols = 6, 7

        game = ConnectFourGame(rows=rows, cols=cols)
        # CHANGED: MCTS使用训练专用的模拟次数
        mcts = MCTS(game, model, device, simulations=MCTS_SIMULATIONS_TRAIN)
        game_history = []
        
        # ... (文件的其余部分代码保持不变) ...
        postfix_stats["Board"] = f"{rows}x{cols}"
        postfix_stats.pop("Moves", None)
        postfix_stats.pop("Result", None)
        pbar.set_postfix(postfix_stats)

        move_count = 0
        while not game.game_over:
            move_count += 1
            postfix_stats["Moves"] = move_count
            pbar.set_postfix(postfix_stats)

            state = game.get_board_state()
            analysis = mcts.get_move_analysis(game, game.cols)
            action_probs = analysis["policy"]
            game_history.append([state, action_probs, game.current_player])
            
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
                
            masked_probs = np.zeros_like(action_probs)
            masked_probs[valid_moves] = action_probs[valid_moves]

            if np.sum(masked_probs) == 0:
                action = np.random.choice(valid_moves)
            else:
                masked_probs /= np.sum(masked_probs)
                action = np.random.choice(len(action_probs), p=masked_probs)
            
            game.make_move(action)

        winner = game.winner
        if winner == -1:
            postfix_stats["Result"] = "Draw"
        else:
            postfix_stats["Result"] = f"P{winner} Won"
        pbar.set_postfix(postfix_stats)

        for state, probs, player in game_history:
            if winner == -1: value = 0
            else: value = 1 if player == winner else -1
            training_data.append({'state': state, 'probs': probs, 'value': value})

    with open(TRAINING_DATA_PATH, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"\nSelf-play finished. Total data points: {len(training_data)}")


if __name__ == "__main__":
    self_play()