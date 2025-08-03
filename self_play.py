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
from heuristics import find_immediate_win_loss_search
from config import SELF_PLAY_GAMES, MODEL_SAVE_PATH, TRAINING_DATA_PATH, DATA_MAX_SIZE, MCTS_SIMULATIONS_TRAIN, TEMPERATURE_THRESHOLD

def augment_data(state, probs, value):
    original_data = {'state': state, 'probs': probs, 'value': value}
    flipped_state = np.flip(state, axis=2).copy()
    flipped_probs = np.flip(probs).copy()
    flipped_data = {'state': flipped_state, 'probs': flipped_probs, 'value': value}
    return [original_data, flipped_data]

def self_play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConnectFourNet().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
            print("Loaded existing universal model.")
        except Exception as e:
            print(f"Could not load model weights: {e}. Starting with a random one.")
    else:
        print("No existing model found, starting with a random one.")
    model.eval()

    if os.path.exists(TRAINING_DATA_PATH):
        with open(TRAINING_DATA_PATH, 'rb') as f:
            training_data = pickle.load(f)
        print(f"Loaded {len(training_data)} existing data points.")
    else:
        training_data = deque(maxlen=DATA_MAX_SIZE)
    
    mcts_player = MCTS(model=model, device=device, simulations=MCTS_SIMULATIONS_TRAIN)
    
    pbar = tqdm(range(SELF_PLAY_GAMES), desc="Self-Play Games (Heuristics-Assisted)")
    heuristic_overrides = 0

    for i in pbar:
        rows, cols = random.randint(5, 8), random.randint(5, 8)
        game = ConnectFourGame(rows=rows, cols=cols)
        game_history = []
        move_count = 0
        
        while not game.game_over:
            state = game.get_board_state()
            
            # [FIX] 正确处理从启发式搜索返回的元组
            heuristic_result = find_immediate_win_loss_search(game)
            
            if heuristic_result is not None:
                # 解包元组，我们只关心棋步 (move)
                _ , priority_move = heuristic_result
                
                action_probs = np.zeros(game.cols)
                action_probs[priority_move] = 1.0 # 使用解包后的整数作为索引
                action = priority_move           # 将整数棋步赋给action
                
                heuristic_overrides += 1
                pbar.set_postfix({"Overrides": heuristic_overrides})
            else:
                # 运行MCTS来获得策略
                # [FIXED] 确保get_move_analysis传递正确的参数
                temp = 1.0 if move_count < TEMPERATURE_THRESHOLD else 1e-3
                analysis = mcts_player.get_move_analysis(game, temp=temp)
                action_probs = analysis["policy"]
                
                valid_moves = game.get_valid_moves()
                if not valid_moves: break
                
                masked_probs = np.zeros_like(action_probs)
                masked_probs[valid_moves] = action_probs[valid_moves]

                if np.sum(masked_probs) == 0:
                    action = np.random.choice(valid_moves)
                else:
                    masked_probs /= np.sum(masked_probs)
                    action = np.random.choice(len(action_probs), p=masked_probs)

            game_history.append([state, action_probs, game.current_player])
            game.make_move(action)

        winner = game.winner
        for state, probs, player in game_history:
            value = 0 if winner == -1 else (1 if player == winner else -1)
            augmented_samples = augment_data(state, probs, value)
            training_data.extend(augmented_samples)

    with open(TRAINING_DATA_PATH, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"\nSelf-play finished. Total data points: {len(training_data)}")
    print(f"Heuristic search overrides occurred {heuristic_overrides} times.")


if __name__ == "__main__":
    self_play()