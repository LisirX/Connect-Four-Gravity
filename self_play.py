import os
import torch
import numpy as np
import pickle
import random
from collections import deque

from game_logic import ConnectFourGame
from neural_network import UniversalConnectFourNet as ConnectFourNet
from mcts import MCTS
# 移除了 'from heuristics import find_immediate_win_loss_search'
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
            print("加载了已存在的通用模型。")
        except Exception as e:
            print(f"无法加载模型权重: {e}。将从一个随机模型开始。")
    else:
        print("未找到现有模型，将从一个随机模型开始。")
    model.eval()

    if os.path.exists(TRAINING_DATA_PATH):
        with open(TRAINING_DATA_PATH, 'rb') as f:
            training_data = pickle.load(f)
        print(f"加载了 {len(training_data)} 条现有数据。")
    else:
        training_data = deque(maxlen=DATA_MAX_SIZE)
    
    mcts_player = MCTS(model=model, device=device, simulations=MCTS_SIMULATIONS_TRAIN)
    
    # 用 print 语句替代进度条描述
    print(f"开始自对弈，共生成 {SELF_PLAY_GAMES} 局游戏数据...")

    for i in range(SELF_PLAY_GAMES):
        rows, cols = random.randint(5, 8), random.randint(5, 8)
        game = ConnectFourGame(rows=rows, cols=cols)
        game_history = []
        move_count = 0
        
        while not game.game_over:
            state = game.get_board_state()
            
            # [已修改] 移除了启发式搜索，总是使用 MCTS
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
            move_count += 1

        winner = game.winner
        for state, probs, player in game_history:
            value = 0 if winner == -1 else (1 if player == winner else -1)
            augmented_samples = augment_data(state, probs, value)
            training_data.extend(augmented_samples)

    with open(TRAINING_DATA_PATH, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"\n自对弈完成。总数据点: {len(training_data)}")


if __name__ == "__main__":
    self_play()