# arena.py
import torch
import numpy as np
import os
import argparse
import json
import random
from tqdm import tqdm

from game_logic import ConnectFourGame
from neural_network import UniversalConnectFourNet as ConnectFourNet
from mcts import MCTS
from config import MODEL_SAVE_PATH, ARENA_GAMES, ARENA_MCTS_SIMULATIONS, RESULTS_FILE_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArenaPlayer:
    def __init__(self, model_path, name, simulations):
        self.name = name
        self.model = ConnectFourNet().to(DEVICE)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}。")
        
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        self.model.eval()
        print(f"玩家 '{self.name}' 已成功加载模型: {model_path}")

        self.mcts = MCTS(model=self.model, device=DEVICE, simulations=simulations)

    def get_move(self, game):
        analysis = self.mcts.get_move_analysis(game, game.cols)
        action_probs = analysis["policy"]
        
        if not game.get_valid_moves(): return None
        
        best_move = np.argmax(action_probs)
        return int(best_move)

def play_game(p1: ArenaPlayer, p2: ArenaPlayer):
    """
    [MODIFIED] 运行一局游戏并返回包含所有细节的日志。
    """
    rows, cols = random.randint(5, 8), random.randint(5, 8)
    game = ConnectFourGame(rows=rows, cols=cols)
    
    move_history = [] # [NEW] 用于记录本局游戏的所有落子

    while not game.game_over:
        current_player_obj = p1 if game.current_player == 1 else p2
        move = current_player_obj.get_move(game)
        
        if move is None or move not in game.get_valid_moves():
            # 处理无效移动，提前结束游戏
            winner = 3 - game.current_player # 对手获胜
            break
            
        game.make_move(move)
        move_history.append(move) # [NEW] 将有效落子添加到历史记录
    
    # [NEW] 构建详细的游戏日志字典
    game_log = {
        "board_size": f"{game.rows}x{game.cols}",
        "player1_red": p1.name,
        "player2_black": p2.name,
        "moves": move_history,
        "winner_player_number": game.winner, # 1, 2, or -1 for draw
        "winner_name": "Draw" if game.winner == -1 else (p1.name if game.winner == 1 else p2.name)
    }
    
    return game_log

def main(old_model_path):
    print(f"设备: {DEVICE}")
    print("--- 开始模型竞技场评估 ---")

    new_player = ArenaPlayer(MODEL_SAVE_PATH, "挑战者(新)", ARENA_MCTS_SIMULATIONS)
    old_player = ArenaPlayer(old_model_path, "冠军(旧)", ARENA_MCTS_SIMULATIONS)

    scores = {"new_model_wins": 0, "old_model_wins": 0, "draws": 0}
    all_game_logs = [] # [NEW] 用于存储所有对局的详细日志

    pbar = tqdm(range(ARENA_GAMES), desc=f"竞技场: {new_player.name} vs {old_player.name}")

    for i in pbar:
        if i % 2 == 0:
            p1, p2 = new_player, old_player
            pbar.set_postfix({"先手": p1.name})
            
            game_log = play_game(p1, p2) # [MODIFIED] 获取完整的游戏日志
            winner_num = game_log["winner_player_number"]
            
            if winner_num == 1: scores["new_model_wins"] += 1
            elif winner_num == 2: scores["old_model_wins"] += 1
            else: scores["draws"] += 1
        else:
            p1, p2 = old_player, new_player
            pbar.set_postfix({"先手": p1.name})

            game_log = play_game(p1, p2) # [MODIFIED] 获取完整的游戏日志
            winner_num = game_log["winner_player_number"]

            if winner_num == 1: scores["old_model_wins"] += 1
            elif winner_num == 2: scores["new_model_wins"] += 1
            else: scores["draws"] += 1
        
        all_game_logs.append(game_log) # [NEW] 将本局日志添加到总列表中

    print("\n--- 竞技场评估结束 ---")

    total_games = ARENA_GAMES
    new_wins = scores["new_model_wins"]
    old_wins = scores["old_model_wins"]
    draws = scores["draws"]
    win_rate = new_wins / total_games if total_games > 0 else 0
    
    # [MODIFIED] 将包含所有游戏日志的列表添加到最终结果中
    results = {
        "summary": {
            "total_games": total_games,
            "new_model_win_rate": f"{win_rate:.2%}",
            "old_model_win_rate": f"{old_wins / total_games:.2%}" if total_games > 0 else "0.00%",
            "draw_rate": f"{draws / total_games:.2%}" if total_games > 0 else "0.00%",
        },
        "raw_scores": scores,
        "models": {"new_model": MODEL_SAVE_PATH, "old_model": old_model_path},
        "config": {"arena_games": ARENA_GAMES, "mcts_simulations": ARENA_MCTS_SIMULATIONS},
        "games": all_game_logs  # [NEW] 这里是所有对局的详细日志
    }

    print("\n[对战结果]")
    print(f"总局数: {total_games}")
    print(f"新模型胜率: {results['summary']['new_model_win_rate']} ({new_wins} 胜)")
    print(f"旧模型胜率: {results['summary']['old_model_win_rate']} ({old_wins} 胜)")
    print(f"平局率:     {results['summary']['draw_rate']} ({draws} 平局)")

    os.makedirs(os.path.dirname(RESULTS_FILE_PATH), exist_ok=True)
    with open(RESULTS_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n详细结果已保存至: {RESULTS_FILE_PATH}")
    
    if win_rate > 0.55: print("\n结论: 新模型表现显著优于旧模型！")
    elif win_rate > 0.5: print("\n结论: 新模型略有优势，但进步不明显。")
    else: print("\n结论: 新模型未能战胜旧模型。")
    
    exit(int(win_rate * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行新旧AI模型之间的竞技场评估。")
    parser.add_argument("--old_model", type=str, required=True, help="旧版本模型的路径 (.pth 文件), 例如: 'models/archive/old_v1.pth'")
    args = parser.parse_args()
    main(args.old_model)