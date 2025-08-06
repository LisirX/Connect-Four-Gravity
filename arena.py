# arena.py (Modified for win_vs_lose rate and versioned logging)

import torch
import numpy as np
import os
import argparse
import json
import random
import re # [新增] 导入正则表达式库，用于提取版本号

from game_logic import ConnectFourGame
from neural_network import UniversalConnectFourNet as ConnectFourNet
from mcts import MCTS
# [修改] 不再从config导入固定的文件名，让它动态生成
from config import MODEL_SAVE_PATH, ARENA_GAMES, ARENA_MCTS_SIMULATIONS, RESULTS_FILE_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArenaPlayer:
    def __init__(self, model_path, name, simulations):
        self.name = name
        self.model = ConnectFourNet().to(DEVICE)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}。")
        
        try:
            # 确保即使是旧模型也能加载
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        except Exception as e:
            print(f"警告: 加载模型 {model_path} 时出现问题，可能是版本不兼容。错误: {e}")
            # 即使加载失败，也继续，但这个玩家的表现会很差
        
        self.model.eval()
        print(f"玩家 '{self.name}' 已成功加载模型: {model_path}")

        self.mcts = MCTS(model=self.model, device=DEVICE, simulations=simulations)

    def get_move(self, game):
        # [修改] 在竞技场中，总是使用确定性策略（temp -> 0）
        analysis = self.mcts.get_move_analysis(game, temp=1e-3)
        action_probs = analysis["policy"]
        
        if not game.get_valid_moves(): return None
        
        # np.argmax 会在有多个最大值时，默认选择第一个。
        # 这里我们随机选择一个，增加一点点多样性，避免两个相同模型对战时完全一样。
        best_actions = np.flatnonzero(action_probs == np.max(action_probs))
        return int(np.random.choice(best_actions))


def play_game(p1: ArenaPlayer, p2: ArenaPlayer):
    """
    运行一局游戏并返回包含所有细节的日志。
    (此函数逻辑不变)
    """
    rows, cols = random.randint(5, 8), random.randint(5, 8)
    game = ConnectFourGame(rows=rows, cols=cols)
    
    move_history = []

    while not game.game_over:
        current_player_obj = p1 if game.current_player == 1 else p2
        move = current_player_obj.get_move(game)
        
        if move is None or move not in game.get_valid_moves():
            winner = 3 - game.current_player
            break
            
        game.make_move(move)
        move_history.append(move)
    
    game_log = {
        "board_size": f"{game.rows}x{game.cols}",
        "player1_red": p1.name,
        "player2_black": p2.name,
        "moves": move_history,
        "winner_player_number": game.winner,
        "winner_name": "Draw" if game.winner == -1 else (p1.name if game.winner == 1 else p2.name)
    }
    
    return game_log

def get_version_from_path(path):
    """
    [新增] 从模型路径中提取版本号 (例如 'v3', 'best_model_25')
    """
    # 使用正则表达式查找 'v' + 数字, 或其他常见的版本模式
    # 例如： models/archive/v3.pth -> v3
    #        models/model_epoch_50.pth -> epoch_50
    #        models/universal_model.pth -> universal_model
    base_name = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r'v\d+|version\d+|epoch_\d+|universal_model', base_name, re.IGNORECASE)
    if match:
        return match.group(0)
    # 如果找不到特定模式，就返回不带扩展名的文件名
    return base_name

def main(old_model_path):
    print(f"设备: {DEVICE}")
    print("--- 开始模型竞技场评估 ---")
    
    new_model_path = MODEL_SAVE_PATH # 新模型总是来自config中的最新路径
    
    # [修改] 动态生成版本名称
    new_model_version = get_version_from_path(new_model_path)
    old_model_version = get_version_from_path(old_model_path)

    new_player = ArenaPlayer(new_model_path, f"挑战者({new_model_version})", ARENA_MCTS_SIMULATIONS)
    old_player = ArenaPlayer(old_model_path, f"冠军({old_model_version})", ARENA_MCTS_SIMULATIONS)

    scores = {"new_model_wins": 0, "old_model_wins": 0, "draws": 0}
    all_game_logs = []

    print(f"竞技场: {new_player.name} vs {old_player.name}, 共 {ARENA_GAMES} 局")
    for i in range(ARENA_GAMES):
        # 轮流执先
        p1, p2 = (new_player, old_player) if i % 2 == 0 else (old_player, new_player)
        is_new_player_p1 = (p1 == new_player)
        
        game_log = play_game(p1, p2)
        winner_num = game_log["winner_player_number"]
        
        if winner_num == 1: # P1 获胜
            if is_new_player_p1: scores["new_model_wins"] += 1
            else: scores["old_model_wins"] += 1
        elif winner_num == 2: # P2 获胜
            if not is_new_player_p1: scores["new_model_wins"] += 1
            else: scores["old_model_wins"] += 1
        else: # 平局
            scores["draws"] += 1
        
        all_game_logs.append(game_log)

    print("\n--- 竞技场评估结束 ---")

    total_games = ARENA_GAMES
    new_wins = scores["new_model_wins"]
    old_wins = scores["old_model_wins"]
    draws = scores["draws"]
    
    # [核心修改] 计算新的胜率指标
    win_vs_lose_denominator = new_wins + old_wins
    if win_vs_lose_denominator > 0:
        win_vs_lose_rate = new_wins / win_vs_lose_denominator
    else:
        win_vs_lose_rate = 0.5 # 如果没有分出胜负的对局，则认为是均势

    results = {
        "summary": {
            "total_games": total_games,
            "new_model_win_vs_lose_rate": f"{win_vs_lose_rate:.2%}",
            "new_model_total_win_rate": f"{(new_wins / total_games):.2%}" if total_games > 0 else "0.00%",
            "old_model_total_win_rate": f"{(old_wins / total_games):.2%}" if total_games > 0 else "0.00%",
            "draw_rate": f"{(draws / total_games):.2%}" if total_games > 0 else "0.00%",
        },
        "raw_scores": scores,
        "models": {"new_model": new_model_path, "old_model": old_model_path},
        "config": {"arena_games": ARENA_GAMES, "mcts_simulations": ARENA_MCTS_SIMULATIONS},
        "games": all_game_logs
    }

    print("\n[对战结果]")
    print(f"总局数: {total_games}")
    print(f"新模型胜场: {new_wins}")
    print(f"旧模型胜场: {old_wins}")
    print(f"平局:       {draws}")
    print("-" * 20)
    print(f"新模型净胜率 (胜 / (胜+负)): {results['summary']['new_model_win_vs_lose_rate']}")
    print("-" * 20)


    # [核心修改] 动态生成日志文件名
    results_dir = os.path.dirname(RESULTS_FILE_PATH)
    versioned_results_filename = f"arena_{new_model_version}_vs_{old_model_version}.json"
    versioned_results_filepath = os.path.join(results_dir, versioned_results_filename)

    os.makedirs(results_dir, exist_ok=True)
    with open(versioned_results_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n详细结果已保存至: {versioned_results_filepath}")
    
    # 结论基于新的净胜率指标
    if win_vs_lose_rate > 0.55: print("\n结论: 新模型表现显著优于旧模型！可以考虑替换。")
    elif win_vs_lose_rate > 0.5: print("\n结论: 新模型略有优势，但进步不明显。")
    else: print("\n结论: 新模型未能战胜旧模型。")
    
    # 使用 exit code 仍然可以反映一个大概的强度，便于自动化脚本处理
    exit(int(win_vs_lose_rate * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行新旧AI模型之间的竞技场评估。")
    parser.add_argument("--old_model", type=str, required=True, help="旧版本模型的路径 (.pth 文件), 例如: 'models/archive/old_v1.pth'")
    args = parser.parse_args()
    main(args.old_model)