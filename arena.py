# arena.py
import torch
import numpy as np
import os
import argparse
import json
import random
from tqdm import tqdm

# 导入项目模块
from game_logic import ConnectFourGame
from neural_network import UniversalConnectFourNet as ConnectFourNet
from mcts import MCTS
from config import MODEL_SAVE_PATH, ARENA_GAMES, ARENA_MCTS_SIMULATIONS, RESULTS_FILE_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArenaPlayer:
    """
    一个封装了AI模型和MCTS的玩家，用于竞技场对战。
    """
    def __init__(self, model_path, name, simulations):
        self.name = name
        self.simulations = simulations
        
        # 加载指定的模型
        self.model = ConnectFourNet().to(DEVICE)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}。请确保路径正确。")
        
        # 使用 weights_only=True 更安全，如果模型是你自己训练的
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        self.model.eval()
        print(f"玩家 '{self.name}' 已成功加载模型: {model_path}")

    def get_move(self, game):
        """
        根据当前棋局，使用MCTS计算并返回最佳动作。
        """
        # 为当前棋局状态创建一个新的MCTS实例
        mcts = MCTS(game, self.model, DEVICE, simulations=self.simulations)
        
        analysis = mcts.get_move_analysis(game, game.cols)
        action_probs = analysis["policy"]
        
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        
        # 在评估时，我们总是选择访问次数最多的“确定性”动作
        best_move = np.argmax(action_probs)
        return int(best_move)

def play_game(p1: ArenaPlayer, p2: ArenaPlayer):
    """
    让两个AI玩家进行一局对战。
    
    Args:
        p1 (ArenaPlayer): 执先手 (玩家1)
        p2 (ArenaPlayer): 执后手 (玩家2)

    Returns:
        int: 返回获胜玩家的编号 (1 或 2), 或 -1 表示平局。
    """
    # 随机选择棋盘尺寸来测试模型的通用性
    rows = random.randint(5, 8)
    cols = random.randint(5, 8)
    game = ConnectFourGame(rows=rows, cols=cols)

    while not game.game_over:
        current_player_obj = p1 if game.current_player == 1 else p2
        move = current_player_obj.get_move(game)
        
        if move is None or move not in game.get_valid_moves():
            # 如果AI返回了一个无效的移动（理论上不应该发生），判其告负
            return 3 - game.current_player 
            
        game.make_move(move)
        
    return game.winner

# arena.py -> main() 函数的完整替换代码

def main(old_model_path):
    print(f"设备: {DEVICE}")
    print("--- 开始模型竞技场评估 ---")

    # 定义新旧模型玩家
    # 注意：在我们的自动化脚本中，“新模型”总是位于主路径上
    new_player = ArenaPlayer(MODEL_SAVE_PATH, "挑战者(新)", ARENA_MCTS_SIMULATIONS)
    old_player = ArenaPlayer(old_model_path, "冠军(旧)", ARENA_MCTS_SIMULATIONS)

    # 初始化计分板
    scores = {
        "new_model_wins": 0,
        "old_model_wins": 0,
        "draws": 0
    }
    
    # 使用tqdm显示进度条
    pbar = tqdm(range(ARENA_GAMES), desc=f"竞技场: {new_player.name} vs {old_player.name}")

    for i in pbar:
        # 每局交替先手以保证公平
        if i % 2 == 0:
            p1, p2 = new_player, old_player
            pbar.set_postfix({"先手": p1.name})
            winner = play_game(p1, p2)
            if winner == 1: scores["new_model_wins"] += 1
            elif winner == 2: scores["old_model_wins"] += 1
            else: scores["draws"] += 1
        else:
            p1, p2 = old_player, new_player
            pbar.set_postfix({"先手": p1.name})
            winner = play_game(p1, p2)
            if winner == 1: scores["old_model_wins"] += 1
            elif winner == 2: scores["new_model_wins"] += 1
            else: scores["draws"] += 1

    print("\n--- 竞技场评估结束 ---")

    total_games = ARENA_GAMES
    new_wins = scores["new_model_wins"]
    old_wins = scores["old_model_wins"]
    draws = scores["draws"]

    win_rate = new_wins / total_games if total_games > 0 else 0
    
    results = {
        "summary": {
            "total_games": total_games,
            "new_model_win_rate": f"{win_rate:.2%}",
            "old_model_win_rate": f"{old_wins / total_games:.2%}" if total_games > 0 else "0.00%",
            "draw_rate": f"{draws / total_games:.2%}" if total_games > 0 else "0.00%",
        },
        "raw_scores": scores,
        "models": {
            "new_model": MODEL_SAVE_PATH,
            "old_model": old_model_path
        },
        "config": {
            "arena_games": ARENA_GAMES,
            "mcts_simulations": ARENA_MCTS_SIMULATIONS
        }
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
    
    # 返回新模型的整数胜率作为退出码，以便批处理脚本捕获
    exit(int(win_rate * 100))


if __name__ == "__main__":
    # 使用argparse来从命令行接收旧模型的路径
    parser = argparse.ArgumentParser(description="运行新旧AI模型之间的竞技场评估。")
    parser.add_argument(
        "--old_model", 
        type=str, 
        required=True, 
        help="旧版本模型的路径 (.pth 文件), 例如: 'models/archive/old_v1.pth'"
    )
    args = parser.parse_args()
    
    main(args.old_model)