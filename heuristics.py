# heuristics.py
import pickle

# --- 定义求解器返回的状态 ---
UNKNOWN = 0
WIN = 1
LOSS = -1
DRAW = 2

# ==============================================================================
#  函数 1: 快速的一步搜索 (用于自我对弈)
# ==============================================================================
def find_immediate_win_loss_search(game):
    """
    [快速版]
    一个简单的深度优先搜索，仅用于查找一步制胜或必须防守的点。
    这个函数速度非常快，适合在自对弈中大量使用。
    """
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        return None
        
    ai_player = game.current_player
    opponent_player = 3 - ai_player

    # 1. 检查AI自己能否一步获胜
    for move in valid_moves:
        game_copy = pickle.loads(pickle.dumps(game))
        game_copy.make_move(move)
        if game_copy.winner == ai_player:
            return ('WIN', move)

    # 2. 检查对手能否一步获胜，如果能，就必须堵住
    for move in valid_moves:
        game_copy = pickle.loads(pickle.dumps(game))
        game_copy.current_player = opponent_player
        game_copy.make_move(move)
        if game_copy.winner == opponent_player:
            return ('BLOCK', move)

    return None
