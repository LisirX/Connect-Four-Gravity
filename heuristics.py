# heuristics.py
import pickle

def find_immediate_win_loss_search(game):
    """
    [MODIFIED] 一个简单的深度优先搜索，用于查找一步制胜或必须防守的点。
    返回: 
        - ('WIN', move) 如果找到AI自己的制胜步。
        - ('BLOCK', move) 如果找到必须防守的对手的制胜步。
        - None 如果没有发现任何紧急情况。
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
            return ('WIN', move) # 返回胜利类型和棋步

    # 2. 检查对手能否一步获胜，如果能，就必须堵住
    for move in valid_moves:
        game_copy = pickle.loads(pickle.dumps(game))
        game_copy.current_player = opponent_player
        game_copy.make_move(move)
        if game_copy.winner == opponent_player:
            return ('BLOCK', move) # 返回防守类型和棋步

    # 如果没有发现任何紧急情况
    return None