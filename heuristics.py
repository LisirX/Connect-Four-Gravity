# heuristics.py
import pickle
from config import HEURISTIC_MAX_DEPTH

# --- 定义求解器返回的状态 ---
# UNKNOWN 表示在当前深度限制内无法确定结果
# WIN 表示当前玩家有必胜策略
# LOSS 表示当前玩家无论如何都会输
# DRAW 表示当前玩家最好也只能逼和，或者游戏本身就是平局
UNKNOWN = 0
WIN = 1
LOSS = -1
DRAW = 2

def solve_position_with_lookahead(game):
    """
    [重构后的主函数]
    使用一个可配置深度的递归求解器来查找必胜或必须防守的棋步。
    
    返回:
        - ('WIN', move) 如果找到AI自己的必胜棋。
        - ('BLOCK', move) 如果找到必须防-守的对手的必胜棋。
        - None 如果在指定深度内没有发现任何确定的结果。
    """
    # 记忆化缓存，避免重复计算相同局面。键是 (board_tuple, player)。
    cache = {}
    ai_player = game.current_player
    opponent_player = 3 - ai_player
    
    # --- 1. 进攻性搜索：寻找AI自己的必胜棋 ---
    # 遍历AI所有可能的下一步棋
    for move in game.get_valid_moves():
        game_copy = pickle.loads(pickle.dumps(game))
        game_copy.make_move(move)

        # 在AI下出这一步后，轮到对手。我们从对手的角度来求解。
        # 如果对手的处境是 LOSS，那么我们这一步就是 WIN。
        result = _recursive_solve(game_copy, HEURISTIC_MAX_DEPTH - 1, cache)
        
        if result == LOSS:
            # 找到了一个能让对手必输的棋步，这就是我们的必胜棋！
            # print(f"Heuristics: Found forced WIN at move {move}")
            return ('WIN', move)

    # --- 2. 防守性搜索：寻找并阻止对手的必胜棋 ---
    # 如果没找到自己的必胜棋，就要检查对手有没有必胜棋。
    # 这至关重要，因为阻止对手获胜的优先级高于一切。
    for move in game.get_valid_moves():
        game_copy = pickle.loads(pickle.dumps(game))
        # 暂时扮演对手，看看如果他下在这里会发生什么
        game_copy.current_player = opponent_player
        game_copy.make_move(move)

        # 在对手下出这一步后，轮到我们。我们从自己的角度求解。
        # 如果我们的处境是 LOSS，说明对手刚才那步棋是制胜棋。
        result = _recursive_solve(game_copy, HEURISTIC_MAX_DEPTH - 1, cache)

        if result == LOSS:
            # 我们必须下在 `move` 这个位置来阻止对手！
            # print(f"Heuristics: Found opponent's forced win. Must BLOCK at {move}")
            return ('BLOCK', move)
            
    # 在指定深度内没有发现任何必胜或必败的局面
    return None


def _recursive_solve(game, depth, cache):
    """
    [内部递归函数]
    使用Minimax逻辑来求解给定游戏状态的结果。
    返回当前玩家的局面状态 (WIN, LOSS, DRAW, UNKNOWN)。
    """
    # --- 缓存和终止条件 ---
    # 1. 检查缓存
    cache_key = (tuple(map(tuple, game.board)), game.current_player)
    if cache_key in cache:
        return cache[cache_key]

    # 2. 检查游戏是否已经结束
    if game.game_over:
        if game.winner == -1:
            return DRAW
        # 如果胜利者是当前玩家，则返回WIN，否则返回LOSS
        # 注意：这里逻辑稍微绕，因为game_over时，胜利者是上一个玩家。
        # 所以如果胜利者不是当前玩家，说明是上一个玩家赢了。
        return LOSS if game.winner == game.current_player else WIN

    # 3. 检查是否达到搜索深度限制
    if depth <= 0:
        return UNKNOWN

    # --- 递归搜索 ---
    valid_moves = game.get_valid_moves()
    
    # 默认为必败，除非能找到更好的出路
    can_force_draw = False

    for move in valid_moves:
        game_copy = pickle.loads(pickle.dumps(game))
        game_copy.make_move(move)

        # 递归调用，获取对手视角下的结果
        result_for_opponent = _recursive_solve(game_copy, depth - 1, cache)
        
        # --- 结果解释与剪枝 ---
        # 这是Minimax的核心逻辑
        if result_for_opponent == LOSS:
            # 如果对手必败，说明我方当前这一步棋可以导向胜利。
            # 这是最佳结果，无需再检查其他分支。这就是剪枝！
            cache[cache_key] = WIN
            return WIN
        
        if result_for_opponent == DRAW:
            # 如果这一步能导向平局，记下来。
            # 这比必败要好，但我们还想找找有没有必胜的机会。
            can_force_draw = True

    # --- 循环结束后的最终判断 ---
    # 如果能走到这里，说明在所有分支中都没有找到必胜棋。
    if can_force_draw:
        # 如果没有必胜棋，但至少有一个分支可以导向平局，那么选择平局。
        cache[cache_key] = DRAW
        return DRAW
    else:
        # 如果所有分支的结果要么是对手必胜(我方必败)，要么是未知，
        # 那么我们只能认为当前局面是必败的（因为我们假设对手会走出让我们输的棋）。
        cache[cache_key] = LOSS
        return LOSS