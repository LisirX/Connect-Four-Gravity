# mcts.py
import numpy as np
import math
import torch
import pickle # <-- 已添加

from config import MCTS_CPUCT, MCTS_SIMULATIONS

class Node:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}  # action -> Node
        self.n_visits = 0
        self.q_value = 0 # -1 到 1 的值，代表当前玩家的期望回报
        self.u_value = 0
        self.p_value = prior_p

    def expand(self, action_priors):
        for action, prob in enumerate(action_priors):
            if action not in self.children:
                self.children[action] = Node(parent=self, prior_p=prob)

    def select_child(self):
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value())

    def update(self, leaf_value):
        self.n_visits += 1
        self.q_value += (leaf_value - self.q_value) / self.n_visits

    def get_value(self):
        # 如果父节点访问次数为0，则u_value也应为0，避免除零错误
        if self.parent.n_visits == 0:
            parent_visits_sqrt = 0
        else:
            parent_visits_sqrt = math.sqrt(self.parent.n_visits)
            
        self.u_value = MCTS_CPUCT * self.p_value * parent_visits_sqrt / (1 + self.n_visits)
        return self.q_value + self.u_value

class MCTS:
    def __init__(self, game, model, device):
        self.game = game
        self.model = model
        self.device = device
        self.root = Node() # 初始化时创建根节点

    def get_move_analysis(self, game_state, num_cols, temp=1e-3):
        """
        运行MCTS模拟并返回每一步的详细分析（策略概率和Q值）。
        """
        # 每次调用分析时，都基于当前游戏状态重置根节点
        self.root = Node()
        
        for _ in range(MCTS_SIMULATIONS):
            # 创建一个游戏副本进行模拟，避免修改原始游戏状态
            sim_game = pickle.loads(pickle.dumps(game_state))
            self.search(sim_game)

        # 从根节点的子节点中提取访问次数和 Q 值
        move_visits = np.zeros(num_cols)
        q_values = np.zeros(num_cols) # -1 到 1 的值
        
        for action, node in self.root.children.items():
            if action < num_cols:
                move_visits[action] = node.n_visits
                q_values[action] = node.q_value

        # --- 计算策略概率 ---
        if np.sum(move_visits) == 0:
            valid_moves = game_state.get_valid_moves()
            probs = np.zeros(num_cols)
            if valid_moves:
                probs[valid_moves] = 1.0 / len(valid_moves)
            policy_probs = probs
        else:
            move_probs = move_visits**(1/temp)
            policy_probs = move_probs / np.sum(move_probs)

        # 返回一个包含所有分析数据的字典
        return {
            "policy": policy_probs,
            "q_values": q_values
        }

    def search(self, game_state):
        """
        执行一次从根节点到叶节点的MCTS搜索。
        """
        node = self.root
        
        # --- SELECTION ---
        while node.children:
            action, node = node.select_child()
            game_state.make_move(action)

        # --- EXPANSION & EVALUATION ---
        if not game_state.game_over:
            board_state_tensor = torch.FloatTensor(game_state.get_board_state()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                log_policy, value = self.model(board_state_tensor)
            
            policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
            valid_moves = game_state.get_valid_moves()
            masked_policy = np.zeros_like(policy)
            if valid_moves:
                masked_policy[valid_moves] = policy[valid_moves]
                if np.sum(masked_policy) > 0:
                     masked_policy /= np.sum(masked_policy)
                else: 
                     masked_policy[valid_moves] = 1.0 / len(valid_moves)

            node.expand(masked_policy)
            leaf_value = -value.item() # 从下一个玩家的角度看，价值是负的
        else:
            if game_state.winner == -1: # Draw
                leaf_value = 0.0
            else: # Win/Loss
                # 如果游戏结束，那么对于上一个玩家来说是胜利
                leaf_value = 1.0 

        # --- BACKPROPAGATION ---
        while node is not None:
            # 价值是相对于当前节点的父节点的玩家而言的
            node.update(leaf_value)
            leaf_value = -leaf_value # 每次向上传播时，视角反转
            node = node.parent