import numpy as np
import math
import torch
from config import MCTS_CPUCT, MCTS_SIMULATIONS

class Node:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}  # action -> Node
        self.n_visits = 0
        self.q_value = 0
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
        self.u_value = MCTS_CPUCT * self.p_value * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.q_value + self.u_value

class MCTS:
    def __init__(self, game, model, device):
        self.game = game
        self.model = model
        self.device = device
    
    def get_move_probs(self, game_state, num_cols, temp=1e-3):
        for _ in range(MCTS_SIMULATIONS):
            self.search(game_state)

        move_visits = np.array([self.root.children.get(a, Node()).n_visits for a in range(num_cols)])
        if np.sum(move_visits) == 0: # 如果没有访问任何子节点，返回均匀分布
            valid_moves = game_state.get_valid_moves()
            probs = np.zeros(COLS)
            if valid_moves:
                probs[valid_moves] = 1.0 / len(valid_moves)
            return probs

        move_probs = move_visits**(1/temp)
        move_probs /= np.sum(move_probs)
        return move_probs

    def search(self, game_state):
        self.root = Node()
        
        # --- SELECTION ---
        node = self.root
        current_game = game_state
        
        while node.children:
            action, node = node.select_child()
            current_game.make_move(action)

        # --- EXPANSION & SIMULATION(EVALUATION) ---
        if not current_game.game_over:
            board_state_tensor = torch.FloatTensor(current_game.get_board_state()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                log_policy, value = self.model(board_state_tensor)
            
            policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
            valid_moves = current_game.get_valid_moves()
            masked_policy = np.zeros_like(policy)
            if valid_moves:
                masked_policy[valid_moves] = policy[valid_moves]
                if np.sum(masked_policy) > 0:
                     masked_policy /= np.sum(masked_policy)
                else: # 防止所有合法走法概率都为0
                     masked_policy[valid_moves] = 1.0 / len(valid_moves)

            node.expand(masked_policy)
            leaf_value = -value.item() # 从下一个玩家的角度看，价值是负的
        else:
            if current_game.winner == -1: # Draw
                leaf_value = 0.0
            else: # Win/Loss
                leaf_value = 1.0 # 假设上一步导致游戏结束，那么上一个玩家赢了

        # --- BACKPROPAGATION ---
        while node is not None:
            node.update(leaf_value)
            leaf_value = -leaf_value
            node = node.parent