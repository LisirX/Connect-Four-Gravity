# mcts.py
import numpy as np
import math
import torch
import pickle

# CHANGED: 导入 MCTS_CPUCT，但不再需要默认的 SIMULATIONS
from config import MCTS_CPUCT

class Node:
    # ... (Node 类的代码保持不变) ...
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}
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
        if self.parent.n_visits == 0:
            parent_visits_sqrt = 0
        else:
            parent_visits_sqrt = math.sqrt(self.parent.n_visits)
        self.u_value = MCTS_CPUCT * self.p_value * parent_visits_sqrt / (1 + self.n_visits)
        return self.q_value + self.u_value

class MCTS:
    # CHANGED: 构造函数接受模拟次数作为参数
    def __init__(self, game, model, device, simulations):
        self.game = game
        self.model = model
        self.device = device
        self.simulations = simulations # NEW: 存储模拟次数
        self.root = Node()

    def get_move_analysis(self, game_state, num_cols, temp=1e-3):
        self.root = Node()
        # CHANGED: 使用实例变量 self.simulations
        for _ in range(self.simulations):
            sim_game = pickle.loads(pickle.dumps(game_state))
            self.search(sim_game)

        move_visits = np.zeros(num_cols)
        q_values = np.zeros(num_cols)
        for action, node in self.root.children.items():
            if action < num_cols:
                move_visits[action] = node.n_visits
                q_values[action] = node.q_value
        
        # ... (get_move_analysis 的其余部分代码保持不变) ...
        if np.sum(move_visits) == 0:
            valid_moves = game_state.get_valid_moves()
            probs = np.zeros(num_cols)
            if valid_moves:
                probs[valid_moves] = 1.0 / len(valid_moves)
            policy_probs = probs
        else:
            if temp < 1e-2:
                policy_probs = np.zeros_like(move_visits)
                max_visits = np.max(move_visits)
                best_actions = np.where(move_visits == max_visits)[0]
                best_action = np.random.choice(best_actions)
                policy_probs[best_action] = 1.0
            else:
                move_probs = move_visits**(1/temp)
                policy_probs = move_probs / np.sum(move_probs)

        if np.isnan(np.sum(policy_probs)):
            print("Warning: NaN detected in policy probabilities. Falling back to uniform distribution.")
            valid_moves = game_state.get_valid_moves()
            policy_probs = np.zeros(num_cols)
            if valid_moves:
                policy_probs[valid_moves] = 1.0 / len(valid_moves)
        
        return {"policy": policy_probs, "q_values": q_values}


    # ... (search 方法的代码保持不变) ...
    def search(self, game_state):
        node = self.root
        while node.children:
            action, node = node.select_child()
            game_state.make_move(action)

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
            leaf_value = -value.item()
        else:
            if game_state.winner == -1: leaf_value = 0.0
            else: leaf_value = 1.0 

        while node is not None:
            node.update(leaf_value)
            leaf_value = -leaf_value
            node = node.parent