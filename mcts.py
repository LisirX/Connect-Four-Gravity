# mcts.py
import numpy as np
import math
import torch
import pickle
import queue

from config import MCTS_CPUCT

class Node:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent; self.children = {}; self.n_visits = 0
        self.q_value = 0; self.u_value = 0; self.p_value = prior_p
    def expand(self, action_priors):
        for action, prob in enumerate(action_priors):
            if action not in self.children: self.children[action] = Node(parent=self, prior_p=prob)
    def select_child(self):
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value())
    def update(self, leaf_value):
        self.n_visits += 1; self.q_value += (leaf_value - self.q_value) / self.n_visits
    def get_value(self):
        parent_visits_sqrt = math.sqrt(self.parent.n_visits) if self.parent and self.parent.n_visits > 0 else 1
        self.u_value = MCTS_CPUCT * self.p_value * parent_visits_sqrt / (1 + self.n_visits)
        return self.q_value + self.u_value

class MCTS:
    def __init__(self, model, device, simulations):
        self.model = model; self.device = device; self.simulations = simulations
        self.root = None
        self.game_state_at_root = None

    def set_game_state(self, game):
        current_board_tuple = tuple(map(tuple, game.board))
        root_board_tuple = None
        if self.game_state_at_root:
            root_board_tuple = tuple(map(tuple, self.game_state_at_root.board))
        if self.root is None or root_board_tuple != current_board_tuple or self.game_state_at_root.current_player != game.current_player:
            self.root = Node()
            self.game_state_at_root = pickle.loads(pickle.dumps(game))
            # print("MCTS root has been reset for the new game state.")

    def ponder(self, num_sims=100):
        if self.root is None or self.game_state_at_root is None: return
        for _ in range(num_sims):
            sim_game = pickle.loads(pickle.dumps(self.game_state_at_root))
            self._search(sim_game, self.root)

    def get_q_values(self, num_cols):
        if self.root is None: return np.full(num_cols, -2.0)
        q_values = np.full(num_cols, -2.0)
        for action, node in self.root.children.items():
            q_values[action] = node.q_value
        return q_values

    def get_move(self, game, update_queue=None, is_thinking_flag=None):
        self.set_game_state(game)
        if self.simulations > 0: update_freq = max(1, self.simulations // 200)
        else: update_freq = 100

        for i in range(self.simulations):
            if is_thinking_flag and not is_thinking_flag[0]: break
            sim_game = pickle.loads(pickle.dumps(game))
            self._search(sim_game, self.root)
            if update_queue and (i + 1) % update_freq == 0:
                try: update_queue.put({'sims': i + 1, 'q_values': self.get_q_values(game.cols)}, block=False)
                except queue.Empty: pass
        
        if not self.root.children: return None
        return max(self.root.children.items(), key=lambda act_node: act_node[1].n_visits)[0]
    
    def get_final_q_values(self, game):
        """ 一个简单的辅助方法，在get_move之后调用，以获取最终的Q值。"""
        return self.get_q_values(game.cols)

    def get_move_analysis(self, game, temp=1e-3):
        """
        [RESTORED & REFACTORED] 重新加入此方法，专门用于生成训练数据。
        它运行完整的模拟，并返回策略向量和Q值。
        """
        self.set_game_state(game)
        for _ in range(self.simulations):
            sim_game = pickle.loads(pickle.dumps(game))
            self._search(sim_game, self.root)

        num_cols = game.cols
        move_visits = np.zeros(num_cols)
        for action, node in self.root.children.items():
            move_visits[action] = node.n_visits
        
        # 计算策略向量
        if np.sum(move_visits) == 0:
            valid_moves = game.get_valid_moves()
            probs = np.zeros(num_cols)
            if valid_moves: probs[valid_moves] = 1.0 / len(valid_moves)
            policy_probs = probs
        else:
            if temp < 1e-2: # 在竞技或评估时，选择访问次数最多的
                policy_probs = np.zeros_like(move_visits)
                best_action = np.argmax(move_visits)
                policy_probs[best_action] = 1.0
            else: # 在自对弈初期，增加探索
                move_probs = move_visits**(1/temp)
                policy_probs = move_probs / np.sum(move_probs)
        
        return {"policy": policy_probs, "q_values": self.get_q_values(num_cols)}


    def _search(self, game_state, node):
        while node.children:
            action, node = node.select_child(); game_state.make_move(action)
        leaf_value = 0.0
        if not game_state.game_over:
            board_state_tensor = torch.FloatTensor(game_state.get_board_state()).unsqueeze(0).to(self.device)
            board = game_state.board; rows, cols = board.shape
            pieces_in_col = np.sum(board != 0, axis=0); open_rows = (rows - pieces_in_col - 1).astype(np.int64)
            target_rows_tensor = torch.from_numpy(open_rows).unsqueeze(0).to(self.device)
            with torch.no_grad():
                log_policy, value = self.model(board_state_tensor, target_rows_tensor)
            policy = torch.exp(log_policy).squeeze(0).cpu().numpy(); valid_moves = game_state.get_valid_moves()
            masked_policy = np.zeros_like(policy)
            if valid_moves:
                masked_policy[valid_moves] = policy[valid_moves]
                if np.sum(masked_policy) > 0: masked_policy /= np.sum(masked_policy)
                else: masked_policy[valid_moves] = 1.0 / len(valid_moves)
            node.expand(masked_policy); leaf_value = -value.item()
        else: leaf_value = 1.0 if game_state.winner != -1 else 0.0
        while node is not None:
            node.update(leaf_value); leaf_value = -leaf_value; node = node.parent