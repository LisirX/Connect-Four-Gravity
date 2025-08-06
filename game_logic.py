# game_logic.py
import numpy as np

class ConnectFourGame:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = 0

    def get_valid_moves(self):
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def make_move(self, col):
        if col not in self.get_valid_moves():
            return False

        r = self.get_next_open_row(col)
        self.board[r][col] = self.current_player
        
        if self.check_win((r, col)):
            self.game_over = True
            self.winner = self.current_player
        elif self.check_draw():
            self.game_over = True
            self.winner = -1

        self.switch_player()
        return True

    def get_next_open_row(self, col):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                return r
        return None

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def check_win(self, last_move):
        if not last_move: return False
        player = self.board[last_move[0]][last_move[1]]
        row, col = last_move

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == player:
                    count += 1
                else: break
            for i in range(1, 4):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == player:
                    count += 1
                else: break
            if count >= 4:
                return True
        return False

    def check_draw(self):
        return 0 not in self.board[0]
        
    def get_board_state(self):
        state = np.zeros((3, self.rows, self.cols))
        state[0] = (self.board == 1)
        state[1] = (self.board == 2)
        state[2] = (self.current_player == 1)
        return state