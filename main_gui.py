import tkinter as tk
from tkinter import messagebox
import time

# 导入新的游戏逻辑和AI接口
from game_logic import ConnectFourGame 
from ai_player import ai_move

# --- 游戏配置 ---
ROWS = 6
COLS = 7
SQUARE_SIZE = 80
RADIUS = int(SQUARE_SIZE / 2 * 0.85)
WINDOW_WIDTH = COLS * SQUARE_SIZE
WINDOW_HEIGHT = (ROWS + 1) * SQUARE_SIZE

# --- 颜色配置 ---
PLAYER1_COLOR = "red"
PLAYER2_COLOR = "yellow"
BOARD_COLOR = "blue"
EMPTY_COLOR = "white"

# --- 玩家定义 ---
PLAYER_HUMAN = 1
PLAYER_AI = 2

class ConnectFourGUI:
    def __init__(self, master):
        self.master = master
        master.title("通用重力四子棋 (CNN+MCTS AI)")
        master.resizable(False, False)

        # Game将在new_game中初始化
        self.game = None
        self.square_size = 80
        self.radius = int(self.square_size / 2 * 0.85)

        self.status_text = tk.StringVar()
        self.status_label = tk.Label(master, textvariable=self.status_text, font=("Arial", 16), pady=10)
        self.status_label.pack()

        # Canvas将在new_game中创建
        self.canvas = None

        self.new_game_button = tk.Button(master, text="新游戏", command=self.prompt_new_game, font=("Arial", 14))
        self.new_game_button.pack(pady=10)

        self.prompt_new_game()

    # NEW: 弹出对话框让用户选择尺寸
    def prompt_new_game(self):
        try:
            res_str = simpledialog.askstring("新游戏", "输入棋盘尺寸 (例如 '6x7'):", initialvalue=f"{DEFAULT_ROWS}x{DEFAULT_COLS}")
            if res_str is None: return # 用户取消
            rows, cols = map(int, res_str.lower().split('x'))
            if rows < 4 or cols < 4:
                messagebox.showerror("尺寸无效", "行和列都必须至少为4。")
                return
            self.new_game(rows, cols)
        except (ValueError, TypeError):
             messagebox.showerror("输入无效", "请输入正确的格式，例如 '6x7'")

    # CHANGED: 接受行列数
    def new_game(self, rows, cols):
        self.game = ConnectFourGame(rows=rows, cols=cols)
        
        # 动态调整窗口和画布大小
        window_width = cols * self.square_size
        window_height = (rows + 1) * self.square_size
        self.master.geometry(f"{window_width}x{window_height}")
        
        if self.canvas:
            self.canvas.destroy()
        
        self.canvas = tk.Canvas(self.master, width=window_width, height=rows * self.square_size, bg="blue")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)

        self.draw_board()
        self.update_status("轮到你了 (红色)")

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                x1, y1 = c * SQUARE_SIZE, r * SQUARE_SIZE
                x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=BOARD_COLOR, outline="black")
                
                player = self.game.board[r][c]
                color = EMPTY_COLOR
                if player == PLAYER_HUMAN: color = PLAYER1_COLOR
                elif player == PLAYER_AI: color = PLAYER2_COLOR
                
                self.canvas.create_oval(
                    x1 + (SQUARE_SIZE // 2 - RADIUS), y1 + (SQUARE_SIZE // 2 - RADIUS),
                    x1 + (SQUARE_SIZE // 2 + RADIUS), y1 + (SQUARE_SIZE // 2 + RADIUS),
                    fill=color, outline=BOARD_COLOR
                )
        self.master.update()

    def update_status(self, message):
        self.status_text.set(message)

    def handle_click(self, event):
        if self.game.game_over or self.game.current_player != PLAYER_HUMAN:
            return

        col = event.x // SQUARE_SIZE
        if col in self.game.get_valid_moves():
            self.process_move(col)

    def process_move(self, col):
        self.game.make_move(col)
        self.draw_board()

        if self.game.game_over:
            self.end_game()
            return
            
        self.update_status("AI (黄色) 正在思考...")
        self.master.after(100, self.ai_turn) # 缩短延迟

    def ai_turn(self):
        if self.game.game_over: return

        # --- 调用新的AI接口 ---
        col = ai_move(self.game.board, self.game.current_player)

        if col is not None:
            self.game.make_move(col)
            self.draw_board()

            if self.game.game_over:
                self.end_game()
                return

        self.update_status("轮到你了 (红色)")

    def end_game(self):
        winner = self.game.winner
        message = ""
        title = "游戏结束"
        if winner == PLAYER_HUMAN:
            message = "恭喜！你赢了！"
            self.update_status("恭喜！你赢了！")
        elif winner == PLAYER_AI:
            message = "很遗憾，AI获胜！"
            self.update_status("很遗憾，AI获胜！")
        elif winner == -1:
            message = "平局！棋盘已满。"
            self.update_status("平局！")
        
        messagebox.showinfo(title, message)


if __name__ == "__main__":
    root = tk.Tk()
    app = ConnectFourGUI(root)
    root.mainloop()