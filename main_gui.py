# main_gui.py
import tkinter as tk
from tkinter import messagebox, simpledialog # <-- 确保 simpledialog 已导入
import time # <-- 已添加

# 导入新的游戏逻辑和AI接口
from game_logic import ConnectFourGame 
from ai_player import ai_move
from config import DEFAULT_ROWS, DEFAULT_COLS # <-- 导入默认尺寸

# --- 游戏配置 ---
SQUARE_SIZE = 80
RADIUS = int(SQUARE_SIZE / 2 * 0.85)

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

        self.game = None
        self.square_size = SQUARE_SIZE
        self.radius = RADIUS

        self.status_text = tk.StringVar()
        self.status_label = tk.Label(master, textvariable=self.status_text, font=("Arial", 16), pady=10)
        self.status_label.pack()

        self.canvas = None

        self.new_game_button = tk.Button(master, text="新游戏", command=self.prompt_new_game, font=("Arial", 14))
        self.new_game_button.pack(pady=10)

        self.prompt_new_game()

    def prompt_new_game(self):
        try:
            res_str = simpledialog.askstring("新游戏", "输入棋盘尺寸 (例如 '6x7'):", initialvalue=f"{DEFAULT_ROWS}x{DEFAULT_COLS}")
            if res_str is None: return
            rows, cols = map(int, res_str.lower().split('x'))
            if rows < 4 or cols < 4:
                messagebox.showerror("尺寸无效", "行和列都必须至少为4。")
                return
            self.new_game(rows, cols)
        except (ValueError, TypeError):
             messagebox.showerror("输入无效", "请输入正确的格式，例如 '6x7'")

    def new_game(self, rows, cols):
        self.game = ConnectFourGame(rows=rows, cols=cols)
        
        window_width = cols * self.square_size
        # 为顶部的胜率显示留出更多空间
        top_margin = 60 
        window_height = (rows * self.square_size) + top_margin
        
        self.master.geometry(f"{window_width}x{window_height+60}")
        
        if self.canvas:
            self.canvas.destroy()
        
        self.canvas = tk.Canvas(self.master, width=window_width, height=window_height, bg="lightgray")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)

        self.draw_board()
        self.update_status("轮到你了 (红色)")

    def draw_board(self):
        self.canvas.delete("all")
        # 棋盘区域有偏移，因为顶部要留给分析文本
        board_y_offset = 60
        
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                x1, y1 = c * self.square_size, r * self.square_size + board_y_offset
                x2, y2 = x1 + self.square_size, y1 + self.square_size
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=BOARD_COLOR, outline="black")
                
                player = self.game.board[r][c]
                color = EMPTY_COLOR
                if player == PLAYER_HUMAN: color = PLAYER1_COLOR
                elif player == PLAYER_AI: color = PLAYER2_COLOR
                
                self.canvas.create_oval(
                    x1 + (self.square_size // 2 - self.radius), y1 + (self.square_size // 2 - self.radius),
                    x1 + (self.square_size // 2 + self.radius), y1 + (self.square_size // 2 + self.radius),
                    fill=color, outline=BOARD_COLOR
                )
        self.master.update()

    def draw_move_analysis(self, q_values):
        """在棋盘顶部绘制每一步的胜率"""
        valid_moves = self.game.get_valid_moves()
        
        for c in range(self.game.cols):
            if c not in valid_moves:
                continue

            win_rate_percent = (q_values[c] + 1) / 2 * 100
            text = f"{win_rate_percent:.1f}%"
            
            x_pos = c * self.square_size + self.square_size / 2
            y_pos = 30 # Y 坐标固定在顶部区域
            
            if win_rate_percent >= 55: fill_color = "#4CAF50" # 亮绿色
            elif win_rate_percent <= 45: fill_color = "#F44336" # 亮红色
            else: fill_color = "black"

            self.canvas.create_text(
                x_pos, y_pos, text=text, font=("Arial", 12, "bold"), fill=fill_color, tag="analysis_text"
            )

    def update_status(self, message):
        self.status_text.set(message)

    def handle_click(self, event):
        if self.game.game_over or self.game.current_player != PLAYER_HUMAN:
            return

        col = event.x // self.square_size
        if col in self.game.get_valid_moves():
            self.process_move(col)

    def process_move(self, col):
        self.game.make_move(col)
        self.draw_board() # 玩家落子后立即重绘，清除AI的分析文本

        if self.game.game_over:
            self.end_game()
            return
            
        self.update_status("AI (黄色) 正在思考...")
        self.master.after(100, self.ai_turn) 

    def ai_turn(self):
        if self.game.game_over: return

        # 调用新的AI接口，现在会返回走法和Q值
        col, q_values = ai_move(self.game.board, self.game.current_player)

        # 在AI落子前，先绘制胜率分析
        self.draw_move_analysis(q_values)
        self.master.update() 
        time.sleep(0.7) # 短暂暂停，让你能看清胜率

        if col is not None:
            self.game.make_move(col)
            # 落子后，重绘整个棋盘 (这会清除旧的胜率文本)
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