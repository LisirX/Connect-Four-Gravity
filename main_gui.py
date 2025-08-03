# main_gui.py
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, simpledialog
import time
import threading
import queue

# 导入游戏逻辑和AI接口
from game_logic import ConnectFourGame
from ai_player import ai_move
from config import DEFAULT_ROWS, DEFAULT_COLS

# --- 游戏配置 ---
SQUARE_SIZE = 80
RADIUS = int(SQUARE_SIZE / 2 * 0.85)
TOP_MARGIN = 50
CONTROLS_HEIGHT = 40

# --- 颜色配置 ---
PLAYER1_COLOR = "Red"
PLAYER2_COLOR = "Yellow"
BOARD_COLOR = "Blue"
EMPTY_COLOR = "White"

class ConnectFourGUI:
    def __init__(self, master):
        self.master = master
        master.title("通用重力四子棋 (CNN+MCTS AI)")
        master.resizable(False, False)

        # --- 状态变量 ---
        self.game = None
        self.PLAYER_HUMAN = 1
        self.PLAYER_AI = 2
        self.is_ai_thinking = False # 新增：用于防止AI思考时重复调用
        self.ai_move_queue = queue.Queue() # 新增：用于线程通信

        # --- UI 控件变量 ---
        self.analysis_mode_var = tk.BooleanVar(value=False)
        self.player_choice_var = tk.StringVar(value="人类先手 (红色)")

        # --- 创建主框架 ---
        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- 1. 创建顶部控制栏 ---
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(5, 10))

        tk.Label(controls_frame, text="选择模式:").pack(side=tk.LEFT, padx=(0, 5))
        player_choices = ["人类先手 (红色)", "AI先手 (黄色)"]
        self.player_choice_menu = ttk.Combobox(
            controls_frame, textvariable=self.player_choice_var, values=player_choices, state="readonly", width=15
        )
        self.player_choice_menu.pack(side=tk.LEFT, padx=5)
        
        self.analysis_check = tk.Checkbutton(
            controls_frame, text="开启分析模式", variable=self.analysis_mode_var, command=self.on_mode_change
        )
        self.analysis_check.pack(side=tk.LEFT, padx=5)

        self.new_game_button = tk.Button(
            controls_frame, text="开始新游戏", command=self.prompt_new_game, font=("Arial", 10, "bold")
        )
        self.new_game_button.pack(side=tk.RIGHT, padx=5)
        
        # --- 2. 创建状态显示标签 ---
        self.status_text = tk.StringVar()
        self.status_label = tk.Label(main_frame, textvariable=self.status_text, font=("Arial", 16), pady=5)
        self.status_label.pack(fill=tk.X)

        # --- 3. Canvas (将在new_game中创建) ---
        self.canvas = None

        self.prompt_new_game()

    def on_mode_change(self):
        if self.game and not self.game.game_over:
             self.update_status("模式已更改，请点击'开始新游戏'应用设置")

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
        self.is_ai_thinking = False
        if self.player_choice_var.get() == "人类先手 (红色)":
            self.PLAYER_HUMAN = 1
            self.PLAYER_AI = 2
        else:
            self.PLAYER_HUMAN = 2
            self.PLAYER_AI = 1

        self.game = ConnectFourGame(rows=rows, cols=cols)

        window_width = cols * SQUARE_SIZE
        canvas_height = (rows * SQUARE_SIZE) + TOP_MARGIN
        total_window_height = canvas_height + CONTROLS_HEIGHT + 60
        
        self.master.geometry(f"{window_width}x{total_window_height}")
        
        if self.canvas: self.canvas.destroy()
        
        self.canvas = tk.Canvas(self.master, width=window_width, height=canvas_height, bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.handle_click)

        self.draw_board()

        if self.analysis_mode_var.get():
            self.update_status("分析模式: 请人类玩家落子")
            self.run_continuous_analysis()
        else:
            if self.game.current_player == self.PLAYER_HUMAN:
                self.update_status("轮到你了 (" + (PLAYER1_COLOR if self.PLAYER_HUMAN==1 else PLAYER2_COLOR) + ")")
            else:
                self.ai_turn()

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                x1, y1 = c * SQUARE_SIZE, r * SQUARE_SIZE + TOP_MARGIN
                x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=BOARD_COLOR, outline="black")
                player = self.game.board[r][c]
                color = EMPTY_COLOR
                if player == 1: color = PLAYER1_COLOR
                elif player == 2: color = PLAYER2_COLOR
                self.canvas.create_oval(
                    x1 + (SQUARE_SIZE // 2 - RADIUS), y1 + (SQUARE_SIZE // 2 - RADIUS),
                    x1 + (SQUARE_SIZE // 2 + RADIUS), y1 + (SQUARE_SIZE // 2 + RADIUS),
                    fill=color, outline=BOARD_COLOR
                )
        self.master.update()

    def draw_move_analysis(self, q_values):
        valid_moves = self.game.get_valid_moves()
        for c in range(self.game.cols):
            if c not in valid_moves: continue
            win_rate_percent = (q_values[c] + 1) / 2 * 100
            text = f"{win_rate_percent:.1f}%"
            x_pos = c * SQUARE_SIZE + SQUARE_SIZE / 2
            y_pos = TOP_MARGIN / 2
            if win_rate_percent >= 55: fill_color = "#4CAF50"
            elif win_rate_percent <= 45: fill_color = "#F44336"
            else: fill_color = "black"
            self.canvas.create_text(x_pos, y_pos, text=text, font=("Arial", 12, "bold"), fill=fill_color)

    def update_status(self, message):
        self.status_text.set(message)

    def handle_click(self, event):
        if self.game.game_over or self.is_ai_thinking: return
        if not self.analysis_mode_var.get() and self.game.current_player != self.PLAYER_HUMAN:
            return
        if event.y < TOP_MARGIN: return
        col = event.x // SQUARE_SIZE
        if col in self.game.get_valid_moves():
            self.process_human_move(col)

    def process_human_move(self, col):
        self.game.make_move(col)
        self.draw_board()
        if self.game.game_over:
            self.end_game()
            return
        if self.analysis_mode_var.get():
            self.update_status("分析模式: 请继续落子")
            self.run_continuous_analysis()
        else:
            self.ai_turn()

    def threaded_ai_move(self):
        """将AI计算放入此函数以在子线程中运行"""
        move, q_values = ai_move(self.game.board, self.game.current_player)
        self.ai_move_queue.put((move, q_values))

    def check_ai_result(self):
        """定期检查队列，看AI是否已返回结果"""
        try:
            move, q_values = self.ai_move_queue.get_nowait()
            self.is_ai_thinking = False
            
            # 在落子前短暂显示分析结果
            if self.analysis_mode_var.get() == False:
                 self.draw_move_analysis(q_values)
                 self.master.update()
                 time.sleep(0.5)

            if move is not None:
                self.game.make_move(move)
                self.draw_board()
                if self.game.game_over:
                    self.end_game()
                    return
            
            self.update_status("轮到你了 (" + (PLAYER1_COLOR if self.PLAYER_HUMAN==1 else PLAYER2_COLOR) + ")")

        except queue.Empty:
            # 如果队列为空，则在100毫秒后再次检查
            self.master.after(100, self.check_ai_result)

    def ai_turn(self):
        if self.game.game_over or self.is_ai_thinking: return
        
        self.is_ai_thinking = True
        self.update_status("AI (" + (PLAYER1_COLOR if self.PLAYER_AI==1 else PLAYER2_COLOR) + ") 正在思考...")
        
        # 创建并启动子线程来运行AI计算
        thread = threading.Thread(target=self.threaded_ai_move, daemon=True)
        thread.start()
        
        # 开始轮询检查结果
        self.master.after(100, self.check_ai_result)

    def run_continuous_analysis(self):
        if self.game.game_over: return
        _, q_values = ai_move(self.game.board, self.game.current_player)
        self.draw_board()
        self.draw_move_analysis(q_values)

    def end_game(self):
        self.is_ai_thinking = False
        winner = self.game.winner
        message = ""
        title = "游戏结束"
        if winner == self.PLAYER_HUMAN: message = "恭喜！你赢了！"
        elif winner == self.PLAYER_AI: message = "很遗憾，AI获胜！"
        elif winner == -1: message = "平局！棋盘已满。"
        self.update_status(message)
        messagebox.showinfo(title, message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ConnectFourGUI(root)
    root.mainloop()