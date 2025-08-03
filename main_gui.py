# main_gui.py
import tkinter as tk
import tkinter.ttk as ttk  # 导入ttk以使用更美观的控件
from tkinter import messagebox, simpledialog
import time

# 导入游戏逻辑和AI接口
from game_logic import ConnectFourGame
from ai_player import ai_move
from config import DEFAULT_ROWS, DEFAULT_COLS

# --- 游戏配置 ---
SQUARE_SIZE = 80
RADIUS = int(SQUARE_SIZE / 2 * 0.85)
TOP_MARGIN = 50  # 顶部留给分析文本的空间
CONTROLS_HEIGHT = 40 # 顶部控制栏的高度

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
        
        # --- UI 控件变量 ---
        self.analysis_mode_var = tk.BooleanVar(value=False)
        self.player_choice_var = tk.StringVar(value="人类先手 (红色)")

        # --- 创建主框架 ---
        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- 1. 创建顶部控制栏 ---
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(5, 10))

        # 先后手选择下拉框
        tk.Label(controls_frame, text="选择模式:").pack(side=tk.LEFT, padx=(0, 5))
        player_choices = ["人类先手 (红色)", "AI先手 (黄色)"]
        self.player_choice_menu = ttk.Combobox(
            controls_frame, textvariable=self.player_choice_var, values=player_choices, state="readonly", width=15
        )
        self.player_choice_menu.pack(side=tk.LEFT, padx=5)
        
        # 分析模式复选框
        self.analysis_check = tk.Checkbutton(
            controls_frame, text="开启分析模式", variable=self.analysis_mode_var, command=self.on_mode_change
        )
        self.analysis_check.pack(side=tk.LEFT, padx=5)

        # 新游戏按钮
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

        # 启动时自动开始一局新游戏
        self.prompt_new_game()

    def on_mode_change(self):
        """当模式改变时，提示用户需要开始新游戏"""
        if self.game and not self.game.game_over:
             self.update_status("模式已更改，请点击'开始新游戏'应用设置")

    def prompt_new_game(self):
        """弹出对话框让用户选择尺寸，然后开始新游戏"""
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
        """核心函数：根据用户选择的设置初始化一局新游戏"""
        # 1. 设置玩家角色
        if self.player_choice_var.get() == "人类先手 (红色)":
            self.PLAYER_HUMAN = 1
            self.PLAYER_AI = 2
        else:
            self.PLAYER_HUMAN = 2
            self.PLAYER_AI = 1

        # 2. 初始化游戏逻辑
        self.game = ConnectFourGame(rows=rows, cols=cols)

        # 3. 动态调整窗口和画布大小
        window_width = cols * SQUARE_SIZE
        canvas_height = (rows * SQUARE_SIZE) + TOP_MARGIN
        total_window_height = canvas_height + CONTROLS_HEIGHT + 60 # 加上控件和标签的高度
        
        self.master.geometry(f"{window_width}x{total_window_height}")
        
        if self.canvas:
            self.canvas.destroy()
        
        self.canvas = tk.Canvas(self.master, width=window_width, height=canvas_height, bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.handle_click)

        # 4. 绘制初始棋盘
        self.draw_board()

        # 5. 根据游戏模式决定下一步
        if self.analysis_mode_var.get():
            self.update_status("分析模式: 请人类玩家落子")
            self.run_continuous_analysis() # 立即显示空棋盘的分析
        else: # 对战模式
            if self.game.current_player == self.PLAYER_HUMAN:
                self.update_status("轮到你了 (" + (PLAYER1_COLOR if self.PLAYER_HUMAN==1 else PLAYER2_COLOR) + ")")
            else:
                self.update_status("AI (" + (PLAYER1_COLOR if self.PLAYER_AI==1 else PLAYER2_COLOR) + ") 正在思考...")
                self.master.after(200, self.ai_turn)

    def draw_board(self):
        """绘制整个棋盘（棋子和网格），会清除所有旧内容"""
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
        """在棋盘顶部绘制每一步的胜率"""
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
        """处理人类玩家的鼠标点击事件"""
        if self.game.game_over: return
        
        # 在对战模式下，只在轮到人类时响应
        if not self.analysis_mode_var.get() and self.game.current_player != self.PLAYER_HUMAN:
            return

        # 在棋盘区域内点击才有效
        if event.y < TOP_MARGIN: return
            
        col = event.x // SQUARE_SIZE
        if col in self.game.get_valid_moves():
            self.process_human_move(col)

    def process_human_move(self, col):
        """处理一步有效的人类走法"""
        self.game.make_move(col)
        self.draw_board()

        if self.game.game_over:
            self.end_game()
            return
        
        # 根据当前模式决定下一步行动
        if self.analysis_mode_var.get():
            self.update_status("分析模式: 请继续落子")
            self.run_continuous_analysis()
        else: # 对战模式
            self.update_status("AI (" + (PLAYER1_COLOR if self.PLAYER_AI==1 else PLAYER2_COLOR) + ") 正在思考...")
            self.master.after(100, self.ai_turn)

    def ai_turn(self):
        """执行一步AI走法（仅在对战模式下调用）"""
        if self.game.game_over: return

        col, q_values = ai_move(self.game.board, self.game.current_player)
        
        # 在落子前短暂显示分析结果
        self.draw_move_analysis(q_values)
        self.master.update()
        time.sleep(0.5)

        if col is not None:
            self.game.make_move(col)
            self.draw_board()

            if self.game.game_over:
                self.end_game()
                return

        self.update_status("轮到你了 (" + (PLAYER1_COLOR if self.PLAYER_HUMAN==1 else PLAYER2_COLOR) + ")")
    
    def run_continuous_analysis(self):
        """执行一次分析并显示结果（仅在分析模式下调用）"""
        if self.game.game_over: return
        
        # 即使是分析模式，也要从当前玩家的视角进行分析
        _, q_values = ai_move(self.game.board, self.game.current_player)
        self.draw_board() # 重绘棋盘以清除上一轮的分析
        self.draw_move_analysis(q_values) # 绘制新一轮的分析

    def end_game(self):
        """游戏结束时显示最终信息"""
        winner = self.game.winner
        message = ""
        title = "游戏结束"

        if winner == self.PLAYER_HUMAN:
            message = "恭喜！你赢了！"
        elif winner == self.PLAYER_AI:
            message = "很遗憾，AI获胜！"
        elif winner == -1:
            message = "平局！棋盘已满。"
        
        self.update_status(message)
        messagebox.showinfo(title, message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ConnectFourGUI(root)
    root.mainloop()