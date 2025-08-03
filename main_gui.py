# main_gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
from enum import Enum, auto
import os
import torch
import pickle

# [MODIFIED] 直接从需要的模块导入
from heuristics import find_immediate_win_loss_search
from game_logic import ConnectFourGame
from mcts import MCTS
from neural_network import UniversalConnectFourNet
from config import *

# --- 全局AI模型实例 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AI_MODEL = UniversalConnectFourNet().to(DEVICE)
if os.path.exists(MODEL_SAVE_PATH):
    try:
        AI_MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
        print(f"AI Player: Loaded universal model from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"AI Player: WARNING - Could not load model. Error: {e}. AI will perform poorly.")
else:
    print("AI Player: WARNING - No trained model found. AI will perform poorly.")
AI_MODEL.eval()

class Style:
    BACKGROUND="#FFFFFF"; CONTROLS_BACKGROUND="#F1F1F1"; BOARD_COLOR="#0053A6"
    PLAYER1_COLOR="#D93025"; PLAYER1_SHADOW="#A3241C"; PLAYER2_COLOR="#F9AB00"
    PLAYER2_SHADOW="#B47B00"; EMPTY_COLOR="#E8F0FE"; ANALYSIS_GOOD="#1E8E3E"
    ANALYSIS_BAD="#D93025"; ANALYSIS_NEUTRAL="#BDC1C6"; STATUS_FONT=("Segoe UI", 12)
    CONTROL_FONT=("Segoe UI", 10); ANALYSIS_FONT=("Segoe UI", 12, "bold")

class AppState(Enum):
    IDLE = auto(); HUMAN_TURN = auto(); AI_THINKING = auto()
    ANALYZING = auto(); ANIMATING = auto(); GAME_OVER = auto()

class BoardCanvas(tk.Canvas):
    # 此类代码与上一版完全相同，无需修改
    def __init__(self, master, style):
        super().__init__(master, bg=style.BACKGROUND, highlightthickness=0)
        self.style = style; self.game = None; self.hovered_col = None; self.q_values = None
        self.square_size = 0; self.radius = 0; self.x_offset = 0; self.y_offset = 0; self.top_margin = 0
    def _calculate_sizes(self):
        canvas_w, canvas_h = self.winfo_width(), self.winfo_height()
        cols = self.game.cols if self.game else DEFAULT_COLS; rows = self.game.rows if self.game else DEFAULT_ROWS
        self.top_margin = min(canvas_w / cols, canvas_h / (rows + 1.5)); h_size = (canvas_h - self.top_margin) / rows
        w_size = canvas_w / cols; self.square_size = max(1, min(h_size, w_size)); self.radius = self.square_size * 0.42
        board_w = self.square_size * cols; board_h = self.square_size * rows
        self.x_offset = (canvas_w - board_w) / 2; self.y_offset = self.top_margin
    def get_col_from_event(self, event):
        if self.square_size <= 0: return None
        col = int((event.x - self.x_offset) // self.square_size)
        return col if self.game and 0 <= col < self.game.cols else None
    def refresh(self, game, hovered_col=None, q_values=None):
        self.game = game; self.hovered_col = hovered_col; self.q_values = q_values
        self._calculate_sizes(); self.delete("all")
        if not self.game:
            self.create_rectangle(self.x_offset, self.y_offset, self.x_offset + DEFAULT_COLS * self.square_size, self.y_offset + DEFAULT_ROWS * self.square_size, fill=self.style.BOARD_COLOR, outline="")
            return
        self._draw_board_and_holes(); self._draw_all_pieces()
        if self.q_values is not None: self._draw_analysis()
    def draw_hover(self, col, player):
        self.delete("hover");
        if col is None or self.square_size <= 1: return
        color = self.style.PLAYER1_COLOR if player == 1 else self.style.PLAYER2_COLOR
        cx = self.x_offset + col * self.square_size + self.square_size / 2; cy = self.top_margin / 2
        self.create_oval(cx - self.radius, cy - self.radius, cx + self.radius, cy + self.radius, fill=color, outline="", stipple="gray25", tags="hover")
    def _draw_board_and_holes(self):
        x0, y0 = self.x_offset, self.y_offset; x1 = x0 + self.game.cols * self.square_size; y1 = y0 + self.game.rows * self.square_size
        self.create_rectangle(x0, y0, x1, y1, fill=self.style.BOARD_COLOR, outline="")
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                cx = self.x_offset + c * self.square_size + self.square_size / 2; cy = self.y_offset + r * self.square_size + self.square_size / 2
                self.create_oval(cx - self.radius, cy - self.radius, cx + self.radius, cy + self.radius, fill=self.style.EMPTY_COLOR, outline="")
    def _draw_all_pieces(self):
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                if self.game.board[r][c] != 0: self._draw_piece(r, c, self.game.board[r][c])
    def _draw_piece(self, r, c, player, custom_coords=None, tags=()):
        if player == 1: main_color, shadow_color = self.style.PLAYER1_COLOR, self.style.PLAYER1_SHADOW
        else: main_color, shadow_color = self.style.PLAYER2_COLOR, self.style.PLAYER2_SHADOW
        if custom_coords: cx, cy = custom_coords
        else: cx = self.x_offset + c * self.square_size + self.square_size / 2; cy = self.y_offset + r * self.square_size + self.square_size / 2
        self.create_oval(cx - self.radius, cy - self.radius, cx + self.radius, cy + self.radius, fill=shadow_color, outline="", tags=tags)
        self.create_oval(cx - self.radius, cy - self.radius * 1.1, cx + self.radius, cy + self.radius * 0.9, fill=main_color, outline="", tags=tags)
    def _get_contrast_color(self, hex_color):
        try:
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            brightness = ((r * 299) + (g * 587) + (b * 114)) / 1000
            return "white" if brightness < 128 else "black"
        except: return "black"
    def _draw_analysis(self):
        if self.q_values is None: return
        for c, q_val in enumerate(self.q_values):
            if q_val <= -1.5: continue
            win_rate = (q_val + 1) / 2; text = f"{win_rate:.1%}"; x1 = self.x_offset + c * self.square_size
            bar_color = self.style.ANALYSIS_NEUTRAL
            if win_rate >= 0.99: bar_color = self.style.ANALYSIS_GOOD
            elif win_rate >= 0.55: bar_color = self.style.ANALYSIS_GOOD
            elif win_rate <= 0.45: bar_color = self.style.ANALYSIS_BAD
            self.create_rectangle(x1, 0, x1 + self.square_size, self.top_margin, fill="#E0E0E0", outline="")
            bar_height = self.top_margin * win_rate
            if bar_height > 0: self.create_rectangle(x1, self.top_margin - bar_height, x1 + self.square_size, self.top_margin, fill=bar_color, outline="")
            text_color = self._get_contrast_color(bar_color if bar_height > self.top_margin * 0.4 else "#E0E0E0")
            self.create_text(x1 + self.square_size / 2, self.top_margin / 2, text=text, font=self.style.ANALYSIS_FONT, fill=text_color)
    def animate_piece_drop(self, col, target_row, player, callback):
        self.delete("hover"); self.refresh(self.game)
        start_y = self.top_margin / 2; end_y = self.y_offset + target_row * self.square_size + self.square_size / 2
        cx = self.x_offset + col * self.square_size + self.square_size / 2
        self._draw_piece(-1, -1, player, custom_coords=(cx, start_y), tags="anim_piece")
        current_y = start_y; velocity = 0; gravity = self.square_size / 150
        def animation_step():
            nonlocal current_y, velocity; velocity += gravity; current_y += velocity
            if current_y >= end_y: self.delete("anim_piece"); self.refresh(self.game); callback(); return
            anim_items = self.find_withtag("anim_piece")
            if len(anim_items) == 2:
                self.coords(anim_items[0], cx - self.radius, current_y - self.radius, cx + self.radius, current_y + self.radius)
                self.coords(anim_items[1], cx - self.radius, current_y - self.radius * 1.1, cx + self.radius, current_y + self.radius * 0.9)
            self.master.after(10, animation_step)
        animation_step()

class ControlsFrame(ttk.Frame):
    def __init__(self, master, app_callbacks, variables):
        super().__init__(master, style="Controls.TFrame")
        ttk.Label(self, text="棋盘:", font=Style.CONTROL_FONT).pack(side=tk.LEFT, padx=(5,2))
        self.board_size_combo = ttk.Combobox(self, textvariable=variables['board_size'], values=["6x5", "6x7", "7x6", "8x7"], width=7, font=Style.CONTROL_FONT, state="readonly")
        self.board_size_combo.pack(side=tk.LEFT, padx=2)
        self.analysis_check = ttk.Checkbutton(self, text="分析模式", variable=variables['analysis_mode'], command=app_callbacks['on_control_change'])
        self.analysis_check.pack(side=tk.LEFT, padx=(15, 5))
        self.ai_red_check = ttk.Checkbutton(self, text="电脑执红", variable=variables['is_ai_red'], command=app_callbacks['on_control_change'])
        self.ai_red_check.pack(side=tk.LEFT, padx=5)
        self.ai_black_check = ttk.Checkbutton(self, text="电脑执黑", variable=variables['is_ai_black'], command=app_callbacks['on_control_change'])
        self.ai_black_check.pack(side=tk.LEFT, padx=5)
        self.reset_button = ttk.Button(self, text="重置游戏", command=app_callbacks['new_game'])
        self.reset_button.pack(side=tk.RIGHT, padx=5)
        self.new_game_button = ttk.Button(self, text="开始新游戏", command=app_callbacks['new_game'])
        self.new_game_button.pack(side=tk.RIGHT, padx=0)
    def set_state(self, app_state, analysis_mode, ai_player_mode):
        is_busy = app_state in [AppState.AI_THINKING, AppState.ANIMATING, AppState.ANALYZING]
        self.new_game_button.config(state=tk.NORMAL if app_state != AppState.ANIMATING else tk.DISABLED)
        self.reset_button.config(state=tk.NORMAL if app_state != AppState.ANIMATING else tk.DISABLED)
        self.board_size_combo.config(state=tk.NORMAL if app_state in [AppState.IDLE, AppState.GAME_OVER] else tk.DISABLED)
        is_game_over = app_state == AppState.GAME_OVER
        if is_game_over:
            for check in [self.analysis_check, self.ai_red_check, self.ai_black_check]: check.config(state=tk.DISABLED)
            return
        if analysis_mode:
            self.analysis_check.config(state=tk.NORMAL); self.ai_red_check.config(state=tk.DISABLED); self.ai_black_check.config(state=tk.DISABLED)
        elif ai_player_mode:
            self.analysis_check.config(state=tk.DISABLED); self.ai_red_check.config(state=tk.NORMAL); self.ai_black_check.config(state=tk.NORMAL)
        else: # Human vs Human
            self.analysis_check.config(state=tk.NORMAL); self.ai_red_check.config(state=tk.NORMAL); self.ai_black_check.config(state=tk.NORMAL)

class StatusBar(ttk.Frame):
    def __init__(self, master, variables):
        super().__init__(master, style="Status.TFrame")
        self.status_label = ttk.Label(self, textvariable=variables['status_text'], font=Style.STATUS_FONT, anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=3, fill=tk.X, expand=True)
        self.sim_count_label = ttk.Label(self, textvariable=variables['sim_count_text'], font=Style.STATUS_FONT, anchor='e')
        self.sim_count_label.pack(side=tk.RIGHT, padx=10, pady=3)

class AIWorker(threading.Thread):
    """[REFACTORED] AI工作线程，现在包含了全部“AI回合”的逻辑。"""
    def __init__(self, app_instance, game, sims):
        super().__init__(daemon=True)
        self.app = app_instance
        self.game = game
        self.simulations = sims
    def run(self):
        # 1. 创建MCTS实例
        mcts = MCTS(AI_MODEL, DEVICE, self.simulations)
        # 2. 运行启发式搜索
        heuristic_result = find_immediate_win_loss_search(self.game)
        final_result = None

        if heuristic_result:
            move_type, move_col = heuristic_result
            if move_type == 'WIN':
                q_values = self.app._get_nn_q_values(self.game)
                q_values[move_col] = 1.0
                final_result = {'move': move_col, 'q_values': q_values, 'status': '必胜!'}
            elif move_type == 'BLOCK':
                mcts.get_move(self.game, self.app.analysis_queue, self.app.is_ai_thinking_flag)
                if not self.app.is_ai_thinking_flag[0]: return
                final_q_values = mcts.get_final_q_values(self.game)
                display_q_values = np.full(self.game.cols, -2.0)
                for move in self.game.get_valid_moves(): display_q_values[move] = -1.0
                display_q_values[move_col] = final_q_values[move_col]
                final_result = {'move': move_col, 'q_values': display_q_values, 'status': '必须防守!'}
        
        if not final_result:
            best_move = mcts.get_move(self.game, self.app.analysis_queue, self.app.is_ai_thinking_flag)
            if not self.app.is_ai_thinking_flag[0]: return
            final_q_values = mcts.get_final_q_values(self.game)
            final_result = {'move': best_move, 'q_values': final_q_values, 'sims': self.simulations}

        if self.app.is_ai_thinking_flag[0]:
            self.app.analysis_queue.put(final_result)

class ConnectFourApp:
    def __init__(self, master):
        self.master = master
        master.title("通用重力四子棋 (专业分析版)"); master.geometry("700x700"); master.minsize(450, 500)
        self._configure_styles(); self.master.configure(bg=Style.BACKGROUND)
        self.master.rowconfigure(1, weight=1); self.master.columnconfigure(0, weight=1)
        self.state = AppState.IDLE; self.game = None; self.hovered_col = None; self.ai_worker = None
        self.is_ai_thinking_flag = [False]; self.analysis_queue = queue.Queue(); self.live_analysis_job = None
        self.vars = {
            'board_size': tk.StringVar(value=f"{DEFAULT_ROWS}x{DEFAULT_COLS}"),
            'analysis_mode': tk.BooleanVar(value=False), 'is_ai_red': tk.BooleanVar(value=False),
            'is_ai_black': tk.BooleanVar(value=False), 'status_text': tk.StringVar(), 'sim_count_text': tk.StringVar()
        }
        app_callbacks = {'new_game': self.new_game, 'on_control_change': self._on_control_change}
        self.controls = ControlsFrame(master, app_callbacks, self.vars); self.controls.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        self.board_canvas = BoardCanvas(master, Style); self.board_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.status_bar = StatusBar(master, self.vars); self.status_bar.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        self.board_canvas.bind("<Configure>", self._on_resize); self.board_canvas.bind("<Motion>", self._on_mouse_move)
        self.board_canvas.bind("<Leave>", self._on_mouse_leave); self.board_canvas.bind("<Button-1>", self._on_click)
        self.master.after(100, self.initialize_ui)
    
    def initialize_ui(self):
        self.game = None; self._set_state(AppState.IDLE)
        self.board_canvas.refresh(None)
        self.vars['status_text'].set("请选择设置并点击“开始/重置游戏”")
        self.vars['sim_count_text'].set("")

    def _configure_styles(self):
        style = ttk.Style(); style.configure("Controls.TFrame", background=Style.CONTROLS_BACKGROUND)
        style.configure("Status.TFrame", background="#E0E0E0")

    def _set_state(self, new_state):
        self.state = new_state; print(f"State changed to: {self.state}")
        analysis_mode = self.vars['analysis_mode'].get()
        ai_player_mode = self.vars['is_ai_red'].get() or self.vars['is_ai_black'].get()
        self.controls.set_state(new_state, analysis_mode, ai_player_mode)
        self.master.config(cursor="watch" if new_state == AppState.AI_THINKING else "")

    def new_game(self):
        self._stop_ai_process()
        try: rows, cols = map(int, self.vars['board_size'].get().split('x'))
        except (ValueError, TypeError): self.vars['board_size'].set(f"{DEFAULT_ROWS}x{DEFAULT_COLS}"); rows, cols = DEFAULT_ROWS, DEFAULT_COLS
        self.game = ConnectFourGame(rows=rows, cols=cols)
        self.board_canvas.refresh(self.game)
        self._on_control_change()

    def _on_control_change(self):
        self._stop_ai_process()
        # 立即更新控件状态以实现互斥
        self._set_state(self.state)
        self.master.after(50, self._check_turn)

    def _check_turn(self):
        if not self.game: self._set_state(AppState.IDLE); return
        if self.game.game_over: self._end_game(); return
        self._set_state(self.state) # 再次确保控件状态正确
        is_analysis_mode = self.vars['analysis_mode'].get()
        is_ai_turn_now = (self.game.current_player == 1 and self.vars['is_ai_red'].get()) or (self.game.current_player == 2 and self.vars['is_ai_black'].get())
        if is_analysis_mode:
            self.vars['status_text'].set(f"分析模式: 轮到 {'红方' if self.game.current_player == 1 else '黑方'}")
            self._start_live_analysis()
        elif is_ai_turn_now:
            self._start_ai_move_computation()
        else:
            player_name = "红方" if self.game.current_player == 1 else "黑方"
            self.vars['status_text'].set(f"轮到你了 ({player_name})")
            self.vars['sim_count_text'].set(""); self._set_state(AppState.HUMAN_TURN)
    
    def _start_ai_move_computation(self):
        self._set_state(AppState.AI_THINKING)
        self.vars['status_text'].set(f"电脑 ({'红方' if self.game.current_player == 1 else '黑方'}) 正在思考...")
        self.vars['sim_count_text'].set(f"Sims: 0 / {MCTS_SIMULATIONS_PLAY}")
        self._stop_ai_process()
        self.is_ai_thinking_flag[0] = True
        self.ai_worker = AIWorker(self, self.game, MCTS_SIMULATIONS_PLAY)
        self.ai_worker.start()
        self.master.after(100, self._process_ai_queue)

    def _start_live_analysis(self):
        self._set_state(AppState.ANALYZING)
        self.vars['status_text'].set("分析模式: 实时计算中...")
        self._stop_ai_process()
        self.mcts_instance = MCTS(AI_MODEL, DEVICE, -1)
        self.mcts_instance.set_game_state(self.game)
        self.total_live_sims = 0
        self._live_analysis_loop()

    def _live_analysis_loop(self):
        if self.state != AppState.ANALYZING: return
        self.mcts_instance.ponder(num_sims=MCTS_SIMULATIONS_LIVE_ANALYSIS)
        q_values = self.mcts_instance.get_q_values(self.game.cols)
        self.board_canvas.refresh(self.game, self.hovered_col, q_values)
        self.total_live_sims += MCTS_SIMULATIONS_LIVE_ANALYSIS
        self.vars['sim_count_text'].set(f"累计Sims: {self.total_live_sims:,}")
        self.live_analysis_job = self.master.after(100, self._live_analysis_loop)

    def _process_ai_queue(self):
        if self.state != AppState.AI_THINKING: return
        try:
            while True:
                update = self.analysis_queue.get_nowait()
                if 'q_values' in update: self.board_canvas.refresh(self.game, self.hovered_col, update['q_values'])
                if 'status' in update: self.vars['sim_count_text'].set(f"分析: {update['status']}")
                elif 'sims' in update: self.vars['sim_count_text'].set(f"Sims: {update['sims']} / {MCTS_SIMULATIONS_PLAY}")
                if 'move' in update: self._stop_ai_process(); self._execute_move(update['move']); return
        except queue.Empty: pass
        self.master.after(100, self._process_ai_queue)
    
    def _get_nn_q_values(self, game):
        q_values = np.full(game.cols, -2.0)
        valid_moves = game.get_valid_moves()
        for move in valid_moves:
            temp_game = pickle.loads(pickle.dumps(game))
            temp_game.make_move(move)
            state_tensor = torch.FloatTensor(temp_game.get_board_state()).unsqueeze(0).to(DEVICE)
            board = temp_game.board; rows, cols = board.shape
            pieces_in_col = np.sum(board != 0, axis=0); open_rows = (rows - pieces_in_col - 1).astype(np.int64)
            target_rows_tensor = torch.from_numpy(open_rows).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, value = AI_MODEL(state_tensor, target_rows_tensor)
            q_values[move] = -value.item()
        return q_values

    def _execute_move(self, col):
        if self.game is None or col is None or col not in self.game.get_valid_moves(): self._check_turn(); return
        self._stop_ai_process()
        self._set_state(AppState.ANIMATING)
        player = self.game.current_player; target_row = self.game.get_next_open_row(col); self.game.make_move(col)
        self.board_canvas.animate_piece_drop(col, target_row, player, self._on_animation_complete)
        
    def _on_animation_complete(self):
        self.board_canvas.refresh(self.game); self._check_turn()

    def _end_game(self):
        if self.state == AppState.GAME_OVER: return
        self._set_state(AppState.GAME_OVER); self._stop_ai_process()
        winner = self.game.winner
        if winner == -1: message = "平局！棋盘已满。"
        else:
            player_name = "红方" if winner == 1 else "黑方"
            is_ai_player = (winner == 1 and self.vars['is_ai_red'].get()) or (winner == 2 and self.vars['is_ai_black'].get())
            message = f"电脑 ({player_name}) 获胜！" if is_ai_player and not self.vars['analysis_mode'].get() else f"恭喜！{player_name} 赢了！"
        self.vars['status_text'].set("游戏结束"); self.vars['sim_count_text'].set("")
        messagebox.showinfo("游戏结束", message)
        self.initialize_ui()

    def _stop_ai_process(self):
        if self.live_analysis_job: self.master.after_cancel(self.live_analysis_job); self.live_analysis_job = None
        if self.ai_worker and self.ai_worker.is_alive():
            self.is_ai_thinking_flag[0] = False; self.ai_worker.join(timeout=0.2)
        self.is_ai_thinking_flag[0] = False

    def _on_resize(self, event):
        self.board_canvas.refresh(self.game, self.hovered_col, self.board_canvas.q_values)
        
    def _on_mouse_move(self, event):
        if self.state not in [AppState.HUMAN_TURN, AppState.ANALYZING]: self.board_canvas.draw_hover(None, None); return
        col = self.board_canvas.get_col_from_event(event)
        if col != self.hovered_col:
            self.hovered_col = col
            if self.game and self.state == AppState.HUMAN_TURN and col is not None and col in self.game.get_valid_moves():
                self.board_canvas.draw_hover(col, self.game.current_player)
            else: self.board_canvas.draw_hover(None, None)

    def _on_mouse_leave(self, event):
        self.hovered_col = None; self.board_canvas.draw_hover(None, None)

    def _on_click(self, event):
        if not self.game: return
        col = self.board_canvas.get_col_from_event(event)
        if col is None or col not in self.game.get_valid_moves(): return
        if self.state in [AppState.HUMAN_TURN, AppState.ANALYZING]:
            self._execute_move(col)

if __name__ == "__main__":
    root = tk.Tk()
    app = ConnectFourApp(root)
    root.mainloop()