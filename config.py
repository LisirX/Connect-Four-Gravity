# config.py

# --- 默认游戏配置 ---
DEFAULT_ROWS = 6
DEFAULT_COLS = 7

# --- MCTS 配置 ---
MCTS_SIMULATIONS = 50      # 对于更大的棋盘，你可能需要增加这个值
MCTS_CPUCT = 1.0

# --- 自对弈 (Self-Play) 配置 ---
SELF_PLAY_GAMES = 200
DATA_MAX_SIZE = 50000
# 模型路径现在不包含尺寸，因为是通用模型
MODEL_SAVE_PATH = "models/universal_model.pth" 
TRAINING_DATA_PATH = "data/training_data.pkl"

# --- 训练 (Training) 配置 ---
EPOCHS = 10
BATCH_SIZE = 32 # 使用更小的批大小，因为不同尺寸的输入会占用不同大小的内存
LEARNING_RATE = 0.001