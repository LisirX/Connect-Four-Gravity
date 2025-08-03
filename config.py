# config.py

# --- 默认游戏配置 ---
DEFAULT_ROWS = 6
DEFAULT_COLS = 7

# --- MCTS 配置 ---
# NEW: 区分训练和对战的思考深度
# 用于自我对弈（self_play.py），追求速度和数据量
MCTS_SIMULATIONS_TRAIN = 100       
# 用于与人类对战或分析（main_gui.py），追求棋力强度
MCTS_SIMULATIONS_PLAY = 4000       
MCTS_CPUCT = 1.0

# --- 自对弈 (Self-Play) 配置 ---
SELF_PLAY_GAMES = 200
DATA_MAX_SIZE = 50000
MODEL_SAVE_PATH = "models/universal_model.pth" 
TRAINING_DATA_PATH = "data/training_data.pkl"

# --- 训练 (Training) 配置 ---
EPOCHS = 10
BATCH_SIZE = 64 
LEARNING_RATE = 0.001