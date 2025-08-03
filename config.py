# config.py

# --- 默认游戏配置 ---
DEFAULT_ROWS = 6
DEFAULT_COLS = 7

# --- MCTS 配置 ---
# NEW: 区分训练和对战的思考深度
# 用于自我对弈（self_play.py），追求速度和数据量
MCTS_SIMULATIONS_TRAIN = 100       
# 用于与人类对战或分析（main_gui.py），追求棋力强度
MCTS_SIMULATIONS_PLAY = 8000
# [NEW] 专门用于分析模式，追求极致的计算深度和准确性
MCTS_SIMULATIONS_ANALYSIS = 800000
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

# --- 竞技场 (Arena) 配置 ---
# 用于模型评估，让新旧模型对战
ARENA_GAMES = 20 # 总共对战的局数 (建议为偶数)
# 评估时使用的MCTS模拟次数，越高棋力越强，速度越慢
ARENA_MCTS_SIMULATIONS = 1000
# 对战结果保存路径
RESULTS_FILE_PATH = "results/arena_results.json" 

# 用于实时分析模式下，每次UI更新循环的MCTS模拟次数
# 这个值不需要太高，因为它会持续不断地运行
MCTS_SIMULATIONS_LIVE_ANALYSIS = 30