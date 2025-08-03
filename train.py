# train.py
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import numpy as np

# CHANGED: 导入 Dataset 和 DataLoader
from torch.utils.data import Dataset, DataLoader

from neural_network import UniversalConnectFourNet as ConnectFourNet
# CHANGED: 导入 BATCH_SIZE
from config import MODEL_SAVE_PATH, TRAINING_DATA_PATH, EPOCHS, LEARNING_RATE, BATCH_SIZE

# --- NEW: 自定义 Dataset 类 ---
# 这是一个好的实践，将数据逻辑与训练逻辑解耦
class ConnectFourDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- NEW: 自定义 Collate Function ---
# 这是实现批处理的核心！它负责将不同尺寸的样本填充并打包。
def custom_collate_fn(batch):
    # batch 是一个列表，列表中的每个元素都是一个 data_point 字典
    
    states = [item['state'] for item in batch]
    probs = [item['probs'] for item in batch]
    values = [item['value'] for item in batch]

    # --- 填充 State ---
    max_rows = max(s.shape[1] for s in states)
    max_cols = max(s.shape[2] for s in states)
    
    padded_states = []
    for s in states:
        _, rows, cols = s.shape
        # 创建一个填满0的大棋盘
        padded_s = np.zeros((3, max_rows, max_cols), dtype=np.float32)
        # 将小棋盘的数据复制进去
        padded_s[:, :rows, :cols] = s
        padded_states.append(padded_s)

    # --- 填充 Policy (Probs) ---
    max_policy_len = max(len(p) for p in probs)
    padded_probs = []
    for p in probs:
        padded_p = np.zeros(max_policy_len, dtype=np.float32)
        padded_p[:len(p)] = p
        padded_probs.append(padded_p)

    return {
        'state': torch.tensor(np.array(padded_states)),
        'probs': torch.tensor(np.array(padded_probs)),
        'value': torch.tensor(values, dtype=torch.float32)
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConnectFourNet().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
        print("Loaded universal model for further training.")
    else:
        print("Initializing a new universal model.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if not os.path.exists(TRAINING_DATA_PATH):
        print("No training data found. Run self_play.py first.")
        return
        
    with open(TRAINING_DATA_PATH, 'rb') as f:
        # pickle加载的数据是deque，我们将其转为list
        training_data = list(pickle.load(f))

    # --- CHANGED: 使用 Dataset 和 DataLoader ---
    dataset = ConnectFourDataset(training_data)
    # DataLoader会使用我们的自定义函数来打包数据
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=custom_collate_fn
    )

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        # --- CHANGED: 循环现在遍历 DataLoader ---
        # pbar 现在包裹 data_loader
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            # 从字典中获取已经打包好的批次数据，并送到GPU
            states = batch['state'].to(device)
            target_policies = batch['probs'].to(device)
            target_values = batch['value'].to(device)

            pred_policies, pred_values = model(states)
            
            # --- 损失计算 ---
            # 策略损失：只计算有效部分，忽略填充部分
            # 我们需要确保预测和目标的长度一致
            target_len = target_policies.size(1)
            pred_policies = pred_policies[:, :target_len]
            
            policy_loss = -torch.sum(target_policies * pred_policies) / states.size(0) # 除以批大小
            value_loss = F.mse_loss(pred_values.squeeze(), target_values)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 计算平均损失
            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix({"Loss": loss.item(), "Avg Loss": f"{avg_loss:.4f}"})

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Universal model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()