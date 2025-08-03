# train.py
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader

from neural_network import UniversalConnectFourNet as ConnectFourNet
from config import MODEL_SAVE_PATH, TRAINING_DATA_PATH, EPOCHS, LEARNING_RATE, BATCH_SIZE

# --- 自定义 Dataset 类 (不变) ---
class ConnectFourDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- [核心修改] 自定义 Collate Function ---
def custom_collate_fn(batch):
    states = [item['state'] for item in batch]
    probs = [item['probs'] for item in batch]
    values = [item['value'] for item in batch]

    # --- 填充 State (逻辑不变) ---
    max_rows = max(s.shape[1] for s in states)
    max_cols = max(s.shape[2] for s in states)
    
    padded_states = []
    for s in states:
        _, rows, cols = s.shape
        padded_s = np.zeros((3, max_rows, max_cols), dtype=np.float32)
        padded_s[:, :rows, :cols] = s
        padded_states.append(padded_s)

    # --- 填充 Policy (逻辑不变) ---
    max_policy_len = max(len(p) for p in probs)
    padded_probs = []
    for p in probs:
        padded_p = np.zeros(max_policy_len, dtype=np.float32)
        padded_p[:len(p)] = p
        padded_probs.append(padded_p)

    # --- [新增] 计算和填充 target_rows ---
    target_rows_list = []
    for s in states:
        # s 的形状是 (3, rows, cols)
        # 通过 state[0] (玩家1棋子) 和 state[1] (玩家2棋子) 重建棋盘占用情况
        board_occupied = (s[0] + s[1])
        rows, cols = board_occupied.shape
        # 对每一列求和，得到该列已有的棋子数
        pieces_in_col = np.sum(board_occupied, axis=0)
        # 下一个可用行的索引 = 总行数 - 棋子数 - 1
        open_rows = (rows - pieces_in_col - 1).astype(np.int64)
        target_rows_list.append(open_rows)

    padded_target_rows = []
    for r in target_rows_list:
        # 使用-1作为填充值，因为行索引不可能是负数
        padded_r = np.full(max_policy_len, -1, dtype=np.int64)
        padded_r[:len(r)] = r
        padded_target_rows.append(padded_r)
        
    return {
        'state': torch.tensor(np.array(padded_states)),
        'probs': torch.tensor(np.array(padded_probs)),
        'value': torch.tensor(values, dtype=torch.float32),
        'target_rows': torch.tensor(np.array(padded_target_rows)) # 新增返回项
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
        training_data = list(pickle.load(f))

    dataset = ConnectFourDataset(training_data)
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=custom_collate_fn
    )

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            states = batch['state'].to(device)
            target_policies = batch['probs'].to(device)
            target_values = batch['value'].to(device)
            # [核心修改] 从批次中获取 target_rows
            target_rows = batch['target_rows'].to(device)

            # [核心修改] 将 target_rows 传入模型
            pred_policies, pred_values = model(states, target_rows)
            
            # --- 损失计算 (逻辑不变) ---
            target_len = target_policies.size(1)
            pred_policies = pred_policies[:, :target_len]
            
            # 忽略在填充策略上的损失
            valid_policy_mask = target_policies > 0
            policy_loss = -torch.sum(target_policies[valid_policy_mask] * pred_policies[valid_policy_mask])
            if valid_policy_mask.sum() > 0:
                policy_loss /= valid_policy_mask.sum()
            
            value_loss = F.mse_loss(pred_values.squeeze(-1), target_values)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix({"Loss": loss.item(), "Avg Loss": f"{avg_loss:.4f}"})

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Universal model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()