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

class ConnectFourDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    states = [item['state'] for item in batch]
    probs = [item['probs'] for item in batch]
    values = [item['value'] for item in batch]

    # --- 填充 State (不变) ---
    max_rows = max(s.shape[1] for s in states)
    max_cols = max(s.shape[2] for s in states)
    
    padded_states = []
    for s in states:
        _, rows, cols = s.shape
        padded_s = np.zeros((3, max_rows, max_cols), dtype=np.float32)
        padded_s[:, :rows, :cols] = s
        padded_states.append(padded_s)

    # --- 填充 Policy (Probs) (不变) ---
    max_policy_len = max(len(p) for p in probs)
    padded_probs = []
    for p in probs:
        padded_p = np.zeros(max_policy_len, dtype=np.float32)
        padded_p[:len(p)] = p
        padded_probs.append(padded_p)

    # --- 计算和填充 target_rows (不变) ---
    target_rows_list = []
    for s in states:
        board_occupied = (s[0] + s[1])
        rows, cols = board_occupied.shape
        pieces_in_col = np.sum(board_occupied, axis=0)
        open_rows = (rows - pieces_in_col - 1).astype(np.int64)
        target_rows_list.append(open_rows)

    padded_target_rows = []
    for r in target_rows_list:
        # [FIX] Ensure padding length for target_rows matches the policy padding length
        padded_r = np.full(max_policy_len, -1, dtype=np.int64)
        padded_r[:len(r)] = r
        padded_target_rows.append(padded_r)
        
    return {
        'state': torch.tensor(np.array(padded_states)),
        'probs': torch.tensor(np.array(padded_probs)),
        'value': torch.tensor(values, dtype=torch.float32),
        'target_rows': torch.tensor(np.array(padded_target_rows))
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConnectFourNet().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
            print("Loaded universal model for further training.")
        except Exception as e:
            print(f"Could not load model weights: {e}. Initializing a new model.")
            
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
            target_rows = batch['target_rows'].to(device)

            pred_policies, pred_values = model(states, target_rows)
            
            target_len = target_policies.size(1)
            pred_policies = pred_policies[:, :target_len]
            
            # 使用掩码来确保只在有效（非填充）的策略上下计算损失
            valid_policy_mask = target_policies > 0
            policy_loss = -torch.sum(target_policies[valid_policy_mask] * pred_policies[valid_policy_mask])
            
            # 正则化，除以有效条目的数量，以获得更稳定的平均损失
            num_valid_entries = valid_policy_mask.sum()
            if num_valid_entries > 0:
                policy_loss /= num_valid_entries
            
            value_loss = F.mse_loss(pred_values.squeeze(-1), target_values)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg Loss": f"{avg_loss:.4f}"})

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Universal model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()