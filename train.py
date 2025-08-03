# train.py
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import random

from neural_network import UniversalConnectFourNet as ConnectFourNet
from config import MODEL_SAVE_PATH, TRAINING_DATA_PATH, EPOCHS, LEARNING_RATE

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConnectFourNet().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("Loaded universal model for further training.")
    else:
        print("Initializing a new universal model.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if not os.path.exists(TRAINING_DATA_PATH):
        print("No training data found. Run self_play.py first.")
        return
        
    with open(TRAINING_DATA_PATH, 'rb') as f:
        training_data = list(pickle.load(f))

    model.train()
    for epoch in range(EPOCHS):
        random.shuffle(training_data)
        total_loss = 0
        pbar = tqdm(training_data, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # CHANGED: 由于尺寸可变，我们一次处理一个样本 (Batch Size = 1)
        for data_point in pbar:
            state = torch.FloatTensor(data_point['state']).unsqueeze(0).to(device)
            target_policy = torch.FloatTensor(data_point['probs']).to(device)
            target_value = torch.FloatTensor([data_point['value']]).to(device)

            pred_policy, pred_value = model(state)
            
            policy_loss = -torch.sum(target_policy * pred_policy)
            value_loss = F.mse_loss(pred_value.squeeze(), target_value.squeeze())
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Universal model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()