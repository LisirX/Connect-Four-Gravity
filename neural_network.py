# neural_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalConnectFourNet(nn.Module):
    def __init__(self):
        super(UniversalConnectFourNet, self).__init__()
        # 卷积主干部分保持不变，它可以处理任意大小的输入
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # --- 价值头 (Value Head) ---
        # 使用全局平均池化，将 (128, N, M) 的特征图变为 (128, 1, 1)
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # --- 策略头 (Policy Head) ---
        # 输出一个策略“图”，每个位置对应一个动作的倾向
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        # 稍后我们将从这个特征图池化到一维的列策略
        self.policy_fc = nn.Linear(32, 1) # 输出每个位置的单一logits

    def forward(self, x, target_rows=None):
        # x 的尺寸是 (batch, 3, rows, cols)
        
        # 1. 通过卷积主干
        x = self.conv_block(x) # 输出尺寸 (batch, 128, rows, cols)

        # 2. 价值头 (逻辑不变)
        v = self.value_conv(x) # 输出 (batch, 32, rows, cols)
        v = F.adaptive_avg_pool2d(v, (1, 1)) # 输出 (batch, 32, 1, 1)
        v = v.view(v.size(0), -1) # 扁平化为 (batch, 32)
        value = torch.tanh(self.value_fc(v)) # 输出 (batch, 1)

        # 3. 策略头
        p = self.policy_conv(x) # 输出 (batch, 32, rows, cols)
        
        # --- [核心修改] 精确提取特征，而非平均化 ---
        batch_size, num_cols = x.size(0), x.size(3)
        
        # 如果提供了 target_rows (来自训练或MCTS)，则使用精确提取
        if target_rows is not None:
            # target_rows 的形状是 (batch, cols)
            # 我们需要用它来从 p 中索引
            
            # 创建一个批次索引 [0, 0, ..., 1, 1, ...]
            batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, num_cols)
            # 创建一个列索引 [0, 1, 2, ..., 0, 1, 2, ...]
            col_idx = torch.arange(num_cols, device=x.device).unsqueeze(0).expand(batch_size, -1)
            
            # 使用高级索引直接提取特征
            # p 的维度是 (batch, channels, rows, cols)
            # 我们想要提取的索引是 (batch_idx, :, target_rows, col_idx)
            # 需要先调整 p 的维度以方便 gather
            p = p.permute(0, 2, 3, 1) # -> (batch, rows, cols, channels=32)
            
            # 注意：只提取有效列的特征
            valid_cols_mask = target_rows != -1 # collate_fn中用-1填充
            
            # 初始化一个零张量来存储结果
            policy_features = torch.zeros(batch_size, num_cols, 32, device=x.device)
            
            # 只对有效的行列组合进行索引
            if valid_cols_mask.any():
                policy_features[valid_cols_mask] = p[batch_idx[valid_cols_mask], target_rows[valid_cols_mask], col_idx[valid_cols_mask]]
            
            p = policy_features # p 的新形状 -> (batch, cols, channels=32)
            
        else: # 如果没有提供 target_rows，则回退到原来的方法，用于兼容
            p = p.mean(dim=2) # 平均掉row维度 -> (batch, 32, cols)
            p = p.transpose(1, 2) # 交换维度 -> (batch, cols, 32)
        
        # --- 后续处理 (逻辑不变) ---
        p = self.policy_fc(p).squeeze(-1) # -> (batch, cols, 1) -> (batch, cols)
        policy = F.log_softmax(p, dim=1) # 在列维度上应用softmax

        return policy, value