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

    def forward(self, x):
        # x 的尺寸是 (batch, 3, rows, cols)
        num_cols = x.size(3)
        
        # 1. 通过卷积主干
        x = self.conv_block(x) # 输出尺寸 (batch, 128, rows, cols)

        # 2. 价值头
        v = self.value_conv(x) # 输出 (batch, 32, rows, cols)
        # 全局平均池化
        v = F.adaptive_avg_pool2d(v, (1, 1)) # 输出 (batch, 32, 1, 1)
        v = v.view(v.size(0), -1) # 扁平化为 (batch, 32)
        value = torch.tanh(self.value_fc(v)) # 输出 (batch, 1)

        # 3. 策略头
        p = self.policy_conv(x) # 输出 (batch, 32, rows, cols)
        
        # 将2D特征图压缩为1D列策略
        # 我们对每一列的所有行取平均值
        p = p.mean(dim=2) # 平均掉row维度 -> (batch, 32, cols)
        p = p.transpose(1, 2) # 交换维度 -> (batch, cols, 32)

        p = self.policy_fc(p).squeeze(-1) # -> (batch, cols, 1) -> (batch, cols)
        policy = F.log_softmax(p, dim=1) # 在列维度上应用softmax

        return policy, value