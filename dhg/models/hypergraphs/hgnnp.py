import torch
import torch.nn as nn
import torch.nn.functional as F
import dhg
from dhg.nn import HGNNPConv

class HGNNP(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bn=True, drop_rate=0.5):
        super(HGNNP, self).__init__()
        self.use_bn = use_bn
        self.drop_rate = drop_rate

        # 超图卷积层
        self.layers = nn.ModuleList([
            HGNNPConv(input_dim, hidden_dim, use_bn=use_bn, drop_rate=drop_rate),
            HGNNPConv(hidden_dim, hidden_dim, use_bn=use_bn, drop_rate=drop_rate),
            HGNNPConv(hidden_dim, hidden_dim, use_bn=use_bn, drop_rate=drop_rate)
        ])

        # 注意力层，用于给异常实例更高的权重
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, 1)

        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.drop = nn.Dropout(drop_rate)

    def compute_attention_weights(self, X):
        """计算每个节点的注意力权重"""
        attention_scores = self.attention(X)  # (num_nodes, 1)
        attention_weights = torch.sigmoid(attention_scores)  # 将分数归一化到0-1
        return attention_weights

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph"):
        # 超图卷积
        for layer in self.layers:
            X = layer(X, hg)

        # 计算注意力权重
        attention_weights = self.compute_attention_weights(X)  # (num_nodes, 1)

        # 应用注意力权重到特征
        X_weighted = X * attention_weights  # 元素wise乘法

        # 特征提取
        X = F.relu(X_weighted)
        X = self.bn(X)
        X = self.drop(X)

        # 全连接层
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))

        # 输出层：预测根因概率
        root_cause_score = torch.sigmoid(self.fc_out(X))

        return root_cause_score, attention_weights