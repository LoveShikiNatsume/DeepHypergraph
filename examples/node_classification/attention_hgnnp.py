import torch
import torch.nn as nn
import torch.nn.functional as F
import dhg
from dhg.models import HGNNP


class AttentionHGNNP(HGNNP):
    def __init__(
            self,
            in_channels: int,
            hid_channels: int,
            num_classes: int,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            num_heads: int = 4,
    ) -> None:
        super().__init__(in_channels, hid_channels, num_classes, use_bn, drop_rate)

        # 添加多头注意力层
        self.attention = nn.MultiheadAttention(hid_channels, num_heads, dropout=drop_rate)

        # 添加层归一化
        self.norm1 = nn.LayerNorm(hid_channels)
        self.norm2 = nn.LayerNorm(hid_channels)

        # 添加前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hid_channels, hid_channels * 4),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hid_channels * 4, hid_channels)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        # 首先通过原始HGNNP层
        out = super().forward(X, hg)

        # 准备注意力输入
        # [N, C] -> [N, 1, C]
        out = out.unsqueeze(1)

        # 应用自注意力机制
        attended_out, attention_weights = self.attention(
            out, out, out
        )

        # 残差连接和归一化
        out = out + attended_out
        out = self.norm1(out)

        # 前馈网络
        ff_out = self.feed_forward(out)
        out = out + ff_out
        out = self.norm2(out)

        # [N, 1, C] -> [N, C]
        out = out.squeeze(1)

        return out

    def get_attention_weights(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        """返回注意力权重以供分析"""
        self.eval()
        with torch.no_grad():
            out = super().forward(X, hg)
            out = out.unsqueeze(1)
            _, attention_weights = self.attention(out, out, out)
            return attention_weights