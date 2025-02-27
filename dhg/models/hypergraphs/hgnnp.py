from dhg.nn import HGNNPConv
import torch
import torch.nn as nn
from typing import Tuple
from dhg.structure.hypergraphs import Hypergraph


class HGNNP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int = 1,
            use_bn: bool = True,
            drop_rate: float = 0.3,
            v2e_aggr: str = "sum",
            e2v_aggr: str = "sum",
            v2e_drop_rate: float = 0.1,
            attention_heads: int = 4,
            use_edge_features: bool = True,
            normalize: str = "both",
            activation: str = "gelu"
    ):
        super(HGNNP, self).__init__()

        # 初始节点特征转换层
        self.node_feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.2),
            nn.Dropout(drop_rate)
        )

        # 第一层卷积：节点特征 → 超边特征 (v2e)
        self.v2e_conv1 = HGNNPConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            bias=True,
            use_bn=use_bn,
            drop_rate=drop_rate,
            is_last=False,
            v2e_aggr=v2e_aggr,
            e2v_aggr=e2v_aggr,
            v2e_drop_rate=v2e_drop_rate,
            attention_heads=attention_heads,
            use_edge_features=False,  # 第一层没有边特征输入
            residual=True,
            normalize=normalize,
            activation=activation
        )

        # 第二层卷积：超边特征 → 节点特征 (e2v)
        self.e2v_conv = HGNNPConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            bias=True,
            use_bn=use_bn,
            drop_rate=drop_rate,
            is_last=False,
            v2e_aggr=v2e_aggr,
            e2v_aggr=e2v_aggr,
            v2e_drop_rate=v2e_drop_rate,
            attention_heads=attention_heads,
            use_edge_features=use_edge_features,
            residual=True,
            normalize=normalize,
            activation=activation
        )

        # 第三层卷积：节点特征 → 超边特征 (v2e)
        self.v2e_conv2 = HGNNPConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            bias=True,
            use_bn=use_bn,
            drop_rate=drop_rate,
            is_last=True,  # 作为最后一层
            v2e_aggr=v2e_aggr,
            e2v_aggr=e2v_aggr,
            v2e_drop_rate=v2e_drop_rate,
            attention_heads=attention_heads,
            use_edge_features=use_edge_features,
            residual=True,
            normalize=normalize,
            activation=activation
        )

        # 超边特征最终处理
        self.edge_feature_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.2),
            nn.Dropout(drop_rate)
        )

        # 多头注意力
        self.att_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // attention_heads),
                nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // attention_heads, 1)
            ) for _ in range(attention_heads)
        ])

        # 特征提取层
        self.feature_extraction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.2),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.2)
        )

        # 输出层
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # 归一化层
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(drop_rate)

    def compute_multi_head_attention(self, X):
        """计算多头注意力"""
        head_outputs = [head(X) for head in self.att_heads]
        combined = torch.cat(head_outputs, dim=1)
        return torch.sigmoid(torch.mean(combined, dim=1, keepdim=True))

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数

        Args:
            X: 输入节点特征矩阵，形状为(|V|, input_dim)
            hg: 超图结构

        Returns:
            root_cause_score: 模型输出，形状为(|E|, output_dim)
            final_attention: 最终的注意力权重
        """
        # 初始特征转换
        X = self.node_feature_transform(X)
        X = self.layer_norm(X)

        # 第一层卷积: 点 → 边 (v2e)
        v_features1, e_features1, attn1 = self.v2e_conv1(X, hg)

        # 第二层卷积: 边 → 点 (e2v)
        v_features2, e_features2, attn2 = self.e2v_conv(v_features1, hg, edge_features=e_features1)

        # 第三层卷积: 点 → 边 (v2e)
        v_features3, e_features3, conv_attention = self.v2e_conv2(v_features2, hg, edge_features=e_features2)

        # 最终超边特征处理
        refined_edge_features = self.edge_feature_refine(e_features3)

        # 计算注意力权重
        attention_weights = self.compute_multi_head_attention(refined_edge_features)

        # 结合注意力来源
        final_attention = (attention_weights + conv_attention) / 2.0

        # 应用注意力权重
        X_weighted = torch.mul(refined_edge_features, final_attention)

        # 特征提取 - 使用跳跃连接
        X_extracted = self.feature_extraction(X_weighted)
        X = X_extracted + X_weighted  # 残差连接

        # 应用归一化和dropout
        X = self.bn(X )
        X = self.drop( X )

        # 输出层：预测根因概率
        root_cause_score = torch.sigmoid(self.fc_out( X ))

        return root_cause_score, final_attention

# class HGNNP(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, use_bn: bool = True, drop_rate: float = 0.3,
#                  v2e_aggr: str = "softmax_then_sum", v2e_drop_rate: float = 0.1,
#                  use_multi_head: bool = True, num_heads: int = 4):
#         super(HGNNP, self).__init__()
#         self.use_bn = use_bn
#         self.drop_rate = drop_rate
#         self.use_multi_head = use_multi_head
#         self.num_heads = num_heads
#
#         # 第一层：节点特征处理
#         self.node_feature_transform = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(drop_rate)
#         )
#
#         # 超图卷积层：转换后的节点特征 -> 超边特征
#         self.v2e_conv = HGNNPConv(
#             in_channels=hidden_dim,  # 已经转换过的特征
#             out_channels=hidden_dim,
#             use_bn=use_bn,
#             drop_rate=drop_rate,
#             v2e_aggr=v2e_aggr,
#             v2e_drop_rate=v2e_drop_rate,
#             use_attention=True,
#             residual=True,
#             activation="leaky_relu",
#             is_last=False
#         )
#
#         # 多头注意力
#         if use_multi_head:
#             head_dim = hidden_dim // num_heads
#             self.att_heads = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Linear(hidden_dim, head_dim),
#                     nn.LeakyReLU(0.2),
#                     nn.Linear(head_dim, 1)
#                 ) for _ in range(num_heads)
#             ])
#         else:
#             self.attention = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim // 2),
#                 nn.LeakyReLU(0.2),
#                 nn.Linear(hidden_dim // 2, 1)
#             )
#
#         # 特征提取层
#         self.feature_extraction = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(drop_rate),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(0.2)
#         )
#
#         # 超边特征进一步处理
#         self.edge_feature_refine = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(drop_rate)
#         )
#
#         # 输出层
#         self.fc_out = nn.Linear(hidden_dim, 1)
#
#         # 归一化层
#         self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#         self.drop = nn.Dropout(drop_rate)
#
#     def compute_multi_head_attention(self, X):
#         if self.use_multi_head:
#             head_outputs = [head(X) for head in self.att_heads]
#             combined = torch.cat(head_outputs, dim=1)
#             return torch.sigmoid(torch.mean(combined, dim=1, keepdim=True))
#         else:
#             return torch.sigmoid(self.attention(X))
#
#     def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> Tuple[torch.Tensor, torch.Tensor]:
#         # 第一层：节点特征处理
#         X_transformed = self.node_feature_transform(X)
#
#         # 应用层归一化
#         X_transformed = self.layer_norm(X_transformed)
#
#         # 超图卷积：转换后的节点特征 -> 超边特征
#         edge_features, conv_attention = self.v2e_conv(X_transformed, hg)
#
#         # 应用层归一化
#         edge_features = self.layer_norm(edge_features)
#
#         # 第二层：超边特征进一步处理
#         refined_edge_features = self.edge_feature_refine(edge_features)
#         refined_edge_features = self.layer_norm(refined_edge_features)
#
#         # 计算注意力权重
#         attention_weights = self.compute_multi_head_attention(refined_edge_features)
#
#         # 结合注意力来源
#         final_attention = (attention_weights + conv_attention) / 2.0
#
#         # 应用注意力权重
#         X_weighted = torch.mul(refined_edge_features, final_attention)
#
#         # 特征提取 - 使用跳跃连接
#         X_extracted = self.feature_extraction(X_weighted)
#         X = X_extracted + X_weighted  # 残差连接
#
#         # 应用归一化和dropout
#         X = self.bn(X)
#         X = self.drop(X)
#
#         # 输出层：预测根因概率
#         root_cause_score = torch.sigmoid(self.fc_out(X))
#
#         return root_cause_score, final_attention


#毕业设计2.0
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import dhg
# from dhg.nn import HGNNPConv
#
# class HGNNP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, use_bn=True, drop_rate=0.5):
#         super(HGNNP, self).__init__()
#         self.use_bn = use_bn
#         self.drop_rate = drop_rate
#
#         # 超图卷积层
#         self.layers = nn.ModuleList([
#             HGNNPConv(input_dim, hidden_dim, use_bn=use_bn, drop_rate=drop_rate),
#             HGNNPConv(hidden_dim, hidden_dim, use_bn=use_bn, drop_rate=drop_rate)
#         ])
#
#         # 注意力层，用于给异常实例更高的权重
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1)
#         )
#
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#
#         self.fc_out = nn.Linear(hidden_dim, 1)
#
#         self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
#         self.drop = nn.Dropout(drop_rate)
#
#     def compute_attention_weights(self, X):
#         """计算每个节点的注意力权重"""
#         attention_scores = self.attention(X)  # (num_nodes, 1)
#         attention_weights = torch.sigmoid(attention_scores)  # 将分数归一化到0-1
#         return attention_weights
#
#     def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph"):
#         # 超图卷积
#         for layer in self.layers:
#             X = layer(X, hg)
#
#         # 计算注意力权重
#         attention_weights = self.compute_attention_weights(X)  # (num_nodes, 1)
#
#         # 应用注意力权重到特征
#         X_weighted = X * attention_weights  # 元素wise乘法
#
#         # 特征提取
#         X = F.relu(X_weighted)
#         X = self.bn(X)
#         X = self.drop(X)
#
#         # 全连接层
#         X = F.relu(self.fc1(X))
#         X = F.relu(self.fc2(X))
#
#         # 输出层：预测根因概率
#         root_cause_score = torch.sigmoid(self.fc_out(X))
#
#         return root_cause_score, attention_weights

# # 毕业设计1.0
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import dhg
# from dhg.nn import HGNNPConv
#
# class HGNNP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, use_bn=True, drop_rate=0.5):
#         super(HGNNP, self).__init__()
#         self.use_bn = use_bn
#         self.drop_rate = drop_rate
#
#         # 超图卷积层
#         self.layers = nn.ModuleList([
#             HGNNPConv(input_dim, hidden_dim, use_bn=use_bn, drop_rate=drop_rate),
#             HGNNPConv(hidden_dim, hidden_dim, use_bn=use_bn, drop_rate=drop_rate)
#         ])
#
#         # 注意力层，用于给异常实例更高的权重
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1)
#         )
#
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#
#         self.fc_out = nn.Linear(hidden_dim, 1)
#
#         self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
#         self.drop = nn.Dropout(drop_rate)
#
#     def compute_attention_weights(self, X):
#         """计算每个节点的注意力权重"""
#         attention_scores = self.attention(X)  # (num_nodes, 1)
#         attention_weights = torch.sigmoid(attention_scores)  # 将分数归一化到0-1
#         return attention_weights
#
#     def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph"):
#         # 超图卷积
#         for layer in self.layers:
#             X = layer(X, hg)
#
#         # 计算注意力权重
#         attention_weights = self.compute_attention_weights(X)  # (num_nodes, 1)
#
#         # 应用注意力权重到特征
#         X_weighted = X * attention_weights  # 元素wise乘法
#
#         # 特征提取
#         X = F.relu(X_weighted)
#         X = self.bn(X)
#         X = self.drop(X)
#
#         # 全连接层
#         X = F.relu(self.fc1(X))
#         X = F.relu(self.fc2(X))
#
#         # 输出层：预测根因概率
#         root_cause_score = torch.sigmoid(self.fc_out(X))
#
#         return root_cause_score, attention_weights