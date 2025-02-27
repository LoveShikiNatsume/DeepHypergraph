import torch
import torch.nn as nn
from typing import Tuple
from dhg.structure.hypergraphs import Hypergraph


class HGNNPConv(nn.Module):
    """进一步增强的HGNN+卷积层，添加了多头注意力和更高级的特征聚合机制"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = True,
            drop_rate: float = 0.2,  # 调整dropout率
            is_last: bool = False,
            v2e_aggr: str = "sum",  # 使用加权求和作为默认聚合方法
            e2v_aggr: str = "sum",  # 添加边到顶点的聚合方法
            v2e_drop_rate: float = 0.1,
            attention_heads: int = 4,  # 多头注意力
            use_edge_features: bool = True,  # 是否使用边特征
            residual: bool = True,
            normalize: str = "both",  # layer_norm, batch_norm, both, or none
            activation: str = "gelu",  # 使用GELU激活函数
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.attention_heads = attention_heads
        self.use_edge_features = use_edge_features
        self.residual = residual
        self.normalize = normalize

        # 激活函数选择
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "elu":
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.GELU()

        self.drop = nn.Dropout(drop_rate)

        # 线性变换层
        self.theta_v = nn.Linear(in_channels, out_channels, bias=bias)

        # 如果使用边特征，添加边特征变换
        if use_edge_features:
            self.theta_e = nn.Linear(out_channels, out_channels, bias=bias)

        # 归一化层
        if normalize in ["layer_norm", "both"]:
            self.layer_norm_v = nn.LayerNorm(out_channels)
            self.layer_norm_e = nn.LayerNorm(out_channels)

        self.v2e_aggr = v2e_aggr
        self.e2v_aggr = e2v_aggr
        self.v2e_drop_rate = v2e_drop_rate

        # 多头注意力机制
        if attention_heads > 0:
            self.v_attention = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_channels, 32),
                    nn.GELU(),
                    nn.Linear(32, 1)
                ) for _ in range(attention_heads)
            ])

            self.e_attention = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(out_channels, 32),
                    nn.GELU(),
                    nn.Linear(32, 1)
                ) for _ in range(attention_heads)
            ])

            # 注意力输出合并层
            self.attn_combine = nn.Linear(attention_heads * out_channels, out_channels)

        # 全局门控机制，控制信息流
        self.gate = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, X: torch.Tensor, hg: Hypergraph, edge_features: torch.Tensor = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播函数

        Args:
            X: 输入节点特征矩阵，形状为(|V|, C_in)
            hg: 包含|V|个顶点的超图结构
            edge_features: 可选的边特征输入，形状为(|E|, C_in)

        Returns:
            v_features: 更新后的顶点特征，形状为(|V|, C_out)
            e_features: 更新后的超边特征，形状为(|E|, C_out)
            attention_weights: 注意力权重，形状为(|E|, 1)
        """
        # 保存原始输入以用于残差连接
        global v_attn_weights
        X_orig = X
        batch_size = X.shape[0] // hg.num_v if X.shape[0] > hg.num_v else 1

        # 确保H在正确的设备上
        H = hg.H.to(X.device)

        # 转置H用于计算
        if isinstance(H, torch.sparse.Tensor):
            H_t = H.t().to_dense() if batch_size == 1 else H.t().to_dense().repeat(batch_size, 1)
        else:
            H_t = H.t() if batch_size == 1 else H.t().repeat(batch_size, 1)

        # 线性变换顶点特征
        X_transformed = self.theta_v(X)

        # 多头注意力处理
        if self.attention_heads > 0:
            # 存储每个注意力头的输出
            multi_head_e_features = []

            for head in range(self.attention_heads):
                # 计算顶点的注意力权重
                v_attn_scores = self.v_attention[head](X).squeeze(-1)  # (|V|, 1)
                v_attn_weights = torch.sigmoid(v_attn_scores)

                # 带注意力的v2e聚合
                weighted_X = X_transformed * v_attn_weights.unsqueeze(1)  # (|V|, C_out)

                # 自定义聚合，支持注意力权重
                head_e_features = torch.matmul(H_t, weighted_X)  # (|E|, C_out)

                # 归一化
                degrees = torch.sum(H_t, dim=1, keepdim=True)  # (|E|, 1)
                head_e_features = head_e_features / torch.clamp(degrees, min=1.0)

                multi_head_e_features.append(head_e_features)

            # 合并多头注意力结果
            e_features = torch.cat(multi_head_e_features, dim=1)  # (|E|, heads*C_out)
            e_features = self.attn_combine(e_features)  # (|E|, C_out)

            # 计算边的注意力权重，用于可视化或辅助任务
            edge_attention = torch.sum(H_t * v_attn_weights.unsqueeze(0), dim=1) / torch.clamp(degrees.squeeze(),
                                                                                               min=1.0)
            edge_attention = edge_attention.unsqueeze(1)  # (|E|, 1)
        else:
            # 使用标准v2e转换
            e_features = hg.v2e(X_transformed, aggr=self.v2e_aggr, drop_rate=self.v2e_drop_rate)
            edge_attention = torch.ones((hg.num_e, 1), device=X.device)  # 虚拟注意力权重

        # 如果有边特征输入，应用边特征
        if self.use_edge_features and edge_features is not None:
            # 变换边特征
            transformed_edge_features = self.theta_e(edge_features)

            # 边特征与聚合特征融合
            e_attention_scores = self.e_attention[0](e_features).squeeze(
                -1) if self.attention_heads > 0 else torch.ones(e_features.size(0), device=X.device)
            e_attention_weights = torch.sigmoid(e_attention_scores).unsqueeze(1)

            # 融合
            e_features = e_features * e_attention_weights + transformed_edge_features * (1 - e_attention_weights)

        # 归一化处理
        if self.normalize in ["layer_norm", "both"]:
            e_features = self.layer_norm_e(e_features)

        if self.normalize in ["batch_norm", "both"] and self.bn is not None and not self.is_last:
            e_features = self.bn(e_features)

        # 信息流回传到顶点：边特征聚合到顶点特征
        v_features = hg.e2v(e_features, aggr=self.e2v_aggr)

        # 应用门控机制
        gate_value = self.gate(torch.cat([X_orig, v_features], dim=1))
        v_features = gate_value * v_features + (1 - gate_value) * X_transformed

        # 归一化顶点特征
        if self.normalize in ["layer_norm", "both"]:
            v_features = self.layer_norm_v(v_features)

        # 残差连接
        if self.residual and not self.is_last:
            if X_orig.shape[1] == v_features.shape[1]:
                v_features = v_features + X_orig

        # 非线性激活和dropout
        if not self.is_last:
            v_features = self.act(v_features)
            e_features = self.act(e_features)

            v_features = self.drop(v_features)
            e_features = self.drop(e_features)

        return v_features, e_features, edge_attention
# import torch
# import torch.nn as nn
# from typing import Tuple
# from dhg.structure.hypergraphs import Hypergraph
#
#
# class HGNNPConv(nn.Module):
#     """增强版HGNN+卷积层，增加了多种聚合方式和注意力机制"""
#
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             bias: bool = True,
#             use_bn: bool = True,  # 默认启用批归一化
#             drop_rate: float = 0.3,  # 降低默认dropout率
#             is_last: bool = False,
#             v2e_aggr: str = "softmax_then_sum",  # 更改默认聚合方法
#             v2e_drop_rate: float = 0.1,
#             use_attention: bool = True,  # 添加注意力机制
#             residual: bool = True,  # 添加残差连接
#             activation: str = "leaky_relu",  # 使用LeakyReLU激活函数
#     ):
#         super().__init__()
#         self.is_last = is_last
#         self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
#
#         # 更灵活的激活函数选择
#         if activation == "relu":
#             self.act = nn.ReLU(inplace=True)
#         elif activation == "leaky_relu":
#             self.act = nn.LeakyReLU(0.2, inplace=True)
#         elif activation == "elu":
#             self.act = nn.ELU(inplace=True)
#         else:
#             self.act = nn.ReLU(inplace=True)
#
#         self.drop = nn.Dropout(drop_rate)
#         self.theta = nn.Linear(in_channels, out_channels, bias=bias)
#
#         # 增加层归一化作为选项
#         self.layer_norm = nn.LayerNorm(out_channels)
#
#         self.v2e_aggr = v2e_aggr
#         self.v2e_drop_rate = v2e_drop_rate
#         self.use_attention = use_attention
#         self.residual = residual
#
#         # 如果使用注意力机制，添加注意力层
#         if use_attention:
#             self.attention = nn.Sequential(
#                 nn.Linear(in_channels, 64),
#                 nn.LeakyReLU(0.2),
#                 nn.Linear(64, 1)
#             )
#
#     def forward(self, X: torch.Tensor, hg: Hypergraph) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         前向传播函数
#
#         Args:
#             X: 输入节点特征矩阵，形状为(|V|, C_in)
#             hg: 包含|V|个顶点的超图结构
#
#         Returns:
#             e_features: 输出超边特征矩阵，形状为(|E|, C_out)
#             attention_weights: 注意力权重，形状为(|E|, 1)
#         """
#         # 保存原始输入以用于残差连接
#         X_orig = X
#
#         # 线性变换顶点特征
#         X = self.theta(X)
#
#         if self.use_attention:
#             # 计算顶点的注意力权重
#             attention_scores = self.attention(X_orig).squeeze(-1)  # (|V|, 1)
#             attention_weights = torch.sigmoid(attention_scores)
#
#             # 带注意力的v2e聚合
#             H = hg.H.to(X.device)  # 确保H在正确的设备上
#
#             # 对于每个超边，聚合带权重的顶点特征
#             if isinstance(H, torch.sparse.Tensor):
#                 H_t = H.t().to_dense()  # (|E|, |V|)
#             else:
#                 H_t = H.t()  # (|E|, |V|)
#
#             # 应用注意力权重到顶点
#             weighted_X = X * attention_weights.unsqueeze(1)  # (|V|, C_out)
#
#             # 自定义聚合，支持注意力权重
#             e_features = torch.matmul(H_t, weighted_X)  # (|E|, C_out)
#
#             # 归一化
#             degrees = torch.sum(H_t, dim=1, keepdim=True)  # (|E|, 1)
#             e_features = e_features / torch.clamp(degrees, min=1.0)
#
#             # 保存注意力权重以用于损失计算
#             edge_attention = torch.sum(H_t * attention_weights.unsqueeze(0), dim=1) / torch.clamp(degrees.squeeze(),
#                                                                                                   min=1.0)
#             edge_attention = edge_attention.unsqueeze(1)  # (|E|, 1)
#         else:
#             # 使用原始v2e转换节点特征到超边特征
#             e_features = hg.v2e(X, aggr=self.v2e_aggr, drop_rate=self.v2e_drop_rate)
#             edge_attention = torch.ones((hg.num_e, 1), device=X.device)  # 虚拟注意力权重
#
#         # 添加层归一化
#         e_features = self.layer_norm(e_features)
#
#         if self.residual and not self.is_last and hg.num_v == hg.num_e:
#             # 残差连接(仅当输入输出维度匹配时)
#             if X_orig.shape[1] == e_features.shape[1]:
#                 # 直接加法
#                 e_features = e_features + hg.v2e(X_orig, aggr=self.v2e_aggr, drop_rate=0.0)
#
#         if not self.is_last:
#             e_features = self.act(e_features)
#             if self.bn is not None:
#                 e_features = self.bn(e_features)
#             e_features = self.drop(e_features)
#
#         return e_features, edge_attention


























































#毕业设计2.0
# import torch
# import torch.nn as nn
#
# from dhg.structure.hypergraphs import Hypergraph
#
#
# class HGNNPConv(nn.Module):
#     r"""The HGNN :sup:`+` convolution layer proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).
#
#     Sparse Format:
#
#     .. math::
#
#         \left\{
#             \begin{aligned}
#                 m_{\beta}^{t} &=\sum_{\alpha \in \mathcal{N}_{v}(\beta)} M_{v}^{t}\left(x_{\alpha}^{t}\right) \\
#                 y_{\beta}^{t} &=U_{e}^{t}\left(w_{\beta}, m_{\beta}^{t}\right) \\
#                 m_{\alpha}^{t+1} &=\sum_{\beta \in \mathcal{N}_{e}(\alpha)} M_{e}^{t}\left(x_{\alpha}^{t}, y_{\beta}^{t}\right) \\
#                 x_{\alpha}^{t+1} &=U_{v}^{t}\left(x_{\alpha}^{t}, m_{\alpha}^{t+1}\right) \\
#             \end{aligned}
#         \right.
#
#     Matrix Format:
#
#     .. math::
#         \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e
#         \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{X} \mathbf{\Theta} \right).
#
#     Args:
#         ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
#         ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
#         ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
#         ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
#         ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
#         ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
#     """
#
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         bias: bool = True,
#         use_bn: bool = False,
#         drop_rate: float = 0.5,
#         is_last: bool = False,
#     ):
#         super().__init__()
#         self.is_last = is_last
#         self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
#         self.act = nn.ReLU(inplace=True)
#         self.drop = nn.Dropout(drop_rate)
#         self.theta = nn.Linear(in_channels, out_channels, bias=bias)
#
#     def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
#         r"""The forward function.
#
#         Args:
#             X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
#             hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
#         """
#         X = self.theta(X)
#         X = hg.v2v(X, aggr="mean")
#         if not self.is_last:
#             X = self.act(X)
#             if self.bn is not None:
#                 X = self.bn(X)
#             X = self.drop(X)
#         return X
# 毕业设计1.0
# import torch
# import torch.nn as nn
# from dhg.structure.hypergraphs import Hypergraph
#
#
# class HGNNPConv(nn.Module):
#     r"""The HGNN :sup:`+` convolution layer proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).
#
#     Sparse Format:
#
#     .. math::
#
#         \left\{
#             \begin{aligned}
#                 m_{\beta}^{t} &=\sum_{\alpha \in \mathcal{N}_{v}(\beta)} M_{v}^{t}\left(x_{\alpha}^{t}\right) \\
#                 y_{\beta}^{t} &=U_{e}^{t}\left(w_{\beta}, m_{\beta}^{t}\right) \\
#                 m_{\alpha}^{t+1} &=\sum_{\beta \in \mathcal{N}_{e}(\alpha)} M_{e}^{t}\left(x_{\alpha}^{t}, y_{\beta}^{t}\right) \\
#                 x_{\alpha}^{t+1} &=U_{v}^{t}\left(x_{\alpha}^{t}, m_{\alpha}^{t+1}\right) \\
#             \end{aligned}
#         \right.
#
#     Matrix Format:
#
#     .. math::
#         \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e
#         \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{X} \mathbf{\Theta} \right).
#
#     Args:
#         ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
#         ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
#         ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
#         ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
#         ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
#         ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
#     """
#
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             bias: bool = True,
#             use_bn: bool = False,
#             drop_rate: float = 0.5,
#             is_last: bool = False,
#     ):
#         super().__init__()
#         self.is_last = is_last
#         self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
#         self.act = nn.ReLU(inplace=True)
#         self.drop = nn.Dropout(drop_rate)
#         self.theta = nn.Linear(in_channels, out_channels, bias=bias)
#
#     def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
#         r"""The forward function.
#
#         Args:
#             X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
#             hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
#         """
#         X = self.theta(X)
#         X = hg.v2v(X, aggr="mean")
#         if not self.is_last:
#             X = self.act(X)
#             if self.bn is not None:
#                 X = self.bn(X)
#             X = self.drop(X)
#         return X
