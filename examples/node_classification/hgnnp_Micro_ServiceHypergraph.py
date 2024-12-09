import os
import glob
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from dhg import Hypergraph
from dhg.models import HGNNP
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from typing import Dict, List, Tuple
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict
from copy import deepcopy

class TemporalMicroServiceData:
    """处理时序微服务数据的加载和预处理"""

    def __init__(self, data_root: str, train_ratio: float = 0.6, val_ratio: float = 0.2):
        self.data_root = data_root
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # 加载所有时间窗口的数据
        self.feature_files = sorted(glob.glob(os.path.join(data_root, "features_*.pkl")))
        self.label_files = sorted(glob.glob(os.path.join(data_root, "labels_*.pkl")))

        # 加载固定的超图结构
        self.edge_list = self._load_pickle(os.path.join(data_root, "edge_list.pkl"))
        self.num_vertices = 10
        self.num_classes = 2  # 故障分类类别数（即是不是有故障出现异常，0代表正常，1代表异常）

        # 划分训练集、验证集和测试集
        self.train_indices, self.val_indices, self.test_indices = self._split_datasets()

    def _load_pickle(self, file_path: str):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _split_datasets(self) -> Tuple[List[int], List[int], List[int]]:
        """划分时间窗口为训练集、验证集和测试集"""
        total_windows = len(self.feature_files)
        train_size = int(total_windows * self.train_ratio)
        val_size = int(total_windows * self.val_ratio)

        indices = list(range(total_windows))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        return train_indices, val_indices, test_indices

    def load_window_data(self, window_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载指定时间窗口的特征和标签"""
        features = self._load_pickle(self.feature_files[window_idx])
        labels = self._load_pickle(self.label_files[window_idx])

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def get_hypergraph(self) -> Hypergraph:
        """返回超图结构"""
        return Hypergraph(self.num_vertices, self.edge_list)

class AttentionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 计算查询、键和值
        query = self.query(X)
        key = self.key(X)
        value = self.value(X)

        # 计算注意力得分
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # 点积
        attention_weights = F.softmax(attention_scores, dim=-1)  # 使用softmax进行归一化

        # 使用注意力权重对特征值加权
        attended_values = torch.matmul(attention_weights, value)
        # print(f"X shape: {X.shape}")
        # print(f"Attention Weights shape: {attention_weights.shape}")
        return attended_values  # 返回加权后的特征

class TemporalMicroServiceTrainer:
    """处理时序微服务数据的训练和验证"""

    def __init__(self, data_handler: TemporalMicroServiceData, model_save_path: str):
        self.data_handler = data_handler
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型、超图和评估器
        self.G = self.data_handler.get_hypergraph().to(self.device)
        X, _ = self.data_handler.load_window_data(0)  # 用第一个窗口初始化特征维度
        self.model = HGNNP(X.shape[1], 32, self.data_handler.num_classes, use_bn=True).to(self.device)
        self.attention_layer = AttentionLayer(X.shape[1], 32).to(self.device)  # 加入注意力机制
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, verbose=True)  # 学习率调度器
        self.evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    def train_window(self, window_idx: int, epoch: int) -> float:
        """训练单个时间窗口的数据，加入注意力机制的损失"""
        self.model.train()
        X, labels = self.data_handler.load_window_data(window_idx)
        X, labels = X.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()

        # 前向传播，获取预测结果和注意力权重
        root_cause_score, fault_score, attention_weights = self.model(X, self.G)

        # 计算基础损失
        fault_loss = F.cross_entropy(fault_score, labels)
        root_cause_loss = F.binary_cross_entropy(root_cause_score.squeeze(), labels.float())

        # 注意力指导损失：鼓励模型对异常实例给予更高的注意力权重
        attention_target = labels.float().unsqueeze(1)  # 将标签转换为与attention_weights相同的形状
        attention_loss = F.mse_loss(attention_weights, attention_target)

        # 组合损失：可以调整权重
        total_loss = fault_loss + root_cause_loss + 10.0 * attention_loss

        # 反向传播
        total_loss.backward()
        self.optimizer.step()

        if epoch % 10 == 0:  # 每10个epoch打印一次详细信息
            print(f"Epoch: {epoch}, Window: {window_idx}")
            print(f"Total Loss: {total_loss.item():.5f}")
            print(f"Fault Loss: {fault_loss.item():.5f}")
            print(f"Root Cause Loss: {root_cause_loss.item():.5f}")
            print(f"Attention Loss: {attention_loss.item():.5f}")
            print(f"Average Attention Weight for Normal: {attention_weights[labels == 0].mean():.5f}")
            print(f"Average Attention Weight for Anomaly: {attention_weights[labels == 1].mean():.5f}")

        return total_loss.item()

    @torch.no_grad()
    def infer(self, window_idx: int, idx_type: str = 'val') -> Dict[str, float]:
        """
        推理单个时间窗口的数据，计算top-k准确率
        """
        self.model.eval()
        X, lbls = self.data_handler.load_window_data(window_idx)
        X, lbls = X.to(self.device), lbls.to(self.device)

        # 获取根因和故障分类的预测结果，以及注意力权重
        root_cause_score, fault_score, attention_weights = self.model(X, self.G)
        root_cause_preds = root_cause_score.squeeze()

        # 找到真实的根因索引
        true_root_cause = torch.where(lbls == 1)[0]

        # 计算不同k值的准确率
        metrics = {}
        for k in [1, 3, 5]:
            # 获取概率最高的k个预测
            _, top_k_indices = torch.topk(root_cause_preds, k, dim=0)
            # 检查真实根因是否在top-k预测中
            hit = any(idx.item() in top_k_indices.tolist() for idx in true_root_cause)
            metrics[f'top{k}_acc'] = float(hit)

        if idx_type == 'test':
            print(f"\nWindow {window_idx} Results:")
            print(f"Root Cause Probabilities: {root_cause_preds.cpu().numpy()}")
            print(f"True Root Cause Index: {true_root_cause.cpu().numpy()}")
            print("Top-K Accuracies:", {k: v for k, v in metrics.items()})
            print(f"Attention Weights: {attention_weights.squeeze().cpu().numpy()}")

        return metrics

    def evaluate_top_k_accuracy(self, window_idx: int, k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        评估单个时间窗口的Top-K准确率

        Args:
            window_idx: 时间窗口索引
            k_values: 需要评估的k值列表

        Returns:
            包含不同k值准确率的字典
        """
        self.model.eval()
        with torch.no_grad():
            # 获取特征和真实标签
            X, true_labels = self.data_handler.load_window_data(window_idx)
            X, true_labels = X.to(self.device), true_labels.to(self.device)

            # 获取模型预测的根因概率
            root_cause_score, _ = self.model(X, self.G)
            root_cause_probs = root_cause_score.squeeze()

            # 找到真实的根因索引
            true_root_cause = torch.where(true_labels == 1)[0]

            # 计算不同k值的准确率
            accuracies = {}
            for k in k_values:
                # 获取概率最高的k个预测
                _, top_k_indices = torch.topk(root_cause_probs, k, dim=0)

                # 检查真实根因是否在top-k预测中
                hit = any(idx in top_k_indices for idx in true_root_cause)
                accuracies[f'top{k}_acc'] = float(hit)

            return accuracies

    def test_all_windows(self):
        """测试所有时间窗口并统计Top-K准确率"""
        # 初始化评估指标
        k_values = [1, 3, 5]
        total_metrics = {f'top{k}_acc': 0.0 for k in k_values}
        window_count = len(self.data_handler.test_indices)

        print("\nTesting all windows...")
        for window_idx in self.data_handler.test_indices:
            # 评估当前窗口
            window_metrics = self.evaluate_top_k_accuracy(window_idx, k_values)

            # 累加各项指标
            for metric, value in window_metrics.items():
                total_metrics[metric] += value

            # 打印当前窗口的评估结果
            print(f"\nWindow {window_idx} results:")
            for metric, value in window_metrics.items():
                print(f"{metric}: {value}")

        # 计算平均准确率
        print("\nOverall Test Results:")
        for metric in total_metrics.keys():
            avg_value = total_metrics[metric] / window_count
            print(f"Average {metric}: {avg_value:.4f}")

        return total_metrics

    def train_all_windows(self, epochs: int = 10):
        """训练所有时间窗口的数据并评估"""
        best_state = None
        best_epoch, best_val = 0, 0

        for epoch in range(epochs):
            print(f"\nStarting Epoch {epoch + 1}/{epochs}")
            # 训练阶段
            epoch_loss = 0.0
            for window_idx in self.data_handler.train_indices:
                loss = self.train_window(window_idx, epoch)
                epoch_loss += loss

            avg_epoch_loss = epoch_loss / len(self.data_handler.train_indices)
            print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.5f}")

            # 验证阶段
            if epoch % 1 == 0:
                val_metrics = defaultdict(list)
                for window_idx in self.data_handler.val_indices:
                    window_metrics = self.infer(window_idx, 'val')
                    for k, v in window_metrics.items():
                        val_metrics[k].append(v)

                # 计算平均验证指标
                avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
                print("\nValidation Results:")
                for metric, value in avg_val_metrics.items():
                    print(f"Average {metric}: {value:.4f}")

                # 使用top1准确率作为主要评估指标
                if avg_val_metrics['top1_acc'] > best_val:
                    print(f"New best validation score: {avg_val_metrics['top1_acc']:.4f}")
                    best_epoch = epoch
                    best_val = avg_val_metrics['top1_acc']
                    best_state = deepcopy(self.model.state_dict())

            # 学习率调度
            self.scheduler.step(avg_epoch_loss)

        print("\nTraining finished!")
        print(f"Best validation score: {best_val:.4f} at epoch {best_epoch}")

        # 测试阶段
        print("\nTesting...")
        self.model.load_state_dict(best_state)
        test_metrics = defaultdict(list)
        for window_idx in self.data_handler.test_indices:
            window_metrics = self.infer(window_idx, 'test')
            for k, v in window_metrics.items():
                test_metrics[k].append(v)

        # 计算并打印最终测试结果
        print("\nFinal Test Results:")
        for metric, values in test_metrics.items():
            avg_value = np.mean(values)
            print(f"Average {metric}: {avg_value:.4f}")

        return test_metrics

def main():
    # 设置数据路径和模型保存路径
    data_root = "../../datasets/test"
    model_save_path = "../../model"

    # 初始化，默认使用 80% 训练，10% 验证，10% 测试
    data_handler = TemporalMicroServiceData(data_root, train_ratio=0.8, val_ratio=0.1)
    trainer = TemporalMicroServiceTrainer(data_handler, model_save_path)

    # 开始训练、验证和测试
    trainer.train_all_windows(epochs=20)

if __name__ == "__main__":
    main()