import os
import glob
import torch
import torch.nn.functional as F
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
        self.num_classes = 5

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        self.evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    def train_window(self, window_idx: int, epoch: int) -> float:
        """训练单个时间窗口的数据"""
        self.model.train()
        X, labels = self.data_handler.load_window_data(window_idx)
        X, labels = X.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(X, self.G)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        self.optimizer.step()

        print(f"Epoch: {epoch}, Window: {window_idx}, Loss: {loss.item():.5f}")
        return loss.item()

    @torch.no_grad()
    def infer(self, window_idx: int, idx_type: str = 'val'):
        """使用原始验证方法进行推理"""
        self.model.eval()
        X, lbls = self.data_handler.load_window_data(window_idx)
        X, lbls = X.to(self.device), lbls.to(self.device)

        outs = self.model(X, self.G)

        # 输出预测结果和真实标签
        preds = torch.argmax(outs, dim=1)
        print(f"Time Window {window_idx}:")
        print(f"Predictions: {preds}")
        print(f"True Labels: {lbls}")

        # 根据是验证还是测试选择评估方式
        if idx_type == 'val':
            res = self.evaluator.validate(lbls, outs)
        else:
            res = self.evaluator.test(lbls, outs)

        return res

    def train_all_windows(self, epochs: int = 10):
        """训练所有时间窗口的数据"""
        best_state = None
        best_epoch, best_val = 0, 0

        for epoch in range(epochs):
            # 训练阶段
            for window_idx in self.data_handler.train_indices:
                loss = self.train_window(window_idx, epoch)

            # 验证阶段
            if epoch % 1 == 0:
                val_scores = []
                for window_idx in self.data_handler.val_indices:
                    val_res = self.infer(window_idx, 'val')
                    val_scores.append(val_res)

                avg_val_score = np.mean(val_scores)
                if avg_val_score > best_val:
                    print(f"Update best: {avg_val_score:.5f}")
                    best_epoch = epoch
                    best_val = avg_val_score
                    best_state = deepcopy(self.model.state_dict())

        print("\nTraining finished!")
        print(f"Best val: {best_val:.5f}")

        # 测试阶段
        print("Testing...")
        self.model.load_state_dict(best_state)
        test_scores = []
        for window_idx in self.data_handler.test_indices:
            test_res = self.infer(window_idx, 'test')
            test_scores.append(test_res)

        # 分别计算每个指标的平均值
        avg_scores = {}
        for metric in test_scores[0].keys():
            avg_scores[metric] = np.mean([score[metric] for score in test_scores])

        print(f"Final result: epoch: {best_epoch}")
        for metric, value in avg_scores.items():
            print(f"Average test {metric}: {value:.5f}")

    def save_model(self, filename: str, metrics: Dict):
        """保存模型和相关指标"""
        save_path = os.path.join(self.model_save_path, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }, save_path)

def main():
    # 设置数据路径和模型保存路径
    data_root = "../../datasets/test"
    model_save_path = "../../model"

    # 初始化，默认使用 60% 训练，20% 验证，20% 测试
    data_handler = TemporalMicroServiceData(data_root, train_ratio=0.6, val_ratio=0.2)
    trainer = TemporalMicroServiceTrainer(data_handler, model_save_path)

    # 开始训练、验证和测试
    trainer.train_all_windows(epochs=10)


if __name__ == "__main__":
    main()