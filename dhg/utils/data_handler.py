import os
import glob
import torch
import pickle
from typing import List, Tuple
from dhg.structure.hypergraphs import Hypergraph

class TemporalMicroServiceData:
    """处理时序微服务数据的加载和预处理"""

    def __init__(self, data_root: str, train_ratio: float, val_ratio: float):
        self.data_root = data_root
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.feature_files = sorted(glob.glob(os.path.join(data_root, "features_*.pkl")))
        self.label_files = sorted(glob.glob(os.path.join(data_root, "labels_*.pkl")))

        self.edge_list = self._load_pickle(os.path.join(data_root, "edge_list.pkl"))
        self.num_vertices = 10

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