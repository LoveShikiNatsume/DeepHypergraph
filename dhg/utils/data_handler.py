
import os
import glob
import torch
import pickle
from typing import List, Tuple, Dict
from dhg.structure.hypergraphs import Hypergraph
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

class TemporalMicroServiceData:
    """处理时序微服务数据的加载和预处理"""

    def __init__(self, data_root: str, train_ratio: float, val_ratio: float):
        self.data_root = data_root
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # 获取并排序所有文件
        self.feature_files = sorted(glob.glob(os.path.join(data_root,"features_*.pkl")))
        self.label_files = sorted(glob.glob(os.path.join(data_root, "labels_*.pkl")))
        self.edge_list_files = sorted(glob.glob(os.path.join(data_root, "edge_list_30", "edge_list_*.pkl")))

        # 验证文件数量匹配
        assert len(self.feature_files) == len(self.label_files), "特征文件和标签文件数量不匹配"
        print(f"找到 {len(self.feature_files)} 个时间窗口的数据")
        print(f"找到 {len(self.edge_list_files)} 个边列表文件")

        self.num_vertices = 10
        # 预加载所有edge_lists以提高性能
        self.edge_lists = self._preload_edge_lists()

        self.train_indices, self.val_indices, self.test_indices = self._split_datasets()

        # 初始化特征预处理器
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 保留95%的方差信息

    def _preload_edge_lists(self) -> Dict[int, List[List[int]]]:
        """预加载所有edge_list文件"""
        edge_lists = {}
        print("预加载边列表文件...")
        for edge_file in tqdm(self.edge_list_files):
            # 从文件名中提取索引（减1以匹配从0开始的特征和标签索引）
            idx = int(os.path.basename(edge_file).split('_')[2].split('.')[0]) - 1
            with open(edge_file, 'rb') as f:
                edge_lists[idx] = pickle.load(f)
        return edge_lists

    def _load_pickle(self, file_path: str):
        """加载pickle文件"""
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

    def get_hypergraph(self, window_idx: int) -> Hypergraph:
        """返回指定时间窗口的超图结构"""
        if window_idx not in self.edge_lists:
            raise ValueError(f"找不到时间窗口 {window_idx} 的边列表")
        return Hypergraph(self.num_vertices, self.edge_lists[window_idx])

    def get_splits(self) -> Tuple[List[int], List[int], List[int]]:
        """返回训练集、验证集和测试集的索引"""
        return self.train_indices, self.val_indices, self.test_indices



































#毕业设计2.0
# import os
# import glob
# import torch
# import pickle
# from typing import List, Tuple, Dict
# from dhg.structure.hypergraphs import Hypergraph
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from tqdm import tqdm
#
# class TemporalMicroServiceData:
#     """处理时序微服务数据的加载和预处理"""
#
#     def __init__(self, data_root: str, train_ratio: float, val_ratio: float):
#         self.data_root = data_root
#         self.train_ratio = train_ratio
#         self.val_ratio = val_ratio
#
#         # 获取并排序所有文件
#         self.feature_files = sorted(glob.glob(os.path.join(data_root, "features_*.pkl")))
#         self.label_files = sorted(glob.glob(os.path.join(data_root, "labels_*.pkl")))
#         self.edge_list_files = sorted(glob.glob(os.path.join(data_root, "edge_list_20", "edge_list_*.pkl")))
#
#         # 验证文件数量匹配
#         assert len(self.feature_files) == len(self.label_files), "特征文件和标签文件数量不匹配"
#         print(f"找到 {len(self.feature_files)} 个时间窗口的数据")
#         print(f"找到 {len(self.edge_list_files)} 个边列表文件")
#
#         self.num_vertices = 10
#         # 预加载所有edge_lists以提高性能
#         self.edge_lists = self._preload_edge_lists()
#
#         self.train_indices, self.val_indices, self.test_indices = self._split_datasets()
#
#         # 初始化特征预处理器
#         self.scaler = StandardScaler()
#         self.pca = PCA(n_components=0.95)  # 保留95%的方差信息
#
#     def _preload_edge_lists(self) -> Dict[int, List[List[int]]]:
#         """预加载所有edge_list文件"""
#         edge_lists = {}
#         print("预加载边列表文件...")
#         for edge_file in tqdm(self.edge_list_files):
#             # 从文件名中提取索引（减1以匹配从0开始的特征和标签索引）
#             idx = int(os.path.basename(edge_file).split('_')[2].split('.')[0]) - 1
#             with open(edge_file, 'rb') as f:
#                 edge_lists[idx] = pickle.load(f)
#         return edge_lists
#
#     def _load_pickle(self, file_path: str):
#         """加载pickle文件"""
#         with open(file_path, 'rb') as f:
#             return pickle.load(f)
#
#     def _split_datasets(self) -> Tuple[List[int], List[int], List[int]]:
#         """划分时间窗口为训练集、验证集和测试集"""
#         total_windows = len(self.feature_files)
#         train_size = int(total_windows * self.train_ratio)
#         val_size = int(total_windows * self.val_ratio)
#
#         indices = list(range(total_windows))
#         train_indices = indices[:train_size]
#         val_indices = indices[train_size:train_size + val_size]
#         test_indices = indices[train_size + val_size:]
#
#         return train_indices, val_indices, test_indices
#
#     def load_window_data(self, window_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         """加载指定时间窗口的特征和标签"""
#         features = self._load_pickle(self.feature_files[window_idx])
#         labels = self._load_pickle(self.label_files[window_idx])
#
#         return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
#
#     def get_hypergraph(self, window_idx: int) -> Hypergraph:
#         """返回指定时间窗口的超图结构"""
#         if window_idx not in self.edge_lists:
#             raise ValueError(f"找不到时间窗口 {window_idx} 的边列表")
#         return Hypergraph(self.num_vertices, self.edge_lists[window_idx])
#
#     def get_splits(self) -> Tuple[List[int], List[int], List[int]]:
#         """返回训练集、验证集和测试集的索引"""
#         return self.train_indices, self.val_indices, self.test_indices




#毕业设计1.0
# import os
# import glob
# import torch
# import pickle
# from typing import List, Tuple
# from dhg.structure.hypergraphs import Hypergraph
#
# class TemporalMicroServiceData:
#     """处理时序微服务数据的加载和预处理"""
#
#     def __init__(self, data_root: str, train_ratio: float, val_ratio: float):
#         self.data_root = data_root
#         self.train_ratio = train_ratio
#         self.val_ratio = val_ratio
#
#         self.feature_files = sorted(glob.glob(os.path.join(data_root, "features_*.pkl")))
#         self.label_files = sorted(glob.glob(os.path.join(data_root, "labels_*.pkl")))
#
#         self.edge_list = self._load_pickle(os.path.join(data_root, "edge_list.pkl"))
#         self.num_vertices = 10
#
#         self.train_indices, self.val_indices, self.test_indices = self._split_datasets()
#
#     def _load_pickle(self, file_path: str):
#         with open(file_path, 'rb') as f:
#             return pickle.load(f)
#
#     def _split_datasets(self) -> Tuple[List[int], List[int], List[int]]:
#         """划分时间窗口为训练集、验证集和测试集"""
#         total_windows = len(self.feature_files)
#         train_size = int(total_windows * self.train_ratio)
#         val_size = int(total_windows * self.val_ratio)
#
#         indices = list(range(total_windows))
#         train_indices = indices[:train_size]
#         val_indices = indices[train_size:train_size + val_size]
#         test_indices = indices[train_size + val_size:]
#
#         return train_indices, val_indices, test_indices
#
#     def load_window_data(self, window_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         """加载指定时间窗口的特征和标签"""
#         features = self._load_pickle(self.feature_files[window_idx])
#         labels = self._load_pickle(self.label_files[window_idx])
#
#         return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
#
#     def get_hypergraph(self) -> Hypergraph:
#         """返回超图结构"""
#         return Hypergraph(self.num_vertices, self.edge_list)
#

