# class Hypergraph:
#     def __init__(self):
#         self.vertices = []  # 顶点列表
#         self.hyperedges = []  # 超边列表，每个超边是一个顶点集合
#
#     def add_vertex(self, vertex):
#         self.vertices.append(vertex)
#
#     def add_hyperedge(self, hyperedge):
#         self.hyperedges.append(hyperedge)
#
#     def print_hypergraph(self):
#         print("Vertices:", self.vertices)
#         print("Hyperedges:", self.hyperedges)
#
#     def H_matrix(self):
#         import numpy as np
#         num_vertices = len(self.vertices)
#         num_hyperedges = len(self.hyperedges)
#         H = np.zeros((num_vertices, num_hyperedges), dtype=int)
#
#         for i, edge in enumerate(self.hyperedges):
#             for vertex in edge:
#                 H[self.vertices.index(vertex), i] = 1
#
#         return H
#
# # 创建超图实例
# hg = Hypergraph()
#
# # 添加顶点
# services = ["W1", "W2", "R1", "R2", "M1", "M2", "L1", "L2", "D1", "D2"]
# for service in services:
#     hg.add_vertex(service)
#
# # 添加超边
# hg.add_hyperedge(["W1", "R1", "M1"])  # E1
# hg.add_hyperedge(["L1", "W2"])  # E2
# hg.add_hyperedge(["L2","R2","D2"])  # E3
# hg.add_hyperedge(["M2","D1"])
# # 打印超图
# hg.print_hypergraph()
#
# # 生成并打印H矩阵
# H_matrix = hg.H_matrix()
# print("H Matrix:")
# print(H_matrix)




# import pickle
# from pathlib import Path
# import numpy as np
#
# class Hypergraph:
#     def __init__(self):
#         self.vertices = []  # 顶点列表
#         self.hyperedges = []  # 超边列表，每个超边是一个顶点集合
#
#     def add_vertex(self, vertex):
#         self.vertices.append(vertex)
#
#     def add_hyperedge(self, hyperedge):
#         self.hyperedges.append(hyperedge)
#
#     def print_hypergraph(self):
#         print("Vertices:", self.vertices)
#         print("Hyperedges:", self.hyperedges)
#
#     def H_matrix(self):
#         import numpy as np
#         num_vertices = len(self.vertices)
#         num_hyperedges = len(self.hyperedges)
#         H = np.zeros((num_vertices, num_hyperedges), dtype=int)
#
#         for i, edge in enumerate(self.hyperedges):
#             for vertex in edge:
#                 if vertex in self.vertices:
#                     H[self.vertices.index(vertex), i] = 1
#
#         return H
#
# # 创建超图实例
# hg = Hypergraph()
#
# # 添加顶点
# vertices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 使用数字作为顶点
# # vertices = ["0-logservice1", "1-logservice2", "2-mobservice1", "3-mobservice2",
# #             "4-redisservice1", "5-redisservice2", "6-dbservice1", "7-dbservice2",
# #             "8-webservice1", "9-webservice2"]
# for vertex in vertices:
#     hg.add_vertex(vertex)
#
# # 添加超边
# hg.add_hyperedge([8, 5, 2])  # E1
# hg.add_hyperedge([0, 9])  # E2
# hg.add_hyperedge([7, 1, 5])  # E3
# hg.add_hyperedge([3, 6])  # E4
#
# # 打印超图
# hg.print_hypergraph()
#
# # 生成并打印H矩阵
# H_matrix = hg.H_matrix()
# print("H Matrix:")
# print(H_matrix)
#
# # 指定保存路径
# base_save_path = Path("D:/6/DeepHypergraph-main/datasets/test")
#
# # 确保目录存在
# base_save_path.mkdir(parents=True, exist_ok=True)
#
# # 保存超边列表到指定文件
# edge_list_path = base_save_path / "edge_list.pkl"
# with open(edge_list_path, 'wb') as f:
#     pickle.dump(hg.hyperedges, f)
#
# print(f"Edge list saved to {edge_list_path}")
#
# # 生成掩码
# num_vertices = len(hg.vertices)
# train_ratio = 0.8
# val_ratio = 0.1
# test_ratio = 0.1
#
# # 生成随机索引
# indices = np.arange(num_vertices)
# np.random.shuffle(indices)
#
# # 计算训练、验证和测试的索引
# train_end = int(train_ratio * num_vertices)
# val_end = train_end + int(val_ratio * num_vertices)
#
# # 创建掩码
# train_mask = np.zeros(num_vertices, dtype=bool)
# val_mask = np.zeros(num_vertices, dtype=bool)
# test_mask = np.zeros(num_vertices, dtype=bool)
#
# train_mask[indices[:train_end]] = True
# val_mask[indices[train_end:val_end]] = True
# test_mask[indices[val_end:]] = True
#
# # 分别保存掩码到文件
# train_mask_path = base_save_path / "train_mask.pkl"
# val_mask_path = base_save_path / "val_mask.pkl"
# test_mask_path = base_save_path / "test_mask.pkl"
#
# with open(train_mask_path, 'wb') as f:
#     pickle.dump(train_mask, f)
#
# with open(val_mask_path, 'wb') as f:
#     pickle.dump(val_mask, f)
#
# with open(test_mask_path, 'wb') as f:
#     pickle.dump(test_mask, f)
#
# print(f"Train mask saved to {train_mask_path}")
# print(f"Val mask saved to {val_mask_path}")
# print(f"Test mask saved to {test_mask_path}")


# # 创建校验码
# import hashlib
#
# def calculate_md5(file_path):
#     md5 = hashlib.md5()
#     with open(file_path, 'rb') as f:  # 打开文件用于读取二进制模式
#         for chunk in iter(lambda: f.read(4096), b""):  # 读取文件块
#             md5.update(chunk)  # 更新hash对象
#     return md5.hexdigest()  # 返回MD5校验和的十六进制表示
#
# # 使用示例
# file_path = r'D:\6\DeepHypergraph-main\datasets\test\features_0.pkl'  # 替换为您的文件路径
# md5_checksum = calculate_md5(file_path)
# print(f"The MD5 checksum of the file is: {md5_checksum}")
import hashlib
import pickle
import subprocess
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

class MicroServiceHypergraphDataset(Dataset):
    def __init__(self, data_root: str, file_indices: List[int]):
        self.data_root = Path(data_root)
        self.file_indices = file_indices
        self.data = self.load_data()

    def load_data(self):
        data = {}
        for key, value in [
            ("labels", "labels_{}.pkl"),
            ("features", "features_{}.pkl"),
        ]:
            files = [self.data_root / value.format(i) for i in self.file_indices]
            data[key] = [pickle.load(open(file, 'rb')) for file in files]
        return data

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        return {
            "labels": torch.tensor(self.data["labels"][idx], dtype=torch.long),
            "features": torch.tensor(self.data["features"][idx], dtype=torch.float),
        }

def calculate_md5(file_path: Path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def update_md5_checksums(file_indices: List[int], data_root: Path):
    for key, value in [
        ("labels", "labels_{}.pkl"),
        ("features", "features_{}.pkl"),
    ]:
        for i in file_indices:
            file_path = data_root / value.format(i)
            md5 = calculate_md5(file_path)
            print(f"MD5 checksum for {file_path}: {md5}")

def run_training_script():
    subprocess.run(["python", "hgnnp_Micro_ServiceHypergraph.py"])

if __name__ == "__main__":
    num_iterations = 200
    data_root = "D:/6/DeepHypergraph-main/datasets/test"
    file_indices = list(range(1099))  # Assuming 1099 files for labels and features

    for epoch in range(num_iterations):
        file_index = epoch % len(file_indices)
        dataset = MicroServiceHypergraphDataset(data_root, [file_index])
        update_md5_checksums([dataset.file_indices[0]], Path(data_root))
        run_training_script()