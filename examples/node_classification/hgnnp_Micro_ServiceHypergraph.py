import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import dhg
from dhg.models import HGNNP
from typing import Dict, List
import numpy as np
from collections import defaultdict
from copy import deepcopy
from dhg.utils import TemporalMicroServiceData
from dhg.utils import ExperimentSaver


def transform_node_to_edge_labels(node_labels: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
    """为每个超图的前19条超边分配标签：前9条设为0，后10条基于节点标签

    参数:
        node_labels: 节点标签张量，形状为 (num_nodes,)
        hg: 超图结构

    返回:
        edge_labels: 超边标签张量，形状为 (num_edges,)
    """
    num_edges = hg.num_e
    device = node_labels.device

    # 创建全零的超边标签张量
    edge_labels = torch.zeros(num_edges, device=device)

    # 只处理前19条超边
    num_edges_to_process = min(19, num_edges)

    # 前9条超边标签为0（已经默认为0了，所以无需额外设置）

    # 对后10条超边设置标签（如果有足够多的超边）
    if num_edges_to_process > 9:
        # 获取超图的邻接矩阵
        H = hg.H  # 形状为 (num_nodes, num_edges)

        # 确保H与node_labels在同一设备上
        if isinstance(H, torch.sparse.Tensor):
            H = H.to(device)
        else:
            H = torch.tensor(H, device=device)

        # 转置H使得每行代表一个超边，每列代表一个节点
        H_t = H.t()  # 形状为 (num_edges, num_nodes)

        # 处理第10条到第19条超边
        for edge_idx in range(9, num_edges_to_process):
            if isinstance(H_t, torch.sparse.Tensor):
                # 对于稀疏张量，直接转为密集张量处理
                # 这对于大型稀疏矩阵可能会消耗大量内存，但对于单个超边应该可以接受
                edge_vector_dense = H_t[edge_idx].to_dense()

                # 检查是否有标签为1的节点
                node_indices = torch.where(edge_vector_dense > 0)[0]
                has_anomaly = torch.any(node_labels[node_indices] > 0.5).item()
            else:
                # 获取当前超边包含的节点的掩码
                edge_mask = H_t[edge_idx] > 0

                # 使用掩码获取这些节点的标签
                node_labels_in_edge = node_labels[edge_mask]

                # 检查是否有任何异常节点
                has_anomaly = torch.any(node_labels_in_edge > 0.5).item()

            # 如果超边中有任何节点标签为1，则将超边标签设为1
            if has_anomaly:
                edge_labels[edge_idx] = 1.0

    return edge_labels

class TemporalMicroServiceTrainer:
    """处理时序微服务数据的训练和验证"""

    def __init__(self, data_handler: TemporalMicroServiceData, config: dict):
        self.data_handler = data_handler
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化保存工具
        self.saver = ExperimentSaver(config['model_save_path'])

        # 初始化模型
        X, _ = self.data_handler.load_window_data(0)
        self.model = HGNNP(
            X.shape[1],
            config['hidden_dim'],
            use_bn=config['use_bn'],
            drop_rate=config['drop_rate']
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            'min',
            patience=config['scheduler_patience'],
            factor=config['scheduler_factor'],
        )

    def train_window(self, window_idx: int, epoch: int) -> float:
        """训练单个时间窗口的数据"""
        self.model.train()
        X, labels = self.data_handler.load_window_data(window_idx)
        X, labels = X.to(self.device), labels.to(self.device)
        G = self.data_handler.get_hypergraph(window_idx)
        # 将节点标签转换为超边标签
        # 方法1：使用超图结构将节点标签聚合到超边
        # 假设每个超边的标签是其包含的节点中，如果有任何节点是异常的，则超边是异常的
        edge_labels = transform_node_to_edge_labels(labels, G)

        self.optimizer.zero_grad()

        root_cause_score, attention_weights = self.model(X, G)

        # 现在使用超边标签进行损失计算
        pred_loss = F.binary_cross_entropy(root_cause_score.squeeze(), edge_labels.float())
        attention_target = edge_labels.float().unsqueeze(1)
        attention_loss = F.mse_loss(attention_weights, attention_target)
        total_loss = pred_loss + self.config['attention_loss_weight'] * attention_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    @torch.no_grad()
    def infer(self, window_idx: int, idx_type: str = 'val') -> Dict[str, float]:
        """推理单个时间窗口的数据，计算超边top-k准确率"""
        self.model.eval()

        # 获取当前时间窗口的特征、标签和超图
        X, labels  = self.data_handler.load_window_data(window_idx)
        X, labels  = X.to(self.device), labels .to(self.device)
        G = self.data_handler.get_hypergraph(window_idx).to(self.device)

        # 将节点标签转换为超边标签
        edge_labels = transform_node_to_edge_labels(labels, G)
        edge_labels  =edge_labels .to(self.device)

        root_cause_score, attention_weights = self.model(X, G)
        root_cause_preds = root_cause_score.squeeze()

        true_root_cause_edges = torch.where(edge_labels == 1)[0]

        metrics = {}
        for k in [1, 3, 5]:
            _, top_k_indices = torch.topk(root_cause_preds, k, dim=0)
            hit = any(idx.item() in top_k_indices.tolist() for idx in true_root_cause_edges)
            metrics[f'top{k}_acc'] = float(hit)

        return metrics

    def train_all_windows(self) -> Dict[str, List[float]]:
        """训练所有时间窗口的数据并评估"""
        # 初始化实验保存目录
        self.saver.initialize_experiment()

        best_state = None
        best_epoch, best_val = 0, 0
        best_metrics = None

        print(f"Starting training with {len(self.data_handler.train_indices)} windows")
        print(f"Using validation set of {len(self.data_handler.val_indices)} windows")
        print(f"Test set contains {len(self.data_handler.test_indices)} windows")

        for epoch in range(self.config['epochs']):
            print(f"\nStarting Epoch {epoch + 1}/{self.config['epochs']}")

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

                avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
                print("\nValidation Results:")
                for metric, value in avg_val_metrics.items():
                    print(f"Average {metric}: {value:.4f}")

                if avg_val_metrics['top1_acc'] > best_val:
                    print(f"New best validation score: {avg_val_metrics['top1_acc']:.4f}")
                    best_epoch = epoch
                    best_val = avg_val_metrics['top1_acc']
                    best_state = deepcopy(self.model.state_dict())
                    best_metrics = avg_val_metrics.copy()

                    # 保存最优模型的状态
                    self.saver.save_best_model(best_state)

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

        final_test_metrics = {metric: np.mean(values) for metric, values in test_metrics.items()}
        print("\nFinal Test Results:")
        for metric, value in final_test_metrics.items():
            print(f"Average {metric}: {value:.4f}")

        # 保存最终的实验结果
        self.saver.save_experiment_results(
            best_state_dict=best_state,
            final_metrics=final_test_metrics,
            best_epoch=best_epoch,
            config=self.config,
            train_indices=self.data_handler.train_indices,
            val_indices=self.data_handler.val_indices,
            test_indices=self.data_handler.test_indices
        )

        return test_metrics



def main():
    # 所有可配置的参数都在这里设置
    config = {
        # 路径配置
        "data_root": "../../datasets/test",
        "model_save_path": "../../model",

        # 数据集划分配置
        "train_ratio": 0.6,
        "val_ratio": 0.2,

        # 模型配置
        "hidden_dim": 128,
        "use_bn": True,
        "drop_rate": 0.5,

        # 训练配置
        "epochs": 10,
        "learning_rate": 0.0003,
        "weight_decay": 5e-4,
        "attention_loss_weight": 100.0,
        "scheduler_patience": 5,
        "scheduler_factor": 0.5
    }

    # 初始化数据处理器
    data_handler = TemporalMicroServiceData(
        config['data_root'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio']
    )

    # 初始化训练器并开始训练
    trainer = TemporalMicroServiceTrainer(data_handler, config)
    trainer.train_all_windows()


if __name__ == "__main__":
    main()








































#毕业设计2.0
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from tqdm import tqdm
# from dhg.models import HGNNP
# from typing import Dict, List
# import numpy as np
# from collections import defaultdict
# from copy import deepcopy
# from dhg.utils import TemporalMicroServiceData
# from dhg.utils import ExperimentSaver
# class TemporalMicroServiceTrainer:
#     """处理时序微服务数据的训练和验证"""
#
#     def __init__(self, data_handler: TemporalMicroServiceData, config: dict):
#         self.data_handler = data_handler
#         self.config = config
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # 初始化保存工具
#         self.saver = ExperimentSaver(config['model_save_path'])
#
#         # 初始化模型
#         X, _ = self.data_handler.load_window_data(0)
#         self.model = HGNNP(
#             X.shape[1],
#             config['hidden_dim'],
#             use_bn=config['use_bn'],
#             drop_rate=config['drop_rate']
#         ).to(self.device)
#
#         self.optimizer = optim.Adam(
#             self.model.parameters(),
#             lr=config['learning_rate'],
#             weight_decay=config['weight_decay']
#         )
#
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer,
#             'min',
#             patience=config['scheduler_patience'],
#             factor=config['scheduler_factor'],
#         )
#
#     def train_window(self, window_idx: int, epoch: int) -> float:
#         """训练单个时间窗口的数据"""
#         self.model.train()
#
#         # 获取当前时间窗口的特征、标签和超图
#         X, labels = self.data_handler.load_window_data(window_idx)
#         X, labels = X.to(self.device), labels.to(self.device)
#         G = self.data_handler.get_hypergraph(window_idx).to(self.device)
#
#         self.optimizer.zero_grad()
#         root_cause_score, attention_weights = self.model(X, G)
#
#         pred_loss = F.binary_cross_entropy(root_cause_score.squeeze(), labels.float())
#         attention_target = labels.float().unsqueeze(1)
#         attention_loss = F.mse_loss(attention_weights, attention_target)
#         total_loss = pred_loss + self.config['attention_loss_weight'] * attention_loss
#
#         total_loss.backward()
#         self.optimizer.step()
#
#         return total_loss.item()
#
#     @torch.no_grad()
#     def infer(self, window_idx: int, idx_type: str = 'val') -> Dict[str, float]:
#         """推理单个时间窗口的数据，计算top-k准确率"""
#         self.model.eval()
#
#         # 获取当前时间窗口的特征、标签和超图
#         X, lbls = self.data_handler.load_window_data(window_idx)
#         X, lbls = X.to(self.device), lbls.to(self.device)
#         G = self.data_handler.get_hypergraph(window_idx).to(self.device)
#
#         root_cause_score, attention_weights = self.model(X, G)
#         root_cause_preds = root_cause_score.squeeze()
#
#         true_root_cause = torch.where(lbls == 1)[0]
#
#         metrics = {}
#         for k in [1, 3, 5]:
#             _, top_k_indices = torch.topk(root_cause_preds, k, dim=0)
#             hit = any(idx.item() in top_k_indices.tolist() for idx in true_root_cause)
#             metrics[f'top{k}_acc'] = float(hit)
#
#         return metrics
#
#     def train_all_windows(self) -> Dict[str, List[float]]:
#         """训练所有时间窗口的数据并评估"""
#         # 初始化实验保存目录
#         self.saver.initialize_experiment()
#
#         best_state = None
#         best_epoch, best_val = 0, 0
#         best_metrics = None
#
#         print(f"Starting training with {len(self.data_handler.train_indices)} windows")
#         print(f"Using validation set of {len(self.data_handler.val_indices)} windows")
#         print(f"Test set contains {len(self.data_handler.test_indices)} windows")
#
#         for epoch in range(self.config['epochs']):
#             print(f"\nStarting Epoch {epoch + 1}/{self.config['epochs']}")
#
#             # 训练阶段
#             epoch_loss = 0.0
#             for window_idx in self.data_handler.train_indices:
#                 loss = self.train_window(window_idx, epoch)
#                 epoch_loss += loss
#
#             avg_epoch_loss = epoch_loss / len(self.data_handler.train_indices)
#             print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.5f}")
#
#             # 验证阶段
#             if epoch % 1 == 0:
#                 val_metrics = defaultdict(list)
#                 for window_idx in self.data_handler.val_indices:
#                     window_metrics = self.infer(window_idx, 'val')
#                     for k, v in window_metrics.items():
#                         val_metrics[k].append(v)
#
#                 avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
#                 print("\nValidation Results:")
#                 for metric, value in avg_val_metrics.items():
#                     print(f"Average {metric}: {value:.4f}")
#
#                 if avg_val_metrics['top1_acc'] > best_val:
#                     print(f"New best validation score: {avg_val_metrics['top1_acc']:.4f}")
#                     best_epoch = epoch
#                     best_val = avg_val_metrics['top1_acc']
#                     best_state = deepcopy(self.model.state_dict())
#                     best_metrics = avg_val_metrics.copy()
#
#                     # 保存最优模型的状态
#                     self.saver.save_best_model(best_state)
#
#             self.scheduler.step(avg_epoch_loss)
#
#         print("\nTraining finished!")
#         print(f"Best validation score: {best_val:.4f} at epoch {best_epoch}")
#
#         # 测试阶段
#         print("\nTesting...")
#         self.model.load_state_dict(best_state)
#         test_metrics = defaultdict(list)
#         for window_idx in self.data_handler.test_indices:
#             window_metrics = self.infer(window_idx, 'test')
#             for k, v in window_metrics.items():
#                 test_metrics[k].append(v)
#
#         final_test_metrics = {metric: np.mean(values) for metric, values in test_metrics.items()}
#         print("\nFinal Test Results:")
#         for metric, value in final_test_metrics.items():
#             print(f"Average {metric}: {value:.4f}")
#
#         # 保存最终的实验结果
#         self.saver.save_experiment_results(
#             best_state_dict=best_state,
#             final_metrics=final_test_metrics,
#             best_epoch=best_epoch,
#             config=self.config,
#             train_indices=self.data_handler.train_indices,
#             val_indices=self.data_handler.val_indices,
#             test_indices=self.data_handler.test_indices
#         )
#
#         return test_metrics

# #毕业设计1.0
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from dhg.models import HGNNP
# from typing import Dict, List
# import numpy as np
# from collections import defaultdict
# from copy import deepcopy
# from dhg.utils import TemporalMicroServiceData
# from dhg.utils import ExperimentSaver
#
#
# class TemporalMicroServiceTrainer:
#     """处理时序微服务数据的训练和验证"""
#
#     def __init__(self, data_handler: TemporalMicroServiceData, config: dict):
#         self.data_handler = data_handler
#         self.config = config
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # 初始化保存工具
#         self.saver = ExperimentSaver(config['model_save_path'])
#
#         # 初始化模型、超图和评估器
#         self.G = self.data_handler.get_hypergraph().to(self.device)
#         X, _ = self.data_handler.load_window_data(0)
#         self.model = HGNNP(
#             X.shape[1],
#             config['hidden_dim'],
#             use_bn=config['use_bn'],
#             drop_rate=config['drop_rate']
#         ).to(self.device)
#
#         self.optimizer = optim.Adam(
#             self.model.parameters(),
#             lr=config['learning_rate'],
#             weight_decay=config['weight_decay']
#         )
#
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer,
#             'min',
#             patience=config['scheduler_patience'],
#             factor=config['scheduler_factor'],
#             verbose=True
#         )
#
#     def train_window(self, window_idx: int, epoch: int) -> float:
#         """训练单个时间窗口的数据"""
#         self.model.train()
#         X, labels = self.data_handler.load_window_data(window_idx)
#         X, labels = X.to(self.device), labels.to(self.device)
#
#         self.optimizer.zero_grad()
#
#         root_cause_score, attention_weights = self.model(X, self.G)
#
#         pred_loss = F.binary_cross_entropy(root_cause_score.squeeze(), labels.float())
#         attention_target = labels.float().unsqueeze(1)
#         attention_loss = F.mse_loss(attention_weights, attention_target)
#         total_loss = pred_loss + self.config['attention_loss_weight'] * attention_loss
#
#         total_loss.backward()
#         self.optimizer.step()
#
#         return total_loss.item()
#
#     @torch.no_grad()
#     def infer(self, window_idx: int, idx_type: str = 'val') -> Dict[str, float]:
#         """推理单个时间窗口的数据，计算top-k准确率"""
#         self.model.eval()
#         X, lbls = self.data_handler.load_window_data(window_idx)
#         X, lbls = X.to(self.device), lbls.to(self.device)
#
#         root_cause_score, attention_weights = self.model(X, self.G)
#         root_cause_preds = root_cause_score.squeeze()
#
#         true_root_cause = torch.where(lbls == 1)[0]
#
#         metrics = {}
#         for k in [1, 3, 5]:
#             _, top_k_indices = torch.topk(root_cause_preds, k, dim=0)
#             hit = any(idx.item() in top_k_indices.tolist() for idx in true_root_cause)
#             metrics[f'top{k}_acc'] = float(hit)
#
#         return metrics
#
#     def train_all_windows(self) -> Dict[str, List[float]]:
#         """训练所有时间窗口的数据并评估"""
#         # 初始化实验保存目录
#         self.saver.initialize_experiment()
#
#         best_state = None
#         best_epoch, best_val = 0, 0
#         best_metrics = None
#
#         for epoch in range(self.config['epochs']):
#             print(f"\nStarting Epoch {epoch + 1}/{self.config['epochs']}")
#
#             epoch_loss = 0.0
#             for window_idx in self.data_handler.train_indices:
#                 loss = self.train_window(window_idx, epoch)
#                 epoch_loss += loss
#
#             avg_epoch_loss = epoch_loss / len(self.data_handler.train_indices)
#             print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.5f}")
#
#             # 验证阶段
#             if epoch % 1 == 0:
#                 val_metrics = defaultdict(list)
#                 for window_idx in self.data_handler.val_indices:
#                     window_metrics = self.infer(window_idx, 'val')
#                     for k, v in window_metrics.items():
#                         val_metrics[k].append(v)
#
#                 avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
#                 print("\nValidation Results:")
#                 for metric, value in avg_val_metrics.items():
#                     print(f"Average {metric}: {value:.4f}")
#
#                 if avg_val_metrics['top1_acc'] > best_val:
#                     print(f"New best validation score: {avg_val_metrics['top1_acc']:.4f}")
#                     best_epoch = epoch
#                     best_val = avg_val_metrics['top1_acc']
#                     best_state = deepcopy(self.model.state_dict())
#                     best_metrics = avg_val_metrics.copy()
#
#                     # 只保存最优模型的状态
#                     self.saver.save_best_model(best_state)
#
#             self.scheduler.step(avg_epoch_loss)
#
#         print("\nTraining finished!")
#         print(f"Best validation score: {best_val:.4f} at epoch {best_epoch}")
#
#         # 测试阶段
#         print("\nTesting...")
#         self.model.load_state_dict(best_state)
#         test_metrics = defaultdict(list)
#         for window_idx in self.data_handler.test_indices:
#             window_metrics = self.infer(window_idx, 'test')
#             for k, v in window_metrics.items():
#                 test_metrics[k].append(v)
#
#         final_test_metrics = {metric: np.mean(values) for metric, values in test_metrics.items()}
#         print("\nFinal Test Results:")
#         for metric, value in final_test_metrics.items():
#             print(f"Average {metric}: {value:.4f}")
#
#         # 保存最终的实验结果
#         self.saver.save_experiment_results(
#             best_state_dict=best_state,
#             final_metrics=final_test_metrics,
#             best_epoch=best_epoch,
#             config=self.config,
#             train_indices=self.data_handler.train_indices,
#             val_indices=self.data_handler.val_indices,
#             test_indices=self.data_handler.test_indices
#         )
#
#         return test_metrics
#
#
# def main():
#     # 所有可配置的参数都在这里设置
#     config = {
#         # 路径配置
#         "data_root": "../../datasets/test",
#         "model_save_path": "../../model",
#
#         # 数据集划分配置
#         "train_ratio": 0.6,
#         "val_ratio": 0.2,
#
#         # 模型配置
#         "hidden_dim": 128,
#         "use_bn": True,
#         "drop_rate": 0.5,
#
#         # 训练配置
#         "epochs": 10,
#         "learning_rate": 0.0003,
#         "weight_decay": 5e-4,
#         "attention_loss_weight": 10.0,
#         "scheduler_patience": 5,
#         "scheduler_factor": 0.5
#     }
#
#     # 初始化数据处理器
#     data_handler = TemporalMicroServiceData(
#         config['data_root'],
#         train_ratio=config['train_ratio'],
#         val_ratio=config['val_ratio']
#     )
#
#     # 初始化训练器并开始训练
#     trainer = TemporalMicroServiceTrainer(data_handler, config)
#     trainer.train_all_windows()
#
#
# if __name__ == "__main__":
#     main()
#
