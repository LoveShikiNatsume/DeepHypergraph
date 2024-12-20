#
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
#         "hidden_dim": 64,
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
import torch
import torch.nn.functional as F
import torch.optim as optim
from dhg.models import HGNNP
from typing import Dict, List
import numpy as np
from collections import defaultdict
from copy import deepcopy
from dhg.utils import TemporalMicroServiceData
from dhg.utils import ExperimentSaver


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
            verbose=True
        )

    def train_window(self, window_idx: int, epoch: int) -> float:
        """训练单个时间窗口的数据"""
        self.model.train()

        # 获取当前时间窗口的特征、标签和超图
        X, labels = self.data_handler.load_window_data(window_idx)
        X, labels = X.to(self.device), labels.to(self.device)
        G = self.data_handler.get_hypergraph(window_idx).to(self.device)

        self.optimizer.zero_grad()
        root_cause_score, attention_weights = self.model(X, G)

        pred_loss = F.binary_cross_entropy(root_cause_score.squeeze(), labels.float())
        attention_target = labels.float().unsqueeze(1)
        attention_loss = F.mse_loss(attention_weights, attention_target)
        total_loss = pred_loss + self.config['attention_loss_weight'] * attention_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    @torch.no_grad()
    def infer(self, window_idx: int, idx_type: str = 'val') -> Dict[str, float]:
        """推理单个时间窗口的数据，计算top-k准确率"""
        self.model.eval()

        # 获取当前时间窗口的特征、标签和超图
        X, lbls = self.data_handler.load_window_data(window_idx)
        X, lbls = X.to(self.device), lbls.to(self.device)
        G = self.data_handler.get_hypergraph(window_idx).to(self.device)

        root_cause_score, attention_weights = self.model(X, G)
        root_cause_preds = root_cause_score.squeeze()

        true_root_cause = torch.where(lbls == 1)[0]

        metrics = {}
        for k in [1, 3, 5]:
            _, top_k_indices = torch.topk(root_cause_preds, k, dim=0)
            hit = any(idx.item() in top_k_indices.tolist() for idx in true_root_cause)
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
        "hidden_dim": 32,
        "use_bn": True,
        "drop_rate": 0.5,

        # 训练配置
        "epochs": 200,
        "learning_rate": 0.0003,
        "weight_decay": 5e-4,
        "attention_loss_weight": 10.0,
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