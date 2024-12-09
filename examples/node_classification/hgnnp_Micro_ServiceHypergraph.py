import torch
import torch.nn.functional as F
import torch.optim as optim
from dhg.models import HGNNP
from typing import Dict
import numpy as np
from collections import defaultdict
from copy import deepcopy
from dhg.utils import TemporalMicroServiceData

class TemporalMicroServiceTrainer:
    """处理时序微服务数据的训练和验证"""

    def __init__(self, data_handler: TemporalMicroServiceData, model_save_path: str):
        self.data_handler = data_handler
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型、超图和评估器
        self.G = self.data_handler.get_hypergraph().to(self.device)
        X, _ = self.data_handler.load_window_data(0)  # 初始化特征维度
        self.model = HGNNP(X.shape[1], 64, use_bn=True).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, verbose=True)

    def train_window(self, window_idx: int, epoch: int) -> float:
        """训练单个时间窗口的数据"""
        self.model.train()
        X, labels = self.data_handler.load_window_data(window_idx)
        X, labels = X.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()

        # 前向传播，获取预测结果和注意力权重
        root_cause_score, attention_weights = self.model(X, self.G)

        # 计算损失
        pred_loss = F.binary_cross_entropy(root_cause_score.squeeze(), labels.float())
        attention_target = labels.float().unsqueeze(1)
        attention_loss = F.mse_loss(attention_weights, attention_target)
        total_loss = pred_loss + 10.0 * attention_loss

        # 反向传播
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    @torch.no_grad()
    def infer(self, window_idx: int, idx_type: str = 'val') -> Dict[str, float]:
        """推理单个时间窗口的数据，计算top-k准确率"""
        self.model.eval()
        X, lbls = self.data_handler.load_window_data(window_idx)
        X, lbls = X.to(self.device), lbls.to(self.device)

        # 获取根因预测结果和注意力权重
        root_cause_score, attention_weights = self.model(X, self.G)
        root_cause_preds = root_cause_score.squeeze()

        # 找到真实的根因索引
        true_root_cause = torch.where(lbls == 1)[0]

        # 计算不同k值的准确率
        metrics = {}
        for k in [1, 3, 5]:
            _, top_k_indices = torch.topk(root_cause_preds, k, dim=0)
            hit = any(idx.item() in top_k_indices.tolist() for idx in true_root_cause)
            metrics[f'top{k}_acc'] = float(hit)

        if idx_type == 'test':
            print(f"\nWindow {window_idx} Results:")
            print(f"Root Cause Probabilities: {root_cause_preds.cpu().numpy()}")
            print(f"True Root Cause Index: {true_root_cause.cpu().numpy()}")
            print("Top-K Accuracies:", {k: v for k, v in metrics.items()})

        return metrics

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
    data_root = "../../datasets/test"
    model_save_path = "../../model"

    data_handler = TemporalMicroServiceData(data_root, train_ratio=0.6, val_ratio=0.2)
    trainer = TemporalMicroServiceTrainer(data_handler, model_save_path)

    trainer.train_all_windows(epochs=200)

if __name__ == "__main__":
    main()