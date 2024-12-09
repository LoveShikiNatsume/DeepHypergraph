import os
import json
import torch
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


class ExperimentSaver:
    """实验结果保存工具类"""

    def __init__(self, base_save_path: str):
        """
        Args:
            base_save_path: 基础保存路径
        """
        self.base_save_path = base_save_path
        os.makedirs(base_save_path, exist_ok=True)
        self.current_save_dir = None

    def initialize_experiment(self) -> str:
        """为新的实验创建保存目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_save_dir = os.path.join(self.base_save_path, timestamp)
        os.makedirs(self.current_save_dir, exist_ok=True)
        return self.current_save_dir

    def save_best_model(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """保存训练过程中的最优模型"""
        if self.current_save_dir is None:
            self.initialize_experiment()

        model_path = os.path.join(self.current_save_dir, "best_model.pth")
        torch.save(state_dict, model_path)

    def save_experiment_results(self,
                                best_state_dict: Dict[str, torch.Tensor],
                                final_metrics: Dict[str, Any],
                                best_epoch: int,
                                config: Dict[str, Any],
                                train_indices: list,
                                val_indices: list,
                                test_indices: list) -> None:
        """保存实验的最终结果

        Args:
            best_state_dict: 最优模型状态字典
            final_metrics: 最终测试指标
            best_epoch: 最优模型对应的epoch
            config: 实验配置参数
            train_indices: 训练集索引
            val_indices: 验证集索引
            test_indices: 测试集索引
        """
        # 确保实验目录已创建
        if self.current_save_dir is None:
            self.initialize_experiment()

        # 保存最优模型
        model_path = os.path.join(self.current_save_dir, "best_model.pth")
        torch.save(best_state_dict, model_path)

        # 保存数据集划分信息
        split_info = {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "split_config": {
                "train_ratio": config['train_ratio'],
                "val_ratio": config['val_ratio'],
                "test_ratio": 1 - config['train_ratio'] - config['val_ratio']
            }
        }

        # 准备完整的实验信息
        experiment_info = {
            "timestamp": os.path.basename(self.current_save_dir),
            "best_epoch": best_epoch,
            "final_metrics": final_metrics,
            "config": config,
            "dataset_split": split_info
        }

        # 保存实验信息
        info_path = os.path.join(self.current_save_dir, "experiment_info.json")
        with open(info_path, 'w') as f:
            json.dump(experiment_info, f, indent=4)

        print(f"\n实验结果已保存到: {self.current_save_dir}")
        print("包含以下文件:")
        print(" - best_model.pth (最优模型)")
        print(" - experiment_info.json (实验配置和结果)")