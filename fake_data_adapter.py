#!/usr/bin/env python3
"""
假数据适配器 - 将假数据生成器与现有训练系统结合

这个模块允许你使用现有的训练器和main函数来训练模型，
但使用假数据而不是真实数据，非常适合测试和验证。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
import argparse

# 导入现有的训练系统
from trainers.enhanced_trainer import EnhancedTrainer
from models.attention_models import AttentionMultimodal
from models.Baseline import ConcatFusion, EnsembleFusion, BaselineMultimodal
from models.tft_models import TFTMultimodal
from models.enhanced_mmtm_models import EnhancedMMTMFusion

# 导入假数据生成器
from fake_data_generator import FakeDataGenerator


def create_fake_dataset(num_samples=100, num_scans=3, num_wavelengths=1000, num_features=10, num_classes=2):
    """创建假数据集"""
    data = []
    generator = FakeDataGenerator()
    
    for i in range(num_samples):
        # 生成光谱数据
        spectra = generator.generate_spectral_data(1, num_scans, num_wavelengths)[0]  # 形状: (num_scans, num_wavelengths)
        
        # 生成表格数据
        tabular = generator.generate_tabular_data(1, num_features)[0]
        
        # 生成标签
        label = generator.generate_labels(1, num_classes)[0].item()
        
        # 生成mask（所有扫描都有效）
        mask = torch.ones(num_scans, dtype=torch.bool)
        
        data.append({
            'spectra': spectra,
            'tabular': tabular,
            'label': label,
            'mask': mask
        })
    
    return data


class FakeDataAdapter:
    """
    假数据适配器类
    
    将假数据生成器与现有的训练系统无缝结合
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化适配器
        
        Args:
            config_path: 配置文件路径（可选）
        """
        self.config_path = config_path
        self.config = self._load_config() if config_path else self._get_default_config()
        
    def _load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_path, "r", encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            "data": {
                "use_fake_data": True,
                "num_samples": 200,
                "num_scans": 3,
                "num_wavelengths": 1000,
                "num_features": 10,
                "num_classes": 2,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15
            },
            "model": {
                "name": "AttentionMultimodal",
                "num_classes": 2,
                "spec_emb": 256,
                "tab_emb": 128,
                "fusion": "enhanced_cross"
            },
            "train": {
                "batch_size": 8,
                "epochs": 20,
                "lr": 0.001,
                "weight_decay": 1e-4,
                "early_stopping_patience": 10,
                "save_dir": "fake_data_results"
            },
            "visualization": {
                "enable": True
            },
            "interpretability": {
                "enable": True
            }
        }
    
    def prepare_fake_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
        """
        准备假数据
        
        Returns:
            (train_loader, val_loader, test_loader, dataset_info)
        """
        print("📊 准备假数据...")
        
        # 获取数据配置
        data_cfg = self.config["data"]
        
        # 创建假数据集
        dataset = create_fake_dataset(
            num_samples=data_cfg["num_samples"],
            num_scans=data_cfg["num_scans"],
            num_wavelengths=data_cfg["num_wavelengths"],
            num_features=data_cfg["num_features"],
            num_classes=data_cfg["num_classes"]
        )
        
        print(f"✅ 假数据集创建完成: {len(dataset)} 个样本")
        
        # 数据划分
        train_ratio = data_cfg["train_ratio"]
        val_ratio = data_cfg["val_ratio"]
        test_ratio = data_cfg["test_ratio"]
        
        # 确保比例和为1
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"📊 数据划分: 训练={len(train_set)}, 验证={len(val_set)}, 测试={len(test_set)}")
        
        # 创建数据加载器
        batch_size = self.config["train"]["batch_size"]
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn
        )
        
        # 数据集信息
        dataset_info = {
            'tab_dim': data_cfg["num_features"],
            'spec_len': data_cfg["num_wavelengths"],
            'num_classes': data_cfg["num_classes"],
            'class_distribution': {0: len(dataset)//2, 1: len(dataset)//2}  # 假数据均匀分布
        }
        
        print(f"📊 数据集信息:")
        print(f"   • 表格特征维度: {dataset_info['tab_dim']}")
        print(f"   • 光谱长度: {dataset_info['spec_len']}")
        print(f"   • 类别数: {dataset_info['num_classes']}")
        print(f"   • 类别分布: {dataset_info['class_distribution']}")
        
        return train_loader, val_loader, test_loader, dataset_info
    
    def _collate_fn(self, batch):
        """批处理函数"""
        spectra = torch.stack([item['spectra'] for item in batch])  # 形状: (batch_size, 1, num_scans, num_wavelengths)
        spectra = spectra.squeeze(1)  # 去掉第1维，变成 (batch_size, num_scans, num_wavelengths)
        tabular = torch.stack([item['tabular'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        
        return {
            'spectra': spectra,
            'tabular': tabular,
            'label': labels,
            'mask': masks
        }
    
    def build_model(self, dataset_info: dict) -> torch.nn.Module:
        """
        构建模型
        
        Args:
            dataset_info: 数据集信息
            
        Returns:
            构建的模型
        """
        model_name = self.config["model"]["name"]
        num_classes = self.config["model"]["num_classes"]
        
        print(f"🏗️  构建模型: {model_name}")
        
        if model_name == "AttentionMultimodal":
            fusion_cfg = self.config["model"].get("fusion", "enhanced_cross")
            fusion_map = {
                "cross": "enhanced_cross",
                "enhanced_cross": "enhanced_cross",
                "concat": "concat"
            }
            fusion_type = fusion_map.get(fusion_cfg, fusion_cfg)
            
            return AttentionMultimodal(
                spec_embedding_dim=self.config["model"].get("spec_emb", 256),
                tab_embedding_dim=self.config["model"].get("tab_emb", 128),
                num_classes=num_classes,
                fusion_type=fusion_type,
                tab_dim=dataset_info['tab_dim']
            )
        
        elif model_name == "ConcatFusion":
            return ConcatFusion(
                spec_dim=self.config["model"].get("spec_emb", 256),
                clin_dim=self.config["model"].get("tab_emb", 128)
            )
        
        elif model_name == "EnsembleFusion":
            return EnsembleFusion(
                spec_dim=self.config["model"].get("spec_emb", 256),
                clin_dim=self.config["model"].get("tab_emb", 128)
            )
        
        elif model_name == "TFTMultimodal":
            return TFTMultimodal(
                tab_dim=dataset_info['tab_dim'],
                spec_len=dataset_info['spec_len'],
                spec_emb=self.config["model"].get("spec_emb", 256),
                tab_emb=self.config["model"].get("tab_emb", 128),
                num_classes=num_classes
            )
        
        elif model_name == "EnhancedMMTMFusion":
            return EnhancedMMTMFusion(
                spec_embedding_dim=self.config["model"].get("spec_emb", 256),
                tab_embedding_dim=self.config["model"].get("tab_emb", 128),
                num_classes=num_classes,
                mmtm_bottleneck=self.config["model"].get("mmtm_bottleneck", 128),
                num_attention_heads=self.config["model"].get("num_attention_heads", 8),
                fusion_strategy=self.config["model"].get("fusion_strategy", "hierarchical"),
                enable_uncertainty=self.config["model"].get("enable_uncertainty", True)
            )
        
        elif model_name == "BaselineMultimodal":
            fusion_type = self.config["model"].get("fusion_type", "concat")
            return BaselineMultimodal(
                spec_embedding_dim=self.config["model"].get("spec_emb", 256),
                tab_embedding_dim=self.config["model"].get("tab_emb", 128),
                num_classes=num_classes,
                fusion_type=fusion_type
            )
        
        else:
            raise ValueError(f"❌ 未知模型名称: {model_name}")
    
    def train_single_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        dataset_info: dict,
        model_name: str = None
    ) -> EnhancedTrainer:
        """
        训练单个模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            dataset_info: 数据集信息
            model_name: 模型名称（可选，覆盖配置中的名称）
            
        Returns:
            训练好的训练器
        """
        # 使用指定的模型名称或配置中的名称
        if model_name:
            self.config["model"]["name"] = model_name
        
        # 构建模型
        model = self.build_model(dataset_info)
        
        # 创建训练器
        trainer = EnhancedTrainer(
            model=model,
            model_name=self.config["model"]["name"],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            lr=float(self.config["train"]["lr"]),
            weight_decay=float(self.config["train"]["weight_decay"]),
            save_dir=self.config["train"]["save_dir"],
            enable_visualization=bool(self.config.get("visualization", {}).get("enable", True)),
            enable_interpretability=bool(self.config.get("interpretability", {}).get("enable", True))
        )
        
        # 打印模型信息
        model_summary = trainer.get_model_summary()
        print(f"\n📊 模型摘要:")
        print(f"   • 模型名称: {model_summary['model_name']}")
        print(f"   • 总参数数: {model_summary['total_parameters']:,}")
        print(f"   • 可训练参数: {model_summary['trainable_parameters']:,}")
        print(f"   • 模型大小: {model_summary['model_size_mb']:.2f} MB")
        print(f"   • 设备: {model_summary['device']}")
        
        # 训练模型
        training_result = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config["train"]["epochs"],
            early_stopping_patience=self.config["train"]["early_stopping_patience"],
            save_best=True
        )
        
        # 测试模型
        print(f"\n🔍 测试 {self.config['model']['name']}...")
        test_result = trainer.evaluate(test_loader, generate_plots=True)
        
        # 保存结果
        results_path = trainer.save_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            import json
            json.dump({
                'model_name': self.config["model"]["name"],
                'training_result': training_result,
                'test_result': {
                    'metrics': test_result['metrics'],
                    'classification_report': test_result['classification_report']
                },
                'model_summary': model_summary
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {self.config['model']['name']} 训练完成!")
        print(f"📊 测试AUC: {test_result['metrics']['auc']:.4f}")
        print(f"📊 测试准确率: {test_result['metrics']['acc']:.4f}")
        
        return trainer
    
    def train_all_models(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        dataset_info: dict
    ) -> list:
        """
        训练所有模型并进行比较
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            dataset_info: 数据集信息
            
        Returns:
            训练器列表
        """
        print(f"\n🚀 开始训练所有模型...")
        print("=" * 60)
        
        # 定义要训练的模型
        models_to_train = self.config.get("models_to_train", [
            "AttentionMultimodal",
            "ConcatFusion", 
            "TFTMultimodal",
            "EnhancedMMTMFusion",
            "BaselineMultimodal"
        ])
        
        trainers = []
        
        for model_name in models_to_train:
            print(f"\n{'='*20} 训练 {model_name} {'='*20}")
            
            try:
                trainer = self.train_single_model(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    dataset_info=dataset_info,
                    model_name=model_name
                )
                trainers.append(trainer)
                
            except Exception as e:
                print(f"❌ 训练 {model_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 模型比较
        if len(trainers) > 1:
            print(f"\n🔄 开始模型比较...")
            from trainers.enhanced_trainer import compare_models
            comparison_result = compare_models(
                trainers=trainers,
                test_loader=test_loader,
                save_dir=self.config["train"]["save_dir"] + "/comparison"
            )
            
            print(f"\n🏆 最佳模型: {comparison_result['best_model']}")
            print(f"📊 最佳AUC: {comparison_result['best_auc']:.4f}")
            
            # 保存比较结果
            comparison_path = Path(self.config["train"]["save_dir"]) / "comparison" / "comparison_summary.json"
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            with open(comparison_path, 'w', encoding='utf-8') as f:
                import json
                json.dump({
                    'best_model': comparison_result['best_model'],
                    'best_auc': comparison_result['best_auc'],
                    'metrics_comparison': comparison_result['metrics'].to_dict()
                }, f, indent=2, ensure_ascii=False)
        
        return trainers


def main():
    """主函数 - 使用假数据训练模型"""
    parser = argparse.ArgumentParser(description="使用假数据训练多模态模型")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（可选）")
    parser.add_argument("--model", type=str, default=None, help="指定单个模型名称")
    parser.add_argument("--train-all", action="store_true", help="训练所有模型")
    parser.add_argument("--samples", type=int, default=200, help="假数据样本数量")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    
    args = parser.parse_args()
    
    print("🚀 使用假数据训练多模态模型")
    print("=" * 60)
    
    # 创建适配器
    adapter = FakeDataAdapter(config_path=args.config)
    
    # 如果通过命令行参数指定了参数，更新配置
    if args.samples:
        adapter.config["data"]["num_samples"] = args.samples
    if args.epochs:
        adapter.config["train"]["epochs"] = args.epochs
    if args.batch_size:
        adapter.config["train"]["batch_size"] = args.batch_size
    
    print(f"📋 配置信息:")
    print(f"   • 样本数量: {adapter.config['data']['num_samples']}")
    print(f"   • 训练轮数: {adapter.config['train']['epochs']}")
    print(f"   • 批次大小: {adapter.config['train']['batch_size']}")
    print(f"   • 保存目录: {adapter.config['train']['save_dir']}")
    
    # 准备数据
    train_loader, val_loader, test_loader, dataset_info = adapter.prepare_fake_data()
    
    if args.train_all:
        # 训练所有模型
        trainers = adapter.train_all_models(train_loader, val_loader, test_loader, dataset_info)
    else:
        # 训练单个模型
        model_name = args.model or adapter.config["model"]["name"]
        trainer = adapter.train_single_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset_info=dataset_info,
            model_name=model_name
        )
    
    print(f"\n🎉 训练完成!")
    print(f"📁 结果保存在: {adapter.config['train']['save_dir']}")


if __name__ == "__main__":
    main()
