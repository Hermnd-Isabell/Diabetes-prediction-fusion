#!/usr/bin/env python3
"""
增强版主训练脚本 - 支持四个模型并包含丰富的可视化和可解释性分析

支持的模型:
- AttentionMultimodal (注意力机制)
- Baseline (ConcatFusion, EnsembleFusion)  
- TFTMultimodal (时序融合Transformer)

功能特性:
- 多模型训练和对比
- 丰富的可视化展示
- 可解释性分析
- 性能指标跟踪
- 模型保存和加载
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import warnings

# 导入数据集和训练器
from datasets.raman_dataset import RamanDataset, collate_fn, preprocess_spectrum
from datasets.embedding_dataset import EmbeddingMultimodalDataset, embedding_collate_fn
from trainers.enhanced_trainer import EnhancedTrainer, compare_models

# 导入所有模型
from models.Baseline import SpectraEncoder, TabularEncoder, ConcatFusion, EnsembleFusion
from models.attention_models import AttentionMultimodal
from models.tft_models import TFTMultimodal
from models.enhanced_mmtm_models import EnhancedMMTMFusion

# 忽略警告
warnings.filterwarnings('ignore')


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, tab_dim: int, spec_len: int) -> torch.nn.Module:
    """
    根据配置构建模型
    
    Args:
        cfg: 配置字典
        tab_dim: 表格特征维度
        spec_len: 光谱长度
    
    Returns:
        构建的模型
    """
    model_name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]
    
    print(f"[BUILD] 构建模型: {model_name}")

    # 根据当前是否使用 embedding 模式，确定各模态 embedding 维度
    use_embedding = cfg.get("data", {}).get("use_embedding", False)
    if use_embedding:
        # 在 embedding 模式下，直接使用数据集中实际的 embedding 维度
        spec_emb_dim = spec_len
        tab_emb_dim = tab_dim
    else:
        # 在 raw 模式下，使用配置中的默认 embedding 维度
        spec_emb_dim = cfg["model"].get("spec_emb", 256)
        tab_emb_dim = cfg["model"].get("tab_emb", 128)
    
    if model_name == "Spectra-only":
        return SpectraEncoder(input_dim=spec_len, hidden_dim=256)
    
    elif model_name == "Clinical-only":
        return TabularEncoder(input_dim=tab_dim, hidden_dim=128)
    
    elif model_name == "ConcatFusion":
        return ConcatFusion(spec_dim=spec_emb_dim, clin_dim=tab_emb_dim)
    
    elif model_name == "EnsembleFusion":
        return EnsembleFusion(spec_dim=spec_emb_dim, clin_dim=tab_emb_dim)
    
    elif model_name == "BaselineMultimodal":
        # BaselineMultimodal 是一个包装类，需要指定 fusion_type
        from models.Baseline import BaselineMultimodal
        fusion_type = cfg["model"].get("fusion_type", "concat")
        return BaselineMultimodal(
            spec_embedding_dim=spec_emb_dim,
            tab_embedding_dim=tab_emb_dim,
            num_classes=num_classes,
            fusion_type=fusion_type
        )
    
    elif model_name == "AttentionMultimodal":
        # 兼容旧配置中的 fusion 取值
        fusion_cfg = cfg["model"].get("fusion", "enhanced_cross")
        fusion_map = {
            "cross": "enhanced_cross",
            "enhanced_cross": "enhanced_cross",
            "concat": "concat"
        }
        fusion_type = fusion_map.get(fusion_cfg, fusion_cfg)

        return AttentionMultimodal(
            spec_embedding_dim=spec_emb_dim,
            tab_embedding_dim=tab_emb_dim,
            num_classes=num_classes,
            fusion_type=fusion_type,
            tab_dim=tab_dim
        )
    
    elif model_name == "TFTMultimodal":
        return TFTMultimodal(
            tab_dim=tab_dim,
            spec_len=spec_len,
            spec_emb=spec_emb_dim,
            tab_emb=tab_emb_dim,
            num_classes=num_classes
        )
    
    
    elif model_name == "EnhancedMMTM":
        return EnhancedMMTMFusion(
            spec_embedding_dim=spec_emb_dim,
            tab_embedding_dim=tab_emb_dim,
            num_classes=num_classes,
            mmtm_bottleneck=cfg["model"].get("mmtm_bottleneck", 128),
            num_attention_heads=cfg["model"].get("num_attention_heads", 8),
            fusion_strategy=cfg["model"].get("fusion_strategy", "hierarchical"),
            enable_uncertainty=cfg["model"].get("enable_uncertainty", True)
        )
    
    else:
        raise ValueError(f"[ERROR] 未知模型名称: {model_name}")


def prepare_data(cfg: dict) -> tuple:
    """
    准备数据集
    
    Args:
        cfg: 配置字典
    
    Returns:
        (train_loader, val_loader, test_loader, dataset_info)
    """
    print("[DATA] 准备数据集...")
    
    use_embedding = cfg["data"].get("use_embedding", False)

    if use_embedding:
        # -----------------------------
        # embedding 模式：从 CSV 加载已对齐的单模态 embedding
        # -----------------------------
        from multimodal.embedding_loader import (
            load_spectrum_embedding,
            load_clinical_embedding,
            align_by_patient_id,
        )

        spec_path = cfg["data"]["spectrum_embedding_path"]
        clin_path = cfg["data"]["clinical_embedding_path"]

        print(f"[DATA] 使用 embedding 模式加载数据")
        print(f"   - 光谱 embedding: {spec_path}")
        print(f"   - 临床 embedding: {clin_path}")

        spectrum_dict = load_spectrum_embedding(spec_path)
        clinical_dict = load_clinical_embedding(clin_path)
        aligned = align_by_patient_id(spectrum_dict, clinical_dict)

        # 获取模态 dropout 配置（仅在训练集使用）
        dropout_cfg = cfg["train"].get("modality_dropout", {"spectra": 0.0, "clinical": 0.0})
        
        # 基于 split 字段构建三个 Dataset
        # 训练集可以使用 dropout，验证/测试集严禁 dropout
        train_set = EmbeddingMultimodalDataset(
            aligned, split="train", dropout_config=dropout_cfg
        )
        val_set = EmbeddingMultimodalDataset(
            aligned, split="val", dropout_config={"spectra": 0.0, "clinical": 0.0}
        )
        test_set = EmbeddingMultimodalDataset(
            aligned, split="test", dropout_config={"spectra": 0.0, "clinical": 0.0}
        )
        
        # 打印 dropout 配置信息
        if dropout_cfg.get("spectra", 0.0) > 0 or dropout_cfg.get("clinical", 0.0) > 0:
            print(f"[DATA] 模态 Dropout 配置:")
            print(f"   - 光谱 dropout 概率: {dropout_cfg.get('spectra', 0.0):.2f}")
            print(f"   - 临床 dropout 概率: {dropout_cfg.get('clinical', 0.0):.2f}")
            print(f"   - 注意: Dropout 仅在训练集生效，验证/测试集不使用")

        print(f"[DATA] Embedding 数据划分: 训练={len(train_set)}, 验证={len(val_set)}, 测试={len(test_set)}")

        batch_size = cfg["train"]["batch_size"]
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, collate_fn=embedding_collate_fn
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, collate_fn=embedding_collate_fn
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, collate_fn=embedding_collate_fn
        )

        # 从任意非空子集推断维度信息
        ref_items = (
            train_set.items if len(train_set) > 0
            else (val_set.items if len(val_set) > 0 else test_set.items)
        )
        if not ref_items:
            raise ValueError("对齐后的 embedding 数据为空，无法构建数据集")

        tab_dim = ref_items[0]["tabular"].shape[0]
        spec_len = ref_items[0]["spectra"].shape[0]
        labels_all = [it["label"] for it in ref_items]

        dataset_info = {
            "tab_dim": tab_dim,
            "spec_len": spec_len,
            "num_classes": len(set(labels_all)),
            "class_distribution": pd.Series(labels_all).value_counts().to_dict(),
        }

        print(f"[DATA] Embedding 数据集信息:")
        print(f"   - 表格特征维度 (Dc): {dataset_info['tab_dim']}")
        print(f"   - 光谱特征维度 (Ds): {dataset_info['spec_len']}")
        print(f"   - 类别数: {dataset_info['num_classes']}")
        print(f"   - 类别分布: {dataset_info['class_distribution']}")

        return train_loader, val_loader, test_loader, dataset_info

    # -----------------------------
    # 原始 raw 模式：RamanDataset + 光谱序列
    # -----------------------------
    # 数据路径
    spectra_csv = cfg["data"]["spectra_csv"]
    clinical_csv = cfg["data"]["clinical_csv"]
    
    # 读取光谱数据获取波长列
    spectra_df = pd.read_csv(spectra_csv, sep=None, engine="python")
    wave_cols = [c for c in spectra_df.columns if c not in ["Sample", "Group"]]
    
    # 创建数据集
    dataset = RamanDataset(
        spectra_csv=spectra_csv,
        clinical_csv=clinical_csv,
        wave_cols=wave_cols,
        label_col=cfg["data"].get("label_col", "Group"),
        preprocess_fn=preprocess_spectrum,
        min_scans=1,
        max_scans=cfg["data"].get("max_scans", 180),
    )
    
    print(f"[OK] 数据集加载完成: {len(dataset)} 个样本")
    
    # 数据划分
    train_ratio = cfg["data"].get("train_ratio", 0.7)
    val_ratio = cfg["data"].get("val_ratio", 0.15)
    test_ratio = cfg["data"].get("test_ratio", 0.15)
    
    # 确保比例和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"[DATA] 数据划分: 训练={len(train_set)}, 验证={len(val_set)}, 测试={len(test_set)}")
    
    # 创建数据加载器
    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # 数据集信息
    dataset_info = {
        'tab_dim': dataset.items[0]["tabular"].shape[0],
        'spec_len': len(wave_cols),
        'num_classes': len(set(item["label"] for item in dataset.items)),
        'class_distribution': pd.Series([item["label"] for item in dataset.items]).value_counts().to_dict()
    }
    
    print(f"[DATA] 数据集信息:")
    print(f"   - 表格特征维度: {dataset_info['tab_dim']}")
    print(f"   - 光谱长度: {dataset_info['spec_len']}")
    print(f"   - 类别数: {dataset_info['num_classes']}")
    print(f"   - 类别分布: {dataset_info['class_distribution']}")
    
    return train_loader, val_loader, test_loader, dataset_info


def train_single_model(
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dataset_info: dict,
    model_name: str = None
) -> EnhancedTrainer:
    """
    训练单个模型
    
    Args:
        cfg: 配置字典
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
        cfg["model"]["name"] = model_name

    # 确保模型的类别数与数据集一致（避免 config 中 num_classes 配错）
    if "num_classes" in dataset_info:
        cfg.setdefault("model", {})
        cfg["model"]["num_classes"] = int(dataset_info["num_classes"])
    
    # 构建模型
    model = build_model(cfg, dataset_info['tab_dim'], dataset_info['spec_len'])
    
    # 创建训练器（确保超参数为数值类型）
    lr_value = float(cfg["train"].get("lr", 1e-3))
    wd_raw = cfg["train"].get("weight_decay", 1e-4)
    weight_decay_value = float(wd_raw) if wd_raw is not None else 0.0
    use_embedding_input = cfg.get("data", {}).get("use_embedding", False)

    trainer = EnhancedTrainer(
        model=model,
        model_name=cfg["model"]["name"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lr=lr_value,
        weight_decay=weight_decay_value,
        save_dir=cfg["train"].get("save_dir", "results"),
        enable_visualization=cfg.get("visualization", {}).get("enable", True),
        enable_interpretability=cfg.get("interpretability", {}).get("enable", True),
        use_embedding_input=use_embedding_input,
    )
    
    # 打印模型信息
    model_summary = trainer.get_model_summary()
    print(f"\n[MODEL] 模型摘要:")
    print(f"   - 模型名称: {model_summary['model_name']}")
    print(f"   - 总参数数: {model_summary['total_parameters']:,}")
    print(f"   - 可训练参数: {model_summary['trainable_parameters']:,}")
    print(f"   - 模型大小: {model_summary['model_size_mb']:.2f} MB")
    print(f"   - 设备: {model_summary['device']}")
    
    # 训练模型
    training_result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["train"]["epochs"],
        early_stopping_patience=cfg["train"].get("early_stopping_patience", 10),
        save_best=True
    )
    
    # 测试模型
    print(f"\n[TEST] 测试 {cfg['model']['name']}...")
    test_result = trainer.evaluate(test_loader, generate_plots=True)
    
    # 保存详细结果（保持向后兼容）
    results_path = trainer.save_dir / "results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        import json
        json.dump({
            'model_name': cfg["model"]["name"],
            'training_result': training_result,
            'test_result': {
                'metrics': test_result['metrics'],
                'classification_report': test_result['classification_report']
            },
            'model_summary': model_summary
        }, f, indent=2, ensure_ascii=False)
    
    # 保存统一格式的指标摘要（供实验汇总脚本使用）
    metrics_summary = {
        'model_name': cfg["model"]["name"],
        'best_val_auc': training_result.get('best_val_auc', None),
        'best_epoch': training_result.get('best_epoch', None),
        'total_time': training_result.get('total_time', None),
        'final_val_auc': training_result.get('val_history', {}).get('auc', [None])[-1] if training_result.get('val_history', {}).get('auc') else None,
        'final_val_acc': training_result.get('val_history', {}).get('acc', [None])[-1] if training_result.get('val_history', {}).get('acc') else None,
        'final_val_f1': training_result.get('val_history', {}).get('f1', [None])[-1] if training_result.get('val_history', {}).get('f1') else None,
        'test_auc': test_result['metrics'].get('auc', None),
        'test_acc': test_result['metrics'].get('acc', None),
        'test_f1': test_result['metrics'].get('f1', None),
        'test_sensitivity@90%spec': test_result['metrics'].get('sensitivity@90%spec', None),
    }
    
    metrics_summary_path = trainer.save_dir / "metrics_summary.json"
    with open(metrics_summary_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] {cfg['model']['name']} 训练完成!")
    print(f"[RESULT] 测试AUC: {test_result['metrics']['auc']:.4f}")
    print(f"[RESULT] 测试准确率: {test_result['metrics']['acc']:.4f}")
    
    return trainer


def train_all_models(
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dataset_info: dict
) -> list:
    """
    训练所有模型并进行比较
    
    Args:
        cfg: 配置字典
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
    models_to_train = cfg.get("models_to_train", [
        "AttentionMultimodal",
        "ConcatFusion", 
        "TFTMultimodal"
    ])
    
    trainers = []
    
    for model_name in models_to_train:
        print(f"\n{'='*20} 训练 {model_name} {'='*20}")
        
        try:
            trainer = train_single_model(
                cfg=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                dataset_info=dataset_info,
                model_name=model_name
            )
            trainers.append(trainer)
            
        except Exception as e:
            print(f"[ERROR] 训练 {model_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 模型比较
    if len(trainers) > 1:
        print(f"\n🔄 开始模型比较...")
        comparison_result = compare_models(
            trainers=trainers,
            test_loader=test_loader,
            save_dir=cfg["train"].get("save_dir", "results") + "/comparison"
        )
        
        print(f"\n🏆 最佳模型: {comparison_result['best_model']}")
        print(f"[RESULT] 最佳AUC: {comparison_result['best_auc']:.4f}")
        
        # 保存比较结果
        comparison_path = Path(cfg["train"].get("save_dir", "results")) / "comparison" / "comparison_summary.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            import json
            json.dump({
                'best_model': comparison_result['best_model'],
                'best_auc': comparison_result['best_auc'],
                'metrics_comparison': comparison_result['metrics'].to_dict()
            }, f, indent=2, ensure_ascii=False)
    
    return trainers


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强版多模态模型训练脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="指定单个模型名称")
    parser.add_argument("--train-all", action="store_true", help="训练所有模型")
    parser.add_argument("--eval-only", type=str, default=None, help="仅评估指定模型")
    
    args = parser.parse_args()
    
    print("🚀 增强版多模态模型训练系统")
    print("=" * 60)
    
    # 加载配置
    cfg = load_config(args.config)
    print(f"📋 配置文件: {args.config}")
    
    # 准备数据
    train_loader, val_loader, test_loader, dataset_info = prepare_data(cfg)
    
    if args.eval_only:
        # 仅评估模式
        print(f"\n[EVAL] 仅评估模式: {args.eval_only}")
        model = build_model(cfg, dataset_info['tab_dim'], dataset_info['spec_len'])
        trainer = EnhancedTrainer(
            model=model,
            model_name=args.eval_only,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            save_dir=cfg["train"].get("save_dir", "results")
        )
        trainer.load_model()
        result = trainer.evaluate(test_loader)
        print(f"[RESULT] 评估结果: {result['metrics']}")
        
    elif args.train_all:
        # 训练所有模型
        trainers = train_all_models(cfg, train_loader, val_loader, test_loader, dataset_info)
        
    else:
        # 训练单个模型
        model_name = args.model or cfg["model"]["name"]
        trainer = train_single_model(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset_info=dataset_info,
            model_name=model_name
        )
    
    print(f"\n🎉 训练完成!")
    print(f"📁 结果保存在: {cfg['train'].get('save_dir', 'results')}")


if __name__ == "__main__":
    main()

