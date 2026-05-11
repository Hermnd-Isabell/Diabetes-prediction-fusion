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
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
import copy
import sys
from typing import Optional

class DualLogger(object):
    """
    Simultaneous writing to console and log file
    """
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.terminal_is_tty = hasattr(stream, 'isatty') and stream.isatty()
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        if self.terminal_is_tty:
            try:
                self.terminal.write(message)
            except (OSError, ValueError):
                pass
        self.log.write(message)
        self.log.flush()  # Ensure real-time writing

    def flush(self):
        if self.terminal_is_tty:
            try:
                self.terminal.flush()
            except (OSError, ValueError):
                pass
        self.log.flush()

    def close(self):
        self.log.close()


# 导入数据集和训练器
from datasets.raman_dataset import RamanDataset, collate_fn, preprocess_spectrum
from datasets.embedding_dataset import EmbeddingMultimodalDataset, embedding_collate_fn
from trainers.enhanced_trainer import EnhancedTrainer, compare_models

# 导入所有模型
from models.Baseline import SpectraEncoder, TabularEncoder, ConcatFusion, EnsembleFusion, SpectraOnlyModel, TabularOnlyModel
from models.attention_models import AttentionMultimodal
from models.tft_models import TFTMultimodal
from models.enhanced_mmtm_models import EnhancedMMTMFusion

# 忽略警告
warnings.filterwarnings('ignore')


def merge_dicts(base: dict, override: dict) -> dict:
    """
    递归合并两个字典，override 中的值覆盖 base 中的值。
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> dict:
    """
    加载配置文件，支持统一基础配置继承。

    若配置文件包含 ``defaults: [experiment_base]``（或类似列表），
    则先加载 ``configs/experiment_base.yaml``，再用当前文件递归覆盖。
    若不含 ``defaults`` 键（旧格式单文件），则直接返回该文件内容。
    """
    with open(config_path, "r", encoding='utf-8') as f:
        specific_cfg = yaml.safe_load(f)

    if specific_cfg is None:
        specific_cfg = {}

    # 检测是否声明了 base 继承
    defaults = specific_cfg.pop("defaults", None)
    if defaults is not None:
        # 支持 formats: "experiment_base", ["experiment_base"], [{...}]
        base_name = None
        if isinstance(defaults, str):
            base_name = defaults
        elif isinstance(defaults, list) and len(defaults) > 0:
            first = defaults[0]
            if isinstance(first, str):
                base_name = first
            elif isinstance(first, dict):
                # hydra-style _target_ skip; take first string
                for item in defaults:
                    if isinstance(item, str):
                        base_name = item
                        break
        if base_name:
            base_path = Path(config_path).parent / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path, "r", encoding='utf-8') as bf:
                    base_cfg = yaml.safe_load(bf)
                merged = merge_dicts(base_cfg or {}, specific_cfg)
                print(f"[CONFIG] 合并基础配置: {base_path.name} -> {Path(config_path).name}")
                return merged
            else:
                print(f"[WARN] 基础配置未找到: {base_path}，将直接加载当前配置")

    return specific_cfg


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
    lite_cfg = cfg["model"].get("lite", {"enabled": False})

    print(f"[BUILD] 构建模型: {model_name}")
    if lite_cfg.get("enabled", False):
        print(f"[BUILD] 轻量化模式启用: {lite_cfg}")

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
        # 使用独立的 SpectraOnlyModel
        return SpectraOnlyModel(input_dim=spec_len, num_classes=num_classes, hidden_dim=spec_emb_dim, lite_cfg=lite_cfg)

    elif model_name == "Clinical-only":
        # 使用独立的 TabularOnlyModel
        return TabularOnlyModel(input_dim=tab_dim, num_classes=num_classes, hidden_dim=tab_emb_dim, lite_cfg=lite_cfg)

    elif model_name == "ConcatFusion":
        return ConcatFusion(spec_dim=spec_emb_dim, clin_dim=tab_emb_dim, num_classes=num_classes, lite_cfg=lite_cfg)

    elif model_name == "EnsembleFusion":
        return EnsembleFusion(spec_dim=spec_emb_dim, clin_dim=tab_emb_dim, num_classes=num_classes, lite_cfg=lite_cfg)

    elif model_name == "BaselineMultimodal":
        # BaselineMultimodal 是一个包装类，需要指定 fusion_type
        from models.Baseline import BaselineMultimodal
        fusion_type = cfg["model"].get("fusion_type", "concat")
        return BaselineMultimodal(
            spec_embedding_dim=spec_emb_dim,
            tab_embedding_dim=tab_emb_dim,
            num_classes=num_classes,
            fusion_type=fusion_type,
            lite_cfg=lite_cfg
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
            tab_dim=tab_dim,
            hidden_dims=cfg["model"].get("hidden_dims", [512, 256, 128]),
            num_heads=cfg["model"].get("num_attention_heads", 8),
            lite_cfg=lite_cfg
        )

    elif model_name == "TFTMultimodal":
        return TFTMultimodal(
            tab_dim=tab_dim,
            spec_len=spec_len,
            spec_emb=spec_emb_dim,
            tab_emb=tab_emb_dim,
            num_classes=num_classes,
            dropout=cfg["model"].get("dropout", 0.1),
            lite_cfg=lite_cfg
        )

    elif model_name == "EnhancedMMTM":
        return EnhancedMMTMFusion(
            spec_embedding_dim=spec_emb_dim,
            tab_embedding_dim=tab_emb_dim,
            num_classes=num_classes,
            mmtm_bottleneck=cfg["model"].get("mmtm_bottleneck", 128),
            num_attention_heads=cfg["model"].get("num_attention_heads", 8),
            fusion_strategy=cfg["model"].get("fusion_strategy", "hierarchical"),
            enable_uncertainty=cfg["model"].get("enable_uncertainty", True),
            tab_input_dim=tab_dim,
            lite_cfg=lite_cfg
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

        # Embedding 信息泄漏审计
        embedding_audit = cfg.get("data", {}).get("embedding", {})
        generation_scope = embedding_audit.get("generation_scope", None)
        generation_log = embedding_audit.get("generation_log", None)

        if generation_scope == "full_dataset":
            print(
                "WARNING: cfg.data.embedding.generation_scope='full_dataset'. "
                "Pre-computed embeddings were likely fit on the full dataset (including test set), "
                "which may cause information leakage. "
                "Consider regenerating embeddings with train-set-only fit and transform on all data."
            )
        elif generation_scope == "train_only":
            print("INFO: Embedding audit passed: declared as train-only fit.")
        else:
            if not generation_log:
                print(
                    "WARNING: Embedding generation scope is not documented. "
                    "Please set data.embedding.generation_scope to 'train_only' or 'full_dataset' for reproducibility."
                )

        # 基于 split 字段构建三个 Dataset
        # 训练集可以使用 dropout，验证/测试集严禁 dropout
        train_set = EmbeddingMultimodalDataset(
            aligned, split="train", dropout_config=dropout_cfg, embedding_audit=embedding_audit
        )
        val_set = EmbeddingMultimodalDataset(
            aligned, split="val", dropout_config={"spectra": 0.0, "clinical": 0.0}, embedding_audit=embedding_audit
        )
        test_set = EmbeddingMultimodalDataset(
            aligned, split="test", dropout_config={"spectra": 0.0, "clinical": 0.0}, embedding_audit=embedding_audit
        )
        
        # 打印 dropout 配置信息
        if dropout_cfg.get("spectra", 0.0) > 0 or dropout_cfg.get("clinical", 0.0) > 0:
            print(f"[DATA] 模态 Dropout 配置:")
            print(f"   - 光谱 dropout 概率: {dropout_cfg.get('spectra', 0.0):.2f}")
            print(f"   - 临床 dropout 概率: {dropout_cfg.get('clinical', 0.0):.2f}")
            print(f"   - 注意: Dropout 仅在训练集生效，验证/测试集不使用")

        print(f"[DATA] Embedding 数据划分: 训练={len(train_set)}, 验证={len(val_set)}, 测试={len(test_set)}")

        batch_size = cfg["train"]["batch_size"]
        advanced_cfg = cfg["train"].get("advanced", None)
        if advanced_cfg and advanced_cfg.get("enabled", False) and batch_size > 8:
            print(f"INFO: Advanced training enabled. Consider using batch_size <= 8 for 142-sample dataset to increase update steps per epoch.")
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

    # 读取新的预处理与聚合配置（向后兼容：若配置缺失则使用默认值）
    preprocess_cfg = cfg.get("data", {}).get("preprocessing", None)
    normalization_method = cfg.get("data", {}).get("preprocessing", {}).get("normalization", {}).get("method", "SNV")
    scan_aggregation = cfg.get("data", {}).get("scan_aggregation", "sequence")

    # 创建数据集
    dataset = RamanDataset(
        spectra_csv=spectra_csv,
        clinical_csv=clinical_csv,
        wave_cols=wave_cols,
        label_col=cfg["data"].get("label_col", "Group"),
        preprocess_fn=preprocess_spectrum,   # 向后兼容：当 preprocess_cfg 为 None 时使用
        preprocess_cfg=preprocess_cfg,       # 新的配置驱动预处理
        normalization_method=normalization_method,
        scan_aggregation=scan_aggregation,
        min_scans=1,
        max_scans=cfg["data"].get("max_scans", 180),
    )
    
    print(f"[OK] 数据集加载完成: {len(dataset)} 个样本")

    if scan_aggregation == "stats":
        print(
            "[WARN] scan_aggregation='stats' 将多扫描聚合为 [3*L] 的统计向量。"
            "AttentionMultimodal / TFTMultimodal 等依赖 scan-level 序列的模型可能无法正常工作。"
            "建议在此模式下使用 Spectra-only / Clinical-only / ConcatFusion 等基线模型。"
        )

    # 数据划分（向后兼容两种格式）
    ratios = cfg["data"].get("train_val_test_ratio", None)
    if ratios is not None and isinstance(ratios, (list, tuple)) and len(ratios) == 3:
        train_ratio, val_ratio, test_ratio = ratios
    else:
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
    
    split_seed = cfg.get("experiment", {}).get("random_seed", 42)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(split_seed)
    )

    # 启用数据增强 (仅针对训练集)
    if cfg["train"].get("use_augmentation", False):
        aug_params = cfg["train"].get("augmentation_params", {})
        noise = aug_params.get("noise", 0.01)
        scale = aug_params.get("scale", 0.1)
        
        print(f"[DATA] 启用训练集数据增强 (Noise={noise}, Scale={scale})")
        train_ds_copy = copy.deepcopy(dataset)
        train_ds_copy.augment = True
        train_ds_copy.aug_noise = noise
        train_ds_copy.aug_scale = scale
        train_set.dataset = train_ds_copy
    
    print(f"[DATA] 数据划分: 训练={len(train_set)}, 验证={len(val_set)}, 测试={len(test_set)}")
    
    # 创建数据加载器
    batch_size = cfg["train"]["batch_size"]
    advanced_cfg = cfg["train"].get("advanced", None)
    if advanced_cfg and advanced_cfg.get("enabled", False) and batch_size > 8:
        print(f"INFO: Advanced training enabled. Consider using batch_size <= 8 for 142-sample dataset to increase update steps per epoch.")
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
    num_wavelengths = len(wave_cols)
    if scan_aggregation == "stats":
        spec_len = num_wavelengths * 3   # stats 模式下输入展平为 mean+std+max
    else:
        spec_len = num_wavelengths

    dataset_info = {
        'tab_dim': dataset.items[0]["tabular"].shape[0],
        'spec_len': spec_len,
        'num_wavelengths': num_wavelengths,
        'num_classes': len(set(item["label"] for item in dataset.items)),
        'class_distribution': pd.Series([item["label"] for item in dataset.items]).value_counts().to_dict(),
        'scan_aggregation': scan_aggregation,
        'normalization_method': normalization_method,
    }

    print(f"[DATA] 数据集信息:")
    print(f"   - 表格特征维度: {dataset_info['tab_dim']}")
    print(f"   - 光谱波长数: {dataset_info['num_wavelengths']}")
    if scan_aggregation == "stats":
        print(f"   - 光谱输入维度 (stats 聚合): {dataset_info['spec_len']} (= 3 x {num_wavelengths})")
    else:
        print(f"   - 光谱序列长度: {dataset_info['spec_len']}")
    print(f"   - 归一化方式: {normalization_method}")
    print(f"   - 扫描聚合方式: {scan_aggregation}")
    print(f"   - 类别数: {dataset_info['num_classes']}")
    print(f"   - 类别分布: {dataset_info['class_distribution']}")

    return train_loader, val_loader, test_loader, dataset_info


def load_full_dataset_raw(cfg: dict):
    """
    返回完整的数据集（不分 train/val/test），用于 CV 外层划分。
    支持 raw 模式和 embedding 模式。
    """
    if cfg.get("data", {}).get("use_embedding", False):
        # Embedding 模式：合并 train/val/test 的 EmbeddingMultimodalDataset
        from multimodal.embedding_loader import load_spectrum_embedding, load_clinical_embedding, align_by_patient_id
        from datasets.embedding_dataset import EmbeddingMultimodalDataset, embedding_collate_fn
        from torch.utils.data import ConcatDataset

        spectrum_path = cfg["data"]["spectrum_embedding_path"]
        clinical_path = cfg["data"]["clinical_embedding_path"]
        spectrum_dict = load_spectrum_embedding(spectrum_path)
        clinical_dict = load_clinical_embedding(clinical_path)
        aligned = align_by_patient_id(spectrum_dict, clinical_dict)

        subsets = []
        for split in ["train", "val", "test"]:
            ds = EmbeddingMultimodalDataset(aligned, split=split, dropout_config={"spectra": 0.0, "clinical": 0.0})
            if len(ds) > 0:
                subsets.append(ds)

        if not subsets:
            print("[ERROR] Embedding 模式 CV：没有任何样本。")
            return None, None

        dataset = ConcatDataset(subsets)
        dataset.collate_fn = embedding_collate_fn
        labels = np.array([dataset[i]["label"] for i in range(len(dataset))])
        print(f"[CV] 加载完整 embedding 数据集: {len(dataset)} 个样本，类别分布: {dict(pd.Series(labels).value_counts().sort_index())}")
        return dataset, labels

    # Raw 模式
    spectra_csv = cfg["data"]["spectra_csv"]
    clinical_csv = cfg["data"]["clinical_csv"]
    spectra_df = pd.read_csv(spectra_csv, sep=None, engine="python")
    wave_cols = [c for c in spectra_df.columns if c not in ["Sample", "Group"]]

    preprocess_cfg = cfg.get("data", {}).get("preprocessing", None)
    normalization_method = cfg.get("data", {}).get("preprocessing", {}).get("normalization", {}).get("method", "SNV")
    scan_aggregation = cfg.get("data", {}).get("scan_aggregation", "sequence")

    dataset = RamanDataset(
        spectra_csv=spectra_csv,
        clinical_csv=clinical_csv,
        wave_cols=wave_cols,
        label_col=cfg["data"].get("label_col", "Group"),
        preprocess_fn=preprocess_spectrum,
        preprocess_cfg=preprocess_cfg,
        normalization_method=normalization_method,
        scan_aggregation=scan_aggregation,
        min_scans=1,
        max_scans=cfg["data"].get("max_scans", 180),
    )

    labels = np.array([dataset[i]["label"] for i in range(len(dataset))])
    print(f"[CV] 加载完整数据集: {len(dataset)} 个样本，类别分布: {dict(pd.Series(labels).value_counts().sort_index())}")
    return dataset, labels


def get_cv_loaders(
    dataset,
    labels: np.ndarray,
    fold_idx: int,
    n_splits: int = 5,
    inner_val_ratio: float = 0.15,
    batch_size: int = 8,
    random_state: int = 42,
):
    """
    为指定 fold 生成 train/val/test DataLoader。
    外层：StratifiedKFold 划分 train_val / test
    内层：train_test_split 划分 train / val（同样分层）
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_indices = np.arange(len(labels))
    splits = list(skf.split(all_indices, labels))
    train_val_idx, test_idx = splits[fold_idx]

    try:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=inner_val_ratio,
            stratify=labels[train_val_idx],
            random_state=random_state + fold_idx,
        )
    except ValueError:
        # 某些 fold 中稀有类别样本数不足，无法进行分层切分，回退到普通切分
        print(f"[WARN] Fold {fold_idx} inner split: 分层切分失败（某类样本数<2），回退到随机切分。")
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=inner_val_ratio,
            random_state=random_state + fold_idx,
        )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    collate = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx)


def run_cross_validation(cfg: dict, model_name: str) -> dict:
    """
    执行 5-Fold Stratified CV（仅支持 raw 模式）。
    返回 cv_summary 字典，并在 results/<model_name>/ 下保存 cv_summary.json 和 fold_results.csv。
    """
    dataset, labels = load_full_dataset_raw(cfg)
    if dataset is None:
        return None

    cv_cfg = cfg.get("evaluation", {}).get("cross_validation", {})
    n_splits = cv_cfg.get("n_splits", 5)
    inner_val_ratio = cv_cfg.get("inner_val_ratio", 0.15)
    random_state = cv_cfg.get("random_state", 42)
    batch_size = cfg["train"]["batch_size"]

    # 推断 dataset_info（与 prepare_data 一致）
    if cfg.get("data", {}).get("use_embedding", False):
        # Embedding 模式：从第一个样本推断维度
        first_item = dataset[0]
        tab_dim = first_item["tabular"].shape[0]
        spec_len = first_item["spectra"].shape[0]
        num_wavelengths = spec_len
        scan_aggregation = "embedding"
        normalization_method = "none"
    else:
        num_wavelengths = len([c for c in pd.read_csv(cfg["data"]["spectra_csv"], sep=None, engine="python").columns if c not in ["Sample", "Group"]])
        scan_aggregation = cfg.get("data", {}).get("scan_aggregation", "sequence")
        spec_len = num_wavelengths * 3 if scan_aggregation == "stats" else num_wavelengths
        tab_dim = dataset.items[0]["tabular"].shape[0]
        normalization_method = cfg.get("data", {}).get("preprocessing", {}).get("normalization", {}).get("method", "SNV")

    dataset_info = {
        "tab_dim": tab_dim,
        "spec_len": spec_len,
        "num_wavelengths": num_wavelengths,
        "num_classes": len(set(labels)),
        "class_distribution": pd.Series(labels).value_counts().to_dict(),
        "scan_aggregation": scan_aggregation,
        "normalization_method": normalization_method,
    }

    fold_results = []
    for fold_idx in range(n_splits):
        train_loader, val_loader, test_loader, indices = get_cv_loaders(
            dataset, labels, fold_idx, n_splits, inner_val_ratio, batch_size, random_state
        )
        train_idx, val_idx, test_idx = indices
        print(f"[CV Fold {fold_idx}] Train: {np.bincount(labels[train_idx], minlength=dataset_info['num_classes'])}, "
              f"Val: {np.bincount(labels[val_idx], minlength=dataset_info['num_classes'])}, "
              f"Test: {np.bincount(labels[test_idx], minlength=dataset_info['num_classes'])}")

        # 每折重新初始化模型
        model = build_model(cfg, dataset_info["tab_dim"], dataset_info["spec_len"])

        # 训练
        trainer = train_single_model(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset_info=dataset_info,
            model_name=model_name,
            fold_idx=fold_idx,
        )

        # 收集该折最佳 test 指标
        test_metrics = {
            "fold": fold_idx,
            "test_auc": trainer.test_result["metrics"].get("auc", np.nan),
            "test_acc": trainer.test_result["metrics"].get("acc", np.nan),
            "test_f1": trainer.test_result["metrics"].get("f1", np.nan),
            "test_macro_f1": trainer.test_result["metrics"].get("macro_f1", np.nan),
            "test_sensitivity@90%spec": trainer.test_result["metrics"].get("sensitivity@90%spec", np.nan),
            "best_val_auc": trainer.best_val_metric if hasattr(trainer, "best_val_metric") else np.nan,
        }
        # 追加扩展指标
        for ext_key in ("macro_auc", "weighted_auc", "cohens_kappa", "qwk"):
            val = trainer.test_result["metrics"].get(ext_key, np.nan)
            test_metrics[f"test_{ext_key}"] = val

        fold_results.append(test_metrics)

    # 汇总统计
    cv_summary = {"model_name": model_name, "n_splits": n_splits}
    metric_keys = [k for k in fold_results[0].keys() if k != "fold"]
    for key in metric_keys:
        values = [r[key] for r in fold_results if key in r and not np.isnan(r[key])]
        if len(values) > 0:
            cv_summary[f"{key}_mean"] = float(np.mean(values))
            cv_summary[f"{key}_std"] = float(np.std(values))
        else:
            cv_summary[f"{key}_mean"] = np.nan
            cv_summary[f"{key}_std"] = np.nan

    # 保存 CV 汇总
    model_dir = Path(cfg["train"].get("save_dir", "results")) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    cv_path = model_dir / "cv_summary.json"
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2, ensure_ascii=False)

    # 保存每折原始值 CSV
    import csv
    csv_path = model_dir / "fold_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold"] + metric_keys)
        writer.writeheader()
        for r in fold_results:
            row = {"fold": r["fold"]}
            row.update({k: (v if not np.isnan(v) else "NA") for k, v in r.items() if k != "fold"})
            writer.writerow(row)

    print(f"[OK] CV 汇总已保存: {cv_path}")
    print(f"[OK] 每折结果已保存: {csv_path}")
    return cv_summary


def train_single_model(
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dataset_info: dict,
    model_name: str = None,
    resume: bool = False,
    fold_idx: Optional[int] = None,
) -> EnhancedTrainer:
    print(f"[DEBUG] train_single_model called with resume={resume}")
    """
    训练单个模型
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        dataset_info: 数据集信息
        model_name: 模型名称（可选，覆盖配置中的名称）
        resume: 是否从检查点恢复训练
    
    Returns:
        训练好的训练器
    """
    # 使用指定的模型名称或配置中的名称
    if model_name:
        cfg["model"]["name"] = model_name

    # 确保模型的类别数与数据集一致（避免 config 中 num_classes 配错）
    # 但如果是 resume 模式，优先信任配置文件/原始模型结构，避免因数据划分导致类别缺失而改变模型结构
    if "num_classes" in dataset_info and not resume:
        cfg.setdefault("model", {})
        cfg["model"]["num_classes"] = int(dataset_info["num_classes"])
    
    # 构建模型
    model = build_model(cfg, dataset_info['tab_dim'], dataset_info['spec_len'])
    
    # 创建训练器（确保超参数为数值类型）
    lr_value = float(cfg["train"].get("lr", 1e-3))
    wd_raw = cfg["train"].get("weight_decay", 1e-4)
    weight_decay_value = float(wd_raw) if wd_raw is not None else 0.0
    use_embedding_input = cfg.get("data", {}).get("use_embedding", False)

    # 计算类别权重
    class_weights = None
    if cfg["train"].get("use_class_weights", False):
        print(f"[INFO] 计算类别权重...")
        dist = dataset_info['class_distribution']
        num_classes = dataset_info['num_classes']
        total = sum(dist.values())
        weights = []
        for i in range(num_classes):
            count = dist.get(i, 0)
            w = total / (num_classes * count) if count > 0 else 1.0
            weights.append(w)
        class_weights = torch.tensor(weights, dtype=torch.float)
        print(f"[INFO] 类别权重: {[f'{w:.2f}' for w in weights]}")

    # 高级训练策略配置
    advanced_cfg = cfg["train"].get("advanced", None)
    if advanced_cfg and advanced_cfg.get("enabled", False):
        print(f"[ADVANCED] 启用高级训练策略: {advanced_cfg}")

    evaluation_cfg = cfg.get("evaluation", {})
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
        class_weights=class_weights,
        advanced_cfg=advanced_cfg,
        num_classes=cfg["model"]["num_classes"],
        fold_idx=fold_idx,
        evaluation_cfg=evaluation_cfg,
    )
    
    
    if resume:
        trainer.load_model()
        
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
    generate_plots = cfg.get("visualization", {}).get("enable", True)
    test_result = trainer.evaluate(test_loader, generate_plots=generate_plots)
    trainer.test_result = test_result  # 供 CV 汇总使用

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
        'n_parameters': model_summary['total_parameters'],
        'trainable_parameters': model_summary['trainable_parameters'],
        'model_size_mb': model_summary['model_size_mb'],
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
    # 追加扩展指标（如果计算了）
    for ext_key in ('macro_auc', 'weighted_auc', 'cohens_kappa', 'qwk'):
        if ext_key in test_result['metrics']:
            metrics_summary[f'test_{ext_key}'] = test_result['metrics'][ext_key]
    
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
    print("\n[TRAIN_ALL] 开始训练所有模型...")
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
    parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    parser.add_argument("--mode", type=str, default=None, choices=["single", "cv"],
                        help="运行模式: single (单折) 或 cv (交叉验证)。默认从配置读取 evaluation.cross_validation.enabled")
    
    args = parser.parse_args()
    print(f"[DEBUG] args.resume={args.resume}")
    
    print(">>> 增强版多模态模型训练系统")
    print("=" * 60)
    
    # 加载配置
    cfg = load_config(args.config)
    print(f">> 配置文件: {args.config}")
    
    # Create log directory if it doesn't exist
    save_dir = Path(cfg['train'].get('save_dir', 'results'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify model name to create subfolder
    if args.model:
        model_dir = save_dir / args.model
    elif args.train_all:
        model_dir = save_dir
    else:
        model_dir = save_dir / cfg['model']['name']
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = model_dir / (cfg['logging'].get('log_file', 'training.log'))
    print(f"[LOG] Redirecting output to: {log_file}")
    
    # Save original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create dual loggers
    stdout_logger = DualLogger(str(log_file), sys.stdout)
    stderr_logger = DualLogger(str(log_file), sys.stderr)
    
    # Redirect
    sys.stdout = stdout_logger
    sys.stderr = stderr_logger
    
    try:
        # 判断运行模式
        cv_enabled_cfg = cfg.get("evaluation", {}).get("cross_validation", {}).get("enabled", False)
        mode = args.mode or ("cv" if cv_enabled_cfg else "single")

        if mode == "cv" and not args.eval_only and not args.train_all:
            # CV 模式（现已支持 raw 和 embedding 模式）
            model_name = args.model or cfg["model"]["name"]
            run_cross_validation(cfg, model_name)
            print(f"\n>>> CV Training Complete!")
            print(f"--- Results saved to: {cfg['train'].get('save_dir', 'results')}")
            return

        # 单折模式（默认）
        train_loader, val_loader, test_loader, dataset_info = prepare_data(cfg)

        if args.eval_only:
            # Eval only mode
            print(f"\n[EVAL] Eval only mode: {args.eval_only}")
            model = build_model(cfg, dataset_info['tab_dim'], dataset_info['spec_len'])
            advanced_cfg = cfg["train"].get("advanced", None)
            evaluation_cfg = cfg.get("evaluation", {})
            trainer = EnhancedTrainer(
                model=model,
                model_name=args.eval_only,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                save_dir=cfg["train"].get("save_dir", "results"),
                advanced_cfg=advanced_cfg,
                num_classes=cfg["model"]["num_classes"],
                evaluation_cfg=evaluation_cfg,
            )
            trainer.load_model()
            result = trainer.evaluate(test_loader)
            print(f"[RESULT] Eval Results: {result['metrics']}")

        elif args.train_all:
            # Train all models
            trainers = train_all_models(cfg, train_loader, val_loader, test_loader, dataset_info)

        else:
            # Train single model
            model_name = args.model or cfg["model"]["name"]
            trainer = train_single_model(
                cfg=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                dataset_info=dataset_info,
                model_name=model_name,
                resume=args.resume
            )
        
        print(f"\n>>> Training Complete!")
        print(f"--- Results saved to: {cfg['train'].get('save_dir', 'results')}")
        
    finally:
        # Restore streams
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        if hasattr(sys.stderr, 'close'):
            sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("traceback.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("!!! 程序发生错误，详情请查看 traceback.log")
        raise e

