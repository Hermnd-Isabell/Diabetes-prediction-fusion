#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准化消融实验运行器（Standardized Experiment Orchestrator）

支持多种运行模式：
  single  -- 固定三分划分 + 多种子平均
  cv      -- Stratified K-Fold 交叉验证
  legacy  -- 与旧版行为完全一致（默认）

支持统一 Lite 切换、基线补全、自动结果聚合。

使用方法:
  # 多种子单折（推荐用于 142 样本场景）
  python scripts/run_multimodal_experiments.py \
      --config configs/experiment_base.yaml \
      --models SpectraOnlyModel ClinicalOnlyModel ConcatFusion EnsembleFusion BaselineMultimodal AttentionMultimodal TFTMultimodal EnhancedMMTM \
      --mode single --seeds 0 1 2 3 4 --lite

  # 5-Fold CV
  python scripts/run_multimodal_experiments.py \
      --config configs/experiment_base.yaml \
      --models AttentionMultimodal \
      --mode cv --n_splits 5 --lite

  # 旧版行为（单种子、单折）
  python scripts/run_multimodal_experiments.py \
      --config configs/enhanced_config.yaml \
      --models AttentionMultimodal EnhancedMMTM
"""

import argparse
import copy
import json
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

# 确保可以从脚本所在目录的上一级导入
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhanced_main import load_config, prepare_data, train_single_model, build_model
from datasets.raman_dataset import RamanDataset, collate_fn, preprocess_spectrum
from datasets.embedding_dataset import EmbeddingMultimodalDataset, embedding_collate_fn
from torch.utils.data import DataLoader, Subset


DEFAULT_MODELS = [
    "SpectraOnlyModel",
    "ClinicalOnlyModel",
    "ConcatFusion",
    "EnsembleFusion",
    "BaselineMultimodal",
    "AttentionMultimodal",
    "TFTMultimodal",
    "EnhancedMMTM",
]


def set_seed(seed: int):
    """统一设置随机种子。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def try_load_metrics(model_dir: Path) -> Dict[str, Any]:
    """从模型结果目录加载 metrics_summary.json。"""
    path = model_dir / "metrics_summary.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] 读取 {path} 时出错: {e}")
    return {}


def run_single_seed(
    cfg: dict,
    model_name: str,
    seed: int,
    save_root: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    对指定模型和种子执行一次单折训练。
    返回 metrics_summary 字典（若已存在且未 overwrite 则直接读取）。
    """
    seed_dir = save_root / model_name / f"seed_{seed}"
    metrics_path = seed_dir / "metrics_summary.json"

    if seed_dir.exists() and metrics_path.exists() and not overwrite:
        print(f"  [SKIP] {model_name} / seed={seed} 已存在，使用 --overwrite 强制重跑")
        return try_load_metrics(seed_dir)

    # 设置种子
    set_seed(seed)
    run_cfg = copy.deepcopy(cfg)
    run_cfg.setdefault("experiment", {})
    run_cfg["experiment"]["random_seed"] = seed
    run_cfg["model"]["name"] = model_name
    # 同步 save_dir，确保训练输出到正确的根目录
    run_cfg.setdefault("train", {})
    run_cfg["train"]["save_dir"] = str(save_root)

    # 数据准备（seed 会影响 random_split）
    train_loader, val_loader, test_loader, dataset_info = prepare_data(run_cfg)

    # 训练
    print(f"  [TRAIN] {model_name} | seed={seed}")
    trainer = train_single_model(
        run_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_info=dataset_info,
        model_name=model_name,
    )

    # 将结果从 src_dir 复制到 seed_dir（仅复制 JSON/CSV/日志，.pt 文件可能较大且可能被占用）
    src_dir = save_root / model_name
    seed_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        import shutil
        for f in list(src_dir.iterdir()):
            if f.is_file() and f.suffix in (".json", ".csv", ".txt", ".log"):
                dst = seed_dir / f.name
                if dst.exists():
                    dst.unlink()
                shutil.copy2(str(f), str(dst))

    metrics = try_load_metrics(seed_dir)
    metrics["seed"] = seed
    return metrics


def run_cv_fold(
    cfg: dict,
    model_name: str,
    fold: int,
    train_idx: List[int],
    val_idx: List[int],
    dataset,
    save_root: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    执行一次 CV fold 训练。
    """
    fold_dir = save_root / model_name / f"fold_{fold}"
    metrics_path = fold_dir / "metrics_summary.json"

    if fold_dir.exists() and metrics_path.exists() and not overwrite:
        print(f"  [SKIP] {model_name} / fold={fold} 已存在")
        return try_load_metrics(fold_dir)

    run_cfg = copy.deepcopy(cfg)
    run_cfg["model"]["name"] = model_name
    run_cfg.setdefault("train", {})
    run_cfg["train"]["save_dir"] = str(save_root)
    batch_size = run_cfg["train"]["batch_size"]

    # 构建 Subset DataLoader
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    # 测试集：在 CV 中验证集即测试集
    test_set = val_set

    # collate_fn 选择
    if isinstance(dataset, EmbeddingMultimodalDataset):
        cfn = embedding_collate_fn
    else:
        cfn = collate_fn

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=cfn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=cfn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=cfn)

    # dataset_info 推断
    if isinstance(dataset, EmbeddingMultimodalDataset):
        ref_items = dataset.items
        tab_dim = ref_items[0]["tabular"].shape[0]
        spec_len = ref_items[0]["spectra"].shape[0]
    else:
        tab_dim = dataset.items[0]["tabular"].shape[0]
        spec_len = len(dataset.wave_cols)

    labels_all = [dataset[i]["label"] for i in range(len(dataset))]
    dataset_info = {
        "tab_dim": tab_dim,
        "spec_len": spec_len,
        "num_classes": len(set(labels_all)),
        "class_distribution": pd.Series(labels_all).value_counts().to_dict(),
    }

    print(f"  [TRAIN] {model_name} | fold={fold} | train={len(train_idx)} val={len(val_idx)}")
    trainer = train_single_model(
        run_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_info=dataset_info,
        model_name=model_name,
    )

    # 将结果从 src_dir 复制到 fold_dir（仅复制 JSON/CSV/日志，跳过 .pt）
    src_dir = save_root / model_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        import shutil
        for f in list(src_dir.iterdir()):
            if f.is_file() and f.suffix in (".json", ".csv", ".txt", ".log"):
                dst = fold_dir / f.name
                if dst.exists():
                    dst.unlink()
                shutil.copy2(str(f), str(dst))

    metrics = try_load_metrics(fold_dir)
    metrics["fold"] = fold
    return metrics


def aggregate_seeds(results: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """对同一模型的多种子结果计算 mean ± std。"""
    numeric_keys = [
        "test_auc", "test_acc", "test_f1",
        "best_val_auc", "final_val_auc", "final_val_acc",
        "n_parameters", "model_size_mb",
    ]
    row = {"model_name": model_name}
    for key in numeric_keys:
        vals = [r[key] for r in results if key in r and r[key] is not None]
        if vals:
            row[key] = np.mean(vals)
            row[f"{key}_std"] = np.std(vals)
        else:
            row[key] = None
            row[f"{key}_std"] = None
    return row


def run_experiments(args) -> pd.DataFrame:
    cfg = load_config(args.config)
    save_root = Path(cfg.get("experiment", {}).get("output_dir", "results"))
    save_root.mkdir(parents=True, exist_ok=True)

    # 统一启用 Lite
    if args.lite:
        cfg.setdefault("model", {})
        cfg["model"].setdefault("lite", {})
        cfg["model"]["lite"]["enabled"] = True
        print(f"[INFO] 统一启用 Lite 模式: {cfg['model']['lite']}")

    # 确定模型列表（强制包含基线）
    model_names = args.models if args.models is not None else DEFAULT_MODELS
    if args.baselines:
        for baseline in ("ConcatFusion", "EnsembleFusion"):
            if baseline not in model_names:
                model_names = list(model_names) + [baseline]
                print(f"[INFO] 强制补全基线模型: {baseline}")

    summary_rows = []

    if args.mode == "legacy":
        # ========== 旧版行为：单种子、单折 ==========
        print("=" * 80)
        print("[INFO] 运行模式: legacy（与旧版行为一致）")
        print("=" * 80)
        train_loader, val_loader, test_loader, dataset_info = prepare_data(cfg)
        for idx, model_name in enumerate(model_names, 1):
            print(f"\n[{idx}/{len(model_names)}] {model_name}")
            this_cfg = copy.deepcopy(cfg)
            this_cfg["model"]["name"] = model_name
            try:
                trainer = train_single_model(
                    this_cfg, train_loader, val_loader, test_loader, dataset_info, model_name
                )
                metrics = try_load_metrics(save_root / model_name)
                row = {"model_name": model_name, **metrics}
                summary_rows.append(row)
            except Exception as e:
                print(f"[ERROR] {model_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                summary_rows.append({"model_name": model_name, "status": "failed", "error": str(e)})

    elif args.mode == "single":
        # ========== 多种子单折 ==========
        print("=" * 80)
        print(f"[INFO] 运行模式: single | seeds={args.seeds} | models={len(model_names)}")
        print("=" * 80)
        for model_name in model_names:
            print(f"\n[INFO] 模型: {model_name}")
            seed_results = []
            for seed in args.seeds:
                metrics = run_single_seed(cfg, model_name, seed, save_root, args.overwrite)
                seed_results.append(metrics)
            # 汇总
            row = aggregate_seeds(seed_results, model_name)
            summary_rows.append(row)

    elif args.mode == "cv":
        # ========== 交叉验证 ==========
        print("=" * 80)
        print(f"[INFO] 运行模式: cv | n_splits={args.n_splits}")
        print("=" * 80)
        set_seed(cfg.get("experiment", {}).get("random_seed", 42))

        # 先准备一次数据以获取完整数据集
        use_embedding = cfg.get("data", {}).get("use_embedding", False)
        if use_embedding:
            from multimodal.embedding_loader import (
                load_spectrum_embedding, load_clinical_embedding, align_by_patient_id
            )
            spec_path = cfg["data"]["spectrum_embedding_path"]
            clin_path = cfg["data"]["clinical_embedding_path"]
            spectrum_dict = load_spectrum_embedding(spec_path)
            clinical_dict = load_clinical_embedding(clin_path)
            aligned = align_by_patient_id(spectrum_dict, clinical_dict)
            dataset = EmbeddingMultimodalDataset(
                aligned, split="all",
                dropout_config={"spectra": 0.0, "clinical": 0.0}
            )
            labels = [item["label"] for item in dataset.items]
        else:
            import pandas as pd
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
            labels = [item["label"] for item in dataset.items]

        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=cfg.get("experiment", {}).get("random_seed", 42))
        splits = list(skf.split(range(len(dataset)), labels))

        for model_name in model_names:
            print(f"\n[INFO] 模型: {model_name}")
            fold_results = []
            for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
                metrics = run_cv_fold(
                    cfg, model_name, fold_idx, list(train_idx), list(val_idx),
                    dataset, save_root, args.overwrite
                )
                fold_results.append(metrics)
            row = aggregate_seeds(fold_results, model_name)
            summary_rows.append(row)

    else:
        raise ValueError(f"未知运行模式: {args.mode}")

    # 保存汇总表
    df = pd.DataFrame(summary_rows)
    summary_path = save_root / "experiments_summary.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 实验汇总已保存: {summary_path}")

    # 自动调用结果聚合
    try:
        print("[INFO] 自动生成对比表格...")
        cmd = [
            sys.executable,
            str(THIS_DIR / "generate_main_results_table.py"),
            "--summary_path", str(summary_path),
            "--config", args.config,
        ]
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"[WARN] 自动结果聚合失败: {e}")

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standardized multimodal experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 多种子单折 + Lite
  python scripts/run_multimodal_experiments.py \\
      --config configs/experiment_base.yaml \\
      --models SpectraOnlyModel ClinicalOnlyModel ConcatFusion \\
      --mode single --seeds 0 1 2 --lite

  # 5-Fold CV
  python scripts/run_multimodal_experiments.py \\
      --config configs/experiment_base.yaml \\
      --models AttentionMultimodal \\
      --mode cv --n_splits 5

  # 旧版行为（默认）
  python scripts/run_multimodal_experiments.py \\
      --config configs/enhanced_config.yaml
"""
    )
    parser.add_argument("--config", type=str, default="configs/experiment_base.yaml")
    parser.add_argument("--models", type=str, nargs="*", default=None)
    parser.add_argument(
        "--mode", type=str, default="legacy",
        choices=["legacy", "single", "cv"],
        help="运行模式: legacy=旧版单折, single=多种子单折, cv=交叉验证"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--lite", action="store_true", help="统一启用 model.lite.enabled=True")
    parser.add_argument("--baselines", action="store_true", help="强制包含 ConcatFusion 和 EnsembleFusion")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有结果")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("[INFO] 标准化消融实验运行器")
    print("=" * 80)
    print(f"[INFO] 配置文件: {args.config}")
    print(f"[INFO] 运行模式: {args.mode}")
    print(f"[INFO] 模型列表: {args.models if args.models else '默认列表'}")
    if args.lite:
        print("[INFO] Lite 模式: 启用")
    print("=" * 80)

    try:
        df = run_experiments(args)
        print("\n" + "=" * 80)
        print("汇总结果预览:")
        print("=" * 80)
        print(df.to_string(index=False))
        print("\n[OK] 所有实验完成！")
    except KeyboardInterrupt:
        print("\n[WARN] 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
