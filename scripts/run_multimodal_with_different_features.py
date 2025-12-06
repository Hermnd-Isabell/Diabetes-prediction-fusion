#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用不同光谱特征文件运行多模态实验

循环使用不同的光谱特征文件（Feature_Interaction_MLP_features.csv, MLP_Baseline_features.csv）
运行多模态实验，并将结果存储在不同的文件夹中。

使用方法:
    python scripts/run_multimodal_with_different_features.py --config configs/enhanced_config.yaml
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# 确保可以从脚本所在目录的上一级（项目根目录）导入 enhanced_main
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入 enhanced_main 中的函数
try:
    from enhanced_main import load_config, prepare_data, train_single_model
except ImportError as e:
    print(f"[ERROR] 无法导入 enhanced_main: {e}")
    print("       请确认在项目根目录运行本脚本，或检查 enhanced_main.py 是否存在。")
    sys.exit(1)

# 默认要跑的模型列表
DEFAULT_MODELS = [
    "BaselineMultimodal",
    "AttentionMultimodal",
    "EnhancedMMTM",
    # "TFTMultimodal",  # 如需一起跑，取消注释
]

# 要使用的光谱特征文件列表（排除当前使用的 Light_CNN_MLP_features.csv）
SPECTRUM_FEATURE_FILES = [
    "Feature_Interaction_MLP_features.csv",
    "MLP_Baseline_features.csv",
]


def get_feature_name(feature_file: str) -> str:
    """从特征文件名提取特征名称（用于文件夹命名）"""
    # 例如: "Feature_Interaction_MLP_features.csv" -> "Feature_Interaction_MLP"
    return feature_file.replace("_features.csv", "").replace(".csv", "")


def try_load_metrics(model_dir: Path) -> Dict[str, Any]:
    """
    尝试从模型结果目录中加载指标 JSON。
    兼容多种命名：results.json / metrics.json / best_metrics.json / metrics_summary.json
    """
    candidate_names = [
        "metrics_summary.json",
        "results.json",
        "metrics.json",
        "best_metrics.json",
    ]
    for name in candidate_names:
        path = model_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data
    print(f"[WARNING] 未在 {model_dir} 中找到指标 JSON 文件，将仅记录模型名称。")
    return {}


def run_experiments_for_feature(
    config_path: str,
    feature_file: str,
    model_names: List[str],
    use_embedding_only: bool = False,
) -> pd.DataFrame:
    """
    使用指定的特征文件运行多模态实验。
    
    Args:
        config_path: 配置文件路径
        feature_file: 光谱特征文件名（例如 "Feature_Interaction_MLP_features.csv"）
        model_names: 要运行的模型列表
        use_embedding_only: 是否强制使用 embedding 模式
    
    Returns:
        汇总结果的 DataFrame
    """
    # 加载配置
    cfg = load_config(config_path)
    
    # 如果只想在 embedding 模式下跑实验，可以强制打开 use_embedding
    if use_embedding_only:
        cfg.setdefault("data", {})
        cfg["data"]["use_embedding"] = True
    
    # 修改配置中的光谱特征文件路径
    feature_name = get_feature_name(feature_file)
    feature_path = f"spectrum_dimension/diabetes_results/{feature_file}"
    
    if not Path(feature_path).exists():
        print(f"[ERROR] 特征文件不存在: {feature_path}")
        return pd.DataFrame()
    
    cfg.setdefault("data", {})
    cfg["data"]["spectrum_embedding_path"] = feature_path
    print(f"\n[INFO] 使用光谱特征文件: {feature_file}")
    print(f"[INFO] 特征文件路径: {feature_path}")
    
    # 数据只准备一次（前提是所有模型使用同一份数据设定）
    train_loader, val_loader, test_loader, dataset_info = prepare_data(cfg)
    
    # 修改保存目录，添加特征名称前缀
    original_save_dir = cfg["train"].get("save_dir", "results")
    feature_save_dir = f"{original_save_dir}_{feature_name}"
    cfg["train"]["save_dir"] = feature_save_dir
    
    save_root = Path(feature_save_dir)
    summary_rows: List[Dict[str, Any]] = []
    
    for model_name in model_names:
        print("\n" + "=" * 80)
        print(f"[INFO] 开始训练模型: {model_name} (特征: {feature_name})")
        print("=" * 80)
        
        # 针对当前模型复制一份 config，并设置模型名称
        this_cfg = copy.deepcopy(cfg)
        this_cfg.setdefault("model", {})
        this_cfg["model"]["name"] = model_name
        
        # 训练当前模型
        try:
            trainer = train_single_model(
                this_cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                dataset_info=dataset_info,
                model_name=model_name,
            )
            
            # 模型结果目录：save_dir / model_name
            model_dir = save_root / model_name
            metrics = try_load_metrics(model_dir)
            
            row = {
                "model_name": model_name,
                "feature_file": feature_file,
                "feature_name": feature_name,
            }
            
            # 将 metrics 展平成一行（只展开第一层 key）
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    # 避免太乱，只收集简单标量
                    if isinstance(v, (int, float, str)):
                        row[k] = v
            
            summary_rows.append(row)
            print(f"[OK] {model_name} 训练完成")
            
        except Exception as e:
            print(f"[ERROR] {model_name} 训练失败: {e}")
            row = {
                "model_name": model_name,
                "feature_file": feature_file,
                "feature_name": feature_name,
                "status": "failed",
                "error": str(e),
            }
            summary_rows.append(row)
    
    # 汇总为 DataFrame
    df_summary = pd.DataFrame(summary_rows)
    
    # 保存 CSV
    summary_path = save_root / f"experiments_summary_{feature_name}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 实验汇总已保存到: {summary_path}")
    
    return df_summary


def main():
    parser = argparse.ArgumentParser(
        description="使用不同光谱特征文件运行多模态实验。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/enhanced_config.yaml",
        help="Path to base YAML config.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Model names to run (override default list).",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="*",
        default=None,
        help="Spectrum feature files to use (override default list).",
    )
    parser.add_argument(
        "--embedding_only",
        action="store_true",
        help="Force use_embedding=True for all experiments.",
    )
    
    args = parser.parse_args()
    
    model_names = args.models if args.models is not None else DEFAULT_MODELS
    feature_files = args.features if args.features is not None else SPECTRUM_FEATURE_FILES
    
    print("=" * 80)
    print("[INFO] 多模态实验 - 不同特征文件循环训练脚本")
    print("=" * 80)
    print(f"[INFO] 配置文件: {args.config}")
    print(f"[INFO] 要运行的特征文件 ({len(feature_files)} 个):")
    for f in feature_files:
        print(f"  - {f}")
    print(f"[INFO] 要运行的模型列表 ({len(model_names)} 个):")
    for name in model_names:
        print(f"  - {name}")
    if args.embedding_only:
        print("[INFO] 强制启用 embedding 模式")
    print("=" * 80)
    
    # 循环运行不同特征文件的实验
    all_summaries = []
    
    for feature_file in feature_files:
        print("\n" + "=" * 80)
        print(f"[INFO] 开始处理特征文件: {feature_file}")
        print("=" * 80)
        
        df = run_experiments_for_feature(
            config_path=args.config,
            feature_file=feature_file,
            model_names=model_names,
            use_embedding_only=args.embedding_only,
        )
        
        if not df.empty:
            all_summaries.append(df)
    
    # 合并所有结果
    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        combined_path = Path("results") / "all_features_experiments_summary.csv"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] 所有特征文件的实验结果已合并保存到: {combined_path}")
        print("\n[INFO] 汇总结果预览：")
        print(combined_df)
    else:
        print("\n[WARNING] 没有成功完成任何实验，无法生成汇总结果。")
    
    print("\n[OK] 所有实验完成！")


if __name__ == "__main__":
    main()

