#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型循环训练脚本

一次性训练多个多模态模型，并汇总实验结果到 CSV 表格。

使用方法:
    python scripts/run_multimodal_experiments.py --config configs/enhanced_config.yaml --embedding_only
    python scripts/run_multimodal_experiments.py --config configs/enhanced_config.yaml --models AttentionMultimodal EnhancedMMTM
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
# 注意：模型名称必须与 enhanced_main.py 中 build_model() 支持的名称一致
DEFAULT_MODELS = [
    "BaselineMultimodal",  # Baseline 多模态融合（默认使用 concat）
    "AttentionMultimodal",
    "EnhancedMMTM",
    # "ConcatFusion",      # Baseline 融合方式 1（可选）
    # "EnsembleFusion",    # Baseline 融合方式 2（可选）
    # "TFTMultimodal",    # 如需一起跑，取消注释
]


def try_load_metrics(model_dir: Path) -> Dict[str, Any]:
    """
    尝试从模型结果目录中加载指标 JSON。
    兼容多种命名：results.json / metrics.json / best_metrics.json / metrics_summary.json
    """
    candidate_names = [
        "metrics_summary.json",  # 优先读取统一格式
        "results.json",
        "metrics.json",
        "best_metrics.json",
    ]
    
    for name in candidate_names:
        path = model_dir / name
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
            except Exception as e:
                print(f"[WARN] 读取 {path} 时出错: {e}，尝试下一个文件")
                continue
    
    print(f"[WARN] 未在 {model_dir} 中找到指标 JSON 文件（{candidate_names}），将仅记录模型名称。")
    return {}


def extract_metrics_from_results(results_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 results.json 中提取关键指标，转换为统一格式。
    """
    metrics = {}
    
    # 从 training_result 中提取验证集最佳指标
    if "training_result" in results_data:
        train_res = results_data["training_result"]
        metrics["best_val_auc"] = train_res.get("best_val_auc", None)
        metrics["best_epoch"] = train_res.get("best_epoch", None)
        metrics["total_time"] = train_res.get("total_time", None)
        
        # 从 val_history 中提取最后一个 epoch 的指标
        if "val_history" in train_res:
            val_hist = train_res["val_history"]
            if val_hist.get("auc"):
                metrics["final_val_auc"] = val_hist["auc"][-1] if val_hist["auc"] else None
            if val_hist.get("acc"):
                metrics["final_val_acc"] = val_hist["acc"][-1] if val_hist["acc"] else None
            if val_hist.get("f1"):
                metrics["final_val_f1"] = val_hist["f1"][-1] if val_hist["f1"] else None
    
    # 从 test_result 中提取测试集指标
    if "test_result" in results_data and "metrics" in results_data["test_result"]:
        test_metrics = results_data["test_result"]["metrics"]
        metrics["test_auc"] = test_metrics.get("auc", None)
        metrics["test_acc"] = test_metrics.get("acc", None)
        metrics["test_f1"] = test_metrics.get("f1", None)
        metrics["test_sensitivity@90%spec"] = test_metrics.get("sensitivity@90%spec", None)
    
    return metrics


def run_experiments(
    config_path: str,
    model_names: List[str],
    use_embedding_only: bool = False,
) -> pd.DataFrame:
    """
    使用统一的 config，循环跑多个模型，并汇总结果为 DataFrame。
    """
    cfg = load_config(config_path)
    
    # 如果只想在 embedding 模式下跑实验，可以强制打开 use_embedding
    if use_embedding_only:
        cfg.setdefault("data", {})
        cfg["data"]["use_embedding"] = True
        print("[INFO] 强制启用 embedding 模式")
    
    # 数据只准备一次（前提是所有模型使用同一份数据设定）
    print("\n" + "=" * 80)
    print("[INFO] 准备数据集（所有模型共享）")
    print("=" * 80)
    train_loader, val_loader, test_loader, dataset_info = prepare_data(cfg)
    
    save_root = Path(cfg["train"].get("save_dir", "results"))
    summary_rows: List[Dict[str, Any]] = []
    
    print(f"\n[INFO] 开始循环训练 {len(model_names)} 个模型")
    print(f"[INFO] 结果将保存到: {save_root}")
    
    for idx, model_name in enumerate(model_names, 1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(model_names)}] 开始训练模型: {model_name}")
        print("=" * 80)
        
        # 针对当前模型复制一份 config，并设置模型名称
        this_cfg = copy.deepcopy(cfg)
        this_cfg.setdefault("model", {})
        this_cfg["model"]["name"] = model_name
        
        # 这里可以根据模型名做一些特别的超参数覆盖（可选）
        # 例如：
        # if model_name == "EnhancedMMTM":
        #     this_cfg["model"]["fusion_strategy"] = "hierarchical"
        
        try:
            # 训练当前模型
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
            
            # 尝试加载指标
            results_data = try_load_metrics(model_dir)
            
            # 提取关键指标
            if results_data:
                if "metrics_summary.json" in str(model_dir / "metrics_summary.json"):
                    # 如果已经有统一格式的 metrics_summary.json，直接使用
                    metrics = results_data
                else:
                    # 否则从 results.json 中提取
                    metrics = extract_metrics_from_results(results_data)
            else:
                metrics = {}
            
            row = {"model_name": model_name}
            # 将 metrics 展平成一行（只展开简单标量）
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    # 避免太乱，只收集简单标量
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        row[k] = v
            
            summary_rows.append(row)
            print(f"[OK] {model_name} 训练完成并记录指标")
            
        except Exception as e:
            print(f"[ERROR] {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            # 即使失败也记录一行，标记为失败
            summary_rows.append({
                "model_name": model_name,
                "status": "failed",
                "error": str(e)
            })
    
    # 汇总为 DataFrame
    df_summary = pd.DataFrame(summary_rows)
    
    # 保存 CSV
    summary_path = save_root / "experiments_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 实验汇总已保存到: {summary_path}")
    
    return df_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple multimodal experiments and summarize results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认模型列表，强制 embedding 模式
  python scripts/run_multimodal_experiments.py --config configs/enhanced_config.yaml --embedding_only
  
  # 指定要跑的模型
  python scripts/run_multimodal_experiments.py --config configs/enhanced_config.yaml --models AttentionMultimodal EnhancedMMTM
  
  # 使用自定义配置
  python scripts/run_multimodal_experiments.py --config configs/my_config.yaml
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/enhanced_config.yaml",
        help="Path to base YAML config (default: configs/enhanced_config.yaml)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Model names to run (override default list). Example: --models AttentionMultimodal EnhancedMMTM",
    )
    parser.add_argument(
        "--embedding_only",
        action="store_true",
        help="Force use_embedding=True for all experiments.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 确定要跑的模型列表
    model_names = args.models if args.models is not None else DEFAULT_MODELS
    
    print("=" * 80)
    print("[INFO] 多模型循环训练脚本")
    print("=" * 80)
    print(f"[INFO] 配置文件: {args.config}")
    print(f"[INFO] 将要运行的模型列表 ({len(model_names)} 个):")
    for name in model_names:
        print(f"  - {name}")
    if args.embedding_only:
        print("[INFO] 强制启用 embedding 模式")
    print("=" * 80)
    
    try:
        df = run_experiments(
            config_path=args.config,
            model_names=model_names,
            use_embedding_only=args.embedding_only,
        )
        
        print("\n" + "=" * 80)
        print("[INFO] 汇总结果预览：")
        print("=" * 80)
        print(df.to_string(index=False))
        print("\n[OK] 所有实验完成！")
        
    except KeyboardInterrupt:
        print("\n[WARN] 用户中断训练")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

