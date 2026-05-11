#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数敏感性扫描（Parameter Sensitivity Analysis）

对单一模型扫描一个超参数维度，记录参数量与 Test AUC 的关系。

使用方法:
  python scripts/run_sensitivity_analysis.py \\
      --config configs/experiment_base.yaml \\
      --model AttentionMultimodal \\
      --dimension hidden_dim \\
      --values 32 64 128 256 \\
      --lite

支持扫描维度:
  hidden_dim   -> 扫描 model.lite.hidden_dim（影响各模块 hidden size）
  dropout      -> 扫描 model.lite.dropout
  weight_decay -> 扫描 train.weight_decay
  batch_size   -> 扫描 data.batch_size
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhanced_main import load_config, prepare_data, train_single_model, build_model


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_override(cfg: dict, dimension: str, value) -> dict:
    """将扫描值注入配置。"""
    cfg = copy.deepcopy(cfg)
    if dimension == "hidden_dim":
        cfg.setdefault("model", {}).setdefault("lite", {})
        cfg["model"]["lite"]["hidden_dim"] = value
        # 联动调整相关维度（若未显式覆盖）
        cfg["model"]["lite"].setdefault("cross_attn_dim", value)
        cfg["model"]["lite"].setdefault("d_model", value)
        cfg["model"]["lite"].setdefault("fusion_dim", value)
        cfg["model"]["lite"].setdefault("bottleneck_dim", value // 2 if value >= 64 else value)
    elif dimension == "dropout":
        cfg.setdefault("model", {}).setdefault("lite", {})
        cfg["model"]["lite"]["dropout"] = value
    elif dimension == "weight_decay":
        cfg.setdefault("train", {})
        cfg["train"]["weight_decay"] = value
    elif dimension == "batch_size":
        cfg.setdefault("train", {})
        cfg["train"]["batch_size"] = value
    else:
        raise ValueError(f"不支持的扫描维度: {dimension}")
    return cfg


def count_parameters(cfg: dict, dataset_info: dict) -> int:
    """实例化模型并计算参数量。"""
    try:
        model = build_model(cfg, dataset_info["tab_dim"], dataset_info["spec_len"])
        n_params = sum(p.numel() for p in model.parameters())
        return n_params
    except Exception as e:
        print(f"[WARN] 计算参数量失败: {e}")
        return 0


def run_sensitivity(
    config_path: str,
    model_name: str,
    dimension: str,
    values: List[Any],
    seed: int = 42,
    lite: bool = False,
    output_dir: str = "results/sensitivity",
    overwrite: bool = False,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    cfg["model"]["name"] = model_name
    if lite:
        cfg.setdefault("model", {}).setdefault("lite", {})
        cfg["model"]["lite"]["enabled"] = True

    set_seed(seed)
    out_root = Path(output_dir) / dimension
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / f"{model_name}_scan.csv"

    rows = []
    # 如果已有 CSV 且未 overwrite，先加载
    if csv_path.exists() and not overwrite:
        existing = pd.read_csv(csv_path)
        existing_values = set(existing["value"].tolist())
        print(f"[INFO] 加载已有结果: {csv_path} ({len(existing)} 行)")
    else:
        existing = pd.DataFrame()
        existing_values = set()

    for value in values:
        if value in existing_values and not overwrite:
            print(f"[SKIP] {dimension}={value} 已存在")
            row = existing[existing["value"] == value].iloc[0].to_dict()
            rows.append(row)
            continue

        run_cfg = apply_override(cfg, dimension, value)
        run_cfg.setdefault("train", {})
        run_cfg["train"]["save_dir"] = str(out_root)
        train_loader, val_loader, test_loader, dataset_info = prepare_data(run_cfg)
        n_params = count_parameters(run_cfg, dataset_info)

        print(f"\n[SCAN] {dimension}={value} | params={n_params:,}")
        trainer = train_single_model(
            run_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset_info=dataset_info,
            model_name=model_name,
        )

        # 读取 metrics
        metrics_path = Path(run_cfg["train"].get("save_dir", "results")) / model_name / "metrics_summary.json"
        metrics = {}
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        row = {
            "value": value,
            "n_params": n_params,
            "test_auc": metrics.get("test_auc", np.nan),
            "val_auc": metrics.get("best_val_auc", np.nan),
            "train_auc": metrics.get("final_val_auc", np.nan),  # 近似
            "best_epoch": metrics.get("best_epoch", np.nan),
        }
        rows.append(row)
        print(f"[RESULT] test_auc={row['test_auc']:.4f} | best_epoch={row['best_epoch']}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 扫描结果已保存: {csv_path}")

    # 绘图
    plot_sensitivity(df, dimension, model_name, out_root)
    return df


def plot_sensitivity(df: pd.DataFrame, dimension: str, model_name: str, out_root: Path):
    """绘制敏感性扫描结果。"""
    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图：Test AUC vs 参数值
    ax = axes[0]
    ax.plot(df["value"], df["test_auc"], marker="o", linewidth=2, markersize=8, color="#2E86AB")
    ax.set_xlabel(dimension, fontsize=12)
    ax.set_ylabel("Test AUC", fontsize=12)
    ax.set_title(f"{model_name}: {dimension} Sensitivity", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 标注 Lite 配置参考点
    if dimension == "hidden_dim":
        ref_val = 64
    elif dimension == "dropout":
        ref_val = 0.5
    elif dimension == "weight_decay":
        ref_val = 1e-3
    elif dimension == "batch_size":
        ref_val = 8
    else:
        ref_val = None

    if ref_val is not None and ref_val in df["value"].values:
        ref_row = df[df["value"] == ref_val].iloc[0]
        ax.axvline(ref_val, color="red", linestyle="--", alpha=0.5, label=f"Lite default ({ref_val})")
        ax.legend()

    # 右图：参数量变化
    ax = axes[1]
    ax.bar([str(v) for v in df["value"]], df["n_params"], color="#A23B72", alpha=0.8)
    ax.set_xlabel(dimension, fontsize=12)
    ax.set_ylabel("# Parameters", fontsize=12)
    ax.set_title("Model Complexity", fontsize=13, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    plot_path = out_root / f"{model_name}_{dimension}_sensitivity.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 敏感性图已保存: {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter Sensitivity Analysis")
    parser.add_argument("--config", type=str, default="configs/experiment_base.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dimension", type=str, required=True,
                        choices=["hidden_dim", "dropout", "weight_decay", "batch_size"])
    parser.add_argument("--values", type=float, nargs="+", required=True,
                        help="扫描值列表（会自动按类型转换）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("[INFO] 参数敏感性扫描")
    print("=" * 80)
    print(f"模型: {args.model} | 维度: {args.dimension} | 种子: {args.seed}")

    # 自动转换类型
    values = args.values
    if args.dimension in ["hidden_dim", "batch_size"]:
        values = [int(v) for v in values]

    run_sensitivity(
        config_path=args.config,
        model_name=args.model,
        dimension=args.dimension,
        values=values,
        seed=args.seed,
        lite=args.lite,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
