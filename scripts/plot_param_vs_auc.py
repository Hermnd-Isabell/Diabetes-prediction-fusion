#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数量-AUC 关系可视化（Model Complexity vs. Generalization）

读取各模型的 metrics_summary.json，绘制"参数量-Test AUC"散点图，
直观展示 142 样本约束下复杂多模态架构的失效。

使用方法:
  python scripts/plot_param_vs_auc.py
  python scripts/plot_param_vs_auc.py --results_dir results --output results/comparison/param_vs_auc.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_all_metrics(results_dir: Path) -> pd.DataFrame:
    """
    遍历 results_dir 下所有子目录，读取 metrics_summary.json 和 cv_summary.json。
    优先使用 cv_summary.json（如果存在），否则使用 metrics_summary.json。
    """
    rows = []
    if not results_dir.exists():
        print(f"[WARN] 结果目录不存在: {results_dir}")
        return pd.DataFrame()

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        # 跳过非模型目录
        if model_name in {"comparison", "sensitivity", "ablation"}:
            continue

        # 1) 尝试多种子汇总
        seed_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
        if seed_dirs:
            seed_metrics = []
            for sd in seed_dirs:
                mp = sd / "metrics_summary.json"
                if mp.exists():
                    with open(mp, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    seed_metrics.append(m)
            if seed_metrics:
                df_seeds = pd.DataFrame(seed_metrics)
                row = {
                    "model_name": model_name,
                    "n_parameters": df_seeds["n_parameters"].mean() if "n_parameters" in df_seeds.columns else np.nan,
                    "test_auc": df_seeds["test_auc"].mean() if "test_auc" in df_seeds.columns else np.nan,
                    "test_auc_std": df_seeds["test_auc"].std() if "test_auc" in df_seeds.columns else np.nan,
                    "source": "multi_seed",
                }
                rows.append(row)
                continue

        # 2) 尝试单折 metrics_summary.json
        mp = model_dir / "metrics_summary.json"
        if mp.exists():
            with open(mp, "r", encoding="utf-8") as f:
                m = json.load(f)
            row = {
                "model_name": model_name,
                "n_parameters": m.get("n_parameters", np.nan),
                "test_auc": m.get("test_auc", np.nan),
                "test_auc_std": np.nan,
                "source": "single_fold",
            }
            rows.append(row)
            continue

        # 3) 尝试旧格式 results.json
        rp = model_dir / "results.json"
        if rp.exists():
            with open(rp, "r", encoding="utf-8") as f:
                m = json.load(f)
            test_auc = m.get("test_result", {}).get("metrics", {}).get("auc", np.nan)
            model_summary = m.get("model_summary", {})
            n_params = model_summary.get("total_parameters", np.nan)
            row = {
                "model_name": model_name,
                "n_parameters": n_params,
                "test_auc": test_auc,
                "test_auc_std": np.nan,
                "source": "legacy",
            }
            rows.append(row)

    return pd.DataFrame(rows)


def classify_model_type(model_name: str) -> str:
    """按模型复杂度分类，用于颜色区分。"""
    simple = {"SpectraOnlyModel", "ClinicalOnlyModel", "ConcatFusion", "EnsembleFusion", "BaselineMultimodal"}
    complex_ = {"AttentionMultimodal", "TFTMultimodal", "EnhancedMMTM"}
    if model_name in simple:
        return "Simple Fusion / Unimodal"
    elif model_name in complex_:
        return "Complex Multimodal"
    return "Other"


def plot_param_vs_auc(df: pd.DataFrame, output_path: Path, title: str = None):
    """绘制参数量-AUC 散点图。"""
    if len(df) == 0:
        print("[WARN] 没有可用数据，跳过绘图")
        return

    df = df.dropna(subset=["n_parameters", "test_auc"]).copy()
    df["model_type"] = df["model_name"].apply(classify_model_type)

    colors = {
        "Simple Fusion / Unimodal": "#2E86AB",
        "Complex Multimodal": "#A23B72",
        "Other": "#F18F01",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for mtype, group in df.groupby("model_type"):
        ax.scatter(
            group["n_parameters"],
            group["test_auc"],
            s=180,
            c=colors.get(mtype, "gray"),
            alpha=0.85,
            edgecolors="white",
            linewidths=1.5,
            label=mtype,
            zorder=3,
        )
        # 标注模型名
        for _, row in group.iterrows():
            ax.annotate(
                row["model_name"],
                (row["n_parameters"], row["test_auc"]),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=8,
                alpha=0.9,
            )

    # 趋势线（线性回归）
    if len(df) >= 3:
        x = df["n_parameters"].values
        y = df["test_auc"].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", alpha=0.4, linewidth=1.5,
                label=f"Trend (R={r_value:.2f}, p={p_value:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("# Parameters (log scale)", fontsize=13)
    ax.set_ylabel("Test AUC", fontsize=13)
    ax.set_title(title or "Model Complexity vs. Generalization (n=142)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_ylim(bottom=max(0.0, df["test_auc"].min() - 0.1), top=min(1.0, df["test_auc"].max() + 0.1))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] 散点图已保存: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Model Parameters vs. Test AUC")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/comparison/param_vs_auc.png")
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("[INFO] 参数量-AUC 散点图生成")
    print("=" * 80)

    df = load_all_metrics(Path(args.results_dir))
    if len(df) == 0:
        print("[ERROR] 未找到任何模型的 metrics_summary.json")
        print("[TIP] 请先运行训练，确保 results/<model_name>/metrics_summary.json 存在")
        sys.exit(1)

    print(f"[INFO] 加载了 {len(df)} 个模型结果")
    print(df[["model_name", "n_parameters", "test_auc", "source"]].to_string(index=False))

    plot_param_vs_auc(df, Path(args.output), title=args.title)
    print("\n[OK] 完成！")


if __name__ == "__main__":
    main()
