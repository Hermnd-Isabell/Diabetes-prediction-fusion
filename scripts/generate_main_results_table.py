#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版结果聚合脚本（Enhanced Results Aggregator）

自动读取多种结果来源（单折 / 多种子 / CV），生成对比表格与统计显著性标注。

使用方法:
  python scripts/generate_main_results_table.py
  python scripts/generate_main_results_table.py --results_dir results --config configs/experiment_base.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import csv
import itertools
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import wilcoxon

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from enhanced_main import build_model, load_config, prepare_data
except ImportError:
    build_model = None
    load_config = None
    prepare_data = None


def count_params_from_json(metrics: dict) -> Optional[int]:
    """从 metrics_summary.json 中提取参数量。"""
    for key in ("n_parameters", "total_parameters", "trainable_parameters"):
        if key in metrics and metrics[key] is not None:
            return int(metrics[key])
    return None


def load_model_results(model_dir: Path) -> Dict[str, Any]:
    """
    从模型结果目录加载最优可用指标。
    优先级：cv_summary.json > seed_*/metrics_summary.json（汇总）> metrics_summary.json（单折）
    """
    model_name = model_dir.name
    row = {"model_name": model_name}

    # 1) CV 汇总
    cv_path = model_dir / "cv_summary.json"
    if cv_path.exists():
        with open(cv_path, "r", encoding="utf-8") as f:
            cv_data = json.load(f)
        row["source"] = "cv"
        row["test_auc"] = cv_data.get("test_auc_mean", cv_data.get("mean_test_auc", np.nan))
        row["test_auc_std"] = cv_data.get("test_auc_std", cv_data.get("std_test_auc", np.nan))
        row["test_acc"] = cv_data.get("test_acc_mean", np.nan)
        row["n_parameters"] = cv_data.get("n_parameters", np.nan)
        # 追加扩展指标（如果 CV 汇总中有）
        for ext_key in ("test_macro_auc", "test_weighted_auc", "test_cohens_kappa", "test_qwk"):
            mean_key = f"{ext_key}_mean"
            std_key = f"{ext_key}_std"
            if mean_key in cv_data:
                row[ext_key] = cv_data[mean_key]
                row[f"{ext_key}_std"] = cv_data.get(std_key, np.nan)
        return row

    # 2) 多种子汇总
    seed_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    if seed_dirs:
        seed_metrics = []
        for sd in seed_dirs:
            mp = sd / "metrics_summary.json"
            if mp.exists():
                with open(mp, "r", encoding="utf-8") as f:
                    seed_metrics.append(json.load(f))
        if seed_metrics:
            df_seeds = pd.DataFrame(seed_metrics)
            row["source"] = "multi_seed"
            row["test_auc"] = df_seeds["test_auc"].mean() if "test_auc" in df_seeds.columns else np.nan
            row["test_auc_std"] = df_seeds["test_auc"].std() if "test_auc" in df_seeds.columns else np.nan
            row["test_acc"] = df_seeds["test_acc"].mean() if "test_acc" in df_seeds.columns else np.nan
            row["best_val_auc"] = df_seeds["best_val_auc"].mean() if "best_val_auc" in df_seeds.columns else np.nan
            row["n_parameters"] = df_seeds["n_parameters"].mean() if "n_parameters" in df_seeds.columns else np.nan
            return row

    # 3) 单折 metrics_summary.json
    mp = model_dir / "metrics_summary.json"
    if mp.exists():
        with open(mp, "r", encoding="utf-8") as f:
            m = json.load(f)
        row["source"] = "single_fold"
        row["test_auc"] = m.get("test_auc", np.nan)
        row["test_acc"] = m.get("test_acc", np.nan)
        row["best_val_auc"] = m.get("best_val_auc", np.nan)
        row["n_parameters"] = count_params_from_json(m)
        return row

    # 4) 旧格式 results.json
    rp = model_dir / "results.json"
    if rp.exists():
        with open(rp, "r", encoding="utf-8") as f:
            m = json.load(f)
        row["source"] = "legacy"
        test_metrics = m.get("test_result", {}).get("metrics", {})
        row["test_auc"] = test_metrics.get("auc", np.nan)
        row["test_acc"] = test_metrics.get("acc", np.nan)
        model_summary = m.get("model_summary", {})
        row["n_parameters"] = model_summary.get("total_parameters", np.nan)
        return row

    return {}


def compute_pairwise_pvalues(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """
    对多种子结果进行成对 t-test，返回 p-value 矩阵。
    仅当两个模型都有 seed_*/metrics_summary.json 时计算。
    """
    models = df["model_name"].tolist()
    pval_matrix = pd.DataFrame(np.nan, index=models, columns=models)

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i >= j:
                continue
            seeds1 = []
            seeds2 = []
            m1_dir = results_dir / m1
            m2_dir = results_dir / m2
            if m1_dir.exists() and m2_dir.exists():
                for sd in m1_dir.iterdir():
                    if sd.is_dir() and sd.name.startswith("seed_"):
                        mp = sd / "metrics_summary.json"
                        if mp.exists():
                            with open(mp, "r", encoding="utf-8") as f:
                                seeds1.append(json.load(f).get("test_auc", np.nan))
                for sd in m2_dir.iterdir():
                    if sd.is_dir() and sd.name.startswith("seed_"):
                        mp = sd / "metrics_summary.json"
                        if mp.exists():
                            with open(mp, "r", encoding="utf-8") as f:
                                seeds2.append(json.load(f).get("test_auc", np.nan))
            if len(seeds1) >= 2 and len(seeds2) >= 2:
                try:
                    _, pval = stats.ttest_ind(seeds1, seeds2, equal_var=False)
                    pval_matrix.loc[m1, m2] = pval
                    pval_matrix.loc[m2, m1] = pval
                except Exception:
                    pass

    return pval_matrix


def pairwise_wilcoxon(model_fold_scores: dict, metric: str = "test_macro_auc", alpha: float = 0.05) -> List[Dict[str, Any]]:
    """
    对 CV fold 级结果进行成对 Wilcoxon Signed-Rank Test。
    model_fold_scores: {'ModelA': [fold0, fold1, ..., fold4], 'ModelB': [...]}
    要求所有模型列表长度相同（同为 n_folds）。
    返回: list of dict，含 model_a, model_b, n_folds, statistic, pvalue, significant
    """
    results = []
    names = list(model_fold_scores.keys())
    for a, b in itertools.combinations(names, 2):
        scores_a = np.array(model_fold_scores[a])
        scores_b = np.array(model_fold_scores[b])
        if len(scores_a) != len(scores_b):
            continue
        mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
        if np.sum(mask) < 3:
            continue
        try:
            stat, p = wilcoxon(scores_a[mask], scores_b[mask], alternative="two-sided")
            results.append({
                "model_a": a,
                "model_b": b,
                "n_folds": int(np.sum(mask)),
                "statistic": float(stat),
                "pvalue": float(p),
                "significant": p < alpha,
            })
        except Exception:
            pass
    return results


def format_value_with_std(val, std) -> str:
    """格式化 mean ± std 字符串。"""
    if pd.isna(val):
        return "N/A"
    if pd.isna(std) or std == 0:
        return f"{val:.3f}"
    return f"{val:.3f} ± {std:.3f}"


def generate_latex_table(df: pd.DataFrame, out_path: Path):
    """生成 LaTeX 表格片段。"""
    cols = ["model_name", "n_parameters", "test_auc", "test_acc", "source"]
    sub = df[[c for c in cols if c in df.columns]].copy()
    sub["n_parameters"] = sub["n_parameters"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
    sub["test_auc"] = sub["test_auc"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    sub["test_acc"] = sub["test_acc"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

    latex = sub.to_latex(index=False, escape=False, float_format="%.3f")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("% LaTeX table generated by generate_main_results_table.py\n")
        f.write(latex)
    print(f"[OK] LaTeX 表格已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced results comparison table")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--config", type=str, default="configs/experiment_base.yaml")
    parser.add_argument("--summary_path", type=str, default=None,
                        help="可选：直接传入 experiments_summary.csv 路径（旧格式兼容）")
    parser.add_argument("--add_params", action="store_true", help="从配置文件计算参数量")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ====== 加载数据 ======
    if args.summary_path and Path(args.summary_path).exists():
        print(f"[INFO] 读取旧格式汇总: {args.summary_path}")
        df = pd.read_csv(args.summary_path)
        # 尝试从 results_dir 补充更丰富的信息
        rows = []
        for _, row in df.iterrows():
            model_name = row.get("model_name")
            if model_name and (results_dir / model_name).exists():
                extra = load_model_results(results_dir / model_name)
                if extra:
                    merged = {**row.to_dict(), **extra}
                    rows.append(merged)
                else:
                    rows.append(row.to_dict())
            else:
                rows.append(row.to_dict())
        df = pd.DataFrame(rows)
    else:
        print(f"[INFO] 扫描结果目录: {results_dir}")
        rows = []
        for model_dir in sorted(results_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_dir.name in {"comparison", "sensitivity", "ablation"}:
                continue
            res = load_model_results(model_dir)
            if res:
                rows.append(res)
        df = pd.DataFrame(rows)

    if len(df) == 0:
        print("[ERROR] 未找到任何模型结果")
        sys.exit(1)

    # ====== 补充参数量（若缺失且请求了 add_params）======
    if args.add_params and build_model is not None and prepare_data is not None:
        print("[INFO] 计算缺失的参数量...")
        try:
            cfg = load_config(args.config)
            _, _, _, dataset_info = prepare_data(cfg)
            for idx, row in df.iterrows():
                if pd.isna(row.get("n_parameters")):
                    model_name = row["model_name"]
                    try:
                        cfg_copy = copy.deepcopy(cfg)
                        cfg_copy["model"]["name"] = model_name
                        model = build_model(cfg_copy, dataset_info["tab_dim"], dataset_info["spec_len"])
                        n_params = sum(p.numel() for p in model.parameters())
                        df.at[idx, "n_parameters"] = n_params
                        print(f"  - {model_name}: {n_params:,} params")
                    except Exception as e:
                        print(f"  [WARN] {model_name}: 计算参数量失败 ({e})")
        except Exception as e:
            print(f"[WARN] 无法加载配置计算参数量: {e}")

    # ====== 加载配置（用于统计检验参数）======
    stat_cfg = {}
    try:
        if load_config is not None:
            stat_cfg = load_config(args.config).get("evaluation", {}).get("statistical_test", {})
    except Exception:
        pass
    alpha = stat_cfg.get("alpha", 0.05)
    metric_for_comparison = stat_cfg.get("metric_for_comparison", "macro_auc")
    # fold_results.csv 中的列名前缀为 test_
    fold_metric_col = f"test_{metric_for_comparison}"

    # ====== 统计显著性（t-test，用于多种子）======
    pval_matrix = compute_pairwise_pvalues(df, results_dir)
    if not pval_matrix.isna().all().all():
        print("[INFO] 成对 t-test p-value 矩阵:")
        print(pval_matrix.to_string())
        # 在表格中标注显著差异
        df["significant_vs_best"] = ""
        if "test_auc" in df.columns and not df["test_auc"].isna().all():
            best_idx = df["test_auc"].idxmax()
            best_model = df.loc[best_idx, "model_name"]
            for idx, row in df.iterrows():
                m = row["model_name"]
                if m == best_model:
                    continue
                pval = pval_matrix.loc.get(m, {}).get(best_model, np.nan)
                if pd.notna(pval) and pval < 0.05:
                    df.at[idx, "significant_vs_best"] = f"* (p={pval:.3f})"

    # ====== 统计显著性（Wilcoxon，用于 CV fold 配对）======
    wilcoxon_results = []
    model_fold_scores = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in {"comparison", "sensitivity", "ablation"}:
            continue
        fold_csv = model_dir / "fold_results.csv"
        if fold_csv.exists():
            try:
                fold_df = pd.read_csv(fold_csv)
                if fold_metric_col in fold_df.columns:
                    scores = fold_df[fold_metric_col].replace("NA", np.nan).astype(float).tolist()
                    model_fold_scores[model_dir.name] = scores
            except Exception as e:
                print(f"[WARN] 读取 {fold_csv} 失败: {e}")

    if len(model_fold_scores) >= 2:
        wilcoxon_results = pairwise_wilcoxon(model_fold_scores, metric=fold_metric_col, alpha=alpha)
        if wilcoxon_results:
            print(f"[INFO] Wilcoxon Signed-Rank Test (metric={fold_metric_col}, alpha={alpha}):")
            for r in wilcoxon_results:
                sig_mark = " *" if r["significant"] else ""
                print(f"  {r['model_a']} vs {r['model_b']}: p={r['pvalue']:.4f}{sig_mark}")
            # 保存 Wilcoxon 结果 CSV
            wilcoxon_csv = out_dir / "wilcoxon_results.csv"
            pd.DataFrame(wilcoxon_results).to_csv(wilcoxon_csv, index=False, encoding="utf-8-sig")
            print(f"[OK] Wilcoxon 结果已保存: {wilcoxon_csv}")

    # ====== 格式化输出列 ======
    display_cols = ["model_name", "n_parameters", "test_auc", "test_acc", "best_val_auc", "source"]
    output_cols = [c for c in display_cols if c in df.columns]
    # 添加 std 列（如果存在）
    if "test_auc_std" in df.columns:
        # 插入 test_auc 之后
        auc_idx = output_cols.index("test_auc") if "test_auc" in output_cols else len(output_cols)
        output_cols.insert(auc_idx + 1, "test_auc_std")

    # 其他未列出的列也保留
    for c in df.columns:
        if c not in output_cols:
            output_cols.append(c)

    df_out = df[output_cols].copy()

    # ====== 保存 CSV ======
    csv_path = out_dir / "model_comparison.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] CSV 已保存: {csv_path}")

    # ====== 保存 Markdown ======
    md_path = out_dir / "model_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Table\n\n")
        f.write("| Model | # Params | Test AUC | Test Acc | Source |\n")
        f.write("|-------|----------|----------|----------|--------|\n")
        for _, row in df_out.iterrows():
            name = row.get("model_name", "N/A")
            nparams = f"{int(row['n_parameters']):,}" if pd.notna(row.get("n_parameters")) else "N/A"
            auc = format_value_with_std(row.get("test_auc"), row.get("test_auc_std"))
            acc = f"{row['test_acc']:.3f}" if pd.notna(row.get("test_acc")) else "N/A"
            src = row.get("source", "unknown")
            sig = row.get("significant_vs_best", "")
            f.write(f"| {name} | {nparams} | {auc} {sig} | {acc} | {src} |\n")
        f.write("\n\n* * indicates p < 0.05 vs best model (two-sample t-test)\n")

        # 追加 Wilcoxon 结果（如果有）
        if wilcoxon_results:
            f.write(f"\n## Pairwise Wilcoxon Signed-Rank Test (paired by CV fold)\n\n")
            f.write(f"Metric: `{fold_metric_col}`, alpha={alpha}\n\n")
            f.write("| Model A | Model B | n_folds | statistic | p-value | Significant |\n")
            f.write("|---------|---------|---------|-----------|---------|-------------|\n")
            for r in wilcoxon_results:
                sig = "*" if r["significant"] else ""
                f.write(f"| {r['model_a']} | {r['model_b']} | {r['n_folds']} | {r['statistic']:.2f} | {r['pvalue']:.4f} | {sig} |\n")
    print(f"[OK] Markdown 已保存: {md_path}")

    # ====== 保存 LaTeX ======
    latex_path = out_dir / "model_comparison.tex"
    try:
        generate_latex_table(df_out, latex_path)
    except Exception as e:
        print(f"[WARN] LaTeX 生成失败: {e}")

    # ====== 打印预览 ======
    print("\n" + "=" * 80)
    print("表格预览:")
    print("=" * 80)
    print(df_out.to_string(index=False))
    print("\n[OK] 结果聚合完成！")


if __name__ == "__main__":
    main()
