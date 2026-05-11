#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化消融实验（Modular Ablation Experiments）

通过配置 override 验证 Lite 改造中每个组件的贡献。

使用方法:
  python scripts/run_ablation.py \\
      --config configs/experiment_base.yaml \\
      --model AttentionMultimodal \\
      --experiments full_model no_cross_attn no_aux_heads \\
      --lite

实验定义（内置）:
  full_model    - 完整模型
  no_cross_attn - 移除 Cross-Attention（使用 concat 融合）
  no_aux_heads  - 移除辅助监督头
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhanced_main import load_config, prepare_data, train_single_model


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 内置消融实验定义
ABLATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "full_model": {
        "description": "完整模型",
        "overrides": {},
    },
    "no_cross_attn": {
        "description": "移除 Cross-Attention，改用 Concat 融合",
        "overrides": {
            "model": {"fusion": "concat"},
        },
    },
    "no_aux_heads": {
        "description": "移除辅助监督头",
        "overrides": {
            "train": {"advanced": {"aux_weight": 0.0}},
        },
    },
    "avg_pool_only": {
        "description": "Attention Pooling 替换为 AvgPool",
        "overrides": {
            "model": {"lite": {"use_attention_pooling": False}},
        },
    },
    "no_lite": {
        "description": "关闭 Lite 模式（完整参数）",
        "overrides": {
            "model": {"lite": {"enabled": False}},
        },
    },
}


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """递归应用配置覆盖。"""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key in cfg and isinstance(cfg[key], dict) and isinstance(value, dict):
            cfg[key] = apply_overrides(cfg[key], value)
        else:
            cfg[key] = value
    return cfg


def run_ablation_experiment(
    base_cfg: dict,
    model_name: str,
    exp_name: str,
    exp_def: dict,
    save_root: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """执行单次消融实验。"""
    exp_dir = save_root / "ablation" / exp_name
    metrics_path = exp_dir / "metrics_summary.json"

    if exp_dir.exists() and metrics_path.exists() and not overwrite:
        print(f"  [SKIP] {exp_name} 已存在")
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    cfg = apply_overrides(base_cfg, exp_def["overrides"])
    cfg["model"]["name"] = model_name
    cfg.setdefault("train", {})
    cfg["train"]["save_dir"] = str(save_root)

    train_loader, val_loader, test_loader, dataset_info = prepare_data(cfg)
    print(f"  [TRAIN] {exp_name}: {exp_def['description']}")
    trainer = train_single_model(
        cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_info=dataset_info,
        model_name=model_name,
    )

    # 将结果从 src_dir 复制到 ablation 目录（仅复制 JSON/CSV/日志，跳过 .pt）
    src_dir = save_root / model_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        import shutil
        for f in list(src_dir.iterdir()):
            if f.is_file() and f.suffix in (".json", ".csv", ".txt", ".log"):
                dst = exp_dir / f.name
                if dst.exists():
                    dst.unlink()
                shutil.copy2(str(f), str(dst))

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return metrics


def run_ablations(
    config_path: str,
    model_name: str,
    experiments: List[str],
    lite: bool = False,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    cfg["model"]["name"] = model_name
    if lite:
        cfg.setdefault("model", {}).setdefault("lite", {})
        cfg["model"]["lite"]["enabled"] = True

    # 固定数据划分种子，确保消融实验之间数据一致
    cfg.setdefault("experiment", {})
    cfg["experiment"]["random_seed"] = seed
    cfg.setdefault("data", {})
    cfg["data"]["split_seed"] = seed

    set_seed(seed)
    save_root = Path(cfg.get("experiment", {}).get("output_dir", "results"))

    rows = []
    for exp_name in experiments:
        if exp_name not in ABLATION_REGISTRY:
            print(f"[WARN] 未知消融实验: {exp_name}，跳过")
            continue
        exp_def = ABLATION_REGISTRY[exp_name]
        print(f"\n[INFO] 消融实验: {exp_name}")
        metrics = run_ablation_experiment(cfg, model_name, exp_name, exp_def, save_root, overwrite)
        row = {
            "experiment": exp_name,
            "description": exp_def["description"],
            "test_auc": metrics.get("test_auc"),
            "test_acc": metrics.get("test_acc"),
            "best_val_auc": metrics.get("best_val_auc"),
            "n_parameters": metrics.get("n_parameters"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = save_root / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 消融汇总已保存: {csv_path}")
    print(df.to_string(index=False))
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Modular Ablation Experiments")
    parser.add_argument("--config", type=str, default="configs/experiment_base.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--experiments", type=str, nargs="+", required=True,
                        help=f"实验名称列表，可选: {list(ABLATION_REGISTRY.keys())}")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("[INFO] 模块化消融实验")
    print("=" * 80)
    print(f"模型: {args.model} | 实验: {args.experiments} | seed: {args.seed}")

    run_ablations(
        config_path=args.config,
        model_name=args.model,
        experiments=args.experiments,
        lite=args.lite,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
