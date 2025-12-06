#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将旧版临床 embedding CSV 转换为多模态框架统一格式：
    PatientID, Label, feature_0, feature_1, ...

适用于当前的 clinical.csv 头部为：
    PatientID, Group, pc_1, pc_2, ...

转换规则：
    - 新增 Label 列 = Group
    - 删除 Group 列
    - 所有 pc_k 按顺序重命名为 feature_{k-1}

使用方法（在项目根目录）：
    conda activate pytorch2.5.0
    cd "C:\\Users\\yuzih\\Desktop\\diabetes code\\Fusion"
    python scripts/convert_clinical_features.py
"""

import pandas as pd
from pathlib import Path


def convert_clinical(path_str: str):
    path = Path(path_str)
    if not path.exists():
        print(f"[ERROR] 找不到文件: {path}")
        return

    print(f"[INFO] 读取临床特征文件: {path}")
    df = pd.read_csv(path)

    cols = list(df.columns)
    print("[INFO] 原始列前几项:", cols[:10])

    if "Label" in df.columns and any(c.startswith("feature_") for c in df.columns):
        print("[INFO] 检测到 Label + feature_* 列，文件似乎已经是新格式，跳过转换。")
        print(df.head(2))
        return

    if "Group" not in df.columns:
        print("[ERROR] 缺少 Group 列，无法确定标签。")
        return

    # 新增 Label 列（整数）
    df["Label"] = df["Group"].astype(int)

    # 找到所有 pc_* 特征列，并按数字顺序排序
    pc_cols = [c for c in df.columns if c.startswith("pc_")]
    if not pc_cols:
        print("[ERROR] 未找到任何 pc_* 特征列，无法转换。")
        return

    def _pc_idx(name: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 10**9

    pc_cols_sorted = sorted(pc_cols, key=_pc_idx)

    # 构造新的 feature_* 列名
    feature_cols = {}
    for i, col in enumerate(pc_cols_sorted):
        feature_cols[col] = f"feature_{i}"

    df_renamed = df.rename(columns=feature_cols)

    # 组装输出列顺序：PatientID, Label, feature_0 ...
    feature_col_names = [feature_cols[c] for c in pc_cols_sorted]
    out_df = pd.concat(
        [
            df_renamed["PatientID"],
            df_renamed["Label"],
            df_renamed[feature_col_names],
        ],
        axis=1,
    )

    print("[INFO] 新列前几项:", list(out_df.columns)[:10])

    backup_path = path.with_suffix(path.suffix + ".bak")
    path.rename(backup_path)
    print(f"[INFO] 已备份原文件到: {backup_path}")

    out_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] 已按统一格式覆盖保存: {path}")


def main():
    convert_clinical("data/clinical.csv")


if __name__ == "__main__":
    main()


