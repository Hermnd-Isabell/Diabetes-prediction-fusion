#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将旧版光谱 embedding CSV 转换为多模态框架统一格式：
    PatientID, Split, Label, feature_0, feature_1, ...

适用于 baseline2.py 早期导出的：
    Sample, Split, Label, Label_Name, Dataset_Index, feature_0, ...

使用方法（在项目根目录）：
    conda activate pytorch2.5.0
    python scripts/convert_spectrum_features.py
"""

import pandas as pd
from pathlib import Path


def extract_patient_id(sample):
    """
    从 Sample 列提取 PatientID。

    规则：
    - 如果形如 DM-92 / C-15 等（前缀是非数字、后缀是数字），则返回后缀数字部分（例如 92, 15），
      以便与临床 PatientID 的数字编号对齐。
    - 否则，取 '-' 前面的部分（例如 100-176.txt -> 100）。
    - 若无 '-'，则直接返回原字符串。
    """
    s = str(sample)
    if "-" in s:
        left, right = s.split("-", 1)
        # 典型光谱 Sample: DM-92, C-15 等 → 与临床 PatientID 数字对齐
        if right.isdigit() and not left.isdigit():
            return right
        return left
    return s


def convert_file(path_str: str):
    path = Path(path_str)
    if not path.exists():
        print(f"[ERROR] 找不到文件: {path}")
        return

    print(f"[INFO] 读取光谱特征文件: {path}")
    df = pd.read_csv(path)

    # 如果已经有 PatientID 列，进一步检查是否需要“重新对齐”
    if "PatientID" in df.columns:
        unique_pids = set(str(x) for x in df["PatientID"].unique())
        all_numeric = all(pid.isdigit() for pid in unique_pids)
        if all_numeric:
            # 已经是数字型 PatientID，认为是正确的新格式
            print("[INFO] 检测到数字型 PatientID 列，文件已是新格式，跳过转换。")
            print(df.head(2))
            return

        # 否则，尝试从 .bak 备份恢复原始 Sample，再重新计算 PatientID
        backup_path = path.with_suffix(path.suffix + ".bak")
        if backup_path.exists():
            print(f"[WARN] 检测到非数字 PatientID（例如 {list(unique_pids)[:5]}），尝试从备份重算 PatientID: {backup_path}")
            df = pd.read_csv(backup_path)
        else:
            print("[WARN] 已存在 PatientID 列但为非数字，且找不到备份文件，无法安全重算 PatientID，暂时跳过。")
            print(df.head(2))
            return

    required = {"Sample", "Split", "Label"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] CSV 缺少必要列: {missing}，无法转换。")
        return

    # 提取 PatientID
    patient_ids = df["Sample"].map(extract_patient_id)

    # 识别并排序特征列
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        print("[ERROR] 未找到任何 feature_* 列，无法转换。")
        return

    def _feat_idx(name: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 10**9

    feature_cols_sorted = sorted(feature_cols, key=_feat_idx)

    # 组装新 DataFrame：PatientID, Split, Label, feature_*
    out_df = pd.concat(
        [
            pd.Series(patient_ids, name="PatientID"),
            df["Split"],
            df["Label"],
            df[feature_cols_sorted].copy(),
        ],
        axis=1,
    )

    print("[INFO] 新列顺序:", list(out_df.columns)[:10], "...")
    backup_path = path.with_suffix(path.suffix + ".bak")
    # 若备份不存在，则创建备份；若已存在，说明之前已经备份过，直接覆盖原文件即可
    if not backup_path.exists():
        path.rename(backup_path)
        print(f"[INFO] 已备份原文件到: {backup_path}")
    else:
        print(f"[INFO] 备份文件已存在: {backup_path}，本次不再重复备份。")

    out_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] 已按统一格式覆盖保存: {path}")


def main():
    # 一次性转换多个光谱特征文件
    paths = [
        "spectrum_dimension/diabetes_results/MLP_Baseline_features.csv",
        "spectrum_dimension/diabetes_results/Feature_Interaction_MLP_features.csv",
        "spectrum_dimension/diabetes_results/Light_CNN_MLP_features.csv",
    ]
    for p in paths:
        print("=" * 80)
        print(f"[INFO] 开始处理: {p}")
        convert_file(p)
    print("=" * 80)
    print("[OK] 所有指定的光谱特征文件已尝试转换完成。")


if __name__ == "__main__":
    main()


