#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成实验设计表（Ablation Table）

自动生成模型组件对比表，输出 CSV 和 Markdown 格式，适合论文使用。

使用方法:
    python scripts/generate_design_table.py
"""

import pandas as pd
from pathlib import Path


# 每个模型有哪些模块，手动维护即可
MODEL_COMPONENTS = {
    "BaselineMultimodal": {
        "Raw Input": "✗",
        "Embedding": "✔",
        "Soft Gating": "✔",  # BaselineMultimodal 已支持 Soft Gating
        "Fusion Gate": "✔",  # BaselineMultimodal 已支持 Fusion Gate
        "Modality Dropout": "✔",
        "MMTM Block": "✗",
    },
    "AttentionMultimodal": {
        "Raw Input": "✗",
        "Embedding": "✔",
        "Soft Gating": "✔",
        "Fusion Gate": "✔",
        "Modality Dropout": "✔",
        "MMTM Block": "✗",
    },
    "EnhancedMMTM": {
        "Raw Input": "✗",
        "Embedding": "✔",
        "Soft Gating": "✔",
        "Fusion Gate": "✔",
        "Modality Dropout": "✔",
        "MMTM Block": "✔",
    },
    "TFTMultimodal": {
        "Raw Input": "✗",
        "Embedding": "✔",
        "Soft Gating": "✔",
        "Fusion Gate": "✔",
        "Modality Dropout": "✔",
        "MMTM Block": "✗",
    },
    # 可选：添加其他模型
    "ConcatFusion": {
        "Raw Input": "✗",
        "Embedding": "✔",
        "Soft Gating": "✗",
        "Fusion Gate": "✔",
        "Modality Dropout": "✔",
        "MMTM Block": "✗",
    },
    "EnsembleFusion": {
        "Raw Input": "✗",
        "Embedding": "✔",
        "Soft Gating": "✗",
        "Fusion Gate": "✔",
        "Modality Dropout": "✔",
        "MMTM Block": "✗",
    },
}


def main():
    """生成实验设计表"""
    df = pd.DataFrame.from_dict(MODEL_COMPONENTS, orient="index")
    
    # 确保列顺序
    column_order = [
        "Raw Input",
        "Embedding",
        "Soft Gating",
        "Fusion Gate",
        "Modality Dropout",
        "MMTM Block",
    ]
    df = df[column_order]
    
    # 输出路径
    out_csv = Path("results/experiment_design_table.csv")
    out_md = Path("results/experiment_design_table.md")
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, encoding="utf-8-sig")
    
    # Markdown 输出（表格格式）
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Experiment Design Table (Ablation Study)\n\n")
        f.write("This table shows the components used in each model.\n\n")
        f.write(df.to_markdown())
        f.write("\n\n")
        f.write("Legend:\n")
        f.write("- ✔: Component enabled\n")
        f.write("- ✗: Component disabled\n")
    
    print("[OK] 实验设计表已生成:")
    print(f"  CSV: {out_csv}")
    print(f"  Markdown: {out_md}")
    print("\n表格预览 (前5行):")
    # 替换特殊字符以避免编码问题
    df_preview = df.head().copy()
    df_preview = df_preview.replace("✔", "[YES]").replace("✗", "[NO]")
    try:
        print(df_preview.to_string())
    except UnicodeEncodeError:
        # 如果仍有编码问题，只打印基本信息
        print(f"  共 {len(df)} 个模型，{len(df.columns)} 个组件列")
        print("  详细内容请查看生成的 CSV 或 Markdown 文件")


if __name__ == "__main__":
    main()

