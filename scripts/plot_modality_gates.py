#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模态权重可视化脚本

读取训练过程中记录的 modality_gate_history.json，绘制模态融合权重随训练轮次的变化曲线。

使用方法:
    python scripts/plot_modality_gates.py
"""

import json
import os
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_modality_history(path: str):
    """加载模态权重历史 JSON 文件"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到模态权重历史文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    return history


def plot_modality_gates(history, save_path: str = None, show_plot: bool = False):
    """
    绘制模态权重曲线
    
    Args:
        history: 模态权重历史列表
        save_path: 保存路径（可选）
        show_plot: 是否显示图表（默认 False）
    """
    if not history:
        print("⚠ 模态权重历史为空，无法绘图")
        return
    
    epochs = [item["epoch"] for item in history]
    w_spec = [item["gate_softmax"][0] for item in history]  # Raman 光谱
    w_clin = [item["gate_softmax"][1] for item in history]  # 临床特征
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, w_spec, marker="o", label="Raman Spectra Weight", linewidth=2, markersize=6)
    plt.plot(epochs, w_clin, marker="s", label="Clinical Features Weight", linewidth=2, markersize=6)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Modality Weight", fontsize=12)
    plt.title("Modality Fusion Weights over Training", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"✅ 模态权重曲线已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="绘制模态融合权重曲线")
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="模态权重历史 JSON 文件路径（默认：results/{model_name}/modality_gate_history.json）"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="AttentionMultimodal",
        help="模型名称（默认：AttentionMultimodal）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片路径（默认：与 JSON 同目录下的 modality_gate_curve.png）"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示图表（默认：仅保存）"
    )
    
    args = parser.parse_args()
    
    # 确定 JSON 路径
    if args.json_path:
        json_path = Path(args.json_path)
    else:
        default_root = Path("results")
        json_path = default_root / args.model_name / "modality_gate_history.json"
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = json_path.parent / "modality_gate_curve.png"
    
    # 加载并绘图
    try:
        history = load_modality_history(str(json_path))
        plot_modality_gates(history, save_path=str(output_path), show_plot=args.show)
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print(f"💡 提示: 请先运行训练，确保已生成 {json_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

