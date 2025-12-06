#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成主要结果表

自动读取 experiments_summary.csv，计算模型参数量，生成论文用表格。

使用方法:
    python scripts/generate_main_results_table.py
    python scripts/generate_main_results_table.py --summary_path results/experiments_summary.csv
"""

import argparse
import time
import torch
import pandas as pd
from pathlib import Path

try:
    from enhanced_main import build_model, load_config, prepare_data
except ImportError:
    print("[ERROR] 错误: 无法导入 enhanced_main，请确保在项目根目录运行此脚本")
    import sys
    sys.exit(1)


def count_params(model):
    """计算模型可训练参数量（百万）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def estimate_inference_time(model, device, input_shape_spectra, input_shape_tabular, num_runs=100):
    """
    估算模型推理时间（毫秒）
    
    Args:
        model: 模型实例
        device: 设备
        input_shape_spectra: 光谱输入形状（用于 raw 模式）
        input_shape_tabular: 表格输入形状
        num_runs: 运行次数取平均
    """
    model.eval()
    model.to(device)
    
    # 创建 dummy 输入
    if hasattr(model, 'use_embedding_input') and model.use_embedding_input:
        # embedding 模式
        dummy_spectra = torch.randn(1, input_shape_tabular[0]).to(device)
        dummy_tabular = torch.randn(1, input_shape_tabular[0]).to(device)
        dummy_input = (
            {"embedding": dummy_spectra, "mask": torch.ones(1, dtype=torch.bool).to(device), "logits": None},
            {"embedding": dummy_tabular, "mask": torch.ones(1, dtype=torch.bool).to(device), "logits": None}
        )
    else:
        # raw 模式
        dummy_spectra = torch.randn(input_shape_spectra).to(device)
        dummy_tabular = torch.randn(1, input_shape_tabular[0]).to(device)
        dummy_mask = torch.ones(input_shape_spectra[:2], dtype=torch.bool).to(device)
        dummy_input = (dummy_spectra, dummy_mask, dummy_tabular)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(*dummy_input) if isinstance(dummy_input, tuple) else model(*dummy_input)
    
    # 同步 GPU（如果有）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(*dummy_input) if isinstance(dummy_input, tuple) else model(*dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = (time.time() - start_time) / num_runs * 1000  # 转换为毫秒
    return elapsed_time


def main():
    parser = argparse.ArgumentParser(description="生成主要结果表")
    parser.add_argument(
        "--summary_path",
        type=str,
        default="results/experiments_summary.csv",
        help="实验汇总 CSV 路径"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/enhanced_config.yaml",
        help="配置文件路径（用于构建模型计算参数量）"
    )
    parser.add_argument(
        "--add_params",
        action="store_true",
        help="是否计算并添加参数量列"
    )
    parser.add_argument(
        "--add_inference_time",
        action="store_true",
        help="是否计算并添加推理时间列（需要 GPU）"
    )
    
    args = parser.parse_args()
    
    summary_path = Path(args.summary_path)
    if not summary_path.exists():
        print(f"[ERROR] 错误: 找不到实验汇总文件: {summary_path}")
        print("[TIP] 提示: 请先运行 scripts/run_multimodal_experiments.py 生成汇总文件")
        return
    
    print(f"[INFO] 读取实验汇总: {summary_path}")
    df = pd.read_csv(summary_path)
    
    if args.add_params or args.add_inference_time:
        print("[INFO] 加载配置和数据集信息...")
        cfg = load_config(args.config)
        _, _, _, dataset_info = prepare_data(cfg)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] 使用设备: {device}")
        
        if args.add_params:
            print("[INFO] 计算模型参数量...")
            param_list = []
            for model_name in df["model_name"]:
                try:
                    cfg_copy = cfg.copy()
                    cfg_copy["model"] = cfg["model"].copy()
                    cfg_copy["model"]["name"] = model_name
                    model = build_model(cfg_copy, dataset_info['tab_dim'], dataset_info['spec_len'])
                    params = count_params(model)
                    param_list.append(params)
                    print(f"  - {model_name}: {params:.2f}M 参数")
                except Exception as e:
                    print(f"  [WARN] {model_name}: 计算参数量失败 ({e})")
                    param_list.append(None)
            df["Params(M)"] = param_list
        
        if args.add_inference_time:
            print("[INFO] 估算推理时间...")
            if device.type == 'cpu':
                print("  [WARN] 警告: CPU 模式下的推理时间可能不准确，建议使用 GPU")
            
            inference_times = []
            for model_name in df["model_name"]:
                try:
                    cfg_copy = cfg.copy()
                    cfg_copy["model"] = cfg["model"].copy()
                    cfg_copy["model"]["name"] = model_name
                    model = build_model(cfg_copy, dataset_info['tab_dim'], dataset_info['spec_len'])
                    
                    # 估算推理时间
                    time_ms = estimate_inference_time(
                        model, device,
                        input_shape_spectra=(1, 180, dataset_info['spec_len']),
                        input_shape_tabular=(dataset_info['tab_dim'],),
                        num_runs=100
                    )
                    inference_times.append(time_ms)
                    print(f"  - {model_name}: {time_ms:.2f} ms")
                except Exception as e:
                    print(f"  [WARN] {model_name}: 估算推理时间失败 ({e})")
                    inference_times.append(None)
            df["Inference Time(ms)"] = inference_times
    
    # 选择要输出的列（优先显示关键指标）
    priority_columns = [
        "model_name",
        "best_val_auc",
        "test_auc",
        "test_acc",
        "test_f1",
    ]
    
    if "Params(M)" in df.columns:
        priority_columns.append("Params(M)")
    if "Inference Time(ms)" in df.columns:
        priority_columns.append("Inference Time(ms)")
    
    # 添加其他列
    other_columns = [c for c in df.columns if c not in priority_columns]
    output_columns = priority_columns + other_columns
    
    df_output = df[output_columns]
    
    # 输出路径
    out_csv = Path("results/main_results_table.csv")
    out_md = Path("results/main_results_table.md")
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(out_csv, index=False, encoding="utf-8-sig")
    
    # Markdown 输出
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Main Results Table\n\n")
        f.write("Performance comparison of different multimodal fusion models.\n\n")
        f.write(df_output.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Notes\n")
        f.write("- All models are trained with the same data split and hyperparameters.\n")
        if "Params(M)" in df_output.columns:
            f.write("- Parameters are counted in millions (M).\n")
        if "Inference Time(ms)" in df_output.columns:
            f.write("- Inference time is measured on a single sample.\n")
    
    print(f"\n[OK] 主要结果表已生成:")
    print(f"  CSV: {out_csv}")
    print(f"  Markdown: {out_md}")
    print("\n表格预览:")
    print(df_output.to_string(index=False))


if __name__ == "__main__":
    main()

