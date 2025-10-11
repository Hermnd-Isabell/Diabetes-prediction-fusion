#!/usr/bin/env python3
"""
增强版训练系统演示脚本

展示如何使用增强版训练系统进行多模态模型训练和比较
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_demo_data():
    """创建演示数据"""
    print("📊 创建演示数据...")
    
    # 创建数据目录
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 创建演示光谱数据
    n_samples = 200
    n_wavelengths = 1024
    
    spectra_data = []
    for i in range(n_samples):
        sample_id = f"P{i+1:04d}-{np.random.randint(1, 6)}"
        group = "DM" if i < n_samples // 2 else "Control"
        
        # 生成合成光谱
        spectrum = np.random.randn(n_wavelengths) * 0.1 + np.random.randn() * 0.5
        spectrum = np.abs(spectrum)  # 确保为正值
        
        row = {"Sample": sample_id, "Group": group}
        for j in range(n_wavelengths):
            row[f"wavelength_{j}"] = spectrum[j]
        
        spectra_data.append(row)
    
    spectra_df = pd.DataFrame(spectra_data)
    spectra_df.to_csv("data/spectra.csv", index=False)
    print(f"✅ 光谱数据创建完成: {len(spectra_df)} 样本")
    
    # 创建演示临床数据
    clinical_data = []
    for i in range(n_samples // 5):  # 每个病人5个扫描
        patient_id = f"P{i+1:04d}"
        group = "DM" if i < (n_samples // 5) // 2 else "Control"
        
        row = {
            "PatientID": patient_id,
            "Group": group,
            "Age": np.random.randint(30, 80),
            "BMI": np.random.uniform(18, 35),
            "Glucose": np.random.uniform(70, 200),
            "HbA1c": np.random.uniform(4, 12),
            "Cholesterol": np.random.uniform(150, 300),
            "Triglycerides": np.random.uniform(50, 400)
        }
        clinical_data.append(row)
    
    clinical_df = pd.DataFrame(clinical_data)
    clinical_df.to_csv("data/clinical.csv", index=False)
    print(f"✅ 临床数据创建完成: {len(clinical_df)} 病人")

def demo_single_model_training():
    """演示单个模型训练"""
    print("\n🚀 演示单个模型训练")
    print("=" * 50)
    
    from enhanced_main import load_config, prepare_data, train_single_model
    
    # 加载配置
    cfg = load_config("configs/enhanced_config.yaml")
    
    # 准备数据
    train_loader, val_loader, test_loader, dataset_info = prepare_data(cfg)
    
    # 训练AttentionMultimodal模型
    print("\n📊 训练AttentionMultimodal模型...")
    trainer = train_single_model(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_info=dataset_info,
        model_name="AttentionMultimodal"
    )
    
    return trainer

def demo_model_comparison():
    """演示模型比较"""
    print("\n🔄 演示模型比较")
    print("=" * 50)
    
    from enhanced_main import load_config, prepare_data, train_all_models
    
    # 加载配置
    cfg = load_config("configs/enhanced_config.yaml")
    
    # 准备数据
    train_loader, val_loader, test_loader, dataset_info = prepare_data(cfg)
    
    # 训练所有模型
    trainers = train_all_models(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_info=dataset_info
    )
    
    return trainers

def demo_visualization():
    """演示可视化功能"""
    print("\n📊 演示可视化功能")
    print("=" * 50)
    
    # 检查是否有训练结果
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ 没有找到训练结果，请先运行训练")
        return
    
    # 查找模型结果
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != "comparison"]
    
    if not model_dirs:
        print("❌ 没有找到模型结果目录")
        return
    
    print(f"📁 找到 {len(model_dirs)} 个模型结果:")
    for model_dir in model_dirs:
        print(f"   • {model_dir.name}")
        
        # 检查可视化文件
        viz_files = list(model_dir.glob("*.png"))
        if viz_files:
            print(f"     📊 可视化文件: {len(viz_files)} 个")
            for viz_file in viz_files:
                print(f"       - {viz_file.name}")
        else:
            print(f"     ⚠️  没有可视化文件")

def demo_interpretability():
    """演示可解释性分析"""
    print("\n🔍 演示可解释性分析")
    print("=" * 50)
    
    # 检查是否有可解释性分析结果
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ 没有找到训练结果，请先运行训练")
        return
    
    # 查找可解释性文件
    interpretability_files = list(results_dir.rglob("*shap*")) + list(results_dir.rglob("*attention*"))
    
    if interpretability_files:
        print(f"📊 找到 {len(interpretability_files)} 个可解释性分析文件:")
        for file in interpretability_files:
            print(f"   • {file.relative_to(results_dir)}")
    else:
        print("⚠️  没有找到可解释性分析文件")

def main():
    """主演示函数"""
    print("🚀 增强版多模态训练系统演示")
    print("=" * 60)
    
    # 检查配置文件
    config_file = "configs/enhanced_config.yaml"
    if not Path(config_file).exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    print("📋 演示选项:")
    print("1. 创建演示数据")
    print("2. 单个模型训练演示")
    print("3. 模型比较演示")
    print("4. 可视化功能演示")
    print("5. 可解释性分析演示")
    print("6. 完整演示流程")
    
    choice = input("\n请选择演示选项 (1-6): ").strip()
    
    if choice == "1":
        create_demo_data()
        
    elif choice == "2":
        if not Path("data/spectra.csv").exists():
            print("📊 创建演示数据...")
            create_demo_data()
        demo_single_model_training()
        
    elif choice == "3":
        if not Path("data/spectra.csv").exists():
            print("📊 创建演示数据...")
            create_demo_data()
        demo_model_comparison()
        
    elif choice == "4":
        demo_visualization()
        
    elif choice == "5":
        demo_interpretability()
        
    elif choice == "6":
        print("🚀 开始完整演示流程...")
        
        # 1. 创建数据
        create_demo_data()
        
        # 2. 单个模型训练
        trainer = demo_single_model_training()
        
        # 3. 可视化演示
        demo_visualization()
        
        # 4. 可解释性演示
        demo_interpretability()
        
        print("\n🎉 完整演示流程完成!")
        
    else:
        print("❌ 无效的选择")
        return
    
    print(f"\n📁 结果保存在: results/")
    print(f"💡 查看可视化结果: results/*/training_curves.png")
    print(f"💡 查看评估结果: results/*/evaluation_plots.png")

if __name__ == "__main__":
    main()

