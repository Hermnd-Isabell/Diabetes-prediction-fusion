#!/usr/bin/env python3
"""
快速实验启动脚本

提供便捷的命令来运行不同的实验配置
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示描述"""
    print(f"\n🚀 {description}")
    print(f"📝 命令: {cmd}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 多模态模型实验启动器")
    print("=" * 60)
    
    # 检查配置文件是否存在
    config_file = "configs/enhanced_config.yaml"
    if not Path(config_file).exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    # 检查数据文件是否存在
    data_files = ["data/spectra.csv", "data/clinical.csv"]
    for data_file in data_files:
        if not Path(data_file).exists():
            print(f"❌ 数据文件不存在: {data_file}")
            return
    
    print("📋 可用的实验选项:")
    print("1. 训练单个模型 (AttentionMultimodal)")
    print("2. 训练单个模型 (ConcatFusion)")
    print("3. 训练单个模型 (MMTMMultimodal)")
    print("4. 训练单个模型 (TFTMultimodal)")
    print("5. 训练所有模型并比较")
    print("6. 仅评估已训练的模型")
    print("7. 自定义实验")
    
    choice = input("\n请选择实验选项 (1-7): ").strip()
    
    if choice == "1":
        cmd = f"python enhanced_main.py --config {config_file} --model AttentionMultimodal"
        run_command(cmd, "训练注意力机制模型")
        
    elif choice == "2":
        cmd = f"python enhanced_main.py --config {config_file} --model ConcatFusion"
        run_command(cmd, "训练基线融合模型")
        
    elif choice == "3":
        cmd = f"python enhanced_main.py --config {config_file} --model MMTMMultimodal"
        run_command(cmd, "训练MMTM多模态融合模型")
        
    elif choice == "4":
        cmd = f"python enhanced_main.py --config {config_file} --model TFTMultimodal"
        run_command(cmd, "训练TFT时序融合模型")
        
    elif choice == "5":
        cmd = f"python enhanced_main.py --config {config_file} --train-all"
        run_command(cmd, "训练所有模型并比较")
        
    elif choice == "6":
        model_name = input("请输入要评估的模型名称: ").strip()
        cmd = f"python enhanced_main.py --config {config_file} --eval-only {model_name}"
        run_command(cmd, f"评估模型 {model_name}")
        
    elif choice == "7":
        print("\n📝 自定义实验选项:")
        print("可用的模型:")
        print("  - AttentionMultimodal")
        print("  - ConcatFusion")
        print("  - EnsembleFusion")
        print("  - MMTMMultimodal")
        print("  - TFTMultimodal")
        print("  - EnhancedMMTM")
        
        model_name = input("请输入模型名称: ").strip()
        epochs = input("请输入训练轮数 (默认50): ").strip() or "50"
        
        # 创建临时配置文件
        temp_config = f"configs/temp_config_{model_name}.yaml"
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # 修改训练轮数
        config_content = config_content.replace('epochs: 50', f'epochs: {epochs}')
        config_content = config_content.replace('name: AttentionMultimodal', f'name: {model_name}')
        
        with open(temp_config, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        cmd = f"python enhanced_main.py --config {temp_config}"
        success = run_command(cmd, f"自定义训练 {model_name}")
        
        # 清理临时文件
        if success and Path(temp_config).exists():
            os.remove(temp_config)
    
    else:
        print("❌ 无效的选择")
        return
    
    print(f"\n🎉 实验完成!")
    print(f"📁 结果保存在: results/")

if __name__ == "__main__":
    main()

