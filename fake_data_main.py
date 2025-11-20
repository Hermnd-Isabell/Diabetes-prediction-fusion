#!/usr/bin/env python3
"""
假数据训练主函数 - 使用现有训练系统训练假数据

这个脚本允许你使用现有的训练器和main函数来训练模型，
但使用假数据而不是真实数据，非常适合测试和验证。

使用方法:
1. 使用默认配置: python fake_data_main.py
2. 指定配置文件: python fake_data_main.py --config configs/fake_data_config.yaml
3. 训练单个模型: python fake_data_main.py --model AttentionMultimodal
4. 训练所有模型: python fake_data_main.py --train-all
5. 自定义参数: python fake_data_main.py --samples 100 --epochs 10 --batch-size 4
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fake_data_adapter import FakeDataAdapter


def main():
    """主函数 - 使用假数据训练模型"""
    parser = argparse.ArgumentParser(
        description="使用假数据训练多模态模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置训练AttentionMultimodal
  python fake_data_main.py
  
  # 使用指定配置文件
  python fake_data_main.py --config configs/fake_data_config.yaml
  
  # 训练单个模型
  python fake_data_main.py --model AttentionMultimodal
  
  # 训练所有模型
  python fake_data_main.py --train-all
  
  # 自定义参数
  python fake_data_main.py --samples 100 --epochs 10 --batch-size 4
  
  # 快速测试（小数据量）
  python fake_data_main.py --samples 50 --epochs 5 --batch-size 4
        """
    )
    
    # 配置文件参数
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/fake_data_config.yaml",
        help="配置文件路径（默认: configs/fake_data_config.yaml）"
    )
    
    # 模型选择参数
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        choices=["AttentionMultimodal", "ConcatFusion", "EnsembleFusion", "TFTMultimodal", "EnhancedMMTMFusion", "BaselineMultimodal"],
        help="指定单个模型名称"
    )
    parser.add_argument(
        "--train-all", 
        action="store_true",
        help="训练所有模型"
    )
    
    # 数据参数
    parser.add_argument(
        "--samples", 
        type=int, 
        default=None,
        help="假数据样本数量"
    )
    parser.add_argument(
        "--scans", 
        type=int, 
        default=None,
        help="每个样本的扫描次数"
    )
    parser.add_argument(
        "--wavelengths", 
        type=int, 
        default=None,
        help="光谱波长数量"
    )
    parser.add_argument(
        "--features", 
        type=int, 
        default=None,
        help="表格特征数量"
    )
    parser.add_argument(
        "--classes", 
        type=int, 
        default=None,
        help="分类类别数"
    )
    
    # 训练参数
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="批次大小"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=None,
        help="学习率"
    )
    parser.add_argument(
        "--weight-decay", 
        type=float, 
        default=None,
        help="权重衰减"
    )
    
    # 其他参数
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default=None,
        help="结果保存目录"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="训练设备"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="随机种子"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        print(f"💡 使用默认配置...")
        config_path = None
    
    print("🚀 使用假数据训练多模态模型")
    print("=" * 60)
    
    # 创建适配器
    adapter = FakeDataAdapter(config_path=str(config_path) if config_path else None)
    
    # 更新配置（如果通过命令行参数指定了参数）
    if args.samples:
        adapter.config["data"]["num_samples"] = args.samples
    if args.scans:
        adapter.config["data"]["num_scans"] = args.scans
    if args.wavelengths:
        adapter.config["data"]["num_wavelengths"] = args.wavelengths
    if args.features:
        adapter.config["data"]["num_features"] = args.features
    if args.classes:
        adapter.config["data"]["num_classes"] = args.classes
        adapter.config["model"]["num_classes"] = args.classes
    
    if args.epochs:
        adapter.config["train"]["epochs"] = args.epochs
    if args.batch_size:
        adapter.config["train"]["batch_size"] = args.batch_size
    if args.lr:
        adapter.config["train"]["lr"] = args.lr
    if args.weight_decay:
        adapter.config["train"]["weight_decay"] = args.weight_decay
    
    if args.save_dir:
        adapter.config["train"]["save_dir"] = args.save_dir
    if args.seed:
        adapter.config["experiment"]["random_seed"] = args.seed
    
    # 设置随机种子
    if args.seed:
        import torch
        import numpy as np
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # 打印配置信息
    print(f"📋 配置信息:")
    print(f"   • 样本数量: {adapter.config['data']['num_samples']}")
    print(f"   • 扫描次数: {adapter.config['data']['num_scans']}")
    print(f"   • 波长数量: {adapter.config['data']['num_wavelengths']}")
    print(f"   • 特征数量: {adapter.config['data']['num_features']}")
    print(f"   • 类别数量: {adapter.config['data']['num_classes']}")
    print(f"   • 训练轮数: {adapter.config['train']['epochs']}")
    print(f"   • 批次大小: {adapter.config['train']['batch_size']}")
    print(f"   • 学习率: {adapter.config['train']['lr']}")
    print(f"   • 保存目录: {adapter.config['train']['save_dir']}")
    
    if args.verbose:
        print(f"\n🔍 详细配置:")
        import yaml
        print(yaml.dump(adapter.config, default_flow_style=False, allow_unicode=True))
    
    try:
        # 准备数据
        train_loader, val_loader, test_loader, dataset_info = adapter.prepare_fake_data()
        
        if args.train_all:
            # 训练所有模型
            print(f"\n🚀 开始训练所有模型...")
            trainers = adapter.train_all_models(train_loader, val_loader, test_loader, dataset_info)
            
            print(f"\n🎉 所有模型训练完成!")
            print(f"📁 结果保存在: {adapter.config['train']['save_dir']}")
            
        else:
            # 训练单个模型
            model_name = args.model or adapter.config["model"]["name"]
            print(f"\n🚀 开始训练模型: {model_name}")
            
            trainer = adapter.train_single_model(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                dataset_info=dataset_info,
                model_name=model_name
            )
            
            print(f"\n🎉 模型训练完成!")
            print(f"📁 结果保存在: {adapter.config['train']['save_dir']}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

