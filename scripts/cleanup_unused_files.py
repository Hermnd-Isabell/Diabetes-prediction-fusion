#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目清理脚本 - 删除不再使用的文件

此脚本会列出可以安全删除的文件，并在用户确认后删除它们。
"""

import os
from pathlib import Path

# 可以安全删除的文件列表
FILES_TO_DELETE = {
    # Fake data相关（如果不再使用假数据）
    "fake_data_main.py",
    "fake_data_adapter.py",
    "fake_data_generator.py",
    "configs/fake_data_config.yaml",
    
    # 旧的trainer（已被enhanced_trainer.py替代）
    "trainers/trainer.py",
    
    # 旧的配置文件（如果不再使用）
    "configs/attention.yaml",
    "configs/ensemble.yaml",
    "configs/tft.yaml",
    "configs/small_data_config.yaml",
    
    # 测试文件
    "clinic_dimension/test_config.py",
    
    # 旧的脚本（如果不再生成合成数据）
    "scripts/gen_synthetic_data.py",
    
    # 旧的单模态模型（如果不再使用）
    "spectrum_dimension/baseline.py",
    "spectrum_dimension/baseline2.py",
    
    # 临时文件
    "best_model.pt",  # 根目录下的旧模型文件
    "noop",  # 空文件
    "romanlabel.csv",  # 根目录下的旧文件
    "gradcam_synthetic.png",  # 临时图片
    "quick_test_results.png",  # 临时图片
    
    # 旧的README文档（如果已经整合到主README）
    "README_Fake_Data_Integration.md",
    "README_Fake_Data_System.md",
    "README_Enhanced_Training.md",
    "README_Model_Enhancements.md",
}

# 可以删除的目录（如果为空或不再使用）
DIRS_TO_DELETE = {
    "fake_data_results/",  # 假数据实验结果
    "small_data_results/",  # 小数据实验结果
    "test_results/",  # 测试结果
}


def check_files_exist(files: set) -> tuple:
    """检查文件是否存在，返回存在的和不存在的文件列表"""
    existing = []
    missing = []
    
    for file_path in files:
        full_path = Path(file_path)
        if full_path.exists():
            existing.append(file_path)
        else:
            missing.append(file_path)
    
    return existing, missing


def main():
    print("=" * 80)
    print("项目清理脚本 - 删除不再使用的文件")
    print("=" * 80)
    print("\n以下文件将被删除：\n")
    
    # 检查文件
    existing_files, missing_files = check_files_exist(FILES_TO_DELETE)
    
    if existing_files:
        print("文件列表：")
        for i, file_path in enumerate(existing_files, 1):
            size = Path(file_path).stat().st_size if Path(file_path).is_file() else 0
            print(f"  {i}. {file_path} ({size:,} bytes)")
    
    if missing_files:
        print(f"\n以下文件不存在（将跳过）：")
        for file_path in missing_files:
            print(f"  - {file_path}")
    
    # 检查目录
    existing_dirs = []
    for dir_path in DIRS_TO_DELETE:
        full_path = Path(dir_path)
        if full_path.exists() and full_path.is_dir():
            existing_dirs.append(dir_path)
    
    if existing_dirs:
        print(f"\n目录列表：")
        for i, dir_path in enumerate(existing_dirs, 1):
            file_count = sum(1 for _ in Path(dir_path).rglob('*') if _.is_file())
            print(f"  {i}. {dir_path} ({file_count} 个文件)")
    
    if not existing_files and not existing_dirs:
        print("\n没有找到需要删除的文件或目录。")
        return
    
    # 确认删除
    print("\n" + "=" * 80)
    response = input("确认删除以上文件？(yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("取消删除操作。")
        return
    
    # 删除文件
    deleted_count = 0
    failed_count = 0
    
    print("\n开始删除文件...")
    for file_path in existing_files:
        try:
            Path(file_path).unlink()
            print(f"  [OK] 已删除: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"  [ERROR] 删除失败 {file_path}: {e}")
            failed_count += 1
    
    # 删除目录
    for dir_path in existing_dirs:
        try:
            import shutil
            shutil.rmtree(dir_path)
            print(f"  [OK] 已删除目录: {dir_path}")
            deleted_count += 1
        except Exception as e:
            print(f"  [ERROR] 删除目录失败 {dir_path}: {e}")
            failed_count += 1
    
    print("\n" + "=" * 80)
    print(f"清理完成！")
    print(f"  成功删除: {deleted_count} 个文件/目录")
    if failed_count > 0:
        print(f"  删除失败: {failed_count} 个文件/目录")


if __name__ == "__main__":
    main()

