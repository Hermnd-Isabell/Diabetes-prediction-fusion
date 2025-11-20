#!/usr/bin/env python3
"""
假数据生成器 - 用于测试 AttentionMultimodal 模型
支持生成不同规模和配置的测试数据
"""

import torch
import numpy as np
from models.attention_models import AttentionMultimodal

class FakeDataGenerator:
    """假数据生成器"""
    
    def __init__(self, seed=42):
        """初始化生成器"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
    
    def generate_spectral_data(self, batch_size, num_scans=3, num_wavelengths=1000):
        """生成光谱数据"""
        spectra = torch.randn(batch_size, num_scans, num_wavelengths)
        mask = torch.ones(batch_size, num_scans, dtype=torch.bool)
        return spectra, mask
    
    def generate_tabular_data(self, batch_size, num_features=10):
        """生成表格数据"""
        tabular = torch.randn(batch_size, num_features)
        return tabular
    
    def generate_labels(self, batch_size, num_classes=2):
        """生成标签"""
        labels = torch.randint(0, num_classes, (batch_size,))
        return labels
    
    def generate_full_data(self, batch_size, num_classes=2, num_features=10):
        """生成完整的数据集"""
        spectra, mask = self.generate_spectral_data(batch_size)
        tabular = self.generate_tabular_data(batch_size, num_features)
        labels = self.generate_labels(batch_size, num_classes)
        
        return {
            'spectra': spectra,
            'mask': mask,
            'tabular': tabular,
            'labels': labels
        }
    
    def generate_pretrained_embeddings(self, batch_size, spec_dim=256, tab_dim=128, num_classes=2):
        """生成预计算的embedding和logits"""
        spec_result = {
            'embedding': torch.randn(batch_size, spec_dim),
            'logits': torch.randn(batch_size, num_classes)
        }
        
        tab_result = {
            'embedding': torch.randn(batch_size, tab_dim),
            'logits': torch.randn(batch_size, num_classes)
        }
        
        return spec_result, tab_result

def test_with_different_configurations():
    """测试不同配置的模型"""
    print("🔧 测试不同配置的模型")
    print("=" * 60)
    
    generator = FakeDataGenerator()
    
    # 测试配置
    configs = [
        {'batch_size': 1, 'num_classes': 2, 'num_features': 5},
        {'batch_size': 4, 'num_classes': 2, 'num_features': 10},
        {'batch_size': 8, 'num_classes': 3, 'num_features': 15},
        {'batch_size': 16, 'num_classes': 2, 'num_features': 20},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n📊 配置 {i}: {config}")
        
        # 生成数据
        data = generator.generate_full_data(**config)
        
        # 创建模型
        model = AttentionMultimodal(
            spec_embedding_dim=256,
            tab_embedding_dim=128,
            num_classes=config['num_classes'],
            fusion_type='enhanced_cross',
            tab_dim=config['num_features']
        )
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            output = model(data['spectra'], data['mask'], data['tabular'])
            
            print(f"   ✅ 成功")
            print(f"   • 主logits形状: {output['logits'].shape}")
            print(f"   • 融合embedding形状: {output['embedding'].shape}")
            print(f"   • 光谱embedding形状: {output['spec_embedding'].shape}")
            print(f"   • 表格embedding形状: {output['tab_embedding'].shape}")

def test_pretrained_embeddings():
    """测试预计算的embedding"""
    print("\n🔗 测试预计算的embedding")
    print("=" * 60)
    
    generator = FakeDataGenerator()
    
    # 生成预计算的数据
    spec_result, tab_result = generator.generate_pretrained_embeddings(
        batch_size=4, spec_dim=256, tab_dim=128, num_classes=2
    )
    
    # 创建模型
    model = AttentionMultimodal(
        spec_embedding_dim=256,
        tab_embedding_dim=128,
        num_classes=2,
        fusion_type='enhanced_cross'
    )
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        output = model(spec_result, tab_result)
        
        print(f"✅ 预计算embedding测试成功")
        print(f"   • 输出键: {list(output.keys())}")
        for key, value in output.items():
            print(f"   • {key}: {value.shape}")

def test_training_loop():
    """测试训练循环"""
    print("\n🏋️ 测试训练循环")
    print("=" * 60)
    
    generator = FakeDataGenerator()
    
    # 生成数据
    data = generator.generate_full_data(batch_size=8, num_classes=2, num_features=10)
    
    # 创建模型
    model = AttentionMultimodal(
        spec_embedding_dim=256,
        tab_embedding_dim=128,
        num_classes=2,
        fusion_type='enhanced_cross',
        tab_dim=10
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练几个epoch
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data['spectra'], data['mask'], data['tabular'])
        
        # 计算损失
        main_loss = torch.nn.CrossEntropyLoss()(output['logits'], data['labels'])
        spec_loss = torch.nn.CrossEntropyLoss()(output['spec_logits'], data['labels'])
        tab_loss = torch.nn.CrossEntropyLoss()(output['tab_logits'], data['labels'])
        aux_spec_loss = torch.nn.CrossEntropyLoss()(output['aux_spec_logits'], data['labels'])
        aux_tab_loss = torch.nn.CrossEntropyLoss()(output['aux_tab_logits'], data['labels'])
        
        total_loss = main_loss + 0.5 * (spec_loss + tab_loss) + 0.3 * (aux_spec_loss + aux_tab_loss)
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        print(f"   Epoch {epoch+1}: 总损失 = {total_loss.item():.4f}")

def main():
    """主函数"""
    print("🎭 假数据生成器测试")
    print("=" * 60)
    
    # 测试不同配置
    test_with_different_configurations()
    
    # 测试预计算embedding
    test_pretrained_embeddings()
    
    # 测试训练循环
    test_training_loop()
    
    print("\n🎉 所有测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()


