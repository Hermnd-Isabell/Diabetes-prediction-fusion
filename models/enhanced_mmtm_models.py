# models/enhanced_mmtm_models.py
"""
Enhanced MMTM (Multimodal Transfer Module) with advanced fusion capabilities.

Key Improvements:
1. Multi-head Cross-Modal Attention
2. Adaptive Gating with Temperature Scaling
3. Hierarchical Fusion Strategy
4. Advanced Classifier with Residual Connections
5. Uncertainty Estimation
6. Multi-scale Feature Interaction

Usage:
    from models.enhanced_mmtm_models import EnhancedMMTMFusion
    model = EnhancedMMTMFusion(
        spec_embedding_dim=256, 
        tab_embedding_dim=128, 
        num_classes=2,
        num_attention_heads=8,
        fusion_strategy='hierarchical'
    )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MultiHeadCrossModalAttention(nn.Module):
    """多头跨模态注意力机制"""
    
    def __init__(self, spec_dim: int, tab_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.spec_dim = spec_dim
        self.tab_dim = tab_dim
        
        # 确保维度能被头数整除
        assert spec_dim % num_heads == 0, f"spec_dim {spec_dim} must be divisible by num_heads {num_heads}"
        assert tab_dim % num_heads == 0, f"tab_dim {tab_dim} must be divisible by num_heads {num_heads}"
        
        self.head_dim = spec_dim // num_heads
        self.tab_head_dim = tab_dim // num_heads
        
        # 投影层
        self.spec_q = nn.Linear(spec_dim, spec_dim)
        self.spec_k = nn.Linear(spec_dim, spec_dim)
        self.spec_v = nn.Linear(spec_dim, spec_dim)
        
        self.tab_q = nn.Linear(tab_dim, tab_dim)
        self.tab_k = nn.Linear(tab_dim, tab_dim)
        self.tab_v = nn.Linear(tab_dim, tab_dim)
        
        # 跨模态注意力
        self.cross_attn_spec = nn.MultiheadAttention(spec_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_tab = nn.MultiheadAttention(tab_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 输出投影
        self.spec_out_proj = nn.Linear(spec_dim, spec_dim)
        self.tab_out_proj = nn.Linear(tab_dim, tab_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_spec = nn.LayerNorm(spec_dim)
        self.layer_norm_tab = nn.LayerNorm(tab_dim)
        
    def forward(self, spec_vec: torch.Tensor, tab_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spec_vec: [B, spec_dim]
            tab_vec: [B, tab_dim]
        Returns:
            enhanced_spec: [B, spec_dim]
            enhanced_tab: [B, tab_dim]
        """
        B = spec_vec.size(0)
        
        # 自注意力增强
        spec_self_attn, _ = self.cross_attn_spec(spec_vec, spec_vec, spec_vec)
        tab_self_attn, _ = self.cross_attn_tab(tab_vec, tab_vec, tab_vec)
        
        # 残差连接和层归一化
        spec_enhanced = self.layer_norm_spec(spec_vec + self.dropout(spec_self_attn))
        tab_enhanced = self.layer_norm_tab(tab_vec + self.dropout(tab_self_attn))
        
        return spec_enhanced, tab_enhanced


class AdaptiveGating(nn.Module):
    """自适应门控机制，支持温度缩放和动态权重"""
    
    def __init__(self, spec_dim: int, tab_dim: int, bottleneck_dim: int = 128):
        super().__init__()
        self.spec_dim = spec_dim
        self.tab_dim = tab_dim
        self.bottleneck_dim = bottleneck_dim
        
        # 投影到瓶颈空间
        self.spec_proj = nn.Sequential(
            nn.Linear(spec_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU()
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU()
        )
        
        # 交互网络 - 更深的结构
        self.interaction_net = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, bottleneck_dim * 2),
            nn.LayerNorm(bottleneck_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.ReLU()
        )
        
        # 门控生成器
        self.gate_spec = nn.Sequential(
            nn.Linear(bottleneck_dim // 2, spec_dim),
            nn.Sigmoid()
        )
        self.gate_tab = nn.Sequential(
            nn.Linear(bottleneck_dim // 2, tab_dim),
            nn.Sigmoid()
        )
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 残差变换
        self.spec_residual = nn.Sequential(
            nn.Linear(spec_dim, spec_dim),
            nn.LayerNorm(spec_dim),
            nn.ReLU()
        )
        self.tab_residual = nn.Sequential(
            nn.Linear(tab_dim, tab_dim),
            nn.LayerNorm(tab_dim),
            nn.ReLU()
        )
        
    def forward(self, spec_vec: torch.Tensor, tab_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            spec_vec: [B, spec_dim]
            tab_vec: [B, tab_dim]
        Returns:
            spec_out: [B, spec_dim]
            tab_out: [B, tab_dim]
            gating_info: dict with gating details
        """
        # 投影到瓶颈空间
        s = self.spec_proj(spec_vec)  # [B, bottleneck]
        t = self.tab_proj(tab_vec)    # [B, bottleneck]
        
        # 交互计算
        z = torch.cat([s, t], dim=1)  # [B, 2*bottleneck]
        h = self.interaction_net(z)   # [B, bottleneck//2]
        
        # 生成门控（带温度缩放）
        gate_s = self.gate_spec(h) * self.temperature
        gate_t = self.gate_tab(h) * self.temperature
        
        # 应用门控
        spec_gated = spec_vec * gate_s
        tab_gated = tab_vec * gate_t
        
        # 残差连接
        spec_out = spec_gated + self.spec_residual(spec_vec)
        tab_out = tab_gated + self.tab_residual(tab_vec)
        
        return spec_out, tab_out, {
            'gate_spec': gate_s,
            'gate_tab': gate_t,
            'temperature': self.temperature,
            'intermediate': h
        }


class HierarchicalFusion(nn.Module):
    """层次化融合策略"""
    
    def __init__(self, spec_dim: int, tab_dim: int, num_scales: int = 3):
        super().__init__()
        self.spec_dim = spec_dim
        self.tab_dim = tab_dim
        self.num_scales = num_scales
        
        # 多尺度投影
        self.spec_scales = nn.ModuleList([
            nn.Sequential(
                nn.Linear(spec_dim, spec_dim // (2**i)),
                nn.LayerNorm(spec_dim // (2**i)),
                nn.ReLU()
            ) for i in range(num_scales)
        ])
        
        self.tab_scales = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tab_dim, tab_dim // (2**i)),
                nn.LayerNorm(tab_dim // (2**i)),
                nn.ReLU()
            ) for i in range(num_scales)
        ])
        
        # 融合权重
        self.fusion_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
    def forward(self, spec_vec: torch.Tensor, tab_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec_vec: [B, spec_dim]
            tab_vec: [B, tab_dim]
        Returns:
            fused_features: [B, fused_dim]
        """
        fused_features = []
        
        for i in range(self.num_scales):
            # 多尺度特征
            spec_scale = self.spec_scales[i](spec_vec)
            tab_scale = self.tab_scales[i](tab_vec)
            
            # 确保维度匹配
            min_dim = min(spec_scale.size(1), tab_scale.size(1))
            spec_scale = spec_scale[:, :min_dim]
            tab_scale = tab_scale[:, :min_dim]
            
            # 计算交互特征
            hadamard = spec_scale * tab_scale
            concat = torch.cat([spec_scale, tab_scale, hadamard], dim=1)
            
            # 加权融合
            weight = F.softmax(self.fusion_weights, dim=0)[i]
            fused_features.append(weight * concat)
        
        # 拼接所有尺度的特征
        return torch.cat(fused_features, dim=1)


class AdvancedClassifier(nn.Module):
    """高级分类器，带残差连接和不确定性估计"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        self.num_classes = num_classes
        
        # 构建残差块
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 主分类头
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # 不确定性估计头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, 1),
            nn.Softplus()  # 确保输出为正
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, input_dim]
        Returns:
            logits: [B, num_classes]
            uncertainty: [B, 1]
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        uncertainty = self.uncertainty_head(features)
        
        return logits, uncertainty


class EnhancedMMTM(nn.Module):
    """增强版MMTM模块"""
    
    def __init__(self, spec_dim: int, tab_dim: int, bottleneck_dim: int = 128, 
                 num_attention_heads: int = 8):
        super().__init__()
        self.spec_dim = spec_dim
        self.tab_dim = tab_dim
        
        # 多头注意力
        self.attention = MultiHeadCrossModalAttention(
            spec_dim, tab_dim, num_attention_heads
        )
        
        # 自适应门控
        self.adaptive_gating = AdaptiveGating(spec_dim, tab_dim, bottleneck_dim)
        
    def forward(self, spec_vec: torch.Tensor, tab_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            spec_vec: [B, spec_dim]
            tab_vec: [B, tab_dim]
        Returns:
            spec_out: [B, spec_dim]
            tab_out: [B, tab_dim]
            mmtm_info: dict with detailed information
        """
        # 1. 多头注意力增强
        spec_attn, tab_attn = self.attention(spec_vec, tab_vec)
        
        # 2. 自适应门控
        spec_gated, tab_gated, gating_info = self.adaptive_gating(spec_attn, tab_attn)
        
        return spec_gated, tab_gated, {
            'attention_spec': spec_attn,
            'attention_tab': tab_attn,
            'gating_info': gating_info
        }


class EnhancedMMTMFusion(nn.Module):
    """增强版MMTM融合模型"""
    
    def __init__(self, spec_embedding_dim: int = 256, tab_embedding_dim: int = 128, 
                 num_classes: int = 2, mmtm_bottleneck: int = 128, 
                 num_attention_heads: int = 8, fusion_strategy: str = 'hierarchical',
                 enable_uncertainty: bool = True):
        """
        Args:
            spec_embedding_dim: 光谱嵌入维度
            tab_embedding_dim: 表格嵌入维度
            num_classes: 分类类别数
            mmtm_bottleneck: MMTM瓶颈维度
            num_attention_heads: 注意力头数
            fusion_strategy: 融合策略 ('hierarchical', 'concat', 'interaction')
            enable_uncertainty: 是否启用不确定性估计
        """
        super().__init__()
        self.spec_embedding_dim = spec_embedding_dim
        self.tab_embedding_dim = tab_embedding_dim
        self.fusion_strategy = fusion_strategy
        self.enable_uncertainty = enable_uncertainty
        
        # 增强版MMTM
        self.enhanced_mmtm = EnhancedMMTM(
            spec_dim=spec_embedding_dim,
            tab_dim=tab_embedding_dim,
            bottleneck_dim=mmtm_bottleneck,
            num_attention_heads=num_attention_heads
        )
        
        # 融合策略
        if fusion_strategy == 'hierarchical':
            self.fusion_module = HierarchicalFusion(spec_embedding_dim, tab_embedding_dim)
            # 计算融合后的维度
            fused_dim = self._calculate_hierarchical_fusion_dim()
        elif fusion_strategy == 'concat':
            if spec_embedding_dim == tab_embedding_dim:
                fused_dim = spec_embedding_dim * 3
            else:
                self.tab_to_spec_proj = nn.Linear(tab_embedding_dim, spec_embedding_dim)
                fused_dim = spec_embedding_dim * 3
        else:  # interaction
            if spec_embedding_dim == tab_embedding_dim:
                fused_dim = spec_embedding_dim
            else:
                self.tab_to_spec_proj = nn.Linear(tab_embedding_dim, spec_embedding_dim)
                fused_dim = spec_embedding_dim
        
        # 高级分类器
        self.classifier = AdvancedClassifier(
            input_dim=fused_dim,
            num_classes=num_classes,
            hidden_dims=[fused_dim // 2, fused_dim // 4]
        )
        
        # 添加内部编码器，用于处理原始输入
        self.internal_spec_encoder = self._build_spectral_encoder()
        self.internal_tab_encoder = self._build_tabular_encoder()
        self.internal_spec_classifier = nn.Linear(spec_embedding_dim, num_classes)
        self.internal_tab_classifier = nn.Linear(tab_embedding_dim, num_classes)
    
    def _build_spectral_encoder(self):
        """构建内部光谱编码器"""
        return nn.Sequential(
            # 扫描级特征提取
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(256, self.spec_embedding_dim),
            nn.LayerNorm(self.spec_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def _build_tabular_encoder(self):
        """构建内部表格编码器"""
        return nn.Sequential(
            nn.Linear(10, 256),  # 假设表格特征维度为10
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.tab_embedding_dim),
            nn.LayerNorm(self.tab_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _calculate_hierarchical_fusion_dim(self) -> int:
        """计算层次化融合后的维度"""
        total_dim = 0
        for i in range(3):  # num_scales = 3
            spec_scale_dim = self.spec_embedding_dim // (2**i)
            tab_scale_dim = self.tab_embedding_dim // (2**i)
            min_dim = min(spec_scale_dim, tab_scale_dim)
            # concat: [spec, tab, hadamard] = 3 * min_dim
            total_dim += 3 * min_dim
        return total_dim
    
    def forward(self, *args):
        """
        支持两种调用方式的前向传播
        
        Args:
            方式1: spectra_result, tabular_result (dict格式)
            方式2: spectra, mask, tabular (原始张量格式)
        """
        if len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            # 原始方式：字典格式输入
            spectra_result, tabular_result = args
            spec_emb = spectra_result['embedding']
            tab_emb = tabular_result['embedding']
            spec_logits = spectra_result['logits']
            tab_logits = tabular_result['logits']
            
        elif len(args) == 3:
            # 新方式：处理原始张量输入
            spectra, mask, tabular = args
            
            # 处理光谱数据：[B, S, L] -> [B*S, L] -> [B*S, 1, L]
            B, S, L = spectra.shape
            spectra_flat = spectra.view(B * S, L).unsqueeze(1)  # [B*S, 1, L]
            
            # 通过内部光谱编码器
            spec_features = self.internal_spec_encoder(spectra_flat)  # [B*S, spec_embedding_dim]
            
            # 重新reshape并做注意力池化
            spec_features = spec_features.view(B, S, -1)  # [B, S, spec_embedding_dim]
            
            # 使用mask进行注意力池化
            if mask is not None:
                # 简单的掩码平均池化
                mask_expanded = mask.unsqueeze(-1).float()  # [B, S, 1]
                spec_emb = (spec_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
            else:
                spec_emb = spec_features.mean(dim=1)  # [B, spec_embedding_dim]
            
            # 处理表格数据
            tab_emb = self.internal_tab_encoder(tabular)  # [B, tab_embedding_dim]
            
            # 生成内部logits（用于损失计算）
            spec_logits = self.internal_spec_classifier(spec_emb)
            tab_logits = self.internal_tab_classifier(tab_emb)
            
        else:
            raise ValueError(f"❌ 不支持的参数数量: {len(args)}")
        
        # 增强版MMTM处理
        spec_enhanced, tab_enhanced, mmtm_info = self.enhanced_mmtm(spec_emb, tab_emb)
        
        # 融合策略
        if self.fusion_strategy == 'hierarchical':
            fused_features = self.fusion_module(spec_enhanced, tab_enhanced)
        elif self.fusion_strategy == 'concat':
            if spec_emb.shape[1] == tab_emb.shape[1]:
                hadamard = spec_enhanced * tab_enhanced
                fused_features = torch.cat([spec_enhanced, tab_enhanced, hadamard], dim=1)
            else:
                tab_proj = self.tab_to_spec_proj(tab_enhanced)
                hadamard = spec_enhanced * tab_proj
                fused_features = torch.cat([spec_enhanced, tab_proj, hadamard], dim=1)
        else:  # interaction
            if spec_emb.shape[1] == tab_emb.shape[1]:
                fused_features = spec_enhanced * tab_enhanced
            else:
                tab_proj = self.tab_to_spec_proj(tab_enhanced)
                fused_features = spec_enhanced * tab_proj
        
        # 分类和不确定性估计
        if self.enable_uncertainty:
            logits, uncertainty = self.classifier(fused_features)
        else:
            logits = self.classifier.classifier(self.classifier.feature_extractor(fused_features))
            uncertainty = torch.zeros(logits.size(0), 1, device=logits.device)
        
        return {
            "embedding": fused_features,
            "logits": logits,
            "uncertainty": uncertainty,
            "spec_embedding": spec_enhanced,
            "tab_embedding": tab_enhanced,
            "spec_logits": spec_logits,
            "tab_logits": tab_logits,
            "mmtm_info": mmtm_info,
            "fusion_strategy": self.fusion_strategy
        }


# 便捷函数
def create_enhanced_mmtm_model(spec_dim: int = 256, tab_dim: int = 128, 
                              num_classes: int = 2, **kwargs) -> EnhancedMMTMFusion:
    """创建增强版MMTM模型的便捷函数"""
    return EnhancedMMTMFusion(
        spec_embedding_dim=spec_dim,
        tab_embedding_dim=tab_dim,
        num_classes=num_classes,
        **kwargs
    )
