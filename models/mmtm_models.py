# models/mmtm_models.py
"""
MMTM (Multimodal Transfer Module) implementation for external model outputs.

Usage:
    from models.mmtm_models import MMTMFusion
    model = MMTMFusion(spec_embedding_dim=256, tab_embedding_dim=128, num_classes=2)
    
    # 输入外部模型结果
    spectra_result = {"embedding": spec_emb, "logits": spec_logits}
    tabular_result = {"embedding": tab_emb, "logits": tab_logits}
    
    output = model(spectra_result, tabular_result)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 原始数据处理组件已移除，现在只处理外部模型输出


# ----------------------------
# MMTM module (bidirectional cross-gating with bottleneck)
# 专门用于处理外部模型输出的特征向量
# ----------------------------
class MMTM(nn.Module):
    def __init__(self, spec_dim, tab_dim, bottleneck_dim=128):
        """
        spec_dim: 光谱嵌入维度
        tab_dim: 表格嵌入维度  
        bottleneck_dim: 共享瓶颈维度
        """
        super().__init__()
        # 将每个模态投影到共享瓶颈空间
        self.spec_proj = nn.Linear(spec_dim, bottleneck_dim)
        self.tab_proj = nn.Linear(tab_dim, bottleneck_dim)

        # 交互网络 -> 为每个模态生成门控
        inter_dim = max(32, bottleneck_dim)
        self.inter = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.ReLU()
        )
        
        # 生成门控向量（映射回原始维度）
        self.gate_spec = nn.Linear(inter_dim, spec_dim)
        self.gate_tab = nn.Linear(inter_dim, tab_dim)

        # 残差变换
        self.spec_residual = nn.Sequential(nn.Linear(spec_dim, spec_dim), nn.ReLU())
        self.tab_residual = nn.Sequential(nn.Linear(tab_dim, tab_dim), nn.ReLU())

    def forward(self, spec_vec, tab_vec):
        """
        spec_vec: [B, spec_dim] - 外部光谱模型输出
        tab_vec:  [B, tab_dim] - 外部表格模型输出
        returns: 重新校准的 spec', tab' (相同形状)
        """
        s = self.spec_proj(spec_vec)   # [B, bottleneck]
        t = self.tab_proj(tab_vec)     # [B, bottleneck]
        z = torch.cat([s, t], dim=1)   # [B, 2*bottleneck]
        h = self.inter(z)              # [B, inter_dim]

        # 生成门控
        gate_s = torch.sigmoid(self.gate_spec(h))  # [B, spec_dim]
        gate_t = torch.sigmoid(self.gate_tab(h))   # [B, tab_dim]

        # 重新校准特征
        spec_out = spec_vec * gate_s + self.spec_residual(spec_vec)
        tab_out = tab_vec * gate_t + self.tab_residual(tab_vec)
        
        return spec_out, tab_out, {
            'gate_spec': gate_s, 
            'gate_tab': gate_t, 
            'intermediate': h
        }


# ----------------------------
# 新的 MMTM 融合模型 - 专门接收外部模型输出
# ----------------------------
class MMTMFusion(nn.Module):
    def __init__(self, spec_embedding_dim=256, tab_embedding_dim=128, num_classes=2, 
                 mmtm_bottleneck=128, fusion_type='concat'):
        """
        spec_embedding_dim: 外部光谱模型输出的嵌入维度
        tab_embedding_dim: 外部表格模型输出的嵌入维度
        num_classes: 分类类别数
        mmtm_bottleneck: MMTM 瓶颈维度
        fusion_type: 融合方式 ('concat' 或 'interaction')
        """
        super().__init__()
        self.spec_embedding_dim = spec_embedding_dim
        self.tab_embedding_dim = tab_embedding_dim
        self.fusion_type = fusion_type
        
        # MMTM 跨模态门控模块
        self.mmtm = MMTM(
            spec_dim=spec_embedding_dim, 
            tab_dim=tab_embedding_dim, 
            bottleneck_dim=mmtm_bottleneck
        )
        
        # 融合分类器
        if fusion_type == 'concat':
            # 简单拼接: [spec', tab', hadamard]
            if spec_embedding_dim == tab_embedding_dim:
                fused_dim = spec_embedding_dim * 3
            else:
                # 维度不同时，需要投影
                self.tab_to_spec_proj = nn.Linear(tab_embedding_dim, spec_embedding_dim)
                fused_dim = spec_embedding_dim * 3
        else:  # interaction
            # 只使用交互特征
            if spec_embedding_dim == tab_embedding_dim:
                fused_dim = spec_embedding_dim
            else:
                # 维度不同时，需要投影
                self.tab_to_spec_proj = nn.Linear(tab_embedding_dim, spec_embedding_dim)
                fused_dim = spec_embedding_dim
            
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fused_dim // 2, num_classes)
        )

    def forward(self, spectra_result, tabular_result):
        """
        接收外部模型输出进行融合
        
        Args:
            spectra_result: dict with keys ['embedding', 'logits']
                - embedding: [B, spec_embedding_dim] 光谱特征
                - logits: [B, num_classes] 光谱模型分类结果
            tabular_result: dict with keys ['embedding', 'logits']  
                - embedding: [B, tab_embedding_dim] 表格特征
                - logits: [B, num_classes] 表格模型分类结果
                
        Returns:
            dict with keys:
                - embedding: [B, fused_dim] 融合后的特征
                - logits: [B, num_classes] 最终分类结果
                - spec_embedding: [B, spec_embedding_dim] 重新校准的光谱特征
                - tab_embedding: [B, tab_embedding_dim] 重新校准的表格特征
                - spec_logits: [B, num_classes] 原始光谱模型输出
                - tab_logits: [B, num_classes] 原始表格模型输出
                - mmtm_info: dict MMTM 内部信息
        """
        # 提取外部模型输出
        spec_emb = spectra_result['embedding']  # [B, spec_embedding_dim]
        tab_emb = tabular_result['embedding']   # [B, tab_embedding_dim]
        spec_logits = spectra_result['logits']  # [B, num_classes]
        tab_logits = tabular_result['logits']   # [B, num_classes]
        
        # MMTM 跨模态门控
        spec_rec, tab_rec, mmtm_info = self.mmtm(spec_emb, tab_emb)
        
        # 构建融合特征
        if self.fusion_type == 'concat':
            if spec_emb.shape[1] == tab_emb.shape[1]:
                # 维度相同，直接计算 Hadamard 积
                hadamard = spec_rec * tab_rec
                fused = torch.cat([spec_rec, tab_rec, hadamard], dim=1)
            else:
                # 维度不同，投影后计算 Hadamard 积
                tab_proj = self.tab_to_spec_proj(tab_rec)
                hadamard = spec_rec * tab_proj
                fused = torch.cat([spec_rec, tab_proj, hadamard], dim=1)
        else:  # interaction
            # 只使用交互特征
            if spec_emb.shape[1] == tab_emb.shape[1]:
                fused = spec_rec * tab_rec
            else:
                tab_proj = self.tab_to_spec_proj(tab_rec)
                fused = spec_rec * tab_proj
        
        # 最终分类
        logits = self.classifier(fused)
        
        return {
            "embedding": fused,              # 融合后的特征
            "logits": logits,                # 最终分类结果
            "spec_embedding": spec_rec,      # 重新校准的光谱特征
            "tab_embedding": tab_rec,        # 重新校准的表格特征
            "spec_logits": spec_logits,      # 原始光谱模型输出
            "tab_logits": tab_logits,        # 原始表格模型输出
            "mmtm_info": mmtm_info           # MMTM 内部信息
        }
