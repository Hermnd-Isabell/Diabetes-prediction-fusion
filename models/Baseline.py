import torch
import torch.nn as nn

# ----------------------------
# 光谱模态编码器（接收外部ML/DL模型输出）
# ----------------------------
class SpectraEncoder(nn.Module):
    def __init__(self, spec_embedding_dim=256, num_classes=2):
        super().__init__()
        # 这个类主要用于接收外部光谱模型的输出
        # 如果需要进一步处理，可以在这里添加额外的层
        self.spec_embedding_dim = spec_embedding_dim
        self.num_classes = num_classes

    def forward(self, spectra_result=None, mask=None, tabular=None):
        """
        接收来自外部光谱处理模型的输出
        spectra_result: dict with keys "embedding" and "logits"
        """
        if spectra_result is None:
            raise ValueError("spectra_result cannot be None")
        
        # 直接返回外部模型的结果，或者可以在这里做进一步处理
        return {
            "embedding": spectra_result["embedding"],  # [B, spec_embedding_dim]
            "logits": spectra_result["logits"]         # [B, num_classes]
        }


# ----------------------------
# 临床模态编码器（接收外部ML/DL模型输出）
# ----------------------------
class TabularEncoder(nn.Module):
    def __init__(self, tab_embedding_dim=128, num_classes=2):
        super().__init__()
        # 这个类主要用于接收外部临床模型的输出
        # 如果需要进一步处理，可以在这里添加额外的层
        self.tab_embedding_dim = tab_embedding_dim
        self.num_classes = num_classes

    def forward(self, spectra=None, mask=None, tabular_result=None):
        """
        接收来自外部临床处理模型的输出
        tabular_result: dict with keys "embedding" and "logits"
        """
        if tabular_result is None:
            raise ValueError("tabular_result cannot be None")
        
        # 直接返回外部模型的结果，或者可以在这里做进一步处理
        return {
            "embedding": tabular_result["embedding"],  # [B, tab_embedding_dim]
            "logits": tabular_result["logits"]         # [B, num_classes]
        }


# ----------------------------
# 融合方式 1: 简单拼接
# ----------------------------
class ConcatFusion(nn.Module):
    def __init__(self, spec_dim=256, clin_dim=128, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(spec_dim + clin_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, spectra, mask=None, tabular=None):
        # 假设输入已经是 embedding
        fused = torch.cat([spectra, tabular], dim=-1)   # [B, 384]
        logits = self.classifier(fused)                 # [B, C]
        return {"embedding": fused, "logits": logits}


# ----------------------------
# 融合方式 2: 模态独立分类 + 平均 (Ensemble)
# ----------------------------
class EnsembleFusion(nn.Module):
    def __init__(self, spec_dim=256, clin_dim=128, num_classes=2):
        super().__init__()
        self.spec_head = nn.Linear(spec_dim, num_classes)
        self.clin_head = nn.Linear(clin_dim, num_classes)

    def forward(self, spectra, mask=None, tabular=None):
        logits_spec = self.spec_head(spectra)        # [B, C]
        logits_clin = self.clin_head(tabular)        # [B, C]
        logits = (logits_spec + logits_clin) / 2     # ensemble 平均

        # embedding 用拼接（方便和其他模型统一）
        fused = torch.cat([spectra, tabular], dim=-1)  # [B, 384]
        return {"embedding": fused, "logits": logits}


# ----------------------------
# 完整的多模态融合模型
# ----------------------------
class BaselineMultimodal(nn.Module):
    def __init__(self, spec_embedding_dim=256, tab_embedding_dim=128, num_classes=2, fusion_type='concat'):
        super().__init__()
        self.spec_encoder = SpectraEncoder(spec_embedding_dim=spec_embedding_dim, num_classes=num_classes)
        self.tab_encoder = TabularEncoder(tab_embedding_dim=tab_embedding_dim, num_classes=num_classes)
        
        if fusion_type == 'concat':
            self.fusion = ConcatFusion(spec_dim=spec_embedding_dim, clin_dim=tab_embedding_dim, num_classes=num_classes)
        elif fusion_type == 'ensemble':
            self.fusion = EnsembleFusion(spec_dim=spec_embedding_dim, clin_dim=tab_embedding_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, spectra_result, tabular_result, mask=None):
        """
        接收来自外部模型的结果并进行融合
        
        Args:
            spectra_result: dict from external spectra model {"embedding": [B, D_spec], "logits": [B, C]}
            tabular_result: dict from external tabular model {"embedding": [B, D_tab], "logits": [B, C]}
            mask: optional mask (for compatibility)
        
        Returns:
            dict with fused results
        """
        # 获取各模态的embedding和logits
        spec_result = self.spec_encoder(spectra_result=spectra_result, mask=mask)
        tab_result = self.tab_encoder(tabular_result=tabular_result, mask=mask)
        
        # 融合两个模态的特征
        fused_result = self.fusion(
            spectra=spec_result["embedding"], 
            tabular=tab_result["embedding"], 
            mask=mask
        )
        
        return {
            "embedding": fused_result["embedding"],
            "logits": fused_result["logits"],
            "spec_embedding": spec_result["embedding"],
            "tab_embedding": tab_result["embedding"],
            "spec_logits": spec_result["logits"],
            "tab_logits": tab_result["logits"]
        }
