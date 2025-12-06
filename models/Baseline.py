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
        self.spec_dim = spec_dim
        self.clin_dim = clin_dim
        self.num_classes = num_classes
        
        # 内部编码器，用于处理原始输入
        self.internal_spec_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, spec_dim),
            nn.LayerNorm(spec_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 表格编码器将在第一次前向传播时动态构建
        self.internal_tab_encoder = None
        self.tab_input_dim = None
        
        self.classifier = nn.Sequential(
            nn.Linear(spec_dim + clin_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, spectra, mask=None, tabular=None):
        # 检查输入格式
        if isinstance(spectra, dict) and isinstance(tabular, dict):
            # 字典格式输入（来自外部模型）
            spec_embedding = spectra["embedding"]
            tab_embedding = tabular["embedding"]
            
            # ===== Learnable Modality Fusion Gate =====
            if not hasattr(self, "fusion_gate"):
                # 两个模态各一个权重，初始化为均等
                self.fusion_gate = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
            # 归一化权重
            gate = torch.softmax(self.fusion_gate, dim=0)
            # 加权模态 embedding
            spec_embedding = spec_embedding * gate[0]
            tab_embedding = tab_embedding * gate[1]
        else:
            # 原始张量输入（来自假数据）
            # 处理光谱数据：[B, num_scans, wavelengths] -> [B, wavelengths]
            if spectra.dim() == 3:
                spectra = spectra.mean(dim=1)  # 平均多个扫描
            spectra = spectra.unsqueeze(1)  # [B, 1, wavelengths]
            
            spec_embedding = self.internal_spec_encoder(spectra)  # [B, spec_dim]
            
            # 动态构建表格编码器
            if self.internal_tab_encoder is None or self.tab_input_dim != tabular.shape[-1]:
                self.tab_input_dim = tabular.shape[-1]
                self.internal_tab_encoder = nn.Sequential(
                    nn.Linear(self.tab_input_dim, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, self.clin_dim),
                    nn.LayerNorm(self.clin_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ).to(tabular.device)
            
            tab_embedding = self.internal_tab_encoder(tabular)    # [B, clin_dim]
        
        # 融合特征
        fused = torch.cat([spec_embedding, tab_embedding], dim=-1)   # [B, spec_dim + clin_dim]
        logits = self.classifier(fused)                              # [B, num_classes]
        return {"embedding": fused, "logits": logits}


# ----------------------------
# 融合方式 2: 模态独立分类 + 平均 (Ensemble)
# ----------------------------
class EnsembleFusion(nn.Module):
    def __init__(self, spec_dim=256, clin_dim=128, num_classes=2):
        super().__init__()
        self.spec_dim = spec_dim
        self.clin_dim = clin_dim
        self.num_classes = num_classes
        
        # 内部编码器，用于处理原始输入
        self.internal_spec_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, spec_dim),
            nn.LayerNorm(spec_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 表格编码器将在第一次前向传播时动态构建
        self.internal_tab_encoder = None
        self.tab_input_dim = None
        
        self.spec_head = nn.Linear(spec_dim, num_classes)
        self.clin_head = nn.Linear(clin_dim, num_classes)

    def forward(self, spectra, mask=None, tabular=None):
        # 检查输入格式
        if isinstance(spectra, dict) and isinstance(tabular, dict):
            # 字典格式输入（来自外部模型）
            spec_embedding = spectra["embedding"]
            tab_embedding = tabular["embedding"]
        else:
            # 原始张量输入（来自假数据）
            # 处理光谱数据：[B, num_scans, wavelengths] -> [B, wavelengths]
            if spectra.dim() == 3:
                spectra = spectra.mean(dim=1)  # 平均多个扫描
            spectra = spectra.unsqueeze(1)  # [B, 1, wavelengths]
            
            spec_embedding = self.internal_spec_encoder(spectra)  # [B, spec_dim]
            
            # 动态构建表格编码器
            if self.internal_tab_encoder is None or self.tab_input_dim != tabular.shape[-1]:
                self.tab_input_dim = tabular.shape[-1]
                self.internal_tab_encoder = nn.Sequential(
                    nn.Linear(self.tab_input_dim, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, self.clin_dim),
                    nn.LayerNorm(self.clin_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ).to(tabular.device)
            
            tab_embedding = self.internal_tab_encoder(tabular)    # [B, clin_dim]
        
        # 独立分类
        logits_spec = self.spec_head(spec_embedding)        # [B, num_classes]
        logits_clin = self.clin_head(tab_embedding)         # [B, num_classes]
        logits = (logits_spec + logits_clin) / 2           # ensemble 平均

        # embedding 用拼接（方便和其他模型统一）
        fused = torch.cat([spec_embedding, tab_embedding], dim=-1)  # [B, spec_dim + clin_dim]
        return {"embedding": fused, "logits": logits}


# ----------------------------
# 完整的多模态融合模型
# ----------------------------
class BaselineMultimodal(nn.Module):
    def __init__(self, spec_embedding_dim=256, tab_embedding_dim=128, num_classes=2, fusion_type='concat'):
        super().__init__()
        self.spec_embedding_dim = spec_embedding_dim
        self.tab_embedding_dim = tab_embedding_dim
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        
        self.spec_encoder = SpectraEncoder(spec_embedding_dim=spec_embedding_dim, num_classes=num_classes)
        self.tab_encoder = TabularEncoder(tab_embedding_dim=tab_embedding_dim, num_classes=num_classes)
        
        if fusion_type == 'concat':
            self.fusion = ConcatFusion(spec_dim=spec_embedding_dim, clin_dim=tab_embedding_dim, num_classes=num_classes)
        elif fusion_type == 'ensemble':
            self.fusion = EnsembleFusion(spec_dim=spec_embedding_dim, clin_dim=tab_embedding_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, spectra_result=None, tabular_result=None, mask=None, spectra=None, tabular=None):
        """
        支持两种调用方式：
        1. 接收来自外部模型的结果并进行融合（原始接口）
        2. 接收原始张量输入（用于假数据训练）
        
        Args:
            spectra_result: dict from external spectra model {"embedding": [B, D_spec], "logits": [B, C]}
            tabular_result: dict from external tabular model {"embedding": [B, D_tab], "logits": [B, C]}
            mask: optional mask (for compatibility)
            spectra: raw spectra tensor [B, num_scans, wavelengths] (for fake data)
            tabular: raw tabular tensor [B, features] (for fake data)
        
        Returns:
            dict with fused results
        """
        # 检查是否是位置参数调用（来自训练器）
        if isinstance(spectra_result, torch.Tensor) and isinstance(tabular_result, torch.Tensor):
            # 位置参数调用：model(spectra, mask, tabular)
            # spectra_result 实际是 spectra，tabular_result 实际是 mask，mask 实际是 tabular
            spectra = spectra_result
            tabular = mask  # mask 参数实际是 tabular
            mask = tabular_result  # tabular_result 参数实际是 mask
            spectra_result = None
            tabular_result = None
        
        # 检查输入格式
        if spectra_result is not None and tabular_result is not None:
            # 字典格式输入（来自外部模型）
            # ===== Soft Gating for Missing Modality =====
            if isinstance(spectra_result, dict):
                spec_emb = spectra_result.get("embedding", None)
                spec_mask = spectra_result.get("mask", None)
                if spec_emb is not None and spec_mask is not None:
                    m = spec_mask.unsqueeze(-1).float()  # [B,1]
                    # 建立可学习的默认 embedding，用于模态缺失时补偿
                    if not hasattr(self, "_soft_gate_default_spec"):
                        # 初始化为全部 0 → 不破坏已有 embedding 分布
                        self._soft_gate_default_spec = nn.Parameter(torch.zeros_like(spec_emb[0:1]))
                    # soft gating
                    spec_emb = spec_emb * m + (1 - m) * self._soft_gate_default_spec
                    spectra_result = {
                        **spectra_result,
                        "embedding": spec_emb
                    }
            if isinstance(tabular_result, dict):
                tab_emb = tabular_result.get("embedding", None)
                tab_mask = tabular_result.get("mask", None)
                if tab_emb is not None and tab_mask is not None:
                    m = tab_mask.unsqueeze(-1).float()  # [B,1]
                    # 建立可学习的默认 embedding，用于模态缺失时补偿
                    if not hasattr(self, "_soft_gate_default_tab"):
                        # 初始化为全部 0 → 不破坏已有 embedding 分布
                        self._soft_gate_default_tab = nn.Parameter(torch.zeros_like(tab_emb[0:1]))
                    # soft gating
                    tab_emb = tab_emb * m + (1 - m) * self._soft_gate_default_tab
                    tabular_result = {
                        **tabular_result,
                        "embedding": tab_emb
                    }

            spec_result = self.spec_encoder(spectra_result=spectra_result, mask=mask)
            tab_result = self.tab_encoder(tabular_result=tabular_result, mask=mask)
            
            # ===== Learnable Modality Fusion Gate =====
            spec_emb = spec_result["embedding"]
            tab_emb = tab_result["embedding"]
            if not hasattr(self, "fusion_gate"):
                # 两个模态各一个权重，初始化为均等
                self.fusion_gate = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
            # 归一化权重
            gate = torch.softmax(self.fusion_gate, dim=0)
            # 加权模态 embedding
            spec_emb = spec_emb * gate[0]
            tab_emb = tab_emb * gate[1]
            
            # 融合两个模态的特征
            fused_result = self.fusion(
                spectra=spec_emb, 
                tabular=tab_emb, 
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
        
        elif spectra is not None and tabular is not None:
            # 原始张量输入（来自假数据）
            fused_result = self.fusion(
                spectra=spectra,
                tabular=tabular,
                mask=mask
            )
            
            return {
                "embedding": fused_result["embedding"],
                "logits": fused_result["logits"]
            }
        
        else:
            raise ValueError("Either (spectra_result, tabular_result) or (spectra, tabular) must be provided")
