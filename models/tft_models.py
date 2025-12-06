import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------
# Enhanced Spectral TFT Encoder with Residual Connections
# ----------------------------
class SpectraTFTEncoder(nn.Module):
    def __init__(self, input_dim=1, d_model=256, nhead=8, num_layers=3, dropout=0.1, seq_len=1024):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced transformer with residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True,
            activation='gelu'  # Better activation for transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Learnable positional encoding with scaling
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Multi-scale feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        self.conv_fusion = nn.Linear(d_model * 3, d_model)
        
        # Attention pooling instead of simple mean
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, spectra):
        # spectra: [B, S, L]  (B=batch, S=scans, L=wavepoints)
        B, S, L = spectra.shape
        x = spectra.view(B*S, L, 1)  # flatten scans
        
        # Input projection with residual
        x_proj = self.input_proj(x)  # [B*S, L, d_model]
        x = x_proj + self.pos_emb[:, :L, :]  # Add positional encoding
        
        # Transformer encoding
        x_transformed = self.transformer(x)  # [B*S, L, d_model]
        
        # Multi-scale convolution features
        x_conv = x_transformed.transpose(1, 2)  # [B*S, d_model, L]
        conv_features = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x_conv))  # [B*S, d_model, L]
            conv_features.append(conv_out.transpose(1, 2))  # [B*S, L, d_model]
        
        # Fuse multi-scale features
        x_multi_scale = torch.cat(conv_features, dim=-1)  # [B*S, L, d_model*3]
        x_fused = self.conv_fusion(x_multi_scale)  # [B*S, L, d_model]
        
        # Residual connection
        x_final = x_transformed + x_fused
        
        # Attention-based pooling
        pool_query = self.pool_query.expand(B*S, -1, -1)  # [B*S, 1, d_model]
        pooled, _ = self.attention_pool(pool_query, x_final, x_final)  # [B*S, 1, d_model]
        pooled = pooled.squeeze(1)  # [B*S, d_model]
        
        return pooled.view(B, S, -1)  # [B, S, d_model]


# ----------------------------
# Enhanced Clinical Encoder with Feature Selection
# ----------------------------
class TabularStaticEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim=128, dropout=0.1):
        super().__init__()
        self.feature_selector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()  # Feature importance weights
        )
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Feature selection
        feature_weights = self.feature_selector(x)  # [B, in_dim]
        x_selected = x * feature_weights  # [B, in_dim]
        
        # Enhanced encoding
        return self.net(x_selected)  # [B, emb_dim]


# ----------------------------
# Cross-Modal Attention Mechanism
# ----------------------------
class CrossModalAttention(nn.Module):
    def __init__(self, spec_dim, tab_dim, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.spec_proj = nn.Linear(spec_dim, d_model)
        self.tab_proj = nn.Linear(tab_dim, d_model)
        
        # Cross-attention layers
        self.spec_to_tab_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.tab_to_spec_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward networks
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, spec_vec, tab_vec):
        # Project to common dimension
        spec_proj = self.spec_proj(spec_vec).unsqueeze(1)  # [B, 1, d_model]
        tab_proj = self.tab_proj(tab_vec).unsqueeze(1)     # [B, 1, d_model]
        
        # Cross-attention: spectral attends to tabular
        spec_attended, _ = self.spec_to_tab_attn(spec_proj, tab_proj, tab_proj)
        spec_attended = self.norm1(spec_attended + spec_proj)  # Residual connection
        spec_attended = self.norm1(spec_attended + self.ffn1(spec_attended))  # FFN + residual
        
        # Cross-attention: tabular attends to spectral
        tab_attended, _ = self.tab_to_spec_attn(tab_proj, spec_proj, spec_proj)
        tab_attended = self.norm2(tab_attended + tab_proj)  # Residual connection
        tab_attended = self.norm2(tab_attended + self.ffn2(tab_attended))  # FFN + residual
        
        # 确保输出是 [B, d_model] 格式
        if len(spec_attended.shape) == 3:
            spec_attended = spec_attended.squeeze(1)  # [B, 1, d_model] -> [B, d_model]
        elif len(spec_attended.shape) != 2:
            spec_attended = spec_attended.view(spec_attended.shape[0], -1)
        
        if len(tab_attended.shape) == 3:
            tab_attended = tab_attended.squeeze(1)  # [B, 1, d_model] -> [B, d_model]
        elif len(tab_attended.shape) != 2:
            tab_attended = tab_attended.view(tab_attended.shape[0], -1)
        
        return spec_attended, tab_attended  # [B, d_model]


# ----------------------------
# Enhanced Multi-Layer Gating Mechanism
# ----------------------------
class EnhancedGating(nn.Module):
    def __init__(self, spec_dim, tab_dim, d_model=256, num_gates=3):
        super().__init__()
        self.num_gates = num_gates
        
        # Multiple gating layers
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tab_dim, spec_dim),
                nn.Sigmoid()
            ) for _ in range(num_gates)
        ])
        
        # Gate fusion network
        self.gate_fusion = nn.Sequential(
            nn.Linear(spec_dim * num_gates, spec_dim),
            nn.LayerNorm(spec_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Residual connection
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, spec_vec, tab_vec):
        # spec_vec: [B, D_spec], tab_vec: [B, D_tab]
        gate_outputs = []
        
        # Apply multiple gates
        for gate in self.gates:
            gate_weights = gate(tab_vec)  # [B, D_spec]
            gated_spec = spec_vec * gate_weights
            gate_outputs.append(gated_spec)
        
        # Fuse gate outputs
        fused_gates = torch.cat(gate_outputs, dim=-1)  # [B, D_spec * num_gates]
        gated_output = self.gate_fusion(fused_gates)  # [B, D_spec]
        
        # Residual connection
        final_output = spec_vec + self.residual_weight * gated_output
        
        return final_output


# ----------------------------
# Enhanced TFT Multimodal Model with Advanced Fusion
# ----------------------------
class TFTMultimodal(nn.Module):
    def __init__(self, tab_dim, spec_len=1024, spec_emb=256, tab_emb=128, num_classes=2, 
                 fusion_dim=256, dropout=0.1):
        super().__init__()
        # Enhanced encoders
        self.spec_encoder = SpectraTFTEncoder(input_dim=1, d_model=spec_emb, seq_len=spec_len, dropout=dropout)
        self.tab_encoder = TabularStaticEncoder(tab_dim, emb_dim=tab_emb, dropout=dropout)
        
        # Advanced fusion components
        self.cross_attention = CrossModalAttention(spec_emb, tab_emb, d_model=fusion_dim, dropout=dropout)
        self.enhanced_gating = EnhancedGating(spec_emb, tab_emb, d_model=fusion_dim)
        
        # Multi-level fusion
        # 第一个层：输入是 [B, fusion_dim * 2]，输出是 [B, fusion_dim]
        # 第二个层：输入是 [B, fusion_dim]，输出是 [B, fusion_dim]
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        
        # Enhanced classifier with auxiliary outputs
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Auxiliary classifiers for intermediate supervision
        self.aux_classifier_spec = nn.Sequential(
            nn.Linear(spec_emb, spec_emb // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(spec_emb // 2, num_classes)
        )
        
        self.aux_classifier_tab = nn.Sequential(
            nn.Linear(tab_emb, tab_emb // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tab_emb // 2, num_classes)
        )
        
        # Attention-based scan pooling
        self.scan_attention = nn.MultiheadAttention(spec_emb, num_heads=8, dropout=dropout, batch_first=True)
        self.scan_query = nn.Parameter(torch.randn(1, 1, spec_emb))

    def forward(self, *args):
        """
        支持两种调用方式的前向传播
        
        Args:
            方式1: spectra, mask, tabular (原始张量格式)
            方式2: spectra_result, tabular_result (字典格式，embedding 模式)
        """
        if len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            # Embedding 模式：字典格式输入
            spectra_result, tabular_result = args
            
            # ===== Soft Gating for Missing Modality =====
            spec_vec = spectra_result['embedding']
            spec_mask = spectra_result.get('mask', None)
            if spec_mask is not None:
                m = spec_mask.unsqueeze(-1).float()  # [B,1]
                # 建立可学习的默认 embedding，用于模态缺失时补偿
                if not hasattr(self, "_soft_gate_default_spec"):
                    self._soft_gate_default_spec = nn.Parameter(torch.zeros_like(spec_vec[0:1]))
                # soft gating
                spec_vec = spec_vec * m + (1 - m) * self._soft_gate_default_spec
            
            tab_emb = tabular_result['embedding']
            tab_mask = tabular_result.get('mask', None)
            if tab_mask is not None:
                m = tab_mask.unsqueeze(-1).float()  # [B,1]
                # 建立可学习的默认 embedding，用于模态缺失时补偿
                if not hasattr(self, "_soft_gate_default_tab"):
                    self._soft_gate_default_tab = nn.Parameter(torch.zeros_like(tab_emb[0:1]))
                # soft gating
                tab_emb = tab_emb * m + (1 - m) * self._soft_gate_default_tab
            
            # ===== 动态投影：将实际的 embedding 维度投影到模型期望的维度 =====
            actual_spec_dim = spec_vec.shape[-1]
            actual_tab_dim = tab_emb.shape[-1]
            expected_spec_dim = self.cross_attention.spec_proj.in_features
            expected_tab_dim = self.cross_attention.tab_proj.in_features
            
            # 保存投影前的 embedding，用于 aux_classifier
            spec_vec_orig = spec_vec
            tab_emb_orig = tab_emb
            
            # 如果实际维度与模型期望维度不同，添加投影层
            if actual_spec_dim != expected_spec_dim:
                if not hasattr(self, "_embedding_proj_spec"):
                    self._embedding_proj_spec = nn.Linear(actual_spec_dim, expected_spec_dim)
                    # 确保投影层被注册为模型参数并移动到正确的设备
                    self._embedding_proj_spec = self._embedding_proj_spec.to(spec_vec.device)
                    # 注册为子模块，确保参数被包含在模型中
                    self.add_module("_embedding_proj_spec", self._embedding_proj_spec)
                spec_vec = self._embedding_proj_spec(spec_vec)
            
            if actual_tab_dim != expected_tab_dim:
                if not hasattr(self, "_embedding_proj_tab"):
                    self._embedding_proj_tab = nn.Linear(actual_tab_dim, expected_tab_dim)
                    # 确保投影层被注册为模型参数并移动到正确的设备
                    self._embedding_proj_tab = self._embedding_proj_tab.to(tab_emb.device)
                    # 注册为子模块，确保参数被包含在模型中
                    self.add_module("_embedding_proj_tab", self._embedding_proj_tab)
                tab_emb = self._embedding_proj_tab(tab_emb)
            
            spec_logits = spectra_result.get('logits', None)
            tab_logits = tabular_result.get('logits', None)
            
        elif len(args) == 3:
            # Raw 模式：处理原始张量输入
            spectra, mask, tabular = args
            B, S, L = spectra.shape

            # Enhanced spectral encoding
            spec_emb_per_scan = self.spec_encoder(spectra)  # [B, S, D_spec]

            # Attention-based scan pooling with mask（raw 模式下使用扫描级 mask）
            scan_query = self.scan_query.expand(B, -1, -1)  # [B, 1, D_spec]
            if mask is not None:
                mask_expanded = mask.unsqueeze(1)  # [B, 1, S]
                spec_vec, _ = self.scan_attention(
                    scan_query, spec_emb_per_scan, spec_emb_per_scan,
                    key_padding_mask=~mask
                )  # [B, 1, D_spec]
            else:
                # 当没有提供序列 mask（例如将 embedding 直接输入时），退化为简单平均
                spec_vec = spec_emb_per_scan.mean(dim=1, keepdim=True)  # [B, 1, D_spec]
            spec_vec = spec_vec.squeeze(1)  # [B, D_spec]

            # Enhanced tabular encoding
            tab_emb = self.tab_encoder(tabular)  # [B, D_tab]
            
            spec_logits = None
            tab_logits = None
        else:
            raise ValueError(f"[ERROR] 不支持的参数数量: {len(args)}")
        
        # ===== Learnable Modality Fusion Gate =====
        if not hasattr(self, "fusion_gate"):
            # 两个模态各一个权重，初始化为均等
            self.fusion_gate = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        # 归一化权重
        gate = torch.softmax(self.fusion_gate, dim=0)
        # 加权模态 embedding
        spec_vec = spec_vec * gate[0]
        tab_emb = tab_emb * gate[1]
        
        # 确保维度匹配：在调用 cross_attention 之前再次检查并应用投影
        # 这确保即使投影层在 Fusion Gate 之后，维度也是正确的
        if spec_vec.shape[-1] != self.cross_attention.spec_proj.in_features:
            if not hasattr(self, "_embedding_proj_spec"):
                self._embedding_proj_spec = nn.Linear(spec_vec.shape[-1], self.cross_attention.spec_proj.in_features)
                self._embedding_proj_spec = self._embedding_proj_spec.to(spec_vec.device)
                self.add_module("_embedding_proj_spec", self._embedding_proj_spec)
            spec_vec = self._embedding_proj_spec(spec_vec)
        if tab_emb.shape[-1] != self.cross_attention.tab_proj.in_features:
            if not hasattr(self, "_embedding_proj_tab"):
                self._embedding_proj_tab = nn.Linear(tab_emb.shape[-1], self.cross_attention.tab_proj.in_features)
                self._embedding_proj_tab = self._embedding_proj_tab.to(tab_emb.device)
                self.add_module("_embedding_proj_tab", self._embedding_proj_tab)
            tab_emb = self._embedding_proj_tab(tab_emb)
        
        # Cross-modal attention
        spec_attended, tab_attended = self.cross_attention(spec_vec, tab_emb)  # [B, fusion_dim]
        
        # 确保输出维度正确（处理可能的维度问题）
        if len(spec_attended.shape) > 2:
            spec_attended = spec_attended.squeeze(1)  # 确保是 [B, d_model]
        if len(tab_attended.shape) > 2:
            tab_attended = tab_attended.squeeze(1)  # 确保是 [B, d_model]
        
        # 验证输出维度
        expected_dim = self.cross_attention.spec_proj.out_features
        if spec_attended.shape[-1] != expected_dim:
            raise RuntimeError(f"spec_attended 维度错误: {spec_attended.shape}，期望最后一维: {expected_dim}")
        if tab_attended.shape[-1] != expected_dim:
            raise RuntimeError(f"tab_attended 维度错误: {tab_attended.shape}，期望最后一维: {expected_dim}")
        
        # Enhanced gating
        gated_spec = self.enhanced_gating(spec_vec, tab_emb)  # [B, D_spec]
        
        # Multi-level fusion
        # 确保 spec_attended 和 tab_attended 都是 2D 张量 [B, d_model]
        if len(spec_attended.shape) != 2:
            spec_attended = spec_attended.view(spec_attended.shape[0], -1)
        if len(tab_attended.shape) != 2:
            tab_attended = tab_attended.view(tab_attended.shape[0], -1)
        
        # 验证维度匹配
        if spec_attended.shape[0] != tab_attended.shape[0]:
            raise RuntimeError(f"Batch size 不匹配: spec_attended.shape={spec_attended.shape}, tab_attended.shape={tab_attended.shape}")
        
        fusion_input = torch.cat([spec_attended, tab_attended], dim=-1)  # [B, fusion_dim * 2]
        # 验证融合输入维度
        expected_fusion_dim = expected_dim * 2
        if fusion_input.shape[-1] != expected_fusion_dim:
            raise RuntimeError(f"fusion_input 维度错误: {fusion_input.shape}，期望最后一维: {expected_fusion_dim}, "
                             f"spec_attended.shape={spec_attended.shape}, tab_attended.shape={tab_attended.shape}, "
                             f"expected_dim={expected_dim}")
        fused_features = fusion_input
        
        for fusion_layer in self.fusion_layers:
            fused_features = fusion_layer(fused_features)  # [B, fusion_dim]
        
        # Final classification
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        # Auxiliary predictions for intermediate supervision
        # 在 embedding 模式下，使用投影前的原始 embedding
        if len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            # Embedding 模式：使用原始维度
            aux_spec_input = spec_vec_orig if 'spec_vec_orig' in locals() else spec_vec
            aux_tab_input = tab_emb_orig if 'tab_emb_orig' in locals() else tab_emb
        else:
            # Raw 模式：使用当前维度
            aux_spec_input = spec_vec
            aux_tab_input = tab_emb
        
        # 确保 aux_classifier 的输入维度正确
        if aux_spec_input.shape[-1] == self.aux_classifier_spec[0].in_features:
            aux_logits_spec = self.aux_classifier_spec(aux_spec_input)  # [B, num_classes]
        else:
            # 如果维度不匹配，创建动态投影层
            if not hasattr(self, "_aux_proj_spec"):
                self._aux_proj_spec = nn.Linear(aux_spec_input.shape[-1], self.aux_classifier_spec[0].in_features)
                self._aux_proj_spec = self._aux_proj_spec.to(aux_spec_input.device)
                self.add_module("_aux_proj_spec", self._aux_proj_spec)
            aux_logits_spec = self.aux_classifier_spec(self._aux_proj_spec(aux_spec_input))
        
        if aux_tab_input.shape[-1] == self.aux_classifier_tab[0].in_features:
            aux_logits_tab = self.aux_classifier_tab(aux_tab_input)     # [B, num_classes]
        else:
            # 如果维度不匹配，创建动态投影层
            if not hasattr(self, "_aux_proj_tab"):
                self._aux_proj_tab = nn.Linear(aux_tab_input.shape[-1], self.aux_classifier_tab[0].in_features)
                self._aux_proj_tab = self._aux_proj_tab.to(aux_tab_input.device)
                self.add_module("_aux_proj_tab", self._aux_proj_tab)
            aux_logits_tab = self.aux_classifier_tab(self._aux_proj_tab(aux_tab_input))
        
        return {
            "embedding": fused_features,        # final fused embedding
            "logits": logits,                   # main prediction
            "aux_logits_spec": aux_logits_spec, # auxiliary spectral prediction
            "aux_logits_tab": aux_logits_tab,   # auxiliary tabular prediction
            "spec_embedding": spec_vec,         # spectral embedding
            "tab_embedding": tab_emb,           # tabular embedding
            "gated_spec": gated_spec            # gated spectral features
        }


# ----------------------------
# Enhanced Loss Function with Auxiliary Supervision
# ----------------------------
class TFTLoss(nn.Module):
    def __init__(self, main_weight=1.0, aux_spec_weight=0.3, aux_tab_weight=0.3, 
                 contrastive_weight=0.1, temperature=0.07):
        super().__init__()
        self.main_weight = main_weight
        self.aux_spec_weight = aux_spec_weight
        self.aux_tab_weight = aux_tab_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        self.ce_loss = nn.CrossEntropyLoss()
        
    def contrastive_loss(self, spec_emb, tab_emb, labels):
        """
        Contrastive loss to encourage similar embeddings for same class
        """
        # Normalize embeddings
        spec_emb = F.normalize(spec_emb, p=2, dim=1)
        tab_emb = F.normalize(tab_emb, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(spec_emb, tab_emb.T) / self.temperature
        
        # Create positive/negative masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = 1 - positive_mask
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity)
        positive_sum = (exp_sim * positive_mask).sum(dim=1)
        negative_sum = (exp_sim * negative_mask).sum(dim=1)
        
        # Avoid division by zero
        positive_sum = torch.clamp(positive_sum, min=1e-8)
        negative_sum = torch.clamp(negative_sum, min=1e-8)
        
        loss = -torch.log(positive_sum / (positive_sum + negative_sum))
        return loss.mean()
    
    def forward(self, outputs, labels):
        """
        Compute combined loss with auxiliary supervision
        """
        # Main classification loss
        main_loss = self.ce_loss(outputs["logits"], labels)
        
        # Auxiliary losses
        aux_spec_loss = self.ce_loss(outputs["aux_logits_spec"], labels)
        aux_tab_loss = self.ce_loss(outputs["aux_logits_tab"], labels)
        
        # Contrastive loss between modalities
        contrastive_loss = self.contrastive_loss(
            outputs["spec_embedding"], 
            outputs["tab_embedding"], 
            labels
        )
        
        # Combined loss
        total_loss = (self.main_weight * main_loss + 
                     self.aux_spec_weight * aux_spec_loss + 
                     self.aux_tab_weight * aux_tab_loss + 
                     self.contrastive_weight * contrastive_loss)
        
        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "aux_spec_loss": aux_spec_loss,
            "aux_tab_loss": aux_tab_loss,
            "contrastive_loss": contrastive_loss
        }
