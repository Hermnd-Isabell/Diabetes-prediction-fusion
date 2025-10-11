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
        
        return spec_attended.squeeze(1), tab_attended.squeeze(1)  # [B, d_model]


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
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(2)
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

    def forward(self, spectra, mask, tabular):
        # spectra: [B, S, L], mask: [B, S], tabular: [B, D]
        B, S, L = spectra.shape
        
        # Enhanced spectral encoding
        spec_emb_per_scan = self.spec_encoder(spectra)  # [B, S, D_spec]
        
        # Attention-based scan pooling with mask
        scan_query = self.scan_query.expand(B, -1, -1)  # [B, 1, D_spec]
        # Apply mask to attention
        mask_expanded = mask.unsqueeze(1)  # [B, 1, S]
        spec_vec, _ = self.scan_attention(scan_query, spec_emb_per_scan, spec_emb_per_scan, 
                                        key_padding_mask=~mask)  # [B, 1, D_spec]
        spec_vec = spec_vec.squeeze(1)  # [B, D_spec]
        
        # Enhanced tabular encoding
        tab_emb = self.tab_encoder(tabular)  # [B, D_tab]
        
        # Cross-modal attention
        spec_attended, tab_attended = self.cross_attention(spec_vec, tab_emb)  # [B, fusion_dim]
        
        # Enhanced gating
        gated_spec = self.enhanced_gating(spec_vec, tab_emb)  # [B, D_spec]
        
        # Multi-level fusion
        fusion_input = torch.cat([spec_attended, tab_attended], dim=-1)  # [B, fusion_dim * 2]
        fused_features = fusion_input
        
        for fusion_layer in self.fusion_layers:
            fused_features = fusion_layer(fused_features)  # [B, fusion_dim]
        
        # Final classification
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        # Auxiliary predictions for intermediate supervision
        aux_logits_spec = self.aux_classifier_spec(spec_vec)  # [B, num_classes]
        aux_logits_tab = self.aux_classifier_tab(tab_emb)     # [B, num_classes]
        
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
