# multimodal_raman.py
# Minimal, well-documented PyTorch skeleton for:
#  - Spectra (many scans per patient, each scan = 1D Raman spectrum)
#  - Tabular clinical features (patient-level)
#  - Patient-level prediction (classification/regression)
#
# Key components:
#  - Preprocessing: AsLS baseline correction, Savitzky-Golay smoothing, SNV normalization
#  - Spectra encoder: multi-scale 1D-CNN (residual blocks)
#  - Patient pooling: attention pooling over scans
#  - Fusion: cross-attention fusion (spectra <-> tabular) or concat baseline
#  - Grad-CAM 1D for explainability on spectra branch
#
# Example usage:
#   python multimodal_raman.py --csv data/spectra.csv --clinical clinical.csv --epochs 20
#
# CSV assumptions:
#  - spectra CSV: columns: Sample (like "100-176.txt"), Group/Label, w1, w2, ..., wL
#    where Sample encodes patientID-scanIndex (e.g., "100-176.txt" or "100-176")
#  - clinical CSV: patient-level rows keyed by patientID (100), columns clinical features and label
#
# This file focuses on readability and modularity, not hyper-optimized performance.

import os, argparse, math, random
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Preprocessing utilities
# ----------------------------
def asls_baseline(y, lam=1e6, p=0.001, niter=10):
    """
    Asymmetric least squares baseline correction.
    y: 1D numpy array
    returns corrected spectrum y - baseline
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = D.dot(D.transpose())  # second derivative
    w = np.ones(L)
    for i in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * D
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    baseline = z
    return y - baseline

def snv(x):
    """Standard Normal Variate: subtract mean, divide by std per spectrum"""
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        return x - mu
    return (x - mu) / sigma

def preprocess_spectrum(y: np.ndarray, apply_asls=True, sg_window=11, sg_poly=2, apply_snv=True):
    """Pipeline: AsLS -> Savitzky-Golay -> SNV"""
    x = y.astype(np.float64)
    if apply_asls:
        try:
            x = asls_baseline(x)
        except Exception as e:
            # fallback: subtract smooth baseline via Savitzky-Golay
            x = x - savgol_filter(x, sg_window, sg_poly)
    # SG smoothing (works as lowpass)
    if sg_window >= 5:
        x = savgol_filter(x, sg_window, sg_poly)
    if apply_snv:
        x = snv(x)
    return x.astype(np.float32)

# ----------------------------
# Data handling
# ----------------------------
def parse_sample_name(sample: str):
    """
    Parse sample string like '100-176.txt' or '100-176' -> return patient_id '100' and scan_idx 176
    """
    s = sample.replace('.txt','')
    parts = s.split('-')
    patient = parts[0]
    scan_idx = int(parts[1]) if len(parts) > 1 else 0
    return patient, scan_idx

class RamanDataset(Dataset):
    """
    Groups raw scan-level spectra into patient-level entries.
    Each item returned: dict with:
      'patient_id', 'spectra': Tensor [num_scans x L], 'tabular': Tensor [D], 'label': int/float
    """
    def __init__(self,
                 spectra_csv: str,
                 clinical_csv: str,
                 wave_cols: List[str],
                 label_col: str = 'Group',
                 patient_id_parser=parse_sample_name,
                 preprocess_fn=preprocess_spectrum,
                 min_scans=1,
                 max_scans=None,
                 tabular_fill_value=0.0):
        # read
        df = pd.read_csv(spectra_csv, sep=None, engine='python')
        clinic = pd.read_csv(clinical_csv, sep=None, engine='python', index_col=0)
        # assemble per-patient
        per_patient = {}
        for _, row in df.iterrows():
            sample = str(row['Sample'])
            patient, scan_idx = patient_id_parser(sample)
            spec = row[wave_cols].values.astype(float)
            spec = preprocess_fn(spec)
            label = row[label_col] if label_col in row.index else None
            if patient not in per_patient:
                per_patient[patient] = {'spectra': [], 'label': label}
            per_patient[patient]['spectra'].append((scan_idx, spec))
        # sort scans and convert
        self.items = []
        for patient, v in per_patient.items():
            scans = sorted(v['spectra'], key=lambda x: x[0])
            specs = np.stack([s for _, s in scans], axis=0)  # num_scans x L
            if specs.shape[0] < min_scans:
                continue
            if max_scans is not None and specs.shape[0] > max_scans:
                specs = specs[:max_scans]
            # clinical features
            if patient in clinic.index:
                tab = clinic.loc[patient].drop(labels=[label_col], errors='ignore') if label_col in clinic.columns else clinic.loc[patient]
                tab = tab.fillna(tabular_fill_value).values.astype(float)
                lab = clinic.loc[patient, label_col] if label_col in clinic.columns else v['label']
            else:
                # fallback: zeros
                tab = np.zeros((clinic.shape[1],), dtype=float) if clinic.shape[1]>0 else np.zeros((1,), dtype=float)
                lab = v['label']
            self.items.append({'patient': patient, 'spectra': specs.astype(np.float32), 'tabular': tab.astype(np.float32), 'label': lab})
        # tabular scaler
        if len(self.items) == 0:
            raise ValueError("No patients found. Check CSV parsing.")
        tab_stack = np.stack([it['tabular'] for it in self.items], axis=0)
        self.tab_scaler = StandardScaler()
        self.tab_scaler.fit(tab_stack)
        for it in self.items:
            it['tabular'] = self.tab_scaler.transform(it['tabular'].reshape(1,-1)).squeeze().astype(np.float32)

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            'patient': it['patient'],
            'spectra': torch.from_numpy(it['spectra']),  # [num_scans, L]
            'tabular': torch.from_numpy(it['tabular']),
            'label': torch.tensor(1 if str(it['label']).strip().upper()=='DM' else 0, dtype=torch.long)  # example binary map
        }

def collate_fn(batch):
    # variable number of scans per patient -> pad to max in batch
    patients = [b['patient'] for b in batch]
    labels = torch.stack([b['label'] for b in batch])
    tabular = torch.stack([b['tabular'] for b in batch])
    scan_counts = [b['spectra'].shape[0] for b in batch]
    max_scans = max(scan_counts)
    L = batch[0]['spectra'].shape[1]
    spectra_padded = torch.zeros((len(batch), max_scans, L), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_scans), dtype=torch.bool)
    for i,b in enumerate(batch):
        n = b['spectra'].shape[0]
        spectra_padded[i, :n, :] = b['spectra']
        mask[i, :n] = True
    return {'patient': patients, 'spectra': spectra_padded, 'mask': mask, 'tabular': tabular, 'label': labels}

# ----------------------------
# Model components
# ----------------------------
class ResidualConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=9, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: [B, C, L]
        r = self.conv1(x)
        r = self.bn1(r)
        r = self.act(r)
        r = self.conv2(r)
        r = self.bn2(r)
        s = self.shortcut(x)
        return self.act(r + s)

class SpectraEncoder(nn.Module):
    """
    专门用于接收外部光谱处理模型的输出
    输入：spectra_result: dict {"embedding": [B, D], "logits": [B, C]}
    输出：直接返回外部模型的结果
    """
    def __init__(self, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, spectra_result):
        """
        接收外部模型输出并直接返回
        
        Args:
            spectra_result: dict from external model {"embedding": [B, D], "logits": [B, C]}
        
        Returns:
            dict with same structure as input
        """
        return {
            "embedding": spectra_result["embedding"],  # [B, emb_dim]
            "logits": spectra_result["logits"]         # [B, num_classes]
        }

# PatientPooling 类已移除，因为现在只处理外部模型输出

class TabularEncoder(nn.Module):
    """
    专门用于接收外部表格处理模型的输出
    输入：tabular_result: dict {"embedding": [B, D], "logits": [B, C]}
    输出：直接返回外部模型的结果
    """
    def __init__(self, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
    
    def forward(self, tabular_result):
        """
        接收外部模型输出并直接返回
        
        Args:
            tabular_result: dict from external model {"embedding": [B, D], "logits": [B, C]}
        
        Returns:
            dict with same structure as input
        """
        return {
            "embedding": tabular_result["embedding"],  # [B, emb_dim]
            "logits": tabular_result["logits"]         # [B, num_classes]
        }

class EnhancedCrossAttentionFusion(nn.Module):
    """
    增强的跨模态注意力融合机制：
    - 双向注意力：光谱->表格 和 表格->光谱
    - 多头注意力：捕获不同层面的特征交互
    - 残差连接：保持梯度流动
    - 层归一化：提升训练稳定性
    """
    def __init__(self, spec_dim, tab_dim, hid_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        
        # 投影层：将不同维度的特征投影到统一维度
        self.spec_proj = nn.Sequential(
            nn.Linear(spec_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 双向多头注意力
        self.spec_to_tab_attn = nn.MultiheadAttention(
            embed_dim=hid_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.tab_to_spec_attn = nn.MultiheadAttention(
            embed_dim=hid_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hid_dim * 4, hid_dim * 2),  # 4个特征拼接：spec, tab, spec_attended, tab_attended
            nn.LayerNorm(hid_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.LayerNorm(hid_dim)
        )
        
        # 门控机制：动态调节不同模态的重要性
        self.gate = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Sigmoid()
        )

    def forward(self, spec_vec, tab_vec):
        """
        双向跨模态注意力融合
        
        Args:
            spec_vec: [B, spec_dim] 光谱特征
            tab_vec: [B, tab_dim] 表格特征
            
        Returns:
            fused_features: [B, hid_dim] 融合后的特征
        """
        # 投影到统一维度
        spec_proj = self.spec_proj(spec_vec).unsqueeze(1)  # [B, 1, hid_dim]
        tab_proj = self.tab_proj(tab_vec).unsqueeze(1)     # [B, 1, hid_dim]
        
        # 双向注意力计算
        # 光谱关注表格特征
        spec_attended, spec_attn_weights = self.spec_to_tab_attn(
            query=spec_proj, 
            key=tab_proj, 
            value=tab_proj
        )
        
        # 表格关注光谱特征  
        tab_attended, tab_attn_weights = self.tab_to_spec_attn(
            query=tab_proj,
            key=spec_proj,
            value=spec_proj
        )
        
        # 残差连接
        spec_attended = spec_attended.squeeze(1) + spec_proj.squeeze(1)  # [B, hid_dim]
        tab_attended = tab_attended.squeeze(1) + tab_proj.squeeze(1)     # [B, hid_dim]
        
        # 特征拼接和融合
        combined_features = torch.cat([
            spec_proj.squeeze(1),    # 原始光谱特征
            tab_proj.squeeze(1),     # 原始表格特征
            spec_attended,           # 光谱关注表格后的特征
            tab_attended             # 表格关注光谱后的特征
        ], dim=-1)  # [B, hid_dim * 4]
        
        # 通过融合层
        fused_features = self.fusion_layer(combined_features)  # [B, hid_dim]
        
        # 门控机制：动态调节光谱和表格的重要性
        gate_input = torch.cat([spec_attended, tab_attended], dim=-1)
        gate_weights = self.gate(gate_input)  # [B, hid_dim]
        final_features = fused_features * gate_weights
        
        return final_features

class EnhancedClassifier(nn.Module):
    """
    增强的分类器：
    - 更深的网络结构：多层感知机
    - 残差连接：保持梯度流动
    - 层归一化：提升训练稳定性
    - 自适应dropout：根据训练阶段调整
    """
    def __init__(self, in_dim, num_classes=2, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        
        # 构建多层网络
        layers = []
        prev_dim = in_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # 线性层
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # 层归一化
            layers.append(nn.LayerNorm(hidden_dim))
            
            # 激活函数
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # 最终分类层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # 残差连接（如果输入维度与第一个隐藏层维度相同）
        self.use_residual = (in_dim == hidden_dims[0])
        if self.use_residual:
            self.residual_proj = nn.Linear(in_dim, hidden_dims[0])

    def forward(self, x):
        """
        前向传播，支持残差连接
        
        Args:
            x: [B, in_dim] 输入特征
            
        Returns:
            logits: [B, num_classes] 分类logits
        """
        if self.use_residual and len(self.classifier) > 4:
            # 保存输入用于残差连接
            residual = x
            
            # 通过前几层
            x = self.classifier[:-4](x)  # 除了最后4层（Linear, LayerNorm, ReLU, Dropout）
            
            # 残差连接
            if residual.shape[-1] == x.shape[-1]:
                x = x + residual
            else:
                x = x + self.residual_proj(residual)
            
            # 通过剩余层
            x = self.classifier[-4:](x)
        else:
            x = self.classifier(x)
        
        return x

# MultimodalModel 类已移除，因为现在只处理外部模型输出
# 请使用 AttentionMultimodal 类进行融合

# GradCAM1D 类已移除，因为现在只处理外部模型输出
# 如果需要可解释性分析，请在外部模型中实现


# ----------------------------
# 新的融合包装器类 - 专门用于接收外部模型输出
# ----------------------------
class EnhancedAttentionMultimodal(nn.Module):
    """
    增强的注意力融合模型：
    - 使用增强的跨模态注意力机制
    - 支持辅助监督训练
    - 更深的分类器网络
    - 可配置的融合策略
    """
    def __init__(self, spec_embedding_dim=256, tab_embedding_dim=128, num_classes=2, 
                 fusion_type='enhanced_cross', use_auxiliary=True, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.spec_encoder = SpectraEncoder(emb_dim=spec_embedding_dim)
        self.tab_encoder = TabularEncoder(emb_dim=tab_embedding_dim)
        self.use_auxiliary = use_auxiliary
        
        # 选择融合策略
        if fusion_type == 'enhanced_cross':
            self.fusion = EnhancedCrossAttentionFusion(
                spec_dim=spec_embedding_dim, 
                tab_dim=tab_embedding_dim, 
                hid_dim=256,
                num_heads=8,
                dropout=0.1
            )
            self.classifier = EnhancedClassifier(256, num_classes=num_classes, hidden_dims=hidden_dims)
        elif fusion_type == 'concat':
            self.fusion = None
            self.classifier = EnhancedClassifier(
                spec_embedding_dim + tab_embedding_dim, 
                num_classes=num_classes, 
                hidden_dims=hidden_dims
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.fusion_type = fusion_type
        
        # 辅助监督分类器
        if self.use_auxiliary:
            self.aux_spec_classifier = EnhancedClassifier(
                spec_embedding_dim, 
                num_classes=num_classes, 
                hidden_dims=[128, 64]
            )
            self.aux_tab_classifier = EnhancedClassifier(
                tab_embedding_dim, 
                num_classes=num_classes, 
                hidden_dims=[64, 32]
            )

    def forward(self, spectra_result, tabular_result):
        """
        增强的注意力融合前向传播
        
        Args:
            spectra_result: dict {"embedding": [B, D_spec], "logits": [B, C]}
            tabular_result: dict {"embedding": [B, D_tab], "logits": [B, C]}
        
        Returns:
            dict with fused results and auxiliary outputs
        """
        # 获取各模态的embedding和logits
        spec_result = self.spec_encoder(spectra_result)
        tab_result = self.tab_encoder(tabular_result)
        
        spec_embedding = spec_result["embedding"]  # [B, emb_dim]
        tab_embedding = tab_result["embedding"]    # [B, emb_dim]
        spec_logits = spec_result["logits"]
        tab_logits = tab_result["logits"]
        
        # 融合两个模态的特征
        if self.fusion_type == 'enhanced_cross':
            fused_embedding = self.fusion(spec_embedding, tab_embedding)
        else:  # concat
            fused_embedding = torch.cat([spec_embedding, tab_embedding], dim=-1)
        
        # 主分类器预测
        fused_logits = self.classifier(fused_embedding)
        
        # 构建返回结果
        result = {
            "embedding": fused_embedding,
            "logits": fused_logits,
            "spec_embedding": spec_embedding,
            "tab_embedding": tab_embedding,
            "spec_logits": spec_logits,
            "tab_logits": tab_logits
        }
        
        # 辅助监督预测
        if self.use_auxiliary:
            aux_spec_logits = self.aux_spec_classifier(spec_embedding)
            aux_tab_logits = self.aux_tab_classifier(tab_embedding)
            
            result.update({
                "aux_spec_logits": aux_spec_logits,
                "aux_tab_logits": aux_tab_logits
            })
        
        return result

# 保持向后兼容性
class AttentionMultimodal(EnhancedAttentionMultimodal):
    """
    向后兼容的注意力融合模型
    默认使用增强的跨模态注意力机制
    
    支持两种调用方式：
    1. model(spectra_result, tabular_result) - 原始方式，需要预计算的embedding/logits
    2. model(spectra, mask, tabular) - 新方式，直接从原始数据计算
    """
    def __init__(self, spec_embedding_dim=256, tab_embedding_dim=128, num_classes=2, fusion_type='enhanced_cross', tab_dim=None):
        super().__init__(
            spec_embedding_dim=spec_embedding_dim,
            tab_embedding_dim=tab_embedding_dim,
            num_classes=num_classes,
            fusion_type=fusion_type,
            use_auxiliary=True,
            hidden_dims=[512, 256, 128]
        )
        
        # 保存属性（确保在调用_build方法之前设置）
        self.num_classes = num_classes
        self.spec_embedding_dim = spec_embedding_dim
        self.tab_embedding_dim = tab_embedding_dim
        
        # 添加内部编码器，用于处理原始输入
        self.internal_spec_encoder = self._build_spectral_encoder()
        self.internal_tab_encoder = self._build_tabular_encoder(tab_dim or 128)
    
    def _build_spectral_encoder(self):
        """构建内部光谱编码器"""
        return nn.Sequential(
            # 扫描级特征提取
            ResidualConv1D(1, 64, kernel_size=7),
            ResidualConv1D(64, 128, kernel_size=5),
            ResidualConv1D(128, 256, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(256, self.spec_embedding_dim),
            nn.LayerNorm(self.spec_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def _build_tabular_encoder(self, tab_dim):
        """构建内部表格编码器"""
        return nn.Sequential(
            nn.Linear(tab_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.tab_embedding_dim),
            nn.LayerNorm(self.tab_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, *args):
        """
        支持两种调用方式的前向传播
        
        Args:
            方式1: spectra_result, tabular_result (dict格式)
            方式2: spectra, mask, tabular (原始张量格式)
        """
        if len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            # 原始方式：直接调用父类方法
            return super().forward(args[0], args[1])
        
        elif len(args) == 3:
            # 新方式：处理原始张量输入，确保两个维度都是embedding和logits格式
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
                spec_embedding = (spec_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                spec_embedding = spec_features.mean(dim=1)  # [B, spec_embedding_dim]
            
            # 处理表格数据
            tab_embedding = self.internal_tab_encoder(tabular)  # [B, tab_embedding_dim]
            
            # 为每个模态创建独立的分类器来生成logits
            spec_logits = self.aux_spec_classifier(spec_embedding)  # [B, num_classes]
            tab_logits = self.aux_tab_classifier(tab_embedding)    # [B, num_classes]
            
            # 创建字典格式 - 确保两个维度都是embedding和logits格式
            spectra_result = {
                "embedding": spec_embedding,
                "logits": spec_logits
            }
            tabular_result = {
                "embedding": tab_embedding,
                "logits": tab_logits
            }
            
            # 调用父类方法
            return super().forward(spectra_result, tabular_result)
        
        else:
            raise ValueError(f"不支持的参数数量: {len(args)}. 支持: (spectra_result, tabular_result) 或 (spectra, mask, tabular)")


# ----------------------------
# 辅助损失函数
# ----------------------------
class MultiTaskLoss(nn.Module):
    """
    多任务损失函数：
    - 主任务损失：融合后的预测
    - 辅助任务损失：各模态独立预测
    - 可配置的权重平衡
    """
    def __init__(self, main_weight=1.0, aux_spec_weight=0.3, aux_tab_weight=0.3, 
                 consistency_weight=0.1, num_classes=2):
        super().__init__()
        self.main_weight = main_weight
        self.aux_spec_weight = aux_spec_weight
        self.aux_tab_weight = aux_tab_weight
        self.consistency_weight = consistency_weight
        self.num_classes = num_classes
        
        # 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss()
        
        # KL散度损失（用于一致性约束）
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, targets):
        """
        计算多任务损失
        
        Args:
            outputs: 模型输出字典
            targets: 真实标签 [B]
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细信息
        """
        # 主任务损失
        main_loss = self.ce_loss(outputs["logits"], targets)
        
        loss_dict = {
            "main_loss": main_loss.item(),
            "total_loss": 0.0
        }
        
        total_loss = self.main_weight * main_loss
        
        # 辅助任务损失
        if "aux_spec_logits" in outputs:
            aux_spec_loss = self.ce_loss(outputs["aux_spec_logits"], targets)
            total_loss += self.aux_spec_weight * aux_spec_loss
            loss_dict["aux_spec_loss"] = aux_spec_loss.item()
        
        if "aux_tab_logits" in outputs:
            aux_tab_loss = self.ce_loss(outputs["aux_tab_logits"], targets)
            total_loss += self.aux_tab_weight * aux_tab_loss
            loss_dict["aux_tab_loss"] = aux_tab_loss.item()
        
        # 一致性损失：确保各模态预测的一致性
        if self.consistency_weight > 0 and "aux_spec_logits" in outputs and "aux_tab_logits" in outputs:
            # 计算预测概率分布
            main_probs = F.softmax(outputs["logits"], dim=-1)
            spec_probs = F.softmax(outputs["aux_spec_logits"], dim=-1)
            tab_probs = F.softmax(outputs["aux_tab_logits"], dim=-1)
            
            # KL散度损失
            consistency_loss = (
                self.kl_loss(F.log_softmax(outputs["logits"], dim=-1), spec_probs) +
                self.kl_loss(F.log_softmax(outputs["logits"], dim=-1), tab_probs) +
                self.kl_loss(F.log_softmax(outputs["aux_spec_logits"], dim=-1), tab_probs)
            ) / 3.0
            
            total_loss += self.consistency_weight * consistency_loss
            loss_dict["consistency_loss"] = consistency_loss.item()
        
        loss_dict["total_loss"] = total_loss.item()
        
        return total_loss, loss_dict
