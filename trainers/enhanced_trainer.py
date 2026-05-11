#!/usr/bin/env python3
"""
增强版训练器 - 支持四个模型并包含丰富的可视化和可解释性分析

支持的模型:
- AttentionMultimodal (注意力机制)
- Baseline (ConcatFusion, EnsembleFusion)
- TFTMultimodal (时序融合Transformer)

功能特性:
- 多模型训练和对比
- 丰富的可视化展示
- 可解释性分析
- 性能指标跟踪
- 模型保存和加载
"""

import os
import json
import time
import warnings
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, roc_curve, precision_recall_curve,
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import shap
from models.tft_models import TFTLoss

# 兼容性修复：处理 Numpy 2.0 checkpiont 在 Numpy 1.x 环境下的加载问题
import numpy as np
try:
    import numpy._core
except ImportError:
    # 如果找不到 numpy._core (Numpy 1.x)，但 checkpoint 引用了它 (Numpy 2.0+)
    # 将 numpy.core 映射到 numpy._core
    if hasattr(np, 'core'):
        import sys
        sys.modules['numpy._core'] = np.core
        # 同时也可能需要 multiarray
        try:
            from numpy.core import multiarray
            sys.modules['numpy._core.multiarray'] = multiarray
        except ImportError:
            pass
        print("[DEBUG] Applied numpy 2.0 -> 1.x compatibility hack")


# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

# ----------------------------
# 高级训练策略：Focal Loss
# ----------------------------
class FocalLoss(nn.Module):
    """Focal Loss with optional label smoothing support."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.weight = weight
        if self.label_smoothing > 0:
            # When label smoothing is used with focal, we use BCEWithLogitsLoss
            self.base_criterion = nn.BCEWithLogitsLoss(weight=weight, reduction='none')
        else:
            self.base_criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            num_classes = inputs.size(1)
            # One-hot with smoothing
            targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
            targets_smooth = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            ce_loss = self.base_criterion(inputs, targets_smooth).sum(dim=1)
        else:
            ce_loss = self.base_criterion(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ----------------------------
# 高级训练策略：SAM (Sharpness-Aware Minimization)
# ----------------------------
class SAM(torch.optim.Optimizer):
    """
    Lightweight SAM wrapper.
    Requires two forward-backward passes per step.
    """
    def __init__(self, params, base_optimizer, rho=0.05):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        # Inject rho into each param_group
        for group in self.param_groups:
            group['rho'] = rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # w -> w + e(w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])  # w + e(w) -> w
        self.base_optimizer.step()  # update at w
        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


class EnhancedTrainer:
    """
    增强版训练器类
    
    支持多模型训练、可视化分析和可解释性研究
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        save_dir: str = "results",
        enable_visualization: bool = True,
        enable_interpretability: bool = True,
        use_embedding_input: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        advanced_cfg: Optional[Dict[str, Any]] = None,
        num_classes: int = 2,
        fold_idx: Optional[int] = None,
        evaluation_cfg: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化增强版训练器

        Args:
            model: 要训练的模型
            model_name: 模型名称
            device: 训练设备
            lr: 学习率
            weight_decay: 权重衰减
            save_dir: 结果保存目录
            enable_visualization: 是否启用可视化
            enable_interpretability: 是否启用可解释性分析
            use_embedding_input: 是否使用了 embedding 输入
            class_weights: 类别权重 (用于处理由于类别不平衡)
            advanced_cfg: 高级训练策略配置字典
            num_classes: 类别数
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.save_dir = Path(save_dir) / model_name
        if fold_idx is not None:
            self.save_dir = self.save_dir / f"fold_{fold_idx}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_embedding_input = use_embedding_input
        self.advanced_cfg = advanced_cfg or {}
        self.advanced_enabled = self.advanced_cfg.get('enabled', False)
        self.num_classes = num_classes
        self.base_lr = lr
        self.base_wd = weight_decay
        self.fold_idx = fold_idx

        # 评估指标配置（独立于 advanced.enabled）
        eval_cfg = evaluation_cfg or {}
        metrics_cfg = eval_cfg.get('metrics', {})
        self.compute_macro_auc = metrics_cfg.get('compute_macro_auc', False)
        self.compute_weighted_auc = metrics_cfg.get('compute_weighted_auc', False)
        self.compute_cohens_kappa = metrics_cfg.get('compute_cohens_kappa', False)
        self.compute_qwk = metrics_cfg.get('compute_qwk', False)

        # 高级配置项解析（向后兼容：advanced 缺失时取默认值）
        self.loss_type = self.advanced_cfg.get('loss_type', 'CE')
        self.focal_gamma = self.advanced_cfg.get('focal_gamma', 2.0)
        self.label_smoothing = self.advanced_cfg.get('label_smoothing', 0.0)
        self.aux_weight = self.advanced_cfg.get('aux_weight', 1.0)
        self.current_aux_weight = self.aux_weight
        self.aux_decay_schedule = self.advanced_cfg.get('aux_decay_schedule', None)
        self.grad_clip_max_norm = self.advanced_cfg.get('grad_clip_max_norm', None)
        self.scheduler_type = self.advanced_cfg.get('scheduler_type', 'plateau')
        self.warmup_epochs = self.advanced_cfg.get('warmup_epochs', 0)
        self.cosine_T_max = self.advanced_cfg.get('cosine_T_max', 100)
        self.early_stop_metric = self.advanced_cfg.get('early_stop_metric', 'auc')
        self.early_stop_patience = self.advanced_cfg.get('early_stop_patience', None)
        self.phase_training_cfg = self.advanced_cfg.get('phase_training', {})
        self.phase_training_enabled = self.phase_training_cfg.get('enabled', False)

        # 训练设置：损失函数
        if self.model_name == 'TFTMultimodal':
            print(f"[TRAIN] TFT模型使用自定义 TFTLoss")
            self.criterion = TFTLoss().to(device)
        else:
            self.criterion = self._build_criterion(class_weights)

        # 训练设置：优化器与调度器
        self.optimizer, self.base_optimizer = self._setup_optimizer(lr, weight_decay)
        self.scheduler = self._setup_scheduler()

        # 功能开关
        self.enable_visualization = enable_visualization
        self.enable_interpretability = enable_interpretability

        # 训练历史（扩展以容纳新指标）
        self.train_history = {
            'loss': [], 'acc': [], 'auc': [], 'f1': []
        }
        self.val_history = {
            'loss': [], 'acc': [], 'auc': [], 'f1': []
        }

        # 最佳模型
        self.best_val_metric = float('-inf')
        self.best_model_state = None
        self.best_epoch = 0

        # 模态权重历史记录（用于可视化）
        self.modality_gate_history = []

        print(f"[INIT] 增强版训练器初始化完成")
        print(f"[MODEL] 模型: {model_name}")
        print(f"[DEVICE] 设备: {device}")
        print(f"[SAVE] 保存目录: {self.save_dir}")
        print(f"[VIS] 可视化: {'启用' if enable_visualization else '禁用'}")
        print(f"[INTERP] 可解释性: {'启用' if enable_interpretability else '禁用'}")
        print(f"[MODE] 输入模式: {'embedding' if self.use_embedding_input else 'raw 序列'}")
        if self.advanced_enabled:
            print(f"[ADVANCED] 高级训练策略已启用: loss={self.loss_type}, scheduler={self.scheduler_type}, optimizer={self.advanced_cfg.get('optimizer', 'AdamW')}")

    def _build_criterion(self, class_weights: Optional[torch.Tensor] = None):
        """构建损失函数（支持 CE / Focal / Label Smoothing）"""
        if not self.advanced_enabled:
            if class_weights is not None:
                return nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            return nn.CrossEntropyLoss()

        if self.loss_type == 'Focal':
            return FocalLoss(
                gamma=self.focal_gamma,
                weight=class_weights.to(self.device) if class_weights is not None else None,
                label_smoothing=self.label_smoothing,
            )
        elif self.loss_type == 'CORN':
            raise NotImplementedError("CORN loss is not yet implemented. Please use 'CE' or 'Focal'.")
        elif self.loss_type == 'CE':
            if self.label_smoothing > 0:
                return nn.CrossEntropyLoss(
                    weight=class_weights.to(self.device) if class_weights is not None else None,
                    label_smoothing=self.label_smoothing
                )
            if class_weights is not None:
                return nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _setup_optimizer(self, lr: float, weight_decay: float):
        """构建优化器（支持分层参数组与 SAM）"""
        optimizer_name = self.advanced_cfg.get('optimizer', 'AdamW') if self.advanced_enabled else 'AdamW'
        param_groups_cfg = self.advanced_cfg.get('param_groups', None) if self.advanced_enabled else None

        if not self.advanced_enabled or param_groups_cfg is None:
            # 默认全局单组
            base_opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            if optimizer_name == 'AdamW_SAM':
                rho = self.advanced_cfg.get('sam_rho', 0.05)
                sam = SAM(self.model.parameters(), base_opt, rho=rho)
                return sam, base_opt
            return base_opt, base_opt

        # 分层参数组
        groups = []
        assigned_ids = set()

        def collect_params(keywords, name):
            params = []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if id(p) in assigned_ids:
                    continue
                if any(kw in n for kw in keywords):
                    params.append(p)
                    assigned_ids.add(id(p))
            return params

        # spectra_encoder
        spec_params = collect_params(['spectra', 'spec'], 'spectra_encoder')
        if spec_params:
            mul = param_groups_cfg.get('spectra_encoder', {}).get('lr_multiplier', 1.0)
            wd_mul = param_groups_cfg.get('spectra_encoder', {}).get('wd_multiplier', 1.0)
            groups.append({'params': spec_params, 'lr': lr * mul, 'weight_decay': weight_decay * wd_mul, 'name': 'spectra_encoder'})

        # clinical_encoder
        clin_params = collect_params(['clinical', 'tabular', 'tab'], 'clinical_encoder')
        if clin_params:
            mul = param_groups_cfg.get('clinical_encoder', {}).get('lr_multiplier', 1.0)
            wd_mul = param_groups_cfg.get('clinical_encoder', {}).get('wd_multiplier', 1.0)
            groups.append({'params': clin_params, 'lr': lr * mul, 'weight_decay': weight_decay * wd_mul, 'name': 'clinical_encoder'})

        # fusion_module
        fusion_params = collect_params(['fusion', 'cross', 'mmtm', 'attention'], 'fusion_module')
        if fusion_params:
            mul = param_groups_cfg.get('fusion_module', {}).get('lr_multiplier', 1.0)
            wd_mul = param_groups_cfg.get('fusion_module', {}).get('wd_multiplier', 1.0)
            groups.append({'params': fusion_params, 'lr': lr * mul, 'weight_decay': weight_decay * wd_mul, 'name': 'fusion_module'})

        # classifier_head
        head_params = collect_params(['classifier', 'head'], 'classifier_head')
        if head_params:
            mul = param_groups_cfg.get('classifier_head', {}).get('lr_multiplier', 1.0)
            wd_mul = param_groups_cfg.get('classifier_head', {}).get('wd_multiplier', 1.0)
            groups.append({'params': head_params, 'lr': lr * mul, 'weight_decay': weight_decay * wd_mul, 'name': 'classifier_head'})

        # 兜底默认组
        remaining = [p for p in self.model.parameters() if id(p) not in assigned_ids and p.requires_grad]
        if remaining:
            groups.append({'params': remaining, 'lr': lr, 'weight_decay': weight_decay, 'name': 'default'})

        print(f"[OPT] 分层参数组 ({len(groups)} 组):")
        for g in groups:
            print(f"  - {g['name']}: n_params={sum(p.numel() for p in g['params'])}, lr={g['lr']:.2e}, wd={g['weight_decay']:.2e}")

        base_opt = optim.AdamW(groups)
        if optimizer_name == 'AdamW_SAM':
            rho = self.advanced_cfg.get('sam_rho', 0.05)
            sam = SAM([g['params'] for g in groups], base_opt, rho=rho)
            return sam, base_opt
        return base_opt, base_opt

    def _setup_scheduler(self):
        """构建学习率调度器"""
        if not self.advanced_enabled or self.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.base_optimizer, mode='min', patience=5, factor=0.5
            )
        elif self.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.base_optimizer, T_max=self.cosine_T_max
            )
        else:
            raise ValueError(f"Unknown scheduler_type: {self.scheduler_type}")

    def _compute_loss(self, outputs, labels):
        """统一损失计算（含辅助损失）"""
        logits = outputs["logits"]

        if self.model_name == 'TFTMultimodal':
            loss_dict = self.criterion(outputs, labels)
            return loss_dict["total_loss"]

        main_loss = self.criterion(logits, labels)
        total_loss = main_loss

        # 辅助损失（AttentionMultimodal / TFTMultimodal 的 aux_spec_logits / aux_tab_logits）
        if self.current_aux_weight > 0:
            aux_spec = outputs.get("aux_spec_logits", None)
            aux_tab = outputs.get("aux_tab_logits", None)
            if aux_spec is not None:
                total_loss = total_loss + self.current_aux_weight * self.criterion(aux_spec, labels)
            if aux_tab is not None:
                total_loss = total_loss + self.current_aux_weight * self.criterion(aux_tab, labels)

        return total_loss

    def _apply_phase_training(self, epoch: int):
        """渐进式训练：冻结/解冻参数"""
        if not self.advanced_enabled or not self.phase_training_enabled:
            return

        phase1_epochs = self.phase_training_cfg.get('phase1_epochs', 20)
        phase1_modules = self.phase_training_cfg.get('phase1_modules', [])
        phase2_modules = self.phase_training_cfg.get('phase2_modules', [])

        if epoch < phase1_epochs:
            active_modules = phase1_modules
        else:
            active_modules = phase1_modules + phase2_modules

        # 将配置中的模块名映射到参数名子串（与 _setup_optimizer 一致）
        keyword_map = {
            'spectra_encoder': ['spectra', 'spec'],
            'clinical_encoder': ['clinical', 'tabular', 'tab'],
            'fusion_module': ['fusion', 'cross', 'mmtm', 'attention'],
            'classifier_head': ['classifier', 'head'],
        }
        active_keywords = []
        for mod in active_modules:
            active_keywords.extend(keyword_map.get(mod, [mod]))

        for name, param in self.model.named_parameters():
            should_train = any(kw in name for kw in active_keywords)
            param.requires_grad = should_train

        if epoch == phase1_epochs:
            print(f"[PHASE] Phase 2 开始 (epoch {epoch+1})，解冻模块: {phase2_modules}")

    def _update_aux_weight(self, epoch: int):
        """辅助损失权重衰减"""
        if not self.advanced_enabled or self.aux_decay_schedule is None:
            return
        decay_every, decay_factor = self.aux_decay_schedule
        if decay_every > 0 and (epoch + 1) % decay_every == 0:
            self.current_aux_weight = max(0.01, self.current_aux_weight * decay_factor)
            print(f"[AUX] aux_weight 衰减至 {self.current_aux_weight:.4f} (epoch {epoch+1})")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch（支持SAM双步与梯度裁剪）"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        all_probs_full = []

        use_sam = self.advanced_enabled and self.advanced_cfg.get('optimizer', 'AdamW') == 'AdamW_SAM'

        pbar = tqdm(train_loader, desc=f"训练 {self.model_name}", file=sys.stdout)
        for step, batch in enumerate(pbar):
            # 数据准备
            spectra = batch["spectra"].to(self.device)
            mask = batch.get("mask", None)
            if not self.use_embedding_input and mask is not None:
                mask = mask.to(self.device)
            tabular = batch["tabular"].to(self.device)
            labels = batch["label"].to(self.device)
            has_spectra = batch.get("has_spectra", None)
            has_tabular = batch.get("has_tabular", None)
            if has_spectra is not None:
                has_spectra = has_spectra.to(self.device)
            if has_tabular is not None:
                has_tabular = has_tabular.to(self.device)

            def forward_step():
                if not self.use_embedding_input:
                    outputs = self.model(spectra, mask, tabular)
                else:
                    if step == 0:
                        print("[INFO] 当前使用 embedding 输入模式：model(spectra_dict, tabular_dict)")
                        if has_spectra is not None and has_tabular is not None:
                            ratio_spec = has_spectra.float().mean().item()
                            ratio_tab = has_tabular.float().mean().item()
                            print(f"[INFO] embedding 模式：本 batch 中有光谱的比例={ratio_spec:.2f}, 有临床的比例={ratio_tab:.2f}")
                    spectra_result = {
                        "embedding": spectra,
                        "mask": has_spectra,
                        "logits": None,
                    }
                    tabular_result = {
                        "embedding": tabular,
                        "mask": has_tabular,
                        "logits": None,
                    }
                    outputs = self.model(spectra_result, tabular_result)
                return outputs

            # 前向传播
            self.optimizer.zero_grad()
            outputs = forward_step()
            loss = self._compute_loss(outputs, labels)

            # SAM 双步训练
            if use_sam:
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                outputs_adv = forward_step()
                loss_adv = self._compute_loss(outputs_adv, labels)
                loss_adv.backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                loss.backward()
                # 梯度裁剪
                if self.advanced_enabled and self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
                elif not self.advanced_enabled:
                    # 默认行为：保持原有硬编码裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            logits = outputs["logits"]
            # 统计
            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            # 收集完整概率矩阵用于多类 Macro-AUC
            if (self.compute_macro_auc or self.compute_weighted_auc) and probs.shape[1] > 1:
                all_probs_full.append(probs.detach().cpu().numpy())

            # 更新进度条
            lr_display = self.base_optimizer.param_groups[0]["lr"] if hasattr(self, 'base_optimizer') else self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr_display:.2e}'
            })

        # 计算指标
        probs_matrix = np.vstack(all_probs_full) if all_probs_full else None
        metrics = self._calculate_metrics(all_labels, all_probs, all_preds, probs_matrix=probs_matrix)
        n_samples = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(all_labels)
        metrics['loss'] = total_loss / max(n_samples, 1)

        return metrics
    
    def eval_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        all_probs_full = []
        all_features = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"验证 {self.model_name}", file=sys.stdout)
            for batch in pbar:
                # 数据准备
                spectra = batch["spectra"].to(self.device)
                mask = batch.get("mask", None)
                if not self.use_embedding_input and mask is not None:
                    mask = mask.to(self.device)
                tabular = batch["tabular"].to(self.device)
                labels = batch["label"].to(self.device)
                has_spectra = batch.get("has_spectra", None)
                has_tabular = batch.get("has_tabular", None)
                if has_spectra is not None:
                    has_spectra = has_spectra.to(self.device)
                if has_tabular is not None:
                    has_tabular = has_tabular.to(self.device)
                
                # 前向传播
                if not self.use_embedding_input:
                    # raw 模式：保持原有调用方式
                    outputs = self.model(spectra, mask, tabular)
                else:
                    # embedding 模式：包装为外部模型输出字典，并携带缺模态 mask
                    spectra_result = {
                        "embedding": spectra,
                        "mask": has_spectra,
                        "logits": None,
                    }
                    tabular_result = {
                        "embedding": tabular,
                        "mask": has_tabular,
                        "logits": None,
                    }
                    outputs = self.model(spectra_result, tabular_result)
                logits = outputs["logits"]

                # 计算损失
                loss = self._compute_loss(outputs, labels)

                # 统计
                total_loss += loss.item() * labels.size(0)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                all_probs.extend(probs[:, 1].detach().cpu().numpy())
                # 收集完整概率矩阵用于多类 Macro-AUC
                if (self.compute_macro_auc or self.compute_weighted_auc) and probs.shape[1] > 1:
                    all_probs_full.append(probs.detach().cpu().numpy())

                # 收集特征用于可视化
                if 'embedding' in outputs:
                    all_features.append(outputs['embedding'].cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 计算指标
        probs_matrix = np.vstack(all_probs_full) if all_probs_full else None
        metrics = self._calculate_metrics(all_labels, all_probs, all_preds, probs_matrix=probs_matrix)
        n_samples = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else len(all_labels)
        metrics['loss'] = total_loss / max(n_samples, 1)

        # 保存特征用于后续分析
        if all_features:
            metrics['features'] = np.vstack(all_features)

        return metrics
    
    def _log_modality_gates(self, epoch: int):
        """
        尝试从模型中提取 fusion_gate（或子模块中的 fusion_gate），
        记录 softmax 后的两模态权重。
        """
        gate_tensor = None
        
        # 常见情况 1：模型本身有 fusion_gate（AttentionMultimodal, EnhancedMMTMFusion）
        if hasattr(self.model, "fusion_gate"):
            gate_tensor = self.model.fusion_gate
        
        # 常见情况 2：BaselineMultimodal 内部的 fusion 模块上有 fusion_gate
        elif hasattr(self.model, "fusion") and hasattr(self.model.fusion, "fusion_gate"):
            gate_tensor = self.model.fusion.fusion_gate
        
        if gate_tensor is None:
            return  # 当前模型没有 fusion_gate，直接跳过
        
        with torch.no_grad():
            gate = torch.nn.functional.softmax(gate_tensor.detach().cpu().float(), dim=0)  # [2]
        
        record = {
            "epoch": int(epoch),
            "gate_raw": gate_tensor.detach().cpu().tolist(),
            "gate_softmax": gate.tolist(),  # [w_spectra, w_clinical]
        }
        self.modality_gate_history.append(record)
    
    def _calculate_metrics(self, y_true: List, y_prob: List, y_pred: List, probs_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """计算评估指标（含扩展指标）

        Args:
            y_true: 真实标签
            y_prob: 第1类的正类概率（向后兼容）
            y_pred: 预测标签
            probs_matrix: 完整的 softmax 概率矩阵 [N, C]，用于多类 Macro-AUC / Weighted-AUC
        """
        from sklearn.metrics import cohen_kappa_score, roc_auc_score
        from sklearn.preprocessing import label_binarize
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = np.array(y_pred)

        # 空输入保护
        if len(y_true) == 0:
            return {
                'acc': np.nan, 'auc': np.nan, 'f1': np.nan,
                'macro_f1': np.nan, 'sensitivity@90%spec': np.nan
            }

        # 统一二分类 / 多分类的评估逻辑
        classes = np.unique(y_true)
        num_classes = len(classes)

        # 准确率 & F1
        acc = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        # OvR 正类（用于 AUC 计算）
        if num_classes <= 2:
            y_binary = y_true
        else:
            pos_label = 1 if 1 in classes else classes.max()
            y_binary = (y_true == pos_label).astype(int)

        try:
            auc = roc_auc_score(y_binary, y_prob)
        except ValueError:
            auc = np.nan

        # 扩展 AUC 指标（Macro / Weighted）
        macro_auc = np.nan
        weighted_auc = np.nan
        if probs_matrix is not None and probs_matrix.ndim == 2 and probs_matrix.shape[1] > 1:
            n_classes_prob = probs_matrix.shape[1]
            if self.compute_macro_auc or self.compute_weighted_auc:
                try:
                    if self.compute_macro_auc:
                        macro_auc = roc_auc_score(y_true, probs_matrix, multi_class='ovr', average='macro')
                    if self.compute_weighted_auc:
                        weighted_auc = roc_auc_score(y_true, probs_matrix, multi_class='ovr', average='weighted')
                except ValueError:
                    # 某折缺少某类 → 退化为逐类手动计算
                    try:
                        all_classes = np.arange(n_classes_prob)
                        y_true_bin = label_binarize(y_true, classes=all_classes)
                        if y_true_bin.shape[1] < n_classes_prob:
                            n_missing = n_classes_prob - y_true_bin.shape[1]
                            y_true_bin = np.pad(y_true_bin, ((0, 0), (0, n_missing)), mode='constant')
                        per_class_aucs = []
                        weighted_per_class = []
                        for i in range(n_classes_prob):
                            col = y_true_bin[:, i]
                            if np.sum(col) > 0 and np.sum(col) < len(col):
                                auc_i = roc_auc_score(col, probs_matrix[:, i])
                                per_class_aucs.append(auc_i)
                                weighted_per_class.append(auc_i * np.sum(col))
                        if per_class_aucs:
                            if self.compute_macro_auc:
                                macro_auc = float(np.mean(per_class_aucs))
                            if self.compute_weighted_auc:
                                total_support = np.sum([np.sum(y_true_bin[:, i]) for i in range(n_classes_prob) if np.sum(y_true_bin[:, i]) > 0])
                                if total_support > 0:
                                    weighted_auc = float(np.sum(weighted_per_class) / total_support)
                    except Exception:
                        pass

        try:
            fpr, tpr, _ = roc_curve(y_binary, y_prob)
            specificity = 1 - fpr
            mask = specificity >= 0.9
            sens_at_90 = tpr[mask].max() if np.any(mask) else np.nan
        except ValueError:
            sens_at_90 = np.nan

        result = {
            'acc': acc,
            'auc': auc,
            'f1': f1_weighted,
            'macro_f1': f1_macro,
            'sensitivity@90%spec': sens_at_90
        }

        if self.compute_macro_auc and not np.isnan(macro_auc):
            result['macro_auc'] = macro_auc
        if self.compute_weighted_auc and not np.isnan(weighted_auc):
            result['weighted_auc'] = weighted_auc

        # Cohen's Kappa / QWK（独立于 advanced.enabled）
        if self.compute_cohens_kappa:
            try:
                result['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
            except Exception:
                result['cohens_kappa'] = np.nan
        if self.compute_qwk:
            try:
                result['qwk'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
            except Exception:
                result['qwk'] = np.nan

        return result
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ) -> Dict[str, Any]:
        """
        完整训练流程（支持 Phase Training / Warmup / Cosine / 多指标早停）
        """
        # 使用 advanced 中的 patience 覆盖外部传入值
        patience = self.early_stop_patience if self.early_stop_patience is not None else early_stopping_patience
        metric_key = self.early_stop_metric  # 'auc' | 'weighted_f1' | 'macro_f1' | 'macro_auc' | 'qwk'
        # 指标键映射：metrics 字典中的键名
        metric_map = {
            'auc': 'auc',
            'weighted_f1': 'f1',
            'macro_f1': 'macro_f1',
            'macro_auc': 'macro_auc',
            'qwk': 'qwk',
        }
        monitor_key = metric_map.get(metric_key, 'auc')

        print(f"\n[TRAIN] 开始训练 {self.model_name}")
        print(f"[DATA] 训练样本: {len(train_loader.dataset)}")
        print(f"[DATA] 验证样本: {len(val_loader.dataset)}")
        print(f"[EPOCH] 训练轮数: {epochs}")
        print(f"[EARLY_STOP] 监控指标: {metric_key} (patience={patience})")
        if self.phase_training_enabled:
            print(f"[PHASE] 渐进式训练: Phase1={self.phase_training_cfg.get('phase1_epochs', 20)} epochs")
        print("=" * 60)

        start_time = time.time()
        start_epoch = 0
        patience_counter = 0
        best_epoch = 0

        if self.train_history.get('loss'):
            start_epoch = len(self.train_history['loss'])
            print(f"[RESUME] 检测到训练历史，从 Epoch {start_epoch+1} 继续训练")
            if self.val_history.get(monitor_key):
                metric_history = self.val_history[monitor_key]
                best_idx = int(np.argmax(metric_history))
                self.best_val_metric = metric_history[best_idx]
                patience_counter = len(metric_history) - 1 - best_idx
                print(f"[RESUME] 重建 Patience: 当前 {patience_counter} / {patience} (最佳 {metric_key} 在 Epoch {best_idx+1})")

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            # 渐进式训练：冻结/解冻参数
            if self.phase_training_enabled:
                self._apply_phase_training(epoch)

            # Warmup（仅在 cosine 模式下，plateau 模式不启用 warmup）
            if self.advanced_enabled and self.warmup_epochs > 0 and self.scheduler_type == 'cosine' and epoch < self.warmup_epochs:
                for param_group in self.base_optimizer.param_groups:
                    param_group['lr'] = self.base_lr * (epoch + 1) / self.warmup_epochs

            # 训练和验证
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)

            # 学习率调度
            if self.scheduler_type == 'plateau':
                self.scheduler.step(val_metrics['loss'])
            elif self.scheduler_type == 'cosine':
                # 跳过 warmup 阶段的 cosine step
                if epoch >= self.warmup_epochs:
                    self.scheduler.step()

            # 辅助损失权重衰减
            self._update_aux_weight(epoch)

            # 记录历史
            for key in self.train_history:
                if key in train_metrics:
                    self.train_history[key].append(train_metrics[key])
            for key in self.val_history:
                if key in val_metrics:
                    self.val_history[key].append(val_metrics[key])

            # 记录模态权重
            self._log_modality_gates(epoch)

            # 保存最佳模型（按配置指标）
            current_metric = val_metrics.get(monitor_key, -float('inf'))
            # 若目标指标未计算（如某折缺少类别导致 macro_auc 为 nan），回退到 weighted_f1
            if current_metric == -float('inf') or np.isnan(current_metric):
                fallback_key = 'f1'
                fallback_val = val_metrics.get(fallback_key, -float('inf'))
                if not np.isnan(fallback_val):
                    print(f"[WARN] {monitor_key} 不可用，回退到 {fallback_key} 作为早停指标")
                    current_metric = fallback_val
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                if save_best:
                    self.best_model_state = self.model.state_dict().copy()
                    self.save_model("best_model.pt")
            else:
                patience_counter += 1

            # 打印进度
            epoch_time = time.time() - epoch_start
            lr_display = self.base_optimizer.param_groups[0]["lr"] if hasattr(self, 'base_optimizer') else self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: Loss={train_metrics['loss']:.4f}, "
                  f"AUC={train_metrics['auc']:.4f}, "
                  f"Acc={train_metrics['acc']:.4f} | "
                  f"Val: Loss={val_metrics['loss']:.4f}, "
                  f"AUC={val_metrics['auc']:.4f}, "
                  f"Acc={val_metrics['acc']:.4f} | "
                  f"LR={lr_display:.2e} | Time={epoch_time:.1f}s")
            sys.stdout.flush()

            # 早停检查
            if patience_counter >= patience:
                print(f"[STOP] 早停触发 (patience={patience}, 监控指标={metric_key})")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n[OK] 训练完成!")
        print(f"[TIME] 总时间: {total_time:.1f}s")
        print(f"[BEST] 最佳验证{self.early_stop_metric.upper()}: {self.best_val_metric:.4f} (Epoch {best_epoch+1})")

        # 保存最佳模型
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.save_model()

        # 生成可视化
        if self.enable_visualization:
            self._generate_training_visualizations()

        # 保存模态权重历史到 JSON
        if self.modality_gate_history:
            gate_path = self.save_dir / "modality_gate_history.json"
            with gate_path.open("w", encoding="utf-8") as f:
                json.dump(self.modality_gate_history, f, ensure_ascii=False, indent=2)
            print(f"[SAVE] 模态权重轨迹已保存到: {gate_path}")

        return {
            'best_val_auc': self.best_val_metric if self.early_stop_metric == 'auc' else (self.val_history.get('auc', [0])[-1] if self.val_history.get('auc') else 0.0),
            'best_val_metric': self.best_val_metric,
            'best_epoch': best_epoch,
            'total_time': total_time,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def evaluate(
        self,
        test_loader: DataLoader,
        generate_plots: bool = True
    ) -> Dict[str, Any]:
        """
        模型评估
        
        Args:
            test_loader: 测试数据加载器
            generate_plots: 是否生成评估图表
        
        Returns:
            评估结果字典
        """
        print(f"\n[EVAL] 评估 {self.model_name}")
        print("=" * 40)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_probs_full = []
        all_features = []
        all_attention_weights = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估", file=sys.stdout):
                # 数据准备
                spectra = batch["spectra"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                labels = batch["label"].to(self.device)

                has_spectra = batch.get("has_spectra", None)
                has_tabular = batch.get("has_tabular", None)
                if has_spectra is not None:
                    has_spectra = has_spectra.to(self.device)
                if has_tabular is not None:
                    has_tabular = has_tabular.to(self.device)

                mask = batch.get("mask", None)
                if not self.use_embedding_input and mask is not None:
                    mask = mask.to(self.device)

                # 前向传播
                if not self.use_embedding_input:
                    # raw 模式：保持原有调用方式
                    outputs = self.model(spectra, mask, tabular)
                else:
                    # embedding 模式：包装为外部模型输出字典，并携带缺模态 mask
                    spectra_result = {
                        "embedding": spectra,
                        "mask": has_spectra,
                        "logits": None,
                    }
                    tabular_result = {
                        "embedding": tabular,
                        "mask": has_tabular,
                        "logits": None,
                    }
                    outputs = self.model(spectra_result, tabular_result)

                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                # 收集结果
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                # 使用第 1 类的概率作为"正类"概率（多分类情况下在 _calculate_metrics 里会做 one-vs-rest 处理）
                pos_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                all_probs.extend(pos_prob.detach().cpu().numpy())
                # 收集完整概率矩阵用于多类 Macro-AUC
                if (self.compute_macro_auc or self.compute_weighted_auc) and probs.shape[1] > 1:
                    all_probs_full.append(probs.detach().cpu().numpy())

                # 收集特征和注意力权重
                if 'embedding' in outputs:
                    all_features.append(outputs['embedding'].cpu().numpy())
                if 'attention_weights' in outputs:
                    all_attention_weights.append(outputs['attention_weights'].cpu().numpy())

        # 计算指标
        probs_matrix = np.vstack(all_probs_full) if all_probs_full else None
        metrics = self._calculate_metrics(all_labels, all_probs, all_preds, probs_matrix=probs_matrix)
        
        # 生成详细报告（兼容多分类情况）
        unique_labels = sorted(np.unique(all_labels))
        # 如果是二分类并且标签是 {0,1}，使用更友好的名称；否则使用字符串化的标签
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            target_names = ['Control', 'DM']
        else:
            target_names = [str(l) for l in unique_labels]

        report = classification_report(
            all_labels,
            all_preds,
            labels=unique_labels,
            target_names=target_names,
            output_dict=True
        )
        
        print(f"[RESULT] 测试结果:")
        print(f"   - 准确率: {metrics['acc']:.4f}")
        print(f"   - AUC: {metrics['auc']:.4f}")
        print(f"   - F1分数: {metrics['f1']:.4f}")
        print(f"   - 敏感性@90%特异性: {metrics['sensitivity@90%spec']:.4f}")
        
        # 生成可视化
        if generate_plots and self.enable_visualization:
            # 确保特征和注意力权重是numpy数组
            features_array = np.vstack(all_features) if all_features else None
            attention_array = np.vstack(all_attention_weights) if all_attention_weights else None
            
            self._generate_evaluation_plots(
                all_labels, all_probs, all_preds,
                features_array, attention_array
            )
        
        # 可解释性分析
        if self.enable_interpretability:
            # 确保特征和注意力权重是numpy数组
            features_array = np.vstack(all_features) if all_features else None
            attention_array = np.vstack(all_attention_weights) if all_attention_weights else None
            
            self._generate_interpretability_analysis(
                test_loader, features_array, attention_array
            )
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'predictions': {
                'labels': all_labels,
                'probabilities': all_probs,
                'predictions': all_preds
            },
            'features': np.vstack(all_features) if all_features else None,
            'attention_weights': np.vstack(all_attention_weights) if all_attention_weights else None
        }
    
    def _generate_training_visualizations(self):
        """生成训练过程可视化"""
        print("[VIS] 生成训练可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} - 训练过程', fontsize=16, fontweight='bold')
        
        # 损失曲线
        axes[0, 0].plot(self.train_history['loss'], label='训练损失', color='blue', alpha=0.7)
        axes[0, 0].plot(self.val_history['loss'], label='验证损失', color='red', alpha=0.7)
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(self.train_history['acc'], label='训练准确率', color='blue', alpha=0.7)
        axes[0, 1].plot(self.val_history['acc'], label='验证准确率', color='red', alpha=0.7)
        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC曲线
        axes[1, 0].plot(self.train_history['auc'], label='训练AUC', color='blue', alpha=0.7)
        axes[1, 0].plot(self.val_history['auc'], label='验证AUC', color='red', alpha=0.7)
        axes[1, 0].set_title('AUC曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1分数曲线
        axes[1, 1].plot(self.train_history['f1'], label='训练F1', color='blue', alpha=0.7)
        axes[1, 1].plot(self.val_history['f1'], label='验证F1', color='red', alpha=0.7)
        axes[1, 1].set_title('F1分数曲线')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] 训练可视化已保存: {self.save_dir / 'training_curves.png'}")
    
    def _generate_evaluation_plots(
        self,
        y_true: List,
        y_prob: List,
        y_pred: List,
        features: Optional[np.ndarray] = None,
        attention_weights: Optional[np.ndarray] = None
    ):
        """生成评估图表"""
        print("[VIS] 生成评估可视化...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.model_name} - 模型评估', fontsize=16, fontweight='bold')
        
        # ROC曲线 & 精确率-召回率曲线（兼容多分类：使用 one-vs-rest 视角）
        y_true_arr = np.array(y_true)
        classes = np.unique(y_true_arr)
        num_classes = len(classes)

        if num_classes <= 2:
            y_binary = y_true_arr
        else:
            pos_label = 1 if 1 in classes else classes.max()
            y_binary = (y_true_arr == pos_label).astype(int)

        try:
            fpr, tpr, _ = roc_curve(y_binary, y_prob)
            auc = roc_auc_score(y_binary, y_prob)
            axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc:.3f})')
            axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
            axes[0, 0].set_xlim([0.0, 1.0])
            axes[0, 0].set_ylim([0.0, 1.05])
            axes[0, 0].set_xlabel('假正率 (FPR)')
            axes[0, 0].set_ylabel('真正率 (TPR)')
            axes[0, 0].set_title('ROC曲线')
            axes[0, 0].legend(loc="lower right")
            axes[0, 0].grid(True, alpha=0.3)
        except ValueError as e:
            axes[0, 0].text(0.5, 0.5, f'ROC 计算失败:\n{e}', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('ROC曲线 (失败)')

        try:
            precision, recall, _ = precision_recall_curve(y_binary, y_prob)
            axes[0, 1].plot(recall, precision, color='blue', lw=2)
            axes[0, 1].set_xlabel('召回率')
            axes[0, 1].set_ylabel('精确率')
            axes[0, 1].set_title('精确率-召回率曲线')
            axes[0, 1].grid(True, alpha=0.3)
        except ValueError as e:
            axes[0, 1].text(0.5, 0.5, f'PR 计算失败:\n{e}', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('精确率-召回率曲线 (失败)')
        
        # 混淆矩阵
        cm = confusion_matrix(y_true_arr, y_pred)
        # 动态标签名称
        unique_labels = sorted(np.unique(y_true_arr))
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            xticklabels = yticklabels = ['Control', 'DM']
        else:
            xticklabels = yticklabels = [str(l) for l in unique_labels]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
                    xticklabels=xticklabels, yticklabels=yticklabels)
        axes[0, 2].set_title('混淆矩阵')
        axes[0, 2].set_xlabel('预测标签')
        axes[0, 2].set_ylabel('真实标签')
        
        # 预测概率分布
        axes[1, 0].hist([y_prob[i] for i in range(len(y_prob)) if y_binary[i] == 0],
                       bins=20, alpha=0.7, label='Control', color='blue')
        axes[1, 0].hist([y_prob[i] for i in range(len(y_prob)) if y_binary[i] == 1],
                       bins=20, alpha=0.7, label='DM', color='red')
        axes[1, 0].set_xlabel('预测概率')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('预测概率分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 特征可视化 (t-SNE)
        if features is not None and len(features) > 10:
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
                features_2d = tsne.fit_transform(features)
                
                scatter = axes[1, 1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                           c=y_true, cmap='viridis', alpha=0.6)
                axes[1, 1].set_title('特征空间可视化 (t-SNE)')
                axes[1, 1].set_xlabel('t-SNE 1')
                axes[1, 1].set_ylabel('t-SNE 2')
                plt.colorbar(scatter, ax=axes[1, 1])
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f't-SNE失败:\n{str(e)}', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('特征空间可视化 (失败)')
        else:
            axes[1, 1].text(0.5, 0.5, '特征数据不足', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('特征空间可视化')
        
        # 注意力权重可视化
        if attention_weights is not None and len(attention_weights) > 0:
            # 显示平均注意力权重
            avg_attention = np.mean(attention_weights, axis=0)
            if len(avg_attention.shape) == 1:
                axes[1, 2].bar(range(len(avg_attention)), avg_attention)
                axes[1, 2].set_title('平均注意力权重')
                axes[1, 2].set_xlabel('特征维度')
                axes[1, 2].set_ylabel('注意力权重')
            else:
                im = axes[1, 2].imshow(avg_attention, cmap='viridis', aspect='auto')
                axes[1, 2].set_title('注意力权重热图')
                plt.colorbar(im, ax=axes[1, 2])
        else:
            axes[1, 2].text(0.5, 0.5, '无注意力权重数据', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('注意力权重')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] 评估可视化已保存: {self.save_dir / 'evaluation_plots.png'}")
    
    def _generate_interpretability_analysis(
        self,
        test_loader: DataLoader,
        features: Optional[np.ndarray] = None,
        attention_weights: Optional[np.ndarray] = None
    ):
        """生成可解释性分析"""
        print("[INTERP] 生成可解释性分析...")
        
        # 获取一些样本进行SHAP分析
        sample_batch = next(iter(test_loader))
        sample_spectra = sample_batch["spectra"][:5].to(self.device)
        sample_mask = sample_batch.get("mask", None)
        if sample_mask is not None:
            sample_mask = sample_mask[:5].to(self.device)
        sample_tabular = sample_batch["tabular"][:5].to(self.device)
        
        # 创建简化的特征重要性分析（替代SHAP）
        try:
            # 使用梯度分析替代SHAP，更简单可靠
            single_spectra = sample_spectra[:1]  # 只分析第一个样本
            single_mask = sample_mask[:1] if sample_mask is not None else None
            single_tabular = sample_tabular[:1]
            
            # 计算梯度重要性
            single_spectra.requires_grad_(True)
            outputs = self.model(single_spectra, single_mask, single_tabular)
            loss = outputs["logits"].sum()
            loss.backward()
            
            # 获取梯度作为特征重要性
            feature_importance = torch.abs(single_spectra.grad).cpu().numpy().flatten()
            
            # 可视化特征重要性
            plt.figure(figsize=(12, 8))
            plt.plot(feature_importance)
            plt.title(f'{self.model_name} - 特征重要性 (基于梯度)')
            plt.xlabel('波长索引')
            plt.ylabel('梯度重要性')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.save_dir / 'shap_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] 特征重要性分析已保存: {self.save_dir / 'shap_analysis.png'}")
            
        except Exception as e:
            print(f"[WARN] SHAP分析失败: {e}")
            # 如果SHAP失败，至少生成一个简单的特征重要性图
            try:
                plt.figure(figsize=(12, 8))
                # 使用简单的特征重要性可视化
                feature_importance = np.abs(sample_spectra.cpu().numpy().mean(axis=0))
                plt.plot(feature_importance.mean(axis=0))
                plt.title(f'{self.model_name} - 特征重要性 (替代SHAP)')
                plt.xlabel('波长索引')
                plt.ylabel('平均强度')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.save_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"[OK] 特征重要性图已保存: {self.save_dir / 'feature_importance.png'}")
            except Exception as e2:
                print(f"[WARN] 特征重要性图也失败: {e2}")
        
        # 特征重要性分析
        if features is not None:
            self._analyze_feature_importance(features)
        
        # 注意力模式分析
        if attention_weights is not None:
            self._analyze_attention_patterns(attention_weights)
    
    def _analyze_feature_importance(self, features: np.ndarray):
        """分析特征重要性"""
        # 使用PCA分析特征重要性
        if features.shape[0] > 1 and features.shape[1] > 1:
            # 确保n_components不超过样本数和特征数
            max_components = min(10, features.shape[0] - 1, features.shape[1])
            if max_components > 0:
                pca = PCA(n_components=max_components)
                pca.fit(features)
                
                # 可视化主成分贡献
                plt.figure(figsize=(10, 6))
                plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                       pca.explained_variance_ratio_)
                plt.xlabel('主成分')
                plt.ylabel('解释方差比')
                plt.title(f'{self.model_name} - 主成分分析')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.save_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"[OK] PCA分析已保存: {self.save_dir / 'pca_analysis.png'}")
            else:
                print("[WARN] 样本数量不足，跳过PCA分析")
        else:
            print("[WARN] 特征数据不足，跳过PCA分析")
    
    def _analyze_attention_patterns(self, attention_weights: np.ndarray):
        """分析注意力模式"""
        if len(attention_weights.shape) >= 2:
            # 计算注意力权重的统计信息
            mean_attention = np.mean(attention_weights, axis=0)
            std_attention = np.std(attention_weights, axis=0)
            
            # 可视化注意力模式
            plt.figure(figsize=(12, 6))
            
            if len(mean_attention.shape) == 1:
                # 一维注意力权重
                plt.subplot(1, 2, 1)
                plt.bar(range(len(mean_attention)), mean_attention)
                plt.title('平均注意力权重')
                plt.xlabel('特征维度')
                plt.ylabel('注意力权重')
                
                plt.subplot(1, 2, 2)
                plt.bar(range(len(std_attention)), std_attention)
                plt.title('注意力权重标准差')
                plt.xlabel('特征维度')
                plt.ylabel('标准差')
            else:
                # 二维注意力权重
                plt.subplot(1, 2, 1)
                plt.imshow(mean_attention, cmap='viridis', aspect='auto')
                plt.title('平均注意力权重')
                plt.colorbar()
                
                plt.subplot(1, 2, 2)
                plt.imshow(std_attention, cmap='viridis', aspect='auto')
                plt.title('注意力权重标准差')
                plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'attention_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] 注意力分析已保存: {self.save_dir / 'attention_analysis.png'}")
    
    def save_model(self, filename: str = "best_model.pt"):
        """保存模型"""
        save_path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'best_val_auc': getattr(self, 'best_val_metric', 0.0),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, save_path)
        print(f"[SAVE] 模型已保存: {save_path}")
    
    def load_model(self, filename: str = "best_model.pt"):
        """加载模型"""
        load_path = self.save_dir / filename
        if load_path.exists():
            checkpoint = torch.load(load_path, map_location=self.device)
            missing, unexpected = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing:
                print(f"[WARN] 恢复模型时缺失键: {missing}")
            if unexpected:
                print(f"[WARN] 恢复模型时出现意外键 (已忽略): {unexpected}")

            loaded_best = checkpoint.get('best_val_auc', 0.0)
            self.best_val_metric = loaded_best
            self.train_history = checkpoint.get('train_history', {'loss': [], 'acc': [], 'auc': [], 'f1': []})
            self.val_history = checkpoint.get('val_history', {'loss': [], 'acc': [], 'auc': [], 'f1': []})

            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            print(f"[LOAD] 模型已加载: {load_path}")
        else:
            print(f"[ERROR] 模型文件不存在: {load_path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'best_val_auc': getattr(self, 'best_val_metric', 0.0),
            'device': str(self.device)
        }


def compare_models(
    trainers: List[EnhancedTrainer],
    test_loader: DataLoader,
    save_dir: str = "results/comparison"
) -> Dict[str, Any]:
    """
    比较多个模型的性能
    
    Args:
        trainers: 训练器列表
        test_loader: 测试数据加载器
        save_dir: 保存目录
    
    Returns:
        比较结果字典
    """
    print(f"\n[COMPARE] 开始模型比较...")
    print("=" * 50)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    all_metrics = []
    
    # 评估每个模型
    for trainer in trainers:
        print(f"\n[EVAL] 评估 {trainer.model_name}...")
        result = trainer.evaluate(test_loader, generate_plots=False)
        results[trainer.model_name] = result
        all_metrics.append({
            'model': trainer.model_name,
            **result['metrics']
        })
    
    # 创建比较表格
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.set_index('model')
    
    print(f"\n[COMPARE] 模型性能比较:")
    print(metrics_df.round(4))
    
    # 保存比较结果
    metrics_df.to_csv(save_dir / 'model_comparison.csv')
    
    # 生成比较可视化
    _generate_comparison_plots(trainers, results, save_dir)
    
    return {
        'metrics': metrics_df,
        'detailed_results': results,
        'best_model': metrics_df['auc'].idxmax(),
        'best_auc': metrics_df['auc'].max()
    }


def _generate_comparison_plots(
    trainers: List[EnhancedTrainer],
    results: Dict[str, Any],
    save_dir: Path
):
    """生成模型比较图表"""
    print("[VIS] 生成模型比较可视化...")
    
    # 提取指标
    model_names = list(results.keys())
    metrics = ['acc', 'auc', 'f1', 'sensitivity@90%spec']
    
    # 创建比较图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('模型性能比较', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        values = [results[name]['metrics'][metric] for name in model_names]
        
        bars = axes[row, col].bar(model_names, values, alpha=0.7)
        axes[row, col].set_title(f'{metric.upper()} 比较')
        axes[row, col].set_ylabel(metric.upper())
        axes[row, col].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC曲线比较
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        y_true = np.asarray(result['predictions']['labels'])
        y_prob = np.asarray(result['predictions']['probabilities'])
        auc = result['metrics']['auc']

        try:
            # 二分类：直接使用 roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        except ValueError:
            # 多分类：计算 OvR 宏平均 ROC
            try:
                from sklearn.preprocessing import label_binarize
                classes = np.unique(y_true)
                y_true_bin = label_binarize(y_true, classes=classes)
                if y_prob.ndim == 1:
                    y_prob = np.column_stack([1 - y_prob, y_prob])
                # 确保概率列数与类别数匹配
                if y_prob.shape[1] != len(classes):
                    print(f"[WARN] {name} 概率维度 {y_prob.shape[1]} 与类别数 {len(classes)} 不匹配，跳过 ROC 曲线")
                    continue
                all_fpr = np.linspace(0, 1, 100)
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(len(classes)):
                    fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
                mean_tpr /= len(classes)
                plt.plot(all_fpr, mean_tpr, label=f'{name} (macro AUC = {auc:.3f})', linewidth=2)
            except Exception as e2:
                print(f"[WARN] {name} ROC 曲线绘制失败: {e2}")
                continue

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC曲线比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] 比较可视化已保存: {save_dir}")


if __name__ == "__main__":
    print("[INFO] 增强版训练器模块")
    print("请通过主脚本使用此训练器")

