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

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')


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
        enable_interpretability: bool = True
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
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.save_dir = Path(save_dir) / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练设置
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 功能开关
        self.enable_visualization = enable_visualization
        self.enable_interpretability = enable_interpretability
        
        # 训练历史
        self.train_history = {
            'loss': [], 'acc': [], 'auc': [], 'f1': []
        }
        self.val_history = {
            'loss': [], 'acc': [], 'auc': [], 'f1': []
        }
        
        # 最佳模型
        self.best_val_auc = 0.0
        self.best_model_state = None
        
        print(f"🚀 增强版训练器初始化完成")
        print(f"📊 模型: {model_name}")
        print(f"💻 设备: {device}")
        print(f"💾 保存目录: {self.save_dir}")
        print(f"📈 可视化: {'启用' if enable_visualization else '禁用'}")
        print(f"🔍 可解释性: {'启用' if enable_interpretability else '禁用'}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(train_loader, desc=f"训练 {self.model_name}")
        for batch in pbar:
            # 数据准备
            spectra = batch["spectra"].to(self.device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device)
            tabular = batch["tabular"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(spectra, mask, tabular)
            logits = outputs["logits"]
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # 计算指标
        metrics = self._calculate_metrics(all_labels, all_probs, all_preds)
        metrics['loss'] = total_loss / len(train_loader.dataset)
        
        return metrics
    
    def eval_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"验证 {self.model_name}")
            for batch in pbar:
                # 数据准备
                spectra = batch["spectra"].to(self.device)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(self.device)
                tabular = batch["tabular"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # 前向传播
                outputs = self.model(spectra, mask, tabular)
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)
                
                # 统计
                total_loss += loss.item() * labels.size(0)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                all_probs.extend(probs[:, 1].detach().cpu().numpy())
                
                # 收集特征用于可视化
                if 'embedding' in outputs:
                    all_features.append(outputs['embedding'].cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算指标
        metrics = self._calculate_metrics(all_labels, all_probs, all_preds)
        metrics['loss'] = total_loss / len(val_loader.dataset)
        
        # 保存特征用于后续分析
        if all_features:
            metrics['features'] = np.vstack(all_features)
        
        return metrics
    
    def _calculate_metrics(self, y_true: List, y_prob: List, y_pred: List) -> Dict[str, float]:
        """计算评估指标"""
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = np.array(y_pred)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # 计算敏感性@90%特异性
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        specificity = 1 - fpr
        mask = specificity >= 0.9
        sens_at_90 = tpr[mask].max() if np.any(mask) else np.nan
        
        return {
            'acc': acc,
            'auc': auc,
            'f1': f1,
            'sensitivity@90%spec': sens_at_90
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ) -> Dict[str, Any]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_best: 是否保存最佳模型
        
        Returns:
            训练结果字典
        """
        print(f"\n🚀 开始训练 {self.model_name}")
        print(f"📊 训练样本: {len(train_loader.dataset)}")
        print(f"📊 验证样本: {len(val_loader.dataset)}")
        print(f"🔄 训练轮数: {epochs}")
        print("=" * 60)
        
        start_time = time.time()
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练和验证
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_metrics['loss'])
            
            # 记录历史
            for key in self.train_history:
                if key in train_metrics:
                    self.train_history[key].append(train_metrics[key])
            for key in self.val_history:
                if key in val_metrics:
                    self.val_history[key].append(val_metrics[key])
            
            # 保存最佳模型
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                best_epoch = epoch
                patience_counter = 0
                if save_best:
                    self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 打印进度
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: Loss={train_metrics['loss']:.4f}, "
                  f"AUC={train_metrics['auc']:.4f}, "
                  f"Acc={train_metrics['acc']:.4f} | "
                  f"Val: Loss={val_metrics['loss']:.4f}, "
                  f"AUC={val_metrics['auc']:.4f}, "
                  f"Acc={val_metrics['acc']:.4f} | "
                  f"Time={epoch_time:.1f}s")
            
            # 早停检查
            if patience_counter >= early_stopping_patience:
                print(f"🛑 早停触发 (patience={early_stopping_patience})")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n✅ 训练完成!")
        print(f"⏱️  总时间: {total_time:.1f}s")
        print(f"🏆 最佳验证AUC: {self.best_val_auc:.4f} (Epoch {best_epoch+1})")
        
        # 保存最佳模型
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.save_model()
        
        # 生成可视化
        if self.enable_visualization:
            self._generate_training_visualizations()
        
        return {
            'best_val_auc': self.best_val_auc,
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
        print(f"\n🔍 评估 {self.model_name}")
        print("=" * 40)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []
        all_attention_weights = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估"):
                # 数据准备
                spectra = batch["spectra"].to(self.device)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(self.device)
                tabular = batch["tabular"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # 前向传播
                outputs = self.model(spectra, mask, tabular)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # 收集结果
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                all_probs.extend(probs[:, 1].detach().cpu().numpy())
                
                # 收集特征和注意力权重
                if 'embedding' in outputs:
                    all_features.append(outputs['embedding'].cpu().numpy())
                if 'attention_weights' in outputs:
                    all_attention_weights.append(outputs['attention_weights'].cpu().numpy())
        
        # 计算指标
        metrics = self._calculate_metrics(all_labels, all_probs, all_preds)
        
        # 生成详细报告
        report = classification_report(
            all_labels, all_preds,
            target_names=['Control', 'DM'],
            output_dict=True
        )
        
        print(f"📊 测试结果:")
        print(f"   • 准确率: {metrics['acc']:.4f}")
        print(f"   • AUC: {metrics['auc']:.4f}")
        print(f"   • F1分数: {metrics['f1']:.4f}")
        print(f"   • 敏感性@90%特异性: {metrics['sensitivity@90%spec']:.4f}")
        
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
        print("📊 生成训练可视化...")
        
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
        
        print(f"✅ 训练可视化已保存: {self.save_dir / 'training_curves.png'}")
    
    def _generate_evaluation_plots(
        self,
        y_true: List,
        y_prob: List,
        y_pred: List,
        features: Optional[np.ndarray] = None,
        attention_weights: Optional[np.ndarray] = None
    ):
        """生成评估图表"""
        print("📊 生成评估可视化...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.model_name} - 模型评估', fontsize=16, fontweight='bold')
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('假正率 (FPR)')
        axes[0, 0].set_ylabel('真正率 (TPR)')
        axes[0, 0].set_title('ROC曲线')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 精确率-召回率曲线
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0, 1].plot(recall, precision, color='blue', lw=2)
        axes[0, 1].set_xlabel('召回率')
        axes[0, 1].set_ylabel('精确率')
        axes[0, 1].set_title('精确率-召回率曲线')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
                   xticklabels=['Control', 'DM'], yticklabels=['Control', 'DM'])
        axes[0, 2].set_title('混淆矩阵')
        axes[0, 2].set_xlabel('预测标签')
        axes[0, 2].set_ylabel('真实标签')
        
        # 预测概率分布
        axes[1, 0].hist([y_prob[i] for i in range(len(y_prob)) if y_true[i] == 0],
                       bins=20, alpha=0.7, label='Control', color='blue')
        axes[1, 0].hist([y_prob[i] for i in range(len(y_prob)) if y_true[i] == 1],
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
        
        print(f"✅ 评估可视化已保存: {self.save_dir / 'evaluation_plots.png'}")
    
    def _generate_interpretability_analysis(
        self,
        test_loader: DataLoader,
        features: Optional[np.ndarray] = None,
        attention_weights: Optional[np.ndarray] = None
    ):
        """生成可解释性分析"""
        print("🔍 生成可解释性分析...")
        
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
            
            print(f"✅ 特征重要性分析已保存: {self.save_dir / 'shap_analysis.png'}")
            
        except Exception as e:
            print(f"⚠️  SHAP分析失败: {e}")
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
                print(f"✅ 特征重要性图已保存: {self.save_dir / 'feature_importance.png'}")
            except Exception as e2:
                print(f"⚠️  特征重要性图也失败: {e2}")
        
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
                
                print(f"✅ PCA分析已保存: {self.save_dir / 'pca_analysis.png'}")
            else:
                print("⚠️  样本数量不足，跳过PCA分析")
        else:
            print("⚠️  特征数据不足，跳过PCA分析")
    
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
            
            print(f"✅ 注意力分析已保存: {self.save_dir / 'attention_analysis.png'}")
    
    def save_model(self, filename: str = "best_model.pt"):
        """保存模型"""
        save_path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'best_val_auc': self.best_val_auc,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, save_path)
        print(f"💾 模型已保存: {save_path}")
    
    def load_model(self, filename: str = "best_model.pt"):
        """加载模型"""
        load_path = self.save_dir / filename
        if load_path.exists():
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
            self.train_history = checkpoint.get('train_history', {'loss': [], 'acc': [], 'auc': [], 'f1': []})
            self.val_history = checkpoint.get('val_history', {'loss': [], 'acc': [], 'auc': [], 'f1': []})
            print(f"📂 模型已加载: {load_path}")
        else:
            print(f"❌ 模型文件不存在: {load_path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'best_val_auc': self.best_val_auc,
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
    print(f"\n🔄 开始模型比较...")
    print("=" * 50)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    all_metrics = []
    
    # 评估每个模型
    for trainer in trainers:
        print(f"\n📊 评估 {trainer.model_name}...")
        result = trainer.evaluate(test_loader, generate_plots=False)
        results[trainer.model_name] = result
        all_metrics.append({
            'model': trainer.model_name,
            **result['metrics']
        })
    
    # 创建比较表格
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.set_index('model')
    
    print(f"\n📊 模型性能比较:")
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
    print("📊 生成模型比较可视化...")
    
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
        y_true = result['predictions']['labels']
        y_prob = result['predictions']['probabilities']
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = result['metrics']['auc']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
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
    
    print(f"✅ 比较可视化已保存: {save_dir}")


if __name__ == "__main__":
    print("🚀 增强版训练器模块")
    print("请通过主脚本使用此训练器")

