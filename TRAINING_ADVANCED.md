# 高级训练策略配置指南 (TRAINING_ADVANCED.md)

## 概述

`configs/enhanced_config.yaml` 中新增的 `train.advanced` 块提供了一系列针对 **142 样本、4 类极端不平衡** 场景的高级训练策略。

**核心原则**：
- `advanced.enabled: false`（或整个 `advanced` 块缺失）时，训练行为与修改前 **完全一致**。
- 所有新策略均通过配置开关控制，默认关闭。
- 修改后的 `EnhancedTrainer` 向后兼容旧 checkpoint（`best_model.pt` 中的 `best_val_auc` 字段被映射到新的 `best_val_metric`）。

---

## 配置结构

```yaml
train:
  # 原有配置（保持不变）
  lr: 0.001
  weight_decay: 0.0001
  batch_size: 8
  epochs: 60
  patience: 15
  use_class_weights: false

  # 新增高级训练策略（全部可选）
  advanced:
    enabled: false          # 总开关；false 时所有 advanced 策略不生效

    # 1. 损失函数
    loss_type: "CE"         # "CE" | "Focal" | "CORN"
    focal_gamma: 2.0        # Focal Loss gamma
    label_smoothing: 0.0    # 0.0=关闭；建议 0.1
    aux_weight: 1.0         # 辅助损失权重（Attention/TFT 的 aux_spec/aux_tab）
    aux_decay_schedule: null # null | [10, 0.5] 每 10 epoch aux_weight *= 0.5

    # 2. 优化器
    optimizer: "AdamW"      # "AdamW" | "AdamW_SAM"
    sam_rho: 0.05           # SAM 邻域半径

    # 3. 学习率调度
    scheduler_type: "plateau"   # "plateau"（默认）| "cosine"
    warmup_epochs: 0            # 预热轮数（仅 cosine 有效）
    cosine_T_max: 100           # CosineAnnealing 周期

    # 4. 分层优化参数组
    param_groups:
      spectra_encoder:
        lr_multiplier: 1.0
        wd_multiplier: 1.0
      clinical_encoder:
        lr_multiplier: 1.0
        wd_multiplier: 1.0
      fusion_module:
        lr_multiplier: 1.0
        wd_multiplier: 1.0
      classifier_head:
        lr_multiplier: 1.0
        wd_multiplier: 1.0

    # 5. 训练稳定性
    grad_clip_max_norm: null    # null=关闭；如 1.0

    # 6. 早停策略
    early_stop_metric: "auc"    # "auc" | "weighted_f1" | "macro_f1"
    early_stop_patience: 15     # 覆盖外层 train.patience

    # 7. 渐进式训练
    phase_training:
      enabled: false
      phase1_epochs: 20
      phase1_modules: ["spectra_encoder", "clinical_encoder"]
      phase2_modules: ["fusion_module", "classifier_head"]
```

---

## 各策略详解与推荐配置

### 1. 损失函数 (loss_type)

| 选项 | 适用场景 | 推荐值 |
|------|---------|--------|
| `CE` | 默认；配合 `label_smoothing` 可缓解过拟合 | `label_smoothing: 0.1` |
| `Focal` | 类别极度不平衡（class 1 仅 4 例） | `focal_gamma: 2.0` |
| `CORN` | 有序回归（尚未实现，预留接口） | — |

**推荐（142 样本）**：
```yaml
loss_type: "Focal"
focal_gamma: 2.0
label_smoothing: 0.0   # Focal 内部已实现平滑兼容
```

> 注意：当 `loss_type: "Focal"` 且 `label_smoothing > 0` 时，内部使用 `BCEWithLogitsLoss` + one-hot smoothing 实现，与 `CrossEntropyLoss` 的 smoothing 语义略有差异。

### 2. 优化器 (optimizer)

| 选项 | 说明 | 推荐场景 |
|------|------|---------|
| `AdamW` | 默认 | 常规训练 |
| `AdamW_SAM` | SAM 包装 AdamW，每 batch 两次前向-后向 | 泛化性优先 |

**SAM 开销**：每个 batch 执行 2 次 forward + 2 次 backward，训练时间翻倍。

### 3. 学习率调度 (scheduler_type)

| 选项 | 说明 | 推荐配置 |
|------|------|---------|
| `plateau` | ReduceLROnPlateau，监控 val loss | 默认；patience=5, factor=0.5 |
| `cosine` | CosineAnnealingLR + 可选 warmup | `warmup_epochs: 3`, `cosine_T_max: 50` |

**142 样本推荐**：
```yaml
scheduler_type: "cosine"
warmup_epochs: 3
cosine_T_max: 50    # 设为 epochs 或 epochs//2
```

理由：ReduceLROnPlateau 在 20-30 样本的验证集上因 AUC 噪声频繁误触发；Cosine 退火更稳定。

### 4. 分层参数组 (param_groups)

按参数名子串自动分组：
- `spectra_encoder`: 匹配 `spectra`, `spec`
- `clinical_encoder`: 匹配 `clinical`, `tabular`, `tab`
- `fusion_module`: 匹配 `fusion`, `cross`, `mmtm`, `attention`
- `classifier_head`: 匹配 `classifier`, `head`

**142 样本推荐**：
```yaml
param_groups:
  spectra_encoder:
    lr_multiplier: 0.5    # 光谱信号强，慢调
    wd_multiplier: 1.0
  clinical_encoder:
    lr_multiplier: 1.0
    wd_multiplier: 2.0    # 临床特征易过拟合，强正则
  fusion_module:
    lr_multiplier: 1.0
    wd_multiplier: 1.0
  classifier_head:
    lr_multiplier: 1.5    # 分类头快速收敛
    wd_multiplier: 1.0
```

### 5. 梯度裁剪 (grad_clip_max_norm)

默认关闭。当启用时，在 `loss.backward()` 后执行 `clip_grad_norm_`。

**注意**：默认模式（`advanced.enabled: false`）下，原硬编码的 `max_norm=1.0` 仍然保留；高级模式下可独立配置。

### 6. 早停指标 (early_stop_metric)

| 选项 | 说明 |
|------|------|
| `auc` | 监控 one-vs-rest AUC（label=1） |
| `weighted_f1` | 监控 sklearn weighted F1 |
| `macro_f1` | 监控 sklearn macro F1 |

**142 样本推荐**：`macro_f1` 对类别不平衡更敏感，比 AUC 更稳定。

### 7. 渐进式训练 (phase_training)

分阶段冻结/解冻参数，模块名通过内置关键词映射到实际参数名：

| 配置名 | 实际匹配子串 |
|--------|-------------|
| `spectra_encoder` | `spectra`, `spec` |
| `clinical_encoder` | `clinical`, `tabular`, `tab` |
| `fusion_module` | `fusion`, `cross`, `mmtm`, `attention` |
| `classifier_head` | `classifier`, `head` |

**典型配置**：
```yaml
phase_training:
  enabled: true
  phase1_epochs: 20
  phase1_modules: ["spectra_encoder", "clinical_encoder"]
  phase2_modules: ["fusion_module", "classifier_head"]
```

Phase 1 仅训练单模态编码器，冻结融合与分类头；Phase 2 联合微调。

---

## 扩展评估指标

启用 `advanced.enabled: true` 后，`_calculate_metrics` 会额外计算：

| 指标 | 键名 | 说明 |
|------|------|------|
| Macro F1 | `macro_f1` | 各类 F1 的未加权平均 |
| Cohen's Kappa | `cohens_kappa` | 一致性系数 |
| Quadratic Weighted Kappa | `qwk` | 适用于有序分级 |

这些指标自动加入 `metrics_summary.json` 和训练历史，可用于早停监控。

---

## 快速启用模板（142 样本推荐）

```yaml
train:
  batch_size: 8
  epochs: 60
  lr: 0.001
  weight_decay: 1e-4

  advanced:
    enabled: true

    # 损失
    loss_type: "Focal"
    focal_gamma: 2.0
    label_smoothing: 0.0
    aux_weight: 0.5
    aux_decay_schedule: [10, 0.5]

    # 优化器
    optimizer: "AdamW"

    # 调度
    scheduler_type: "cosine"
    warmup_epochs: 3
    cosine_T_max: 50

    # 分层参数
    param_groups:
      spectra_encoder:
        lr_multiplier: 0.5
        wd_multiplier: 1.0
      clinical_encoder:
        lr_multiplier: 1.0
        wd_multiplier: 2.0
      fusion_module:
        lr_multiplier: 1.0
        wd_multiplier: 1.0
      classifier_head:
        lr_multiplier: 1.5
        wd_multiplier: 1.0

    # 稳定性
    grad_clip_max_norm: 1.0

    # 早停
    early_stop_metric: "macro_f1"
    early_stop_patience: 10

    # 渐进训练
    phase_training:
      enabled: true
      phase1_epochs: 15
      phase1_modules: ["spectra_encoder", "clinical_encoder"]
      phase2_modules: ["fusion_module", "classifier_head"]
```

---

## 冒烟测试清单

运行 `python test_training_advanced.py` 验证以下 7 项：

1. **默认行为不变** — `advanced` 缺失时，SpectraOnlyModel / AttentionMultimodal 各 3 epoch，Loss 曲线与修改前一致。
2. **Focal Loss + Label Smoothing** — TabularOnlyModel 1 epoch，Loss 有限值，aux_weight 初始值正确。
3. **SAM + Cosine + Warmup** — ConcatFusion 5 epoch，每 batch 两次前向传播，LR 先升后降。
4. **分层参数组** — 打印 4 组参数，lr/wd 与配置乘数一致，无参数遗漏。
5. **Phase Training** — AttentionMultimodal 10 epoch，Phase 1 冻结融合层，Phase 2 解冻，不报错。
6. **梯度裁剪** — grad_clip_max_norm=1.0，1 epoch 无报错。
7. **早停指标切换** — auc / weighted_f1 / macro_f1 分别监控，指标值与 JSON 一致。
