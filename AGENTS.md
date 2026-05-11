# AGENTS.md — 多模态糖尿病预测项目

> 本文件面向 AI 编程助手。阅读者应对本项目一无所知，所有信息均基于代码库实际内容，不做假设。

---

## 1. 项目概述

本项目是一个**多模态糖尿病预测**研究代码库，核心任务是融合拉曼光谱（Raman spectroscopy）数据与临床表格特征，进行 4 分类预测。代码库包含两个**完全独立**的子系统：

1. **深度学习多模态融合系统**（项目根目录）：基于 PyTorch，包含多种融合架构（交叉注意力、TFT、MMTM 等）。
2. **传统机器学习基线**（`clinic_dimension/`）：基于 LightGBM 的完整流水线，用于二分类（有无并发症/有无精神症状）。

**关键数据约束**：主数据集仅含 **142 个样本**，4 类标签分布极度不平衡（0: 22, 1: 4, 2: 46, 3: 70）。类 1 仅 4 个样本（2.8%）。所有深度学习模型均面临严重的数据稀缺和过拟合问题。复杂模型（500K–1.35M 参数）系统性地劣于简单单模态基线和 LightGBM 基线（AUC ~0.79 vs DL 最佳 ~0.64）。

---

## 2. 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3 |
| 深度学习框架 | PyTorch |
| 传统 ML | LightGBM, scikit-learn |
| 数据处理 | pandas, numpy, scipy |
| 可视化 | matplotlib, seaborn |
| 可解释性 | SHAP |
| 配置管理 | YAML (`pyyaml`) |
| 其他 | tqdm, joblib, openpyxl |

**注意**：项目根目录**没有** `requirements.txt`、`pyproject.toml` 或 `setup.py`。唯一的依赖清单在 `clinic_dimension/requirements.txt` 中。根目录的深度学习子系统依赖需手动安装（通过代码中的 import 推断）。

---

## 3. 项目结构

```
Fusion/
├── enhanced_main.py              # DL 子系统主入口：数据准备、模型构建、训练、评估
├── configs/                      # YAML 实验配置文件
│   ├── enhanced_config.yaml      # AttentionMultimodal / Baseline 默认配置
│   ├── tft_config.yaml           # TFTMultimodal 专用配置
│   └── mmtm_config.yaml          # EnhancedMMTM 专用配置
├── data/                         # 原始输入数据
│   ├── spectra.csv               # ~18,106 行 × ~802 列（每行一个扫描）
│   ├── clinical.csv              # 142 行 × ~81 列（每行一个病人）
│   └── clinic.xlsx               # 临床数据原始 Excel
├── datasets/                     # PyTorch Dataset 定义
│   ├── raman_dataset.py          # 原始光谱模式：RamanDataset + 预处理 + collate_fn
│   └── embedding_dataset.py      # Embedding 模式：EmbeddingMultimodalDataset
├── models/                       # PyTorch 模型定义
│   ├── Baseline.py               # 单模态基线、ConcatFusion、EnsembleFusion
│   ├── attention_models.py       # AttentionMultimodal（交叉注意力 + 多尺度 CNN）
│   ├── tft_models.py             # TFTMultimodal（Transformer 时序融合）
│   └── enhanced_mmtm_models.py   # EnhancedMMTM（多尺度门控层次融合）
├── trainers/
│   └── enhanced_trainer.py       # EnhancedTrainer：统一训练循环、评估、可视化
├── multimodal/
│   └── embedding_loader.py       # Embedding 加载与按 PatientID 对齐工具
├── scripts/                      # 实验自动化脚本
│   ├── run_multimodal_experiments.py      # 批量训练多模型并汇总 CSV
│   ├── generate_main_results_table.py     # 读取 metrics_summary.json 生成比较表
│   ├── plot_modality_gates.py             # 可视化模态门控权重轨迹
│   ├── run_multimodal_with_different_features.py  # 不同特征组合消融
│   ├── run_ablation_experiments.ps1       # PowerShell 批处理脚本
│   └── cleanup_unused_files.py            # 清理未使用文件
├── spectrum_dimension/           # 光谱单模态分析子系统（独立）
│   ├── baseline.py, baseline2.py
│   └── diabetes_results/         # 光谱模型输出（含 embedding CSV）
├── clinic_dimension/             # LightGBM 传统 ML 子系统（完全独立）
│   ├── code/                     # 数据清洗、特征工程、训练、评估脚本
│   ├── data/
│   ├── outputs-4分类/            # 4 分类输出
│   ├── outputs-有无并发/         # 二分类输出（并发症）
│   ├── outputs-有无精神/         # 二分类输出（精神症状）
│   └── requirements.txt          # 该子系统唯一依赖文件
├── results/                      # DL 训练输出（模型、图表、指标 JSON）
│   └── <model_name>/
│       ├── best_model.pt
│       ├── metrics_summary.json  # 统一指标摘要（实验汇总脚本依赖）
│       ├── results.json
│       ├── training_curves.png
│       ├── evaluation_plots.png
│       ├── pca_analysis.png
│       ├── shap_analysis.png
│       └── modality_gate_history.json
├── test_preprocessing.py         # 预处理冒烟测试（纯 numpy，不依赖 PyTorch 训练）
├── test_model_compression.py     # 模型轻量化冒烟测试（验证参数量与前后向兼容）
└── debug_clinical_csv.py         # 临床数据调试脚本
```

---

## 4. 构建与运行命令

### 4.1 深度学习子系统（项目根目录）

```bash
# 单模型训练
python enhanced_main.py --config configs/enhanced_config.yaml --model AttentionMultimodal

# 训练配置中 models_to_train 列表里的所有模型
python enhanced_main.py --config configs/enhanced_config.yaml --train-all

# 从 checkpoint 恢复训练
python enhanced_main.py --config configs/enhanced_config.yaml --model AttentionMultimodal --resume

# 仅评估（加载 best_model.pt）
python enhanced_main.py --config configs/enhanced_config.yaml --eval-only AttentionMultimodal
```

**配置文件说明**：
- `configs/enhanced_config.yaml`：默认配置，支持 AttentionMultimodal / ConcatFusion / TFTMultimodal
- `configs/tft_config.yaml`：TFT 专用（batch_size=4, lr=0.0005, max_scans=64）
- `configs/mmtm_config.yaml`：EnhancedMMTM 专用（batch_size=4, lr=0.0005）

### 4.2 传统 ML 子系统（`clinic_dimension/`）

```bash
cd clinic_dimension
pip install -r requirements.txt
python code/run_all.py
```

流水线自动执行：`data_cleaning.py` → `feature_processing.py` → `model_train.py` → `evaluate_visualize.py`

输出目录由 `code/config.py` 中的 `output_dir` 控制（`outputs-4分类/`、`outputs-有无并发/` 或 `outputs-有无精神/`）。

### 4.3 冒烟测试

```bash
# 测试预处理逻辑（无需 GPU）
python test_preprocessing.py

# 测试模型架构轻量化兼容性
python test_model_compression.py
```

---

## 5. 数据加载与输入模式

`enhanced_main.py::prepare_data()` 支持两种**互斥**的数据加载模式，由 `cfg["data"]["use_embedding"]` 控制：

### 5.1 原始光谱模式（`use_embedding: false`）
- **Dataset**：`datasets/raman_dataset.py::RamanDataset`
- **输入**：`spectra.csv`（每行一个扫描，一个病人有多个扫描）+ `clinical.csv`（每行一个病人）
- **光谱预处理**（配置驱动）：
  1. AsLS 基线校正（`lam=1e6, p=0.001, niter=10`）
  2. Savitzky-Golay 平滑（`window=11, polyorder=2`）
  3. SNV 归一化（逐扫描）或 patient_zscore（患者级 Z-score）
- **序列处理**：多扫描序列 padding / truncate 到 `max_scans`（默认 180），输出 `[S, L]` + mask
- **替代聚合**：`scan_aggregation: "stats"` 可将多扫描聚合为 `[3*L]` 的统计向量（mean/std/max）
- **数据划分**：随机划分 train/val/test（默认 70%/15%/15%）

### 5.2 Embedding 模式（`use_embedding: true`，当前实验主要使用）
- **Dataset**：`datasets/embedding_dataset.py::EmbeddingMultimodalDataset`
- **输入**：预计算的单模态 embedding CSV
  - 光谱 embedding：`spectrum_embedding_path`（如 `spectrum_dimension/diabetes_results/Light_CNN_MLP_features.csv`）
  - 临床 embedding：`clinical_embedding_path`（如 `data/clinical.csv`）
- **CSV 列要求**：`PatientID`, `Label`, `Split`（光谱端）, `feature_*`（特征列）
- **数据划分**：由 CSV 中的 `Split` 列预定义（train/val/test），不再随机划分
- **对齐逻辑**：`multimodal/embedding_loader.py::align_by_patient_id` 以光谱端 patient_id 为主进行对齐
- **模态 Dropout**：仅在训练集生效（默认 spectra=0.15, clinical=0.10），验证/测试集强制为 0

### 5.3 标签列推断风险
`RamanDataset` 会自动推断临床标签列：如果配置的 `label_col` 不存在，会**静默回退**到 `"Label"`，再回退到 `"Group"`。这可能导致光谱与临床标签不匹配。**务必检查 stdout 中输出的 `Detected Clinical Label`。**

---

## 6. 模型架构

所有模型在 `enhanced_main.py::build_model()` 中实例化。模型维度是动态的：embedding 模式下从数据集推断 `spec_emb_dim` 和 `tab_emb_dim`；raw 模式下从配置读取。

| 模型 | 文件 | 关键特性 | 参数量 |
|------|------|----------|--------|
| `Spectra-only` | `Baseline.py` | 1D-CNN 编码器 → 分类器 | ~100K |
| `Clinical-only` | `Baseline.py` | MLP 编码器 → 分类器 | ~100K |
| `ConcatFusion` | `Baseline.py` | 拼接融合 + 分类器 | ~200K |
| `EnsembleFusion` | `Baseline.py` | 独立单模态头 + 可学习融合门 | ~200K |
| `BaselineMultimodal` | `Baseline.py` | 包装器，选择 concat/ensemble | ~200K |
| `AttentionMultimodal` | `attention_models.py` | 多尺度 1D-CNN + 注意力池化 + 双向交叉注意力 + 增强分类器 | ~500K–800K |
| `TFTMultimodal` | `tft_models.py` | Transformer 编码器 + 多尺度卷积 + 特征选择门 + 辅助损失 + 对比损失 | ~600K–900K |
| `EnhancedMMTM` | `enhanced_mmtm_models.py` | 多头自注意力增强 + 自适应门控 + 层次化融合（global/meso/local）+ 不确定性估计 | ~900K–1.35M |

### 6.1 模型输出约定
所有模型的 `forward()` 必须返回一个 `dict`，至少包含：
- `"logits"`: `[B, num_classes]` — 主分类输出

可选返回（用于可视化/训练）：
- `"embedding"`: 融合特征向量（用于 t-SNE/PCA）
- `"spec_embedding"` / `"tab_embedding"`: 单模态嵌入
- `"aux_spec_logits"` / `"aux_tab_logits"`: 辅助任务输出（AttentionMultimodal 和 TFT 使用）
- `"attention_weights"`: 注意力权重
- `"gated_spec"`: 门控分析用

### 6.2 轻量化模式
所有模型支持 `lite_cfg`（在 `model.lite` 下配置）。启用后参数量可减少 85%–94%，适用于小样本场景。`test_model_compression.py` 验证所有模型的 lite 兼容性。

---

## 7. 训练与评估

### 7.1 训练器
`trainers/enhanced_trainer.py::EnhancedTrainer` 是统一训练循环：
- **优化器**：AdamW（lr 来自配置，默认 1e-3）
- **学习率调度**：`ReduceLROnPlateau(mode='min', patience=5, factor=0.5)`，监控验证 loss
- **损失函数**：`CrossEntropyLoss`（可选类别权重）；`TFTMultimodal` 使用自定义 `TFTLoss`（主 CE + 2 辅助 CE + 对比损失）
- **早停**：基于验证 AUC，耐心值可配置（默认 10–20）
- **梯度裁剪**：max_norm=1.0
- **类别权重**：当 `use_class_weights: true` 时，自动计算训练分布的逆频率权重

### 7.2 评估指标
- 准确率（Accuracy）
- 加权 F1（Weighted F1）
- AUC：二分类直接使用；多分类时退化为**标签 1 的 one-vs-rest AUC**（若标签 1 不存在则取最大标签）——这是一个硬编码简化
- 敏感性@90%特异性（sensitivity@90%spec）

### 7.3 输出目录结构
训练完成后 `results/<model_name>/` 包含：
- `best_model.pt`：checkpoint（权重 + 历史记录 dict）
- `metrics_summary.json`：统一指标摘要（实验汇总脚本依赖此文件）
- `results.json`：详细结果
- `training_curves.png`：loss/acc/auc/f1 每轮曲线
- `evaluation_plots.png`：ROC、PR、混淆矩阵、概率分布
- `pca_analysis.png` / `shap_analysis.png` / `feature_importance.png`：可解释性图表
- `modality_gate_history.json`：epoch-by-epoch 融合门控权重（如果模型有 fusion_gate）

### 7.4 Numpy 兼容性
`enhanced_trainer.py` 导入时包含一个 monkey-patch：若运行在 numpy 1.x 环境但 checkpoint 由 numpy 2.x 保存，会将 `numpy.core` 映射到 `numpy._core`，以避免加载失败。

---

## 8. 实验自动化

| 脚本 | 功能 |
|------|------|
| `scripts/run_multimodal_experiments.py` | 批量顺序训练多个模型，自动读取 `metrics_summary.json` 汇总为 CSV。支持 `--embedding_only` 和 `--models <列表>` |
| `scripts/generate_main_results_table.py` | 读取所有 `results/*/metrics_summary.json`，生成比较 markdown/CSV 表格 |
| `scripts/plot_modality_gates.py` | 加载 `modality_gate_history.json`，绘制门控权重随 epoch 变化轨迹 |
| `scripts/run_multimodal_with_different_features.py` | 不同特征组合的消融实验 |

---

## 9. 代码风格与开发约定

### 9.1 语言与注释
- 代码注释和文档字符串以**中文**为主，夹杂英文技术术语。
- 配置项、变量名、类名使用英文。
- 打印日志和错误信息以中文为主（如 `[BUILD] 构建模型`、`[ERROR] 未知模型名称`）。

### 9.2 编码规范
- 无 formal linter/formatter 配置（无 `.pylintrc`、`.flake8`、`pyproject.toml`）。
- 使用 4 空格缩进。
- 函数和类文档字符串使用中文描述参数和返回值。
- 模型定义中广泛使用 `lite_cfg` 字典进行条件化架构切换，保持向后兼容。

### 9.3 输入兼容性
模型 `forward()` 需要兼容多种调用方式：
1. **Raw 模式位置参数**：`model(spectra, mask, tabular)`
2. **Embedding 模式字典参数**：`model(spectra_dict, tabular_dict)`，其中字典包含 `"embedding"`、`"mask"`、`"logits"` 键
3. 部分模型（如 `BaselineMultimodal`）在内部通过类型检查自动分发到不同路径。

### 9.4 配置驱动开发
几乎所有超参数（模型名、数据路径、划分比例、batch size、学习率、类别权重、模态 dropout、增强参数）均通过 YAML 控制。修改行为时优先改配置而非硬编码。

---

## 10. 测试策略

本项目**没有**使用 pytest、unittest 等正式测试框架。测试以**冒烟测试脚本**形式存在：

1. **`test_preprocessing.py`**：纯 numpy 测试，验证：
   - `sequence` 模式的 padding/mask 逻辑
   - `patient_zscore` 归一化分布正确性
   - `stats` 聚合模式输出形状 `[3*L]`
   - 基线校正/平滑配置参数可配置性
   - Embedding 审计逻辑（split 缺失、分布异常）

2. **`test_model_compression.py`**：PyTorch 测试，验证：
   - 默认构建与 Lite 构建的向后兼容性
   - Lite 模式参数量减少目标（各模型有具体阈值，如 AttentionMultimodal ≥90%）
   - Raw 与 Embedding 两种输入模式的前向传播输出形状正确
   - 所有模型返回 dict 且包含 `"logits"`

**运行方式**：直接 `python test_preprocessing.py` / `python test_model_compression.py`。

---

## 11. 关键约束与注意事项

### 11.1 数据稀缺与过拟合
- **142 个样本**，测试集仅 20–30 个样本，指标统计不稳定。**不要信任模型间微小的 AUC 差异。**
- 复杂模型（AttentionMultimodal、EnhancedMMTM、TFTMultimodal）有 500K–1.35M 参数，会记忆训练集。当前正则化（dropout 0.1, weight decay 1e-4）不足。
- 如果继续 DL 工作，必须考虑：更重的 dropout（≥0.5）、迁移学习、数据增强、类别合并（如 0+1 vs 2+3）、或序数回归。

### 11.2 类别不平衡
- 类 1 只有 4 个样本（2.8%）。4 分类对该类几乎不可学习。
- `use_class_weights: true` 可缓解，但在如此极端的分布下效果有限。

### 11.3 Embedding 信息泄漏风险
- Embedding 模式使用预计算特征。如果 embedding 是在**全数据集**上 fit 的（包括测试集），会导致信息泄漏。
- 配置中有审计字段 `data.embedding.generation_scope`（应设为 `"train_only"`），但未强制约束生成过程。**修改或生成新 embedding 时，务必确保仅在训练集上 fit。**

### 11.4 AUC 计算简化
- 多分类时，AUC 和 sensitivity@90%spec 计算的是**单一正类**（优先 label=1，否则最大标签）的 one-vs-rest 指标。**这不是真正的多分类 AUC。**

### 11.5 标签推断风险
- `RamanDataset` 会静默回退标签列名。始终验证 stdout 中的 `Detected Clinical Label` 是否与预期一致。

### 11.6 两个子系统独立性
- `clinic_dimension/` 的 LightGBM 流水线与根目录的 PyTorch 代码**完全独立**，不共享数据加载器、配置或模型。
- `spectrum_dimension/` 的光谱单模态分析也是独立子系统，但其输出 embedding CSV 可被根目录 DL 系统消费。

---

## 12. 安全与部署

- **无网络服务**：本项目是纯离线研究和实验代码，没有 API、Web 服务或网络接口。
- **无敏感数据硬编码**：代码中未出现密钥、密码、Token。
- **数据文件**：`data/spectra.csv` 被 `.gitignore` 排除（过大）。`clinical.csv` 未排除，但应确认其中不含患者隐私信息后再提交。
- **模型文件**：`.pt`、`.pth`、`.pkl`、`.joblib` 均被 `.gitignore` 排除。
- **部署**：本项目为研究性质，无生产部署流程。`results/` 和 `logs/` 目录也被 `.gitignore` 排除。
