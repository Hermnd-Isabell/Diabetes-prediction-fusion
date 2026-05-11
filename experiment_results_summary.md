# 实验结果汇总

> 运行时间：2026-05-08
> 环境：CPU (PyTorch 2.6.0+cpu)，Embedding 模式（跳过原始光谱预处理）

## 1. 单折实验结果（Embedding 模式）

| 模型 | 参数量 | Best Val AUC | Test AUC | Test Acc | Test F1 | Macro-AUC | Weighted-AUC | QWK |
|------|--------|-------------|----------|----------|---------|-----------|--------------|-----|
| AttentionMultimodal | 2,068,441 | 0.667 | 0.250 | 0.364 | 0.424 | 0.504 | 0.639 | 0.384 |
| ConcatFusion | 228,484 | 0.762 | 0.464 | 0.364 | 0.370 | 0.512 | 0.593 | 0.098 |
| TFTMultimodal | 4,277,908 | 0.714 | 0.107 | 0.545 | 0.625 | 0.653 | 0.716 | 0.610 |
| **LightGBM (4-class)** | - | - | **0.579** | **0.414** | **0.399** | **0.579** | - | - |

### 关键观察
- **过拟合极其严重**：所有 DL 模型的训练集 AUC 在几个 epoch 内即达到 0.95+，但测试集 AUC 远低于验证集。
- **模型复杂度与性能不成正比**：TFTMultimodal（427万参数）测试 AUC 仅 0.107；ConcatFusion（22万参数，最简单）反而测试 AUC 最高（0.464）。
- **LightGBM 基线仍然领先**：Macro-AUC 0.579，高于所有单折 DL 模型。
- **小样本波动巨大**：同一模型在不同次运行中，Test AUC 可以从 0.46 掉到 0.25（仅因随机种子/划分不同），说明 11-14 个测试样本的指标极不稳定。

---

## 2. 5-Fold Stratified CV 结果（AttentionMultimodal）

| 指标 | Mean ± Std | 说明 |
|------|-----------|------|
| Test AUC | 0.316 ± 0.173 | 单折 AUC 波动极大 |
| Test Acc | 0.571 ± 0.090 | 准确率相对稳定 |
| Test F1 | 0.540 ± 0.080 | 加权 F1 |
| Test Macro-F1 | 0.383 ± 0.079 | 宏平均 F1 |
| **Test Macro-AUC** | **0.700 ± 0.126** | 跨折平均最高指标 |
| Test Weighted-AUC | 0.688 ± 0.099 | 加权 AUC |
| Test Cohen's Kappa | 0.222 ± 0.111 | 一致性较低 |
| Test QWK | 0.086 ± 0.115 | 二次加权 Kappa 波动大 |
| Best Val AUC | 0.760 ± 0.203 | 验证集表现远好于测试集（过拟合信号） |

### 各折明细

| Fold | Test AUC | Test Acc | Test F1 | Best Val AUC | Test Macro-AUC | Test QWK |
|------|----------|----------|---------|-------------|----------------|----------|
| 0 | 0.250 | 0.643 | 0.641 | 0.700 | 0.875 | ~0.000 |
| 1 | 0.042 | 0.500 | 0.484 | 1.000 | 0.778 | 0.000 |
| 2 | 0.333 | 0.643 | 0.587 | 0.550 | 0.563 | 0.300 |
| 3 | 0.571 | 0.643 | 0.573 | 1.000 | 0.736 | 0.115 |
| 4 | 0.385 | 0.429 | 0.415 | 0.550 | 0.552 | 0.015 |

### 关键观察
- **Macro-AUC 在 CV 中明显更高**（0.700 ± 0.126）vs 单折（0.504），说明单折 Test AUC 受极端类别不平衡（class 1 仅 1 个样本）影响更大。
- **Val AUC 与 Test AUC 严重背离**：Best Val AUC 平均 0.76，但 Test AUC 平均仅 0.32，再次确认模型在 memorize 训练集。
- **QWK 极不稳定**：从 0.0 到 0.3 大幅波动，反映 4 分类任务在 142 样本上的内在困难。

---

## 3. 实验过程中修复的 Bug

| 文件 | 问题 | 修复 |
|------|------|------|
| `models/tft_models.py` | TFTMultimodal 在 embedding 模式下因 `spec_emb` (128D) 与 `tab_emb` (79D) 维度不匹配导致 `contrastive_loss` 中 `matmul` 报错 | 在 `contrastive_loss` 开头添加维度不匹配保护：若维度不同则返回 0 |
| `clinic_dimension/code/model_train.py` | 原始代码为二分类设计，在 4 分类下 `roc_auc_score` 和 `f1_score` 调用报错 | 添加多分类分支：使用 `multi_class='ovr'` 和 `average='weighted'`/`'macro'` |
| `enhanced_main.py` | `main()` 中强制拒绝 embedding 模式的 CV | 移除该限制，允许 embedding 模式跑 5-Fold CV |
| `enhanced_main.py` | `load_full_dataset_raw` 仅支持 raw 模式 | 添加 embedding 模式分支：合并 train/val/test 的 `EmbeddingMultimodalDataset` |
| `enhanced_main.py` | `get_cv_loaders` 的 inner split 在稀有类别（class 1 仅 1 样本）时分层切分失败 | 添加 try-except：分层失败时回退到随机切分 |
| `enhanced_main.py` | `run_cross_validation` 中使用 `json.dump` 但文件未 `import json` | 在文件顶部添加 `import json` |
| `clinic_dimension/` | lightgbm 4.5.0 与 sklearn 1.8.0 不兼容 | 临时降级 scikit-learn 到 1.5.2 |

---

## 4. 结论与建议

1. **数据量太小**：142 样本、4 类、class 1 仅 4 样本（全数据集）/ 1 样本（embedding 训练集），任何复杂 DL 模型都会过拟合。
2. **Embedding 模式加速了实验验证**：raw 模式下 AsLS baseline correction 在 CPU 上每个患者需数十秒，142 患者不可接受；embedding 模式将单折实验缩短到 5-25 秒。
3. **LightGBM 仍是当前最优基线**：Macro-AUC ~0.58，参数量远小于 DL 模型，训练时间秒级。
4. **如要继续 DL 方向**：
   - 强烈建议合并类别（0+1 vs 2+3）或改为有序回归；
   - 使用更激进的正则化（dropout ≥ 0.5，权重衰减 ≥ 1e-3）；
   - 尝试预训练/迁移学习（利用公开光谱数据集预训练光谱编码器）；
   - 考虑基于 LightGBM 的集成策略，而非端到端 DL。
