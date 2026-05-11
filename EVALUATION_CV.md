# 评估验证层文档（Evaluation & Cross-Validation）

> 本文档对应评估验证层补全（第四部分），涵盖 5-Fold Stratified CV、Macro-AUC/QWK 指标、Wilcoxon 统计检验。

---

## 1. 5-Fold Stratified Cross-Validation

### 1.1 为什么需要 CV？

本项目仅 **142 样本**，单次随机划分的测试集仅有 20–30 个样本。AUC 的标准差可能高达 ±0.08，导致：
- 种子间波动远大于模型间真实差异
- 无法可靠比较两个模型的优劣

5-Fold Stratified CV 将每个样本恰好作为测试集一次，同时保持每折的类别比例与总体一致，显著降低评估方差。

### 1.2 双层划分策略

```
外层：StratifiedKFold(n_splits=5)  → train_val / test
内层：train_test_split(stratify=...) → train / val
```

- **外层** 保证测试集的类别分布与总体一致
- **内层** 同样分层，保证验证集的类别分布与 train_val 一致
- 每折的 random_state 不同（`random_state + fold_idx`），避免内层划分重复

### 1.3 使用方式

**命令行：**
```bash
python enhanced_main.py --config configs/experiment_base.yaml --model ConcatFusion --mode cv
```

**配置文件中启用：**
```yaml
evaluation:
  cross_validation:
    enabled: true
    n_splits: 5
    shuffle: true
    random_state: 42
    inner_val_ratio: 0.15
```

### 1.4 输出结构

```
results/
  ConcatFusion/
    fold_0/
      metrics_summary.json
      best_model.pt
    fold_1/
      ...
    cv_summary.json          # mean ± std 汇总
    fold_results.csv         # 每折原始值
```

### 1.5 约束与回退

- **仅支持 Raw 模式**：Embedding 模式使用预定义划分，不支持 CV 外层折叠。若 `use_embedding=true` 且启用了 CV，会自动打印 WARNING 并回退到单折模式。
- **class 1 仅 4 样本**：5-Fold 下某些 test fold 可能完全没有 class 1。代码已处理缺失类别的退化逻辑（见 Macro-AUC 章节）。

---

## 2. Macro-AUC 与 Weighted-AUC

### 2.1 临床意义

在 4 类糖尿病分级（0/1/2/3）中：
- **Macro-AUC**：对每个类别做 One-vs-Rest AUC，然后取平均。对所有类别一视同仁，不因为 class 1 样本少而降低权重。适合评估模型对稀有类别的识别能力。
- **Weighted-AUC**：按各类支持度（样本数）加权平均。反映模型在整体样本上的综合排序能力。
- 对于医学分级任务，Macro-AUC 比单纯 accuracy 更有参考价值，因为罕见并发症（class 1）的漏诊代价更高。

### 2.2 实现细节

`trainers/enhanced_trainer.py::_calculate_metrics()` 现在接收可选参数 `probs_matrix`（完整的 softmax 概率矩阵 `[N, C]`）：

```python
metrics = trainer._calculate_metrics(
    y_true, y_prob_pos_class, y_pred,
    probs_matrix=softmax_probs  # [N, num_classes]
)
```

计算流程：
1. 优先使用 `sklearn.metrics.roc_auc_score(..., multi_class='ovr', average='macro')`
2. 若失败（某折缺少某类），退化为**逐类手动计算**：
   - `label_binarize` 构建 OvR 标签
   - 对存在的类别逐个计算 AUC
   - 取平均（Macro）或按支持度加权（Weighted）

### 2.3 配置

```yaml
evaluation:
  metrics:
    compute_macro_auc: true
    compute_weighted_auc: true
```

### 2.4 早停支持

`early_stop_metric` 现在支持：
- `auc`（默认，OvR 正类 AUC）
- `weighted_f1`
- `macro_f1`
- `macro_auc`
- `qwk`

若某折因缺失类别导致目标指标为 NaN，自动回退到 `weighted_f1` 并打印 WARNING。

---

## 3. QWK（Quadratic Weighted Kappa）解耦

### 3.1 临床意义

QWK 是医学影像和分级诊断中广泛使用的指标，特点：
- 惩罚与“真实距离”成正比的误判（如把 class 2 判为 class 3 比判为 class 0 的惩罚小）
- 对有序分级任务（如 0→1→2→3 代表病情递增）比 accuracy 更敏感
- 取值 [-1, 1]，1 表示完全一致，0 表示随机一致

### 3.2 解耦

**修改前**：QWK 仅在 `advanced.enabled=true` 时计算，与 `train.advanced` 强耦合。

**修改后**：QWK / Cohen's Kappa 完全由 `evaluation.metrics` 配置控制，独立于 `advanced.enabled`：

```yaml
evaluation:
  metrics:
    compute_cohens_kappa: true   # 线性 Kappa
    compute_qwk: true            # 二次加权 Kappa
```

当配置为 `true` 时，无论 advanced 是否启用，指标都会计算并写入 `metrics_summary.json`。

---

## 4. Wilcoxon 配对检验 vs t-test

### 4.1 使用场景区分

| 场景 | 推荐检验 | 原因 |
|------|----------|------|
| 多种子运行（seed 0/1/2...） | **t-test** | 种子间相互独立，属于独立样本比较 |
| CV fold 级比较（fold 0/1/2/3/4） | **Wilcoxon signed-rank** | 同一 fold 下不同模型使用相同数据划分，属于配对样本 |

### 4.2 输出示例

运行 `scripts/generate_main_results_table.py` 后，若检测到 `fold_results.csv`，Markdown 输出末尾自动追加：

```markdown
## Pairwise Wilcoxon Signed-Rank Test (paired by CV fold)

Metric: `test_macro_auc`, alpha=0.05

| Model A | Model B | n_folds | statistic | p-value | Significant |
|---------|---------|---------|-----------|---------|-------------|
| SpectraOnlyModel | ClinicalOnlyModel | 5 | 0.00 | 0.0625 | |
| SpectraOnlyModel | AttentionMultimodal | 5 | 0.00 | 0.0312 | * |
```

### 4.3 配置

```yaml
evaluation:
  statistical_test:
    enabled: true
    alpha: 0.05
    metric_for_comparison: "macro_auc"   # 用于 Wilcoxon 的指标列名
```

---

## 5. 关键约束与向后兼容

1. **单折模式零改动**：`evaluation.cross_validation.enabled: false` 且未指定 `--mode cv` 时，`main()` 完全保持原有单折逻辑。
2. **Embedding 模式回退**：CV 不支持 Embedding mode，自动回退到单折，不抛异常。
3. **Trainer 路径兼容**：`fold_idx=None` 时保存路径与修改前完全一致；仅当 `fold_idx` 不为 None 时才创建 `fold_N/` 子目录。
4. **Metrics JSON 兼容**：原有 `auc`, `accuracy`, `weighted_f1` 等键始终存在。新增 `macro_auc`, `weighted_auc`, `cohens_kappa`, `qwk` 等键追加写入，不覆盖旧键。
5. **空输入保护**：`train_epoch` / `eval_epoch` / `evaluate` 均增加了空 batch / 空 dataset 保护，避免在极端数据划分下崩溃。

---

## 6. 冒烟测试摘要

运行：`python test_evaluation_cv.py`

| # | 测试项 | 状态 |
|---|--------|------|
| 1 | StratifiedKFold 分层有效性 | PASS |
| 2 | Macro-AUC 计算正确性（完美预测=1.0，缺失类别不退化） | PASS |
| 3 | QWK 解耦（advanced.enabled=false 仍可计算） | PASS |
| 4 | Wilcoxon 检验输出（CSV + Markdown） | PASS |
| 5 | 单折向后兼容（路径、metrics_summary 键） | PASS |
| 6 | Embedding 模式回退 | PASS |
| 7 | Trainer 保存路径（fold_idx=None / 2） | PASS |

---

## 7. 修改文件清单

| 文件 | 变更 |
|------|------|
| `enhanced_main.py` | 新增 `load_full_dataset_raw`, `get_cv_loaders`, `run_cross_validation`；`main()` 支持 `--mode cv`；`train_single_model` 支持 `fold_idx`；metrics_summary 追加扩展指标 |
| `trainers/enhanced_trainer.py` | `__init__` 增加 `fold_idx` / `evaluation_cfg`；`_calculate_metrics` 实现 Macro-AUC / Weighted-AUC（含缺失类别退化）并解耦 QWK；`train_epoch`/`eval_epoch` 收集完整 prob 矩阵；空输入保护 |
| `scripts/generate_main_results_table.py` | 新增 `pairwise_wilcoxon`；读取 `fold_results.csv`；Markdown/CSV 追加 Wilcoxon 表格；CV 汇总读取扩展指标 |
| `configs/experiment_base.yaml` | `evaluation` 块补充 `compute_macro_auc`, `compute_weighted_auc`, `compute_cohens_kappa`, `compute_qwk`, `cross_validation.n_splits/shuffle/random_state/inner_val_ratio`, `statistical_test.alpha/metric_for_comparison` |
| `test_evaluation_cv.py` | 新建，7 项冒烟测试 |
| `EVALUATION_CV.md` | 本文档 |
