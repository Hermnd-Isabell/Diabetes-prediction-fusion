# 实验编排与消融层文档

> 本文档对应实验管理层重构（Experiment Orchestration & Ablation Refactoring），涵盖配置继承、多种子运行、敏感性扫描、消融实验、结果聚合与可视化等 6 项增强。

---

## 1. 配置体系（Config Inheritance）

### 1.1 设计目标
- **消除重复**：所有共享参数（数据路径、训练超参、可视化开关等）统一放在 `configs/experiment_base.yaml`。
- **子配置只写差异**：`enhanced_config.yaml`、`tft_config.yaml`、`mmtm_config.yaml` 仅需指定模型名称和少数覆盖项。
- **向后兼容**：旧格式单文件（无 `defaults`）仍可被 `load_config()` 直接加载。

### 1.2 用法示例

```yaml
# configs/enhanced_config.yaml
defaults:
  - experiment_base

experiment:
  name: "enhanced_attention"
model:
  name: AttentionMultimodal
```

```python
from enhanced_main import load_config

cfg = load_config("configs/enhanced_config.yaml")
print(cfg["train"]["batch_size"])   # 8  （来自 base）
print(cfg["model"]["name"])          # AttentionMultimodal （子配置覆盖）
```

### 1.3 关键字段说明
| 配置块 | 作用 |
|--------|------|
| `experiment` | 实验名称、随机种子、输出目录 |
| `data` | 数据路径、划分比例、预处理参数 |
| `model` | 模型名称、`lite` 子开关、类别数 |
| `train` | `epochs`、`batch_size`、`lr`、早停、增强 |
| `visualization` | `enable` 总开关及各子图开关 |
| `interpretability` | SHAP、注意力、特征重要性、PCA/t-SNE |
| `comparison` | 对比表格、统计检验开关 |

---

## 2. 多种子实验（Multi-Seed Experiments）

### 2.1 为什么要跑多种子？
本项目仅 **142 样本**，单次随机划分会让测试 AUC 出现 ±0.05 以上的抖动。跑 3–5 个种子并取 **mean ± std** 是评估模型差异的最低要求。

### 2.2 命令行用法

```bash
# 单模型、3 个种子
python scripts/run_multimodal_experiments.py \
  --config configs/enhanced_config.yaml \
  --mode single \
  --model AttentionMultimodal \
  --seeds 0 1 2 \
  --lite \
  --overwrite

# 所有模型一起跑
python scripts/run_multimodal_experiments.py \
  --config configs/enhanced_config.yaml \
  --mode single \
  --seeds 0 1 2 \
  --lite \
  --baselines
```

### 2.3 输出结构
```
results/
  AttentionMultimodal/
    seed_0/
      metrics_summary.json
      results.json
    seed_1/
      metrics_summary.json
    seed_2/
      metrics_summary.json
    seed_summary.csv          # 自动汇总 mean ± std
```

### 2.4 142 样本场景的标准协议
| 步骤 | 建议 |
|------|------|
| 种子数 | **≥ 3**（推荐 5） |
| 划分比例 | `train_val_test_ratio: [0.7, 0.15, 0.15]` 或固定 `train_ratio: 0.7` |
| Lite 模式 | **必须开启**（参数量下降 90%+） |
| Epochs | 30–60，配合 `early_stopping_patience: 10–15` |
| Baseline | 强制跑 `Spectra-only`、`Clinical-only`、`ConcatFusion` 作为参照 |

---

## 3. 交叉验证模式（CV Mode）

```bash
python scripts/run_multimodal_experiments.py \
  --config configs/enhanced_config.yaml \
  --mode cv \
  --model AttentionMultimodal \
  --n_splits 5 \
  --lite
```

输出结构：
```
results/
  AttentionMultimodal/
    fold_0/
    fold_1/
    ...
    cv_summary.json
```

> **注意**：142 样本做 5-Fold 时，每折训练集约 113 样本，验证集约 29 样本。StratifiedKFold 可保持类别比例，但 class 1（仅 4 样本）仍可能不出现在某些折中。

---

## 4. 参数敏感性扫描（Sensitivity Analysis）

### 4.1 用途
在超参空间中做**单维度控制扫描**，快速发现哪些参数对 AUC 影响最大。

### 4.2 命令行用法

```bash
python scripts/run_sensitivity_analysis.py \
  --config configs/enhanced_config.yaml \
  --model ConcatFusion \
  --dimension hidden_dim \
  --values 16 32 64 128 \
  --seed 42 \
  --lite \
  --output_dir results/sensitivity
```

支持的扫描维度：`hidden_dim`、`dropout`、`weight_decay`、`batch_size`。

### 4.3 输出
- `results/sensitivity/<dimension>/<model>_scan.csv`
- `results/sensitivity/<dimension>/<model>_<dimension>_sensitivity.png`

---

## 5. 消融实验（Ablation Study）

### 5.1 内置实验注册表
脚本 `scripts/run_ablation.py` 内置了以下消融配置：

| 实验名称 | 描述 | 配置覆盖 |
|----------|------|----------|
| `full_model` | 完整模型 | 无覆盖 |
| `no_cross_attn` | 去除跨模态注意力 | `cross_attn_dim: 0` |
| `no_aux_heads` | 去除辅助分类头 | 移除 auxiliary 配置 |
| `avg_pool_only` | 仅使用平均池化 | `pooling: avg` |
| `no_lite` | 关闭 Lite 模式，使用原始维度 | `lite.enabled: false` |

### 5.2 命令行用法

```bash
python scripts/run_ablation.py \
  --config configs/enhanced_config.yaml \
  --model AttentionMultimodal \
  --experiments full_model no_cross_attn no_lite \
  --lite \
  --seed 42 \
  --overwrite
```

### 5.3 输出
```
results/ablation/
  ablation_summary.csv
  full_model/
  no_cross_attn/
  no_lite/
```

---

## 6. 参数量 vs AUC 可视化

### 6.1 用途
验证**奥卡姆剃刀原则**：在 142 样本场景下，参数量越大的模型是否反而 AUC 更低？

### 6.2 命令行用法

```bash
python scripts/plot_param_vs_auc.py \
  --results_dir results \
  --output_path results/comparison/param_vs_auc.png
```

该脚本会自动递归读取：
- `*/metrics_summary.json`（单折）
- `*/seed_*/metrics_summary.json`（多种子）
- `*/cv_summary.json`（交叉验证）

并在图中用颜色区分来源，添加线性回归趋势线。

---

## 7. 结果聚合与对比（Results Aggregation）

### 7.1 功能
将分散在 `results/` 下的各模型结果统一汇总为一张对比表，支持：
- 自动识别单折 / 多种子 / CV 三种来源
- 对多种子结果计算 **mean ± std**
- 成对 t 检验（p-value 矩阵）
- 输出 CSV / Markdown / LaTeX 三种格式

### 7.2 命令行用法

```bash
python scripts/generate_main_results_table.py \
  --results_dir results \
  --output_dir results/comparison
```

### 7.3 输出
```
results/comparison/
  model_comparison.csv
  model_comparison.md
  model_comparison.tex
  pvalue_matrix.csv
```

---

## 8. 冒烟测试（Smoke Test）

```bash
python test_experiment_orchestration.py
```

测试内容：

| 编号 | 测试项 | 说明 |
|------|--------|------|
| 1 | 配置继承 | 验证 base + child YAML 合并正确，向后兼容单文件 |
| 2 | 多种子运行 | Mock 训练，验证目录结构、aggregate_seeds 汇总 |
| 3 | 敏感性扫描 | Mock 训练，验证 CSV + 图片生成 |
| 4 | 消融实验 | Mock 训练，验证 ablation_summary.csv |
| 5 | 参数量-AUC 图 | 假数据验证加载与绘图 |
| 6 | 结果聚合 | 假数据验证 CSV/Markdown 生成、多源识别 |

> **注意**：由于原数据层/训练层在特定 split 下存在偶发性挂起（与实验编排代码无关），冒烟测试对涉及训练的第 2–4 项采用 `unittest.mock` 注入假训练函数，以确保测试稳定、快速（< 30 秒）。

---

## 9. 关键约束与建议

1. **样本量**：142 样本意味着测试集仅 20–30 样本，AUC 标准差可能 > 0.08。不要因单次 AUC 0.62 vs 0.58 的微小差异做过度解读。
2. **Class Imbalance**：class 1 仅 4 样本。若任务允许，建议合并为 2 类（0+1 vs 2+3）或改用 Ordinal Regression。
3. **Lite 模式**：在 142 样本下，> 100k 参数的模型必然过拟合。`--lite` 是强制选项。
4. **可视化/可解释性开销**：SHAP、PCA、t-SNE 在 CPU 上非常慢。大规模批量实验时，建议在 config 中关闭 `interpretability.enable` 和 `visualization.enable`。
5. **Embedding vs Raw**：当前实验以 Embedding 模式为主。若修改 Raw 模式相关代码，请同时验证两条路径。

---

## 10. 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `configs/experiment_base.yaml` | 新建 | 统一基础配置 |
| `configs/enhanced_config.yaml` | 重写 | 继承 base |
| `configs/tft_config.yaml` | 重写 | 继承 base |
| `configs/mmtm_config.yaml` | 重写 | 继承 base |
| `enhanced_main.py` | 修改 | `merge_dicts`、`load_config`、`build_model` num_classes 修复、metrics_summary 增强 |
| `scripts/run_multimodal_experiments.py` | 重写 | 多种子 / CV / Legacy 模式 |
| `scripts/run_sensitivity_analysis.py` | 新建 | 参数敏感性扫描 |
| `scripts/plot_param_vs_auc.py` | 新建 | 参数量-AUC 散点图 |
| `scripts/run_ablation.py` | 新建 | 消融实验注册表与执行 |
| `scripts/generate_main_results_table.py` | 重写 | 多源读取、统计检验、多格式输出 |
| `test_experiment_orchestration.py` | 新建 | 6 项冒烟测试 |
| `EXPERIMENT_ORCHESTRATION.md` | 新建 | 本文档 |
