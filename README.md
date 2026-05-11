# Multimodal Diabetes Prediction via Deep Learning

This repository implements a PyTorch-based deep learning system for multimodal diabetes prediction, fusing Raman spectroscopy with clinical tabular data. It also contains an independent LightGBM baseline under `clinic_dimension/`.

> **Data constraint**: The primary dataset contains **142 patients** with **4-class labels** (0: 22, 1: 4, 2: 46, 3: 70). Class 1 has only 4 samples. All deep learning models are trained under severe data scarcity and exhibit significant overfitting compared to the LightGBM baseline.

---

## 1. Dataset Specification

### Raw Data Files

| File | Shape | Description |
|------|-------|-------------|
| `data/spectra.csv` | ~18,106 rows x ~802 cols | One row per scan. Columns: `Sample` (patient-scanID, e.g. `100-176`), `Group` (label), and wavelength columns (~800 points per scan). |
| `data/clinical.csv` | 142 rows x ~81 cols | One row per patient. Columns: `PatientID`, label (`Label` or `Group`), and ~79 clinical features. |

### Two Data Loading Modes

The training script `enhanced_main.py::prepare_data()` supports two mutually exclusive modes controlled by `cfg["data"]["use_embedding"]`:

**Raw mode** (`use_embedding: false`)
- Uses `datasets/raman_dataset.py::RamanDataset`.
- Groups scan-level spectra into patient-level tensors of shape `[num_scans, num_wavelengths]`.
- Variable-length scan sequences are padded/truncated to `max_scans` (default 180).
- `collate_fn` handles batching of variable-length scan counts with padding masks.
- Label column inference: if the configured `label_col` is missing from `clinical.csv`, the code silently falls back to `"Label"` then `"Group"`. This can cause silent label mismatches.

**Embedding mode** (`use_embedding: true`)
- Uses `datasets/embedding_dataset.py::EmbeddingMultimodalDataset`.
- Loads pre-computed single-vector embeddings from `spectrum_embedding_path` and `clinical_embedding_path`.
- Expects CSVs with a patient ID column and an embedding vector per row.
- Splits are pre-defined by a `split` column (`train`/`val`/`test`) rather than random split.
- This is the mode used by recent experiments.

### Preprocessing Pipeline

`datasets/raman_dataset.py::preprocess_spectrum` applies:
1. **AsLS baseline correction** (`lam=1e6, p=0.001, niter=10`) via `scipy.sparse`
2. **Savitzky-Golay smoothing** (`window=11, polyorder=2`)
3. **SNV normalization** (per-spectrum mean subtraction and division by std)

---

## 2. Model Architectures

All models are instantiated in `enhanced_main.py::build_model()`. The table below describes the *actual implementation* as found in the source files, not aspirational designs.

| Model | Source File | Architecture Summary |
|-------|-------------|---------------------|
| `Spectra-only` | `models/Baseline.py` | `SpectraOnlyModel`: `input_dim -> Linear(256) -> LayerNorm -> ReLU -> Linear(hidden_dim) -> LayerNorm -> ReLU -> classifier` |
| `Clinical-only` | `models/Baseline.py` | `TabularOnlyModel`: same pattern as above for tabular data. |
| `ConcatFusion` | `models/Baseline.py` | Concatenates spec embedding + tab embedding -> `Linear(256) -> ReLU -> classifier`. |
| `EnsembleFusion` | `models/Baseline.py` | Separate unimodal heads + learnable fusion gate (softmax weights) that combines unimodal logits. |
| `BaselineMultimodal` | `models/Baseline.py` | Wrapper selecting `fusion_type="concat"` or `"ensemble"`. |
| `AttentionMultimodal` | `models/attention_models.py` | Multi-scale 1D-CNN encoder (kernels 7, 5, 3) with residual blocks -> attention pooling over scans -> `EnhancedCrossAttentionFusion` (bidirectional multi-head cross-attention, 8 heads, hid_dim=256) -> `EnhancedClassifier` (default `[512, 256, 128]`). Supports auxiliary supervision heads for spectra and tabular branches. |
| `TFTMultimodal` | `models/tft_models.py` | `SpectraTFTEncoder`: `Linear(input_dim, d_model=256)` + learnable positional encoding + `TransformerEncoderLayer` (nhead=8, num_layers=3, gelu) + multi-scale 1D conv (kernels 3, 5, 7) + attention pooling. `TabularStaticEncoder`: feature selection gate (Sigmoid) + MLP. `CrossModalAttention` (bidirectional) + `EnhancedGating` + multi-layer fusion. `TFTLoss`: main CE + 2 auxiliary CE + contrastive loss. |
| `EnhancedMMTM` | `models/enhanced_mmtm_models.py` | `MultiHeadCrossModalAttention` (self-attention enhancement per modality, 8 heads) -> `AdaptiveGating` (bottleneck projection to 128-D, 3-layer interaction net, temperature-scaled sigmoid gates) -> `HierarchicalFusion` (3 scales: global/meso/local, each with Hadamard product + concat) -> classifier with optional `uncertainty_head`. |

### Model Output Contract

All models must return a `dict` from `forward()` containing at minimum:
- `"logits"`: `Tensor[B, num_classes]` — the main classification output.

Optionally returned for visualization/training:
- `"embedding"`: fused feature vector for t-SNE/PCA
- `"spec_embedding"` / `"tab_embedding"`: unimodal embeddings
- `"aux_spec_logits"` / `"aux_tab_logits"`: auxiliary task outputs (used by AttentionMultimodal and TFT)
- `"attention_weights"`: for attention heatmap visualization
- `"gated_spec"`: for gating analysis

### Parameter Counts (Empirical)

| Model | Approx. Parameters |
|-------|-------------------|
| `Spectra-only` / `Clinical-only` | ~100K |
| `ConcatFusion` / `EnsembleFusion` | ~200K |
| `AttentionMultimodal` | ~500K–800K (depends on `hidden_dims`) |
| `TFTMultimodal` | ~600K–900K |
| `EnhancedMMTM` (hierarchical) | ~1.35M |
| `EnhancedMMTM` (interaction) | ~898K |

---

## 3. Training & Evaluation

### Entry Point: `enhanced_main.py`

```bash
# Single model training
python enhanced_main.py --config configs/enhanced_config.yaml --model AttentionMultimodal

# Train all models listed in config's `models_to_train`
python enhanced_main.py --config configs/enhanced_config.yaml --train-all

# Resume from checkpoint (loads `results/<model_name>/best_model.pt`)
python enhanced_main.py --config configs/enhanced_config.yaml --model AttentionMultimodal --resume

# Evaluation-only mode
python enhanced_main.py --config configs/enhanced_config.yaml --eval-only AttentionMultimodal
```

**Configuration files:**
- `configs/enhanced_config.yaml` — default for AttentionMultimodal / Baseline
- `configs/tft_config.yaml` — TFT-specific settings
- `configs/mmtm_config.yaml` — EnhancedMMTM-specific settings

Key config fields:
- `data.use_embedding` — toggles between raw and embedding mode
- `data.modality_dropout` — `{"spectra": 0.15, "clinical": 0.10}`; **only applied to training split**
- `train.use_class_weights` — computes inverse-frequency weights from training distribution
- `train.use_augmentation` — Gaussian noise + intensity scaling (training split only)
- `model.num_classes` — must match dataset; code auto-overrides from `dataset_info` unless `--resume`

### Trainer: `trainers/enhanced_trainer.py::EnhancedTrainer`

Core training loop:
- **Optimizer**: AdamW (`lr` from config, default 1e-3)
- **Scheduler**: `ReduceLROnPlateau(mode='max', patience=5)` monitoring validation AUC
- **Loss**: `nn.CrossEntropyLoss` (with optional `weight` tensor for class balancing)
- **Early stopping**: based on validation AUC; patience configurable (default 10–15)
- **Metrics computed**: Accuracy, weighted F1, AUC (one-vs-rest for label=1 in multiclass), sensitivity@90% specificity

**Important metric behavior**: For multiclass (>2 classes), AUC and sensitivity@90%spec are computed as **one-vs-rest for label=1** (or the maximum label if 1 is absent). This is a hardcoded simplification in `_calculate_metrics()`.

### Output Directory Structure

After training, `results/<model_name>/` contains:

```
results/
├── <model_name>/
│   ├── best_model.pt                 # checkpoint (weights + history dict)
│   ├── metrics_summary.json          # unified metrics for aggregation
│   ├── results.json                  # detailed results (metrics + classification_report)
│   ├── training_curves.png           # loss/acc/auc/f1 per epoch
│   ├── evaluation_plots.png          # ROC, PR, confusion matrix, probability distribution
│   ├── pca_analysis.png              # PCA/t-SNE of embeddings
│   ├── shap_analysis.png             # SHAP feature importance (gradient-based approximation)
│   ├── feature_importance.png        # alternate feature importance plot
│   └── modality_gate_history.json    # epoch-by-epoch fusion gate weights (if model has fusion_gate)
└── comparison/
    ├── model_comparison.csv
    ├── model_comparison.png
    └── roc_comparison.png
```

**Numpy compatibility note**: `enhanced_trainer.py` contains a compatibility hack at import time that monkey-patches `numpy._core` to `numpy.core` to allow loading checkpoints saved with numpy 2.x under numpy 1.x environments.

---

## 4. Experiment Automation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_multimodal_experiments.py` | Batch-trains multiple models sequentially and aggregates `metrics_summary.json` into a single CSV. Supports `--embedding_only` and `--models <list>`. |
| `scripts/generate_main_results_table.py` | Reads all `results/*/metrics_summary.json` files and produces a comparison markdown/CSV table. |
| `scripts/plot_modality_gates.py` | Loads `modality_gate_history.json` and plots gate weight trajectories across epochs. |

---

## 5. Ablation Design

The following table is reproduced from `results/experiment_design_table.md` and shows which architectural components are active per model in the current embedding-mode experiments:

|                     | Raw Input | Embedding | Soft Gating | Fusion Gate | Modality Dropout | MMTM Block |
|:--------------------|:----------|:----------|:------------|:------------|:-----------------|:-----------|
| `BaselineMultimodal`| ✗         | ✔         | ✔           | ✔           | ✔                | ✗          |
| `AttentionMultimodal`| ✗        | ✔         | ✔           | ✔           | ✔                | ✗          |
| `EnhancedMMTM`      | ✗         | ✔         | ✔           | ✔           | ✔                | ✔          |
| `TFTMultimodal`     | ✗         | ✔         | ✔           | ✔           | ✔                | ✗          |
| `ConcatFusion`      | ✗         | ✔         | ✗           | ✔           | ✔                | ✗          |
| `EnsembleFusion`    | ✗         | ✔         | ✗           | ✔           | ✔                | ✗          |

- **Raw Input**: whether the model consumes raw spectral sequences (`[B, S, L]`)
- **Embedding**: whether the model consumes pre-computed embedding vectors
- **Soft Gating**: whether the model uses learnable soft gates to weight modalities
- **Fusion Gate**: whether a fusion mechanism combines modalities
- **Modality Dropout**: whether random modality dropout is applied during training
- **MMTM Block**: whether the MMTM transfer module is used

---

## 6. Performance Baseline (Empirical)

The following are approximate test-set results from `results/*/metrics_summary.json` on the 142-sample dataset. Test sets contain roughly 20–30 samples; metrics are noisy.

| Model | Test AUC | Test Accuracy | Notes |
|-------|----------|---------------|-------|
| `Spectra-only` | 0.786 | 0.364 | Best DL AUC; high variance due to small test set |
| `Clinical-only` | 0.652 | 0.529 | |
| `BaselineMultimodal` | 0.643 | 0.364 | |
| `AttentionMultimodal` | 0.500 | 0.588 | |
| `TFTMultimodal` | 0.424 | 0.412 | |
| `EnhancedMMTM` | 0.379 | 0.412 | Most complex model; worst generalization |
| **LightGBM (clinic_dimension)** | **0.790** | **0.828** | Independent binary task (complications); not directly comparable but indicative of dataset ceiling |

**Observation**: More complex multimodal architectures consistently underperform simpler unimodal baselines and the LightGBM baseline due to severe overfitting on 142 samples.

---

## 7. Independent Subsystem: `clinic_dimension/`

A complete LightGBM pipeline for binary classification of diabetes complications, completely separate from the PyTorch code.

```bash
cd clinic_dimension
pip install -r requirements.txt
python code/run_all.py
```

- **Input**: `原始数据/糖尿病标签2分类.xlsx` (first column = label 0/1)
- **Pipeline**: `data_cleaning.py` -> `feature_processing.py` -> `model_train.py` -> `evaluate_visualize.py`
- **Output**: `outputs-有无并发/` or `outputs-有无精神/` depending on target configured in `code/config.py`
- **Artifacts**: `cleaned_data.csv`, `model_pipeline.joblib`, `metrics.json`, ROC/PR/confusion matrix plots

---

## 8. Known Issues & Constraints

1. **Sample size**: 142 total samples. Test metrics are statistically unstable. Do not interpret small AUC differences between models as meaningful.
2. **Class imbalance**: Class 1 has 4 samples (2.8%). 4-class classification is effectively unlearnable for this class. Consider merging classes (e.g., 0+1 vs 2+3) or ordinal regression.
3. **Overfitting**: Complex models (AttentionMultimodal, EnhancedMMTM, TFTMultimodal) have 500K–1.35M parameters. With 142 samples they memorize the training set. Current regularization (dropout 0.1, weight decay 1e-4) is insufficient.
4. **Label inference hazard**: `RamanDataset` silently changes `label_col` if the configured column is absent. Always verify `Detected Clinical Label` printed in stdout.
5. **AUC simplification for multiclass**: The trainer computes AUC as one-vs-rest for a single positive label (priority: 1, fallback: max label). This is not a true multiclass AUC.
6. **ConcatFusion result missing**: `results/ConcatFusion/` exists but is empty in the current workspace; no metrics are available.
