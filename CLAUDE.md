# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **multimodal diabetes prediction** project combining Raman spectroscopy data with clinical tabular features. It contains two independent subsystems:

1. **Deep Learning multimodal fusion** (`enhanced_main.py`, `models/`, `datasets/`, `trainers/`)
2. **Traditional ML pipeline** (`clinic_dimension/` — LightGBM for binary classification of complications)

The DL subsystem is severely data-constrained: **142 total samples**, 4-class classification with extreme imbalance (class 1 has only 4 samples). Deep learning models systematically underperform compared to the LightGBM baseline (AUC ~0.79 vs DL best ~0.64). Any architectural changes to DL models must account for this constraint.

## Architecture

### Data Flow

There are **two data loading modes** controlled by `cfg["data"]["use_embedding"]`:

- **Raw mode** (`use_embedding: false`): Loads `spectra.csv` (one row per scan, many scans per patient) and `clinical.csv` (one row per patient) via `RamanDataset`. Each patient has variable-length scan sequences padded to `max_scans` (~180).
- **Embedding mode** (`use_embedding: true`): Loads pre-computed single-vector embeddings from `spectrum_embedding_path` and `clinical_embedding_path` via `EmbeddingMultimodalDataset`. This is the mode used by current experiments.

`enhanced_main.py::prepare_data()` handles both paths and returns `(train_loader, val_loader, test_loader, dataset_info)`.

### Model Layer

All models are instantiated in `enhanced_main.py::build_model()`:

| Model | File | Key Characteristic |
|-------|------|-------------------|
| `Spectra-only` / `Clinical-only` | `models/Baseline.py` | Single-modality baselines |
| `ConcatFusion` / `EnsembleFusion` | `models/Baseline.py` | Simple concat or ensemble of unimodal heads |
| `BaselineMultimodal` | `models/Baseline.py` | Wrapper selecting fusion type |
| `AttentionMultimodal` | `models/attention_models.py` | Cross-attention + multi-scale CNN spectra encoder |
| `TFTMultimodal` | `models/tft_models.py` | Transformer with temporal fusion on spectral sequence |
| `EnhancedMMTM` | `models/enhanced_mmtm_models.py` | Multi-head cross-attention + adaptive gating + hierarchical fusion |

Model dimensions are dynamic: `spec_emb_dim` and `tab_emb_dim` are inferred from the dataset in embedding mode, or taken from config in raw mode.

### Training Layer

`trainers/enhanced_trainer.py::EnhancedTrainer` is the unified training loop used by all models. Key behaviors:
- Optimizer: AdamW with ReduceLROnPlateau scheduler
- Loss: CrossEntropyLoss (with optional class-weighting via `use_class_weights`)
- Early stopping based on validation AUC
- Automatic visualization (ROC, PR, confusion matrix, SHAP, PCA/t-SNE, training curves) saved to `results/<model_name>/`
- Metrics summary written to `results/<model_name>/metrics_summary.json`

There is a numpy 2.0→1.x compatibility hack at the top of `enhanced_trainer.py` for loading checkpoints across numpy versions.

### Experiment Orchestration

- `scripts/run_multimodal_experiments.py` runs systematic ablation studies across feature combinations
- `scripts/generate_main_results_table.py` aggregates `metrics_summary.json` files into comparison CSVs
- `scripts/plot_modality_gates.py` visualizes gating weights from attention-based models

## Common Commands

### Deep Learning (root directory)

```bash
# Train a single model
python enhanced_main.py --config configs/enhanced_config.yaml --model AttentionMultimodal

# Train all models in config
python enhanced_main.py --config configs/enhanced_config.yaml --train-all

# Resume from checkpoint
python enhanced_main.py --config configs/enhanced_config.yaml --model AttentionMultimodal --resume

# Evaluate a saved model without training
python enhanced_main.py --config configs/enhanced_config.yaml --eval-only AttentionMultimodal
```

Config files available: `configs/enhanced_config.yaml`, `configs/tft_config.yaml`, `configs/mmtm_config.yaml`.

### Traditional ML (`clinic_dimension/`)

```bash
cd clinic_dimension
pip install -r requirements.txt
python code/run_all.py
```

Outputs go to `outputs-有无并发/` or `outputs-有无精神/` depending on the target label configured in `code/config.py`.

## Key Configuration Notes

- YAML configs control everything: model name, data paths, train/val/test ratios, batch size, learning rate, class weights, modality dropout, augmentation
- `modality_dropout` (spectra/clinical) is **only applied to the training split**; val/test get zero dropout
- When `use_class_weights: true`, weights are auto-computed as inverse frequency from the training split distribution
- Results directories are auto-created under `results/<model_name>/` (or `results/comparison/` for multi-model runs)

## Critical Constraints for Development

1. **Sample size**: 142 samples total. Test set metrics are statistically unstable (~20-30 samples). Do not trust small AUC swings between models.
2. **Class imbalance**: Class 1 has 4 samples. 4-class classification is effectively impossible for minority class. Consider merging classes (e.g., 0+1 vs 2+3) or switching to ordinal regression.
3. **Overfitting**: Complex multimodal models (AttentionMultimodal, EnhancedMMTM) have 500k+ parameters. They will memorize the training set. Regularization, heavy dropout (≥0.5), and transfer learning from public datasets are essential if continuing DL work.
4. **Embedding mode vs Raw mode**: Most recent experiments use embedding mode. If modifying `RamanDataset` or raw-mode preprocessing, verify both code paths in `prepare_data()`.
5. **Label column inference**: `RamanDataset` auto-detects clinical label columns by trying `"Label"` then `"Group"` if the configured `label_col` is missing. This can cause silent mismatches between spectra and clinical labels if column names are inconsistent.

## File Structure Convention

- `data/` — Raw input: `spectra.csv`, `clinical.csv`
- `configs/` — YAML experiment configs
- `models/` — PyTorch model definitions
- `datasets/` — PyTorch Dataset classes and collate functions
- `trainers/` — Training loops and evaluation logic
- `multimodal/` — Embedding loaders and alignment utilities
- `scripts/` — Experiment automation and result aggregation
- `results/` — DL outputs (models, plots, metrics JSON)
- `clinic_dimension/` — Independent LightGBM pipeline with its own `code/`, `outputs*/`, `requirements.txt`
