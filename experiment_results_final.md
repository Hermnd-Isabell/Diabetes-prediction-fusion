# Raw Mode Experiment Results - Final Report

**Date**: 2026-05-11
**Dataset**: 105 patients (after intersection), 4-class classification
**Mode**: Raw (AsLS baseline correction + SNV + sequence aggregation)
**Device**: CPU only

---

## 1. Single-Fold Results (Raw Mode)

| Model | Test AUC | Test Acc | Parameters | Status |
|-------|----------|----------|------------|--------|
| **ConcatFusion** | **0.439** | ~0.41 | ~500K | Completed |
| AttentionMultimodal | 0.348 | ~0.35 | 2.18M | Completed |
| TFTMultimodal | N/A | N/A | 8.23M | **OOM Crash** |

**TFTMultimodal Crash Analysis**:
Model attempts to allocate 9.4GB memory during forward pass. Raw mode input `[B=8, S=180, L=800]` causes the Transformer encoder to create tensors of shape `[B*S, L, d_model]`, which explodes memory usage. This is an **architectural incompatibility** - TFTMultimodal is designed for embedding mode only.

---

## 2. 5-Fold Cross-Validation Results (Raw Mode)

**Model**: AttentionMultimodal
**Config**: `configs/full_cv_raw_cpu.yaml`

| Metric | Mean | Std | Notes |
|--------|------|-----|-------|
| test_auc | 0.4473 | 0.1845 | High variance across folds |
| test_macro_auc | **0.5665** | 0.0335 | More stable than simple AUC |
| test_weighted_auc | 0.5646 | 0.0335 | |
| test_acc | 0.4667 | 0.0356 | Near random (4-class = 25%) |
| test_f1 | 0.4506 | 0.0510 | |
| test_macro_f1 | 0.2919 | 0.0714 | Very poor |
| best_val_auc | 0.7238 | 0.1274 | Severe overfitting |
| test_cohens_kappa | 0.0910 | 0.0938 | Near chance agreement |
| test_qwk | 0.1507 | 0.2453 | Poor ordinal agreement |

**Key Observation**: CV results show severe overfitting - best validation AUC (0.72) far exceeds test AUC (0.45), confirming the model memorizes training data but fails to generalize.

---

## 3. Baseline Comparison

| Model | Mode | Test AUC / Macro-AUC | Status |
|-------|------|---------------------|--------|
| **LightGBM** | 4-class tabular | **0.579** | Completed (best overall) |
| ConcatFusion | Raw multimodal | 0.439 | Completed |
| AttentionMultimodal (CV) | Raw multimodal | 0.447 | Completed |
| AttentionMultimodal | Raw multimodal | 0.348 | Completed |
| TFTMultimodal | Embedding | 0.107 | Completed |
| TFTMultimodal | Raw | N/A | OOM |

---

## 4. Key Findings

### 4.1 Deep Learning Underperforms Baseline
On this 142-sample dataset:
- **LightGBM** (macro-auc: 0.579) outperforms all DL models
- Best DL model (ConcatFusion raw: 0.439) is 24% worse than LightGBM
- Complex models (AttentionMultimodal 2.18M params, TFTMultimodal 8.23M params) severely overfit

### 4.2 Severe Overfitting
Training metrics from CV logs:
- Training AUC rapidly approaches 0.93-0.98
- Validation AUC plateaus around 0.50-0.65
- Test AUC collapses to 0.35-0.45

### 4.3 Architectural Issues
- **TFTMultimodal** incompatible with raw mode due to memory requirements
- **AttentionMultimodal** works but overfits heavily with 2M+ parameters on 73 training samples
- **ConcatFusion** (simplest model) achieves best DL performance, suggesting model complexity hurts generalization

### 4.4 Data Constraints
- Class 1 has only 3-4 samples across folds
- 4-class classification effectively impossible for minority class
- Raw mode preprocessing (AsLS) adds ~8-10 minutes per experiment startup but doesn't improve results

---

## 5. Recommendations

1. **Accept LightGBM as baseline**: At 0.579 macro-auc, it's the practical ceiling for this dataset size
2. **Switch to binary classification**: Merge classes (e.g., 0+1 vs 2+3) to handle class imbalance
3. **Consider ordinal regression**: Classes may have natural ordering (0 < 1 < 2 < 3)
4. **Use embedding mode for speed**: Raw mode adds significant preprocessing time without accuracy benefit
5. **If continuing DL work**:
   - Heavy regularization (dropout >= 0.5)
   - Transfer learning from public Raman datasets
   - Merge classes to reduce complexity
   - Use much smaller models (<100K parameters)

---

## 6. Files Generated

- `results/AttentionMultimodal/metrics_summary.json`
- `results/ConcatFusion/metrics_summary.json`
- `results/TFTMultimodal/metrics_summary.json` (embedding mode)
- `results_cv/AttentionMultimodal/cv_summary.json`
- `results_cv/AttentionMultimodal/fold_results.csv`
- `clinic_dimension/outputs-4分类/metrics.json` (LightGBM)

---

## 7. Bug Fixes Applied During This Run

1. `datasets/raman_dataset.py`: Optimized AsLS baseline correction (CSC precomputation)
2. `enhanced_main.py`: Fixed `DualLogger` pipe blocking issue for background processes
3. `trainers/enhanced_trainer.py`: Fixed multiclass ROC curve comparison in `compare_models`
4. `models/tft_models.py`: Added contrastive loss dimension guard for embedding mode

---

*Report generated automatically after raw-mode experiment completion.*
