# 🫁 ICBHI 2017 Respiratory Sound Classification

> A two-level classification pipeline for lung health detection using Deep Learning and Machine Learning on the ICBHI 2017 Challenge dataset.

---

## 📌 Overview

This project implements a full end-to-end pipeline to classify respiratory sounds at two levels:

| Level | Task | Classes |
|-------|------|---------|
| **Level 1** | Patient health status | `Healthy` / `Unhealthy` |
| **Level 2** | Sound anomaly detection | `None` / `Crackle` / `Wheeze` / `Both` |

Two notebooks are included, each approaching the problem with different model families and feature representations.

---

## 📂 Repository Structure

```
├── ml-updated-withml-dl.ipynb       # DL + ML combined pipeline (primary)
├── notebook-ml-final.ipynb          # ML-only pipeline with K-Means clustering
└── README.md
```

---

## 📓 Notebook Descriptions

### 1. `ml-updated-withml-dl.ipynb` — Full DL + ML Pipeline

A comprehensive pipeline combining Deep Learning image classifiers and traditional ML models.

**Pipeline stages:**
1. Install dependencies & GPU check
2. Imports & global configuration
3. Dataset inspection & path auto-detection
4. Audio preprocessing → 2.5s overlapping segments
5. Patient-stratified train/test split
6. DL image generation (Mel-Spectrogram, MFCC, Recurrence Plot)
7. Sample image visualization per level/sub-level per model
8. **DL models**: ResNet18, MobileNetV2 — trained on both levels with full evaluation metrics & training curves
9. **ML models**: Random Forest, Extra Trees, XGBoost, SVM, KNN, Logistic Regression, Decision Tree — using tabular MFCC + Recurrence Plot features
10. Output summary with leaderboard

**Key design choices:**
- Audio resampled to **4 kHz**, bandpass filtered (50–1800 Hz), RMS-normalised
- Segments: **2.5 seconds**, 50% overlap
- All metrics (Accuracy, F1, Precision, Recall) derived strictly from confusion matrix
- Cross-validation accuracy curves, per-class bar charts, model comparison leaderboard
- Results saved to `ml_results/`

---

### 2. `notebook-ml-final.ipynb` — ML + K-Means Clustering Pipeline

Focuses on traditional ML classification with unsupervised K-Means analysis.

**Pipeline stages:**
1. Feature extraction: **80-dimensional MFCC vectors** (40 mean + 40 std) at 22050 Hz
2. Stratified 80/20 train-test split for both levels
3. **ML classifiers**: Random Forest, XGBoost, LightGBM, SVM (RBF)
4. Confusion matrix PNGs saved per model per level
5. **K-Means clustering** with elbow-method + silhouette score selection
6. PCA (2D, 50D) and t-SNE projections for cluster visualization
7. Cluster composition heatmaps vs. ground-truth labels
8. Centroid feature heatmaps (MFCC mean & std)
9. All outputs saved to `/kaggle/working/results/`

---

## 🗂️ Dataset

**ICBHI 2017 Challenge Respiratory Sound Database**

| Property | Detail |
|----------|--------|
| Source | [Kaggle — nimalanparameshwaran](https://www.kaggle.com/datasets/nimalanparameshwaran/icbhi-2017-challenge-respiratory-sound-database) |
| Diagnosis labels | [Kaggle — santhoshsanka/patient-diagnosis](https://www.kaggle.com/datasets/santhoshsanka/patient-diagnosis) |
| Audio format | `.wav` with `.txt` annotation files |
| Annotations | Per-breath cycle: start time, end time, crackle flag, wheeze flag |

Expected Kaggle input paths:
```
/kaggle/input/datasets/nimalanparameshwaran/icbhi-2017-challenge-respiratory-sound-database/ICBHI_final_database/
/kaggle/input/datasets/santhoshsanka/patient-diagnosis/patient_diagnosis.csv
```

---

## ⚙️ Setup & Requirements

### Dependencies

```bash
pip install librosa soundfile scipy pywavelets pyts \
            scikit-learn xgboost lightgbm imbalanced-learn \
            tqdm matplotlib pillow torch torchvision torchaudio \
            seaborn
```

### Hardware
- GPU strongly recommended for the DL notebook (ResNet18, MobileNetV2 training)
- CPU-only is feasible for the ML-only notebook

---

## 🧪 Models & Methods

### Deep Learning (Notebook 1)

| Model | Input | Levels |
|-------|-------|--------|
| ResNet18 | Mel-Spectrogram / MFCC / Recurrence Plot images (128×128) | L1 & L2 |
| MobileNetV2 | Mel-Spectrogram / MFCC / Recurrence Plot images (128×128) | L1 & L2 |

Hyperparameters: `batch=32`, `epochs=25`, `lr=1e-3`, `image_size=128`, `n_mfcc=40`

### Machine Learning (Both Notebooks)

| Model | Notebook |
|-------|----------|
| Random Forest | Both |
| Extra Trees | Notebook 1 |
| XGBoost | Both |
| LightGBM | Notebook 2 |
| SVM (RBF) | Both |
| KNN | Notebook 1 |
| Logistic Regression | Notebook 1 |
| Decision Tree | Notebook 1 |

### Unsupervised (Notebook 2)

- **K-Means Clustering** with PCA (50 components) preprocessing
- Optimal k selected via elbow method + silhouette score
- Visualized with PCA 2D and t-SNE projections

---

## 📊 Outputs

### Notebook 1 (`ml_results/`)
- Per-model metrics CSVs
- CV accuracy curves
- Per-class bar charts
- Model comparison leaderboard

### Notebook 2 (`/kaggle/working/results/`)
- `all_metrics.csv` — accuracy & weighted F1 for all models × both levels
- `cm_<level>_<model>.png` — confusion matrix heatmaps
- `kmeans_pca2d.png` — K-Means clusters in PCA 2D space
- `tsne_projections.png` — t-SNE colored by L1, L2, and cluster labels
- `cluster_comp_<level>_k<k>.png` — cluster composition vs. ground truth
- `centroid_heatmap_k<k>.png` — MFCC centroid feature heatmaps
- `cluster_assignments.csv` — per-sample cluster + ground-truth labels

---

## 🔬 Audio Preprocessing Details

| Parameter | Notebook 1 | Notebook 2 |
|-----------|-----------|-----------|
| Sample rate | 4,000 Hz | 22,050 Hz |
| Segment length | 2.5 s | Per annotation cycle |
| Overlap | 50% | — |
| Bandpass filter | 50–1800 Hz | — |
| Normalization | RMS | — |
| Features (ML) | MFCC + Recurrence Plot | 40 MFCC (mean + std) → 80-dim |

---

## 🚀 Running on Kaggle

1. Add both datasets to your Kaggle notebook (see dataset links above)
2. Enable GPU accelerator (recommended for Notebook 1)
3. Run all cells top to bottom — intermediate outputs are cached to avoid re-computation on reruns

---

## 📄 License

This project is for research and educational purposes. The ICBHI dataset is subject to its own terms of use — please refer to the [official ICBHI 2017 challenge](https://bhichallenge.med.auth.gr/) for details.

---

## 🙏 Acknowledgements

- ICBHI 2017 Scientific Challenge organizers
- Kaggle dataset contributor: [nimalanparameshwaran](https://www.kaggle.com/nimalanparameshwaran) 
