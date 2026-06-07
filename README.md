# Reproducible Machine Learning Benchmark: Parkinson’s Disease and Forest Cover Type Classification

[![CI](https://github.com/diogoalexandreramalho/data-science/actions/workflows/ci.yml/badge.svg)](https://github.com/diogoalexandreramalho/data-science/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This repository presents a reproducible comparative study of machine learning models on two contrasting tabular classification tasks: 
Parkinson’s disease detection from high-dimensional speech features and forest cover type prediction from cartographic data. 
The full report is in [`reports/report.pdf`](reports/report.pdf).

## Overview

### Datasets


| | Parkinson's Disease | Forest Covertype |
|---|---|---|
| **Task** | Binary classification | Multiclass (7 classes) |
| **Instances** | 756 | 581,012 (subsampled to 7,000 balanced) |
| **Features** | 754 | 54 (10 continuous, 44 binary) |
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification) | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Covertype) |
| **Primary metric** | F1 (positive class) | Macro F1 |
| **CV** | 10-fold stratified group (patient-aware) | 10-fold stratified |


**Parkinson's Disease** — speech recordings from 188 PD patients and 64
healthy controls (Istanbul University). 754 features extracted via MFCCs,
wavelet transforms, vocal fold features, and Tunable Q-Factor Wavelet
Transform (TQWT). Patient-aware cross-validation prevents leakage from
multiple recordings per patient.

**Forest Covertype** — cartographic data from the US Forest Service for
30×30 m cells. 10 continuous features (elevation, slope, distances) plus
44 one-hot encoded soil-type and wilderness-area indicators. Target is one
of 7 tree species. The full dataset is heavily imbalanced; experiments use
a balanced subsample of 1,000 records per class (7,000 total).

### Methodology

A three-stage pipeline evaluates six classical classifiers — Naïve Bayes,
kNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost — across
multiple preprocessing configurations (raw, scaled, scaled + feature
selection, scaled + PCA on PD).

1. **Stage 1** — 10-fold CV of every (preprocessing × classifier) cell at
   default hyperparameters. Selects each classifier's best preprocessing.
2. **Stage 2** — per-classifier grid search on its Stage-1-best preprocessing.
   Selects each classifier's best hyperparameters.
3. **Final** — refit the overall winner on the full training set, evaluate
   on a held-out 20% test split.

In parallel, a **sweeps** step produces the report's analytical figures
(feature-selection sweeps, PCA sweeps, per-classifier hyperparameter sweeps).
Patient-grouped CV on Parkinson's prevents leakage from multiple recordings
per patient. Random seed = 42 throughout.


### Results

Cross-validation primary-metric scores after Stage-2 hyperparameter tuning,
per classifier and dataset.

| Classifier | Parkinson's (F1) | Covertype (Macro F1) |
|---|---|---|
| Naïve Bayes | 0.8424 | 0.5528 |
| Decision Tree | 0.8439 | 0.7365 |
| kNN | 0.8859 | 0.7885 |
| Random Forest | 0.8949 | 0.8240 |
| XGBoost | 0.8880 | 0.8281 |
| **Gradient Boosting (chosen)** | **0.8997** | **0.8285** |

**Final held-out test evaluation:**

| Metric | Parkinson's | Covertype |
|---|---|---|
| Accuracy | 0.850 | 0.827 |
| F1 (primary) | 0.905 | 0.824 (macro) |
| ROC-AUC | 0.938 | n/a (multiclass) |

The chosen model on both datasets is **Gradient Boosting** with
`learning_rate=0.1`, `max_features=sqrt`; on Parkinson's `scaled` features,
`max_depth=5`, `n_estimators=50`; on Covertype `raw` features,
`max_depth=10`, `n_estimators=200`.

## Reproduce the report

This repository regenerates everything in `reports/report.pdf` from scratch.
Prerequisites: [`uv`](https://docs.astral.sh/uv/) for Python environment
management, and a TeX distribution providing `pdflatex` for the PDF.

```bash
make install     # uv sync --frozen
make download    # fetch both UCI datasets into data/raw/
make reproduce   # stage_1 -> sweeps -> stage_2 -> final, for both datasets, then build PDF
```

`make reproduce` runs end-to-end (CV preprocessing matrix → preprocessing and
classifier-hyperparameter sweeps → grid-search tuning → held-out evaluation
→ LaTeX compile). Roughly 30–60 minutes on a modern laptop; Covertype's
Stage 2 + classifier sweeps dominate.

You can also run each stage individually:

```bash
make stage1      # 24 (PD) + 18 (CT) preprocessing-matrix CV runs
make sweeps      # Feature-selection sweep, PCA sweep (PD), per-classifier hyperparam sweeps
make stage2      # Grid search on each classifier's Stage-1-best preprocessing
make final       # Refit Stage-2 winner on full train, evaluate on held-out test
make report      # pdflatex reports/report.tex (two passes)
make test        # uv run pytest
```

The pipeline is also addressable directly via the installed CLI:

```bash
uv run data-science --help
uv run data-science download --dataset PD
uv run data-science stage-1 --config configs/parkinsons.yaml
uv run data-science sweeps  --config configs/parkinsons.yaml
uv run data-science stage-2 --config configs/parkinsons.yaml
uv run data-science final   --config configs/parkinsons.yaml
uv run data-science reproduce
```


## Expected outputs

`make reproduce` populates `artifacts/final/{parkinsons,covertype}/`:

```
artifacts/final/<dataset>/
├── stage_1_results.csv               # Preprocessing matrix CV scores
├── tuning_results.csv                # Stage-2 grid-search winners
├── final_metrics.json                # Held-out test metrics (chosen model)
├── confusion_matrix.png              # Held-out test confusion matrix
├── per_class_metrics.csv             # (Covertype only) per-class P/R/F1
├── feature_selection_sweep.csv       # SelectKBest sweep, all classifiers
├── feature_selection_per_class.csv   # Per-class F1 across SelectKBest k
├── pca_sweep.csv                     # (PD only) PCA n_components sweep
├── classifier_sweep_<model>.csv      # Per-classifier hyperparameter sweep (5 files)
└── plots/                            # All PNGs referenced by reports/report.tex
```

The LaTeX report at `reports/report.tex` reads its figures from
`artifacts/final/<dataset>/plots/` via `\graphicspath`. The compiled
`reports/report.pdf` (~30 pages) presents the comparative analysis with
tables, sweep figures, and confusion matrices.

## Repository layout

```
data-science/
├── README.md                         # This file
├── Makefile                          # Reproduction targets
├── pyproject.toml + uv.lock          # Dependencies (managed via uv)
│
├── configs/                          # Per-dataset YAML configs
│   ├── parkinsons.yaml
│   ├── covertype.yaml
│   ├── hyperparameters.yaml          # Reduced Stage-2 grids
│   └── hyperparameters_full.yaml     # Original (200–560 combos)
│
├── data/raw/                         # UCI datasets, populated by `make download`
│
├── artifacts/final/                  # Pipeline outputs, populated by `make reproduce`
│   ├── parkinsons/
│   └── covertype/
│
├── reports/
│   ├── report.tex                    # LaTeX source
│   └── report.pdf                    # Built by `make report`
│
├── src/data_science/                 # The installable package
│   ├── main.py                       # CLI entry point (subcommands)
│   ├── data/                         # Dataset loaders, splitters, samplers
│   ├── features/                     # Scaling, feature selection, PCA builders
│   ├── models/                       # Classifier registry + sklearn Pipelines
│   ├── experiments/
│   │   ├── _context.py               # Shared bootstrap (load + split + CV)
│   │   ├── stage_1.py                # Preprocessing matrix at defaults
│   │   ├── stage_2.py                # GridSearchCV per classifier
│   │   ├── sweeps.py                 # Preprocessing + hyperparameter sweeps
│   │   ├── final.py                  # Refit winner, evaluate on test
│   │   └── reproduce.py              # `reproduce_all()` chains the four
│   ├── evaluation/                   # Metrics + plots (incl. multi-metric sweep grids)
│   └── utils/                        # YAML config loader
│
└── tests/                            # pytest suite (smoke tests)
```



## Development

```bash
uv sync                 # install runtime + dev dependencies
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pytest -q        # run tests
```

CI (`.github/workflows/lint.yml`) runs `ruff check` and `ruff format --check`
on every push and pull request.

## Tech stack

Python 3.11+ · scikit-learn · XGBoost · pandas · NumPy · Matplotlib · uv

## Authors

Diogo Ramalho
